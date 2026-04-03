"""NeuralForecast backend adapter."""

from __future__ import annotations

import faulthandler
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from lightning.pytorch.callbacks import Callback, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from neuralforecast import NeuralForecast
from neuralforecast.models import MLP, NBEATS, NHITS

from brain_automl.core.protocols import BaseLibraryBackend
from brain_automl.core.registry import BACKEND_REGISTRY
from brain_automl.model_zoo.time_series_ai.data_preparation import to_neuralforecast_format


class _NeuralForecastTrainingCallback(Callback):
    """Bridge Lightning events into Brain-AI logs and notebook output."""

    def __init__(self, run_logger: logging.Logger) -> None:
        self.run_logger = run_logger

    def on_fit_start(self, trainer, pl_module) -> None:  # type: ignore[no-untyped-def]
        self.run_logger.info(
            "[neuralforecast] Lightning fit started | max_epochs=%s max_steps=%s",
            getattr(trainer, "max_epochs", None),
            getattr(trainer, "max_steps", None),
        )

    def on_train_epoch_start(self, trainer, pl_module) -> None:  # type: ignore[no-untyped-def]
        epoch_idx = int(getattr(trainer, "current_epoch", 0)) + 1
        self.run_logger.info("[neuralforecast] Epoch %s started", epoch_idx)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # type: ignore[no-untyped-def]
        global_step = int(getattr(trainer, "global_step", 0))
        max_steps = getattr(trainer, "max_steps", None)
        if max_steps is not None and int(max_steps) > 0:
            self.run_logger.info(
                "[neuralforecast] Train step %s/%s completed",
                global_step,
                int(max_steps),
            )
        else:
            self.run_logger.info("[neuralforecast] Train step %s completed", global_step)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[no-untyped-def]
        metrics = getattr(trainer, "callback_metrics", {}) or {}
        val_loss = metrics.get("ptl/val_loss")
        if val_loss is not None:
            try:
                val_loss = float(val_loss)
                self.run_logger.info("[neuralforecast] Validation loss: %.6f", val_loss)
            except Exception:
                self.run_logger.info("[neuralforecast] Validation loss: %s", val_loss)

    def on_fit_end(self, trainer, pl_module) -> None:  # type: ignore[no-untyped-def]
        self.run_logger.info("[neuralforecast] Lightning fit finished")

    def on_exception(self, trainer, pl_module, exception) -> None:  # type: ignore[no-untyped-def]
        self.run_logger.error("[neuralforecast] Training exception: %s", exception)


class _HeartbeatLogger:
    """Emit periodic progress messages while a library call is running."""

    def __init__(self, run_logger: logging.Logger, label: str, interval_seconds: int = 15) -> None:
        self.run_logger = run_logger
        self.label = label
        self.interval_seconds = max(5, int(interval_seconds))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0

    def start(self) -> None:
        self._started_at = time.perf_counter()
        self.run_logger.info("[neuralforecast] %s started", self.label)
        self._thread = threading.Thread(target=self._run, name=f"{self.label}_heartbeat", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            elapsed = time.perf_counter() - self._started_at
            self.run_logger.info(
                "[neuralforecast] %s still running... elapsed=%.1fs",
                self.label,
                elapsed,
            )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        elapsed = time.perf_counter() - self._started_at
        self.run_logger.info("[neuralforecast] %s finished in %.1fs", self.label, elapsed)


def _resolve_model_name(raw_model_name: Any) -> str:
    model_name = str(raw_model_name or "MLP").strip().upper()
    supported = {"MLP", "NBEATS", "NHITS"}
    if model_name not in supported:
        raise ValueError(f"Unsupported neuralforecast model '{raw_model_name}'. Supported: {sorted(supported)}")
    return model_name


def _build_neuralforecast_model(
    model_name: str,
    prediction_length: int,
    input_size: int,
    max_steps: int,
    callbacks: list[Callback],
    csv_logger: CSVLogger,
    kwargs: dict[str, Any],
) -> Any:
    dataloader_kwargs = dict(kwargs.get("dataloader_kwargs") or {})
    dataloader_kwargs.setdefault("num_workers", 0)
    dataloader_kwargs.setdefault("persistent_workers", False)
    common_kwargs = {
        "h": prediction_length,
        "input_size": input_size,
        "max_steps": max_steps,
        "val_check_steps": 1,
        "enable_progress_bar": True,
        "logger": csv_logger,
        "enable_checkpointing": False,
        "log_every_n_steps": 1,
        "callbacks": callbacks,
        "accelerator": kwargs.get("accelerator", "cpu"),
        "devices": kwargs.get("devices", 1),
        "precision": kwargs.get("precision", "32-true"),
        "num_sanity_val_steps": int(kwargs.get("num_sanity_val_steps", 0)),
        "enable_model_summary": bool(kwargs.get("enable_model_summary", False)),
        "deterministic": bool(kwargs.get("deterministic", True)),
        "dataloader_kwargs": dataloader_kwargs,
    }
    if model_name == "MLP":
        return MLP(
            num_layers=int(kwargs.get("num_layers", 2)),
            hidden_size=int(kwargs.get("hidden_size", 256)),
            batch_size=int(kwargs.get("batch_size", 32)),
            windows_batch_size=int(kwargs.get("windows_batch_size", 256)),
            **common_kwargs,
        )
    if model_name == "NBEATS":
        return NBEATS(**common_kwargs)
    return NHITS(**common_kwargs)


@BACKEND_REGISTRY.register("neuralforecast")
class NeuralForecastBackend(BaseLibraryBackend):
    name = "neuralforecast"
    modality = "time_series"
    task_types = ("forecasting",)

    @classmethod
    def is_available(cls) -> bool:
        try:
            import neuralforecast  # noqa: F401

            return True
        except Exception:
            return False

    def fit(self, x_train: Any, y_train: Any = None, **kwargs: Any) -> Any:
        prediction_length = int(kwargs.get("prediction_length") or kwargs.get("horizon") or 14)
        frequency = kwargs.get("frequency", "B")
        model_name = _resolve_model_name(kwargs.get("neuralforecast_model") or kwargs.get("model_name"))
        max_steps = int(kwargs.get("max_steps", 3))
        input_size = int(kwargs.get("input_size", min(max(24, prediction_length // 2), 64)))
        max_allowed_val_size = max(1, len(x_train) - 1)
        val_size = int(kwargs.get("val_size", min(max(8, prediction_length // 6), 32, max_allowed_val_size)))
        output_dir = Path(kwargs.get("output_dir") or self.name)
        run_logger = kwargs.get("run_logger") or logging.getLogger(__name__)
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_log_dir = output_dir / "lightning_logs"
        fault_log_path = output_dir / "python_fault.log"

        train_data = to_neuralforecast_format(x_train)
        run_logger.info(
            "[neuralforecast] Preparing training | model=%s rows=%s unique_ids=%s freq=%s prediction_length=%s input_size=%s val_size=%s max_steps=%s accelerator=%s devices=%s",
            model_name,
            len(train_data),
            train_data["unique_id"].nunique(),
            frequency,
            prediction_length,
            input_size,
            val_size,
            max_steps,
            kwargs.get("accelerator", "cpu"),
            kwargs.get("devices", 1),
        )
        (output_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "backend": self.name,
                    "model_name": model_name,
                    "frequency": frequency,
                    "prediction_length": prediction_length,
                    "input_size": input_size,
                    "val_size": val_size,
                    "max_steps": max_steps,
                    "rows": int(len(train_data)),
                    "unique_ids": int(train_data["unique_id"].nunique()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        csv_logger = CSVLogger(save_dir=str(csv_log_dir), name=model_name.lower())
        callbacks = [
            TQDMProgressBar(refresh_rate=1),
            _NeuralForecastTrainingCallback(run_logger),
        ]
        run_logger.info("[neuralforecast] Constructing %s model", model_name)
        model = _build_neuralforecast_model(
            model_name=model_name,
            prediction_length=prediction_length,
            input_size=input_size,
            max_steps=max_steps,
            callbacks=callbacks,
            csv_logger=csv_logger,
            kwargs=kwargs,
        )
        run_logger.info("[neuralforecast] %s model constructed", model_name)
        run_logger.info("[neuralforecast] Constructing NeuralForecast wrapper")
        forecaster = NeuralForecast(models=[model], freq=frequency)
        run_logger.info("[neuralforecast] NeuralForecast wrapper constructed")
        run_logger.info("[neuralforecast] Starting forecaster.fit()")
        fit_heartbeat = _HeartbeatLogger(run_logger, "forecaster.fit", interval_seconds=15)
        fit_heartbeat.start()
        fault_log_handle = fault_log_path.open("a", encoding="utf-8")
        try:
            fault_log_handle.write("\n=== NeuralForecast fit trace start ===\n")
            fault_log_handle.flush()
            faulthandler.enable(file=fault_log_handle, all_threads=True)
            faulthandler.dump_traceback_later(30, repeat=True, file=fault_log_handle)
            forecaster.fit(df=train_data, val_size=val_size)
        finally:
            faulthandler.cancel_dump_traceback_later()
            try:
                faulthandler.disable()
            except Exception:
                pass
            fault_log_handle.flush()
            fault_log_handle.close()
            fit_heartbeat.stop()
        run_logger.info(
            "[neuralforecast] Fit completed | artifacts=%s",
            csv_log_dir,
        )
        return {
            "backend": self.name,
            "forecaster": forecaster,
            "prediction_length": prediction_length,
            "model_name": model_name,
            "output_dir": str(output_dir),
        }

    def predict(self, model: Any, x_test: Any, **kwargs: Any) -> Any:
        run_logger = kwargs.get("run_logger") or logging.getLogger(__name__)
        output_dir = Path(model.get("output_dir") or kwargs.get("output_dir") or self.name)
        fault_log_path = output_dir / "python_fault.log"
        run_logger.info("[neuralforecast] Starting prediction | rows=%s", len(x_test))
        predict_heartbeat = _HeartbeatLogger(run_logger, "forecaster.predict", interval_seconds=15)
        predict_heartbeat.start()
        fault_log_handle = fault_log_path.open("a", encoding="utf-8")
        try:
            fault_log_handle.write("\n=== NeuralForecast predict trace start ===\n")
            fault_log_handle.flush()
            faulthandler.enable(file=fault_log_handle, all_threads=True)
            faulthandler.dump_traceback_later(30, repeat=True, file=fault_log_handle)
            forecast_df = model["forecaster"].predict()
        finally:
            faulthandler.cancel_dump_traceback_later()
            try:
                faulthandler.disable()
            except Exception:
                pass
            fault_log_handle.flush()
            fault_log_handle.close()
            predict_heartbeat.stop()
        model_name = model["model_name"]
        run_logger.info("[neuralforecast] Prediction completed | forecast_rows=%s", len(forecast_df))
        return forecast_df[["unique_id", "ds", model_name]].rename(columns={model_name: "prediction"})