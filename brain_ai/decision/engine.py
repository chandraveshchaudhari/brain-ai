from itertools import product
from typing import Any, Dict, Iterable, List, Optional

from .spec import PipelineSpec
from ..fusion.early import EarlyFusion
from ..fusion.intermediate import IntermediateFusion
from ..fusion.late import LateFusion
from ..granularity.attention import AttentionGranularity
from ..granularity.pooling import PoolingGranularity
from ..granularity.resample import ResampleGranularity
from ..models.adapters.autogluon import AutoGluonAdapter
from ..models.adapters.sklearn import SKLearnAdapter
from ..models.adapters.tpot import TPOTAdapter


def generate_pipeline_combinations(
    modalities: List[str],
    granularity_strategies: Iterable[str],
    fusion_strategies: Iterable[str],
    model_backends: Iterable[str],
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> List[PipelineSpec]:
    """Generate a Cartesian product of pipeline specifications."""
    hparams = hyperparameters or {}
    specs: List[PipelineSpec] = []
    for granularity, fusion, backend in product(granularity_strategies, fusion_strategies, model_backends):
        specs.append(
            PipelineSpec(
                modalities=list(modalities),
                granularity_strategy=granularity,
                fusion_strategy=fusion,
                model_backend=backend,
                hyperparameters=dict(hparams),
            )
        )
    return specs


class DecisionEngine:
    """Decision layer that resolves a spec to runnable components."""

    def __init__(self, granularity_registry=None, fusion_registry=None, model_registry=None):
        self.granularity_registry = granularity_registry or {
            "resample": ResampleGranularity,
            "pooling": PoolingGranularity,
            "attention": AttentionGranularity,
        }
        self.fusion_registry = fusion_registry or {
            "early": EarlyFusion,
            "late": LateFusion,
            "intermediate": IntermediateFusion,
        }
        self.model_registry = model_registry or {
            "sklearn": SKLearnAdapter,
            "autogluon": AutoGluonAdapter,
            "tpot": TPOTAdapter,
        }

    def resolve_components(self, spec: PipelineSpec) -> Dict[str, Any]:
        granularity_cls = self.granularity_registry.get(spec.granularity_strategy)
        fusion_cls = self.fusion_registry.get(spec.fusion_strategy)
        model_cls = self.model_registry.get(spec.model_backend)

        if granularity_cls is None:
            raise KeyError(f"Unknown granularity strategy: {spec.granularity_strategy}")
        if fusion_cls is None:
            raise KeyError(f"Unknown fusion strategy: {spec.fusion_strategy}")
        if model_cls is None:
            raise KeyError(f"Unknown model backend: {spec.model_backend}")

        return {
            "granularity": granularity_cls(),
            "fusion": fusion_cls(),
            "model_adapter": self._build_model_adapter(model_cls, spec.hyperparameters),
        }

    @staticmethod
    def _build_model_adapter(model_cls, hyperparameters: Dict[str, Any]):
        # Keep adapters thin: inject backend object if provided, otherwise use default lightweight model.
        if model_cls is SKLearnAdapter:
            from sklearn.ensemble import RandomForestRegressor

            estimator = hyperparameters.get("estimator")
            if estimator is None:
                rf_params = {
                    "n_estimators": hyperparameters.get("n_estimators", 20),
                    "random_state": hyperparameters.get("random_state", 42),
                }
                estimator = RandomForestRegressor(**rf_params)
            return model_cls(estimator=estimator)

        if model_cls is AutoGluonAdapter:
            return model_cls(predictor=hyperparameters.get("predictor"))

        if model_cls is TPOTAdapter:
            return model_cls(tpot_estimator=hyperparameters.get("tpot_estimator"))

        return model_cls()
