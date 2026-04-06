"""Time-series backend adapters.

Importing this package registers available adapters in BACKEND_REGISTRY.
"""

from brain_automl.model_zoo.time_series_ai.backends import (  # noqa: F401
    autogluon_timeseries,
    chronos_timeseries,
    flaml_timeseries,
    h2o_timeseries,
    optuna_tuner,
    pycaret_timeseries,
    statsforecast_timeseries,
)
