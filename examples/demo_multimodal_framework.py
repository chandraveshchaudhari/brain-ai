"""Demo: registry-driven time-series backends."""

import sys

import pandas as pd

# Import the core architecture - this self-registers backends
from brain_automl.model_zoo.time_series_ai import TimeSeriesAutoML
from brain_automl.core import BACKEND_REGISTRY


def main():
    """Run a simple demo."""
    print("=" * 60)
    print("Brain-AI Multimodal Framework Demo")
    print("=" * 60)

    # Show registered backends
    print("\n1. Available time-series backends:")
    print("-" * 40)
    registered = BACKEND_REGISTRY.items()
    for name, backend_cls in registered.items():
        available = "✓ Available" if backend_cls.is_available() else "✗ Not installed"
        print(f"   {name:<30} {available}")

    # Create a simple synthetic dataset
    print("\n2. Creating synthetic time-series data:")
    print("-" * 40)
    import numpy as np

    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    data = pd.DataFrame({
        "date": dates,
        "value": np.sin(np.arange(100) * 0.1) + np.random.normal(0, 0.1, 100),
    })
    print(f"   Dataset shape: {data.shape}")
    print(f"   Date range: {data['date'].min()} to {data['date'].max()}")

    # Run TimeSeriesAutoML with available backends
    print("\n3. Running TimeSeriesAutoML:")
    print("-" * 40)
    executor = TimeSeriesAutoML()

    try:
        results = executor.run(data, task="forecasting")
        print(f"   Ran {len(results)} backend(s)")
        for result in results:
            print(f"   - {result.backend}: OK")
    except Exception as e:
        print(f"   Error: {e}")
        return 1

    print("\n" + "=" * 60)
    print("Demo complete. Architecture is working!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
