from typing import Dict

import numpy as np


def generate_multimodal_regression_data(
    n_samples: int = 120,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """Create deterministic multimodal data for integration tests and examples."""

    rng = np.random.default_rng(random_state)

    tabular = rng.normal(size=(n_samples, 4))
    sensor = rng.normal(size=(n_samples, 3))
    text_embed = rng.normal(size=(n_samples, 5))

    # Build target with modality-specific effects and interactions.
    y = (
        0.5 * tabular[:, 0]
        - 0.8 * tabular[:, 2]
        + 0.7 * sensor[:, 1]
        + 0.3 * text_embed[:, 3]
        + 0.5 * tabular[:, 1] * sensor[:, 0]
        + rng.normal(scale=0.1, size=n_samples)
    )

    return {
        "modalities": {
            "tabular": tabular,
            "sensor": sensor,
            "text": text_embed,
        },
        "y": y,
    }
