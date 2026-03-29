"""
data/__init__.py
Dataset loading utilities for LatentTokenGuard.
"""

from data.pope_loader import POPELoader
from data.mmhal_loader import MMHalLoader
from data.coco_indoor_loader import COCOIndoorLoader

DATASET_REGISTRY = {
    "pope": POPELoader,
    "mmhal": MMHalLoader,
    "coco_indoor": COCOIndoorLoader,
}


def load_dataset(name: str, config: dict, **kwargs):
    """Factory to load any supported dataset."""
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name](config, **kwargs)
