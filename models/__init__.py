"""
models/__init__.py
Model registry for LatentTokenGuard.
"""

from models.phi3_vision_wrapper import Phi3VisionWrapper
from models.llava_wrapper import LLaVAWrapper

MODEL_REGISTRY = {
    "phi3": Phi3VisionWrapper,
    "llava": LLaVAWrapper,
}


def load_model(model_name: str, config: dict):
    """Factory function to load any supported LVLM wrapper."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](config)
