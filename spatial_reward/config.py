import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SpatialRewardConfig:
    """Configuration for SpatialReward evaluator."""

    device: str = "cuda"

    # Model identifiers (HuggingFace repo IDs or local paths)
    grounding_dino_model: str = "IDEA-Research/grounding-dino-base"
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    orient_repo: str = "Viglong/Orient-Anything"
    orient_ckpt: str = "croplargeEX2/dino_weight.pt"
    dinov2_model: str = "facebook/dinov2-large"
    qwen_model: Optional[str] = None  # None = QwenVL parser disabled

    # Feature toggles
    use_ocr: bool = True
    use_clip_color: bool = True
    soft_confidence: bool = False

    # Detection thresholds
    box_threshold: float = 0.3
    text_threshold: float = 0.3
    max_objects: int = 16
    position_threshold: float = 0.1

    # Cache
    cache_dir: Optional[str] = None  # None = HuggingFace default (~/.cache/huggingface/)

    @classmethod
    def from_env(cls) -> "SpatialRewardConfig":
        """Create config with overrides from SPATIAL_REWARD_* environment variables."""
        kwargs = {}
        env_map = {
            "SPATIAL_REWARD_DEVICE": "device",
            "SPATIAL_REWARD_GROUNDING_DINO": "grounding_dino_model",
            "SPATIAL_REWARD_DEPTH_MODEL": "depth_model",
            "SPATIAL_REWARD_ORIENT_REPO": "orient_repo",
            "SPATIAL_REWARD_ORIENT_CKPT": "orient_ckpt",
            "SPATIAL_REWARD_DINOV2_MODEL": "dinov2_model",
            "SPATIAL_REWARD_QWEN_MODEL": "qwen_model",
            "SPATIAL_REWARD_CACHE_DIR": "cache_dir",
        }
        for env_var, field_name in env_map.items():
            val = os.environ.get(env_var)
            if val is not None:
                kwargs[field_name] = val

        bool_map = {
            "SPATIAL_REWARD_USE_OCR": "use_ocr",
            "SPATIAL_REWARD_USE_CLIP_COLOR": "use_clip_color",
            "SPATIAL_REWARD_SOFT_CONFIDENCE": "soft_confidence",
        }
        for env_var, field_name in bool_map.items():
            val = os.environ.get(env_var)
            if val is not None:
                kwargs[field_name] = val.lower() in ("1", "true", "yes")

        float_map = {
            "SPATIAL_REWARD_BOX_THRESHOLD": "box_threshold",
            "SPATIAL_REWARD_TEXT_THRESHOLD": "text_threshold",
        }
        for env_var, field_name in float_map.items():
            val = os.environ.get(env_var)
            if val is not None:
                kwargs[field_name] = float(val)

        return cls(**kwargs)
