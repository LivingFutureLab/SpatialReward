"""
SpatialReward: Evaluate spatial understanding in text-to-image generation.
"""

from .evaluator import SpatialRewardEvaluator as SpatialReward
from .config import SpatialRewardConfig
from .prompt_parser import PromptParser, parse_prompt_to_metadata

__version__ = "0.1.0"
__all__ = [
    "SpatialReward",
    "SpatialRewardConfig",
    "PromptParser",
    "parse_prompt_to_metadata",
]
