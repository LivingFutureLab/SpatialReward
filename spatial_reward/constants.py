from pathlib import Path

# HuggingFace model identifiers for DINOv2
DINO_SMALL = "facebook/dinov2-small"
DINO_BASE = "facebook/dinov2-base"
DINO_LARGE = "facebook/dinov2-large"
DINO_GIANT = "facebook/dinov2-giant"

# Color labels for CLIP color classification
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]

# Default thresholds
DEFAULT_BOX_THRESHOLD = 0.3
DEFAULT_TEXT_THRESHOLD = 0.3
DEFAULT_MAX_OBJECTS = 16
DEFAULT_POSITION_THRESHOLD = 0.1

# Assets path (resolved relative to package root)
ASSETS_DIR = Path(__file__).parent.parent / "assets"
