# SpatialReward

A multi-dimensional reward model for evaluating spatial understanding in text-to-image generation. Given a generated image and a text prompt, SpatialReward automatically assesses whether the image correctly represents the described spatial relationships, object counts, orientations, depth ordering, and text content.

## Features

- **Object Counting** - Verify correct number of objects
- **2D Spatial Positioning** - Evaluate left/right/above/below relations
- **3D Depth Relations** - Assess in-front-of/behind/inside using depth estimation
- **Object Orientation** - Check facing direction (left/right/forward/backward)
- **Color Accuracy** - Verify object colors via CLIP classification
- **Text/OCR Verification** - Check text content on objects
- **Automatic Prompt Parsing** - Rule-based or optional LLM-powered (Qwen2.5-VL)

## Installation

```bash
# Core installation
pip install -e .

# With OCR support (for text evaluation)
pip install -e ".[ocr]"

# With QwenVL smart parser (requires ~14GB VRAM)
pip install -e ".[qwen]"

# Everything
pip install -e ".[all]"
```

## Quick Start

### Python API

```python
from spatial_reward import SpatialReward

# Initialize (models auto-download from HuggingFace on first use)
evaluator = SpatialReward(device="cuda")

# Evaluate a single image
scores, rewards, results = evaluator.evaluate(
    "path/to/image.png",
    "a red car to the left of a blue house"
)

print(f"Score: {scores[0]:.3f}")       # Continuous score [0, 1]
print(f"Correct: {results[0]['correct']}")  # Binary correctness
print(f"Task: {results[0]['tag']}")     # Detected task type
```

### Command Line

```bash
# Single image
python evaluate.py --image photo.png --prompt "a cat to the left of a dog"

# Batch evaluation from JSONL
python evaluate.py --input batch.jsonl --output results.json --json

# With options
python evaluate.py --image img.png --prompt "..." --device cpu --no-ocr --verbose
```


## Models

All models are automatically downloaded from HuggingFace on first use:

| Model | HuggingFace ID | Size | Purpose |
|-------|---------------|------|---------|
| GroundingDINO | `IDEA-Research/grounding-dino-base` | ~900MB | Object detection |
| DINOv2-Large | `facebook/dinov2-large` | ~1.2GB | Orientation backbone |
| Orient-Anything | `Viglong/Orient-Anything` | ~50MB | Orientation prediction |
| Depth-Anything-V2 | `depth-anything/Depth-Anything-V2-Small-hf` | ~100MB | Depth estimation |
| CLIP ViT-L-14 | OpenAI (via open_clip) | ~900MB | Color classification |
| PaddleOCR | Auto-download | ~150MB | Text recognition (optional) |
| Qwen2.5-VL-7B | User-provided path | ~14GB | Smart parsing (optional) |

## Configuration

### Environment Variables

```bash
export SPATIAL_REWARD_DEVICE=cuda
export SPATIAL_REWARD_GROUNDING_DINO=IDEA-Research/grounding-dino-base
export SPATIAL_REWARD_CACHE_DIR=/path/to/cache
export SPATIAL_REWARD_QWEN_MODEL=/path/to/qwen-model
export SPATIAL_REWARD_USE_OCR=true
export SPATIAL_REWARD_BOX_THRESHOLD=0.3
```

### Programmatic Configuration

```python
from spatial_reward import SpatialReward, SpatialRewardConfig

config = SpatialRewardConfig(
    device="cuda",
    use_ocr=True,
    use_clip_color=True,
    box_threshold=0.3,
    text_threshold=0.3,
    cache_dir="/path/to/cache",
    # qwen_model="/path/to/Qwen2.5-VL-7B-Instruct",  # Enable smart parsing
)

evaluator = SpatialReward(config=config)
```

## API Reference

### `SpatialReward(config=None, device=None, qwen_model_path=None)`

Main evaluator class.

**Methods:**
- `evaluate(images, prompts)` - Evaluate image-prompt pairs (auto-parses prompts)
- `evaluate_with_metadata(images, metadatas)` - Evaluate with pre-built metadata

**Returns:** `(scores, rewards, results)` where:
- `scores`: List of continuous scores [0, 1]
- `rewards`: List of binary values (1.0 if fully correct, 0.0 otherwise)
- `results`: List of dicts with `tag`, `prompt`, `correct`, `score`, `metadata`

### `SpatialRewardConfig`

Configuration dataclass with fields:
- `device`, `grounding_dino_model`, `depth_model`, `orient_repo`, `orient_ckpt`
- `dinov2_model`, `qwen_model`, `use_ocr`, `use_clip_color`, `soft_confidence`
- `box_threshold`, `text_threshold`, `max_objects`, `cache_dir`

### `PromptParser`

Rule-based parser: `parser.parse(prompt) -> metadata dict`

## Citation


```bibtex
@inproceedings{zhou2026spatialreward,
  title={SpatialReward: Verifiable Spatial Reward Modeling for Fine-Grained Spatial Consistency in Text-to-Image Generation},
  author={Zhou, Sashuai and Zhou, Qiang and Ma, Junpeng and Cao, Yue and Hu, Ruofan and Zhang, Ziang and Yang, Xiaoda and Wang, Zhibin and Song, Jun and Yu, Cheng and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={647--658},
  year={2026}
}
```


## License

Apache 2.0
