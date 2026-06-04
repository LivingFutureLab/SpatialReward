"""
Example: Evaluate a single image against a prompt.

Usage:
    python examples/evaluate_single.py
"""

from spatial_reward import SpatialReward, SpatialRewardConfig


def main():
    # Option 1: Default config (auto-downloads models from HuggingFace)
    evaluator = SpatialReward(device="cuda")

    # Option 2: Custom config
    # config = SpatialRewardConfig(
    #     device="cuda",
    #     use_ocr=False,          # Disable OCR if not needed
    #     use_clip_color=False,   # Disable color classification
    # )
    # evaluator = SpatialReward(config=config)

    # Evaluate a single image
    image_path = "path/to/your/image.png"
    prompt = "a red car to the left of a blue house"

    scores, rewards, results = evaluator.evaluate(image_path, prompt)

    print(f"Score: {scores[0]:.3f}")
    print(f"Correct: {results[0]['correct']}")
    print(f"Task type: {results[0]['tag']}")
    print(f"Details: {results[0]}")


if __name__ == "__main__":
    main()
