"""
Command-line interface for SpatialReward.
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="SpatialReward: Evaluate spatial understanding in generated images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image evaluation
  spatial-reward --image photo.png --prompt "a red car to the left of a house"

  # Batch evaluation from JSONL
  spatial-reward --input batch.jsonl --output results.json

  # With custom model paths
  spatial-reward --image img.png --prompt "..." --device cpu --no-ocr
        """,
    )

    # Input options
    input_group = parser.add_argument_group("Input")
    input_group.add_argument("--image", type=str, help="Path to a single image file")
    input_group.add_argument("--prompt", type=str, help="Text prompt (required with --image)")
    input_group.add_argument(
        "--input", type=str,
        help='Path to JSONL file (each line: {"image": "...", "prompt": "..."})',
    )

    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output", type=str, help="Output file path (default: stdout)")
    output_group.add_argument("--json", action="store_true", help="Output as JSON")
    output_group.add_argument("--verbose", action="store_true", help="Show detailed logs")

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--device", type=str, default=None, help="Device (default: cuda)")
    model_group.add_argument(
        "--grounding-dino-model", type=str, default=None,
        help="HF ID or local path for GroundingDINO",
    )
    model_group.add_argument(
        "--depth-model", type=str, default=None,
        help="HF ID or local path for depth model",
    )
    model_group.add_argument(
        "--qwen-model", type=str, default=None,
        help="Path to Qwen2.5-VL model (enables smart parsing)",
    )
    model_group.add_argument("--no-ocr", action="store_true", help="Disable OCR evaluation")
    model_group.add_argument("--no-color", action="store_true", help="Disable CLIP color check")
    model_group.add_argument("--cache-dir", type=str, default=None, help="HuggingFace cache dir")
    model_group.add_argument(
        "--box-threshold", type=float, default=None, help="Detection box threshold",
    )
    model_group.add_argument(
        "--text-threshold", type=float, default=None, help="Detection text threshold",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.image and not args.input:
        parser.error("Either --image or --input is required")
    if args.image and not args.prompt:
        parser.error("--prompt is required when using --image")

    # Suppress logs unless verbose
    if not args.verbose:
        import warnings
        warnings.filterwarnings("ignore")
        import logging
        logging.disable(logging.WARNING)

    # Build config
    from .config import SpatialRewardConfig
    config = SpatialRewardConfig.from_env()

    if args.device:
        config.device = args.device
    if args.grounding_dino_model:
        config.grounding_dino_model = args.grounding_dino_model
    if args.depth_model:
        config.depth_model = args.depth_model
    if args.qwen_model:
        config.qwen_model = args.qwen_model
    if args.no_ocr:
        config.use_ocr = False
    if args.no_color:
        config.use_clip_color = False
    if args.cache_dir:
        config.cache_dir = args.cache_dir
    if args.box_threshold is not None:
        config.box_threshold = args.box_threshold
    if args.text_threshold is not None:
        config.text_threshold = args.text_threshold

    # Initialize evaluator
    from .evaluator import SpatialRewardEvaluator
    evaluator = SpatialRewardEvaluator(config=config)

    # Run evaluation
    results = []

    if args.image:
        # Single image mode
        scores, rewards, res = evaluator.evaluate(args.image, args.prompt)
        results = res
    else:
        # Batch mode from JSONL
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {args.input}", file=sys.stderr)
            sys.exit(1)

        images = []
        prompts = []
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                images.append(entry["image"])
                prompts.append(entry["prompt"])

        if not images:
            print("Error: No valid entries in input file", file=sys.stderr)
            sys.exit(1)

        # Process in batches
        batch_size = 8
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_prompts = prompts[i:i + batch_size]
            _, _, batch_results = evaluator.evaluate(batch_images, batch_prompts)
            results.extend(batch_results)

    # Output results
    output_lines = []
    for r in results:
        if args.json:
            output_lines.append(json.dumps(r, ensure_ascii=False))
        else:
            output_lines.append(
                f"[{r['tag']}] score={r['score']:.3f} correct={r['correct']} "
                f"prompt=\"{r['prompt'][:60]}...\""
                if len(r['prompt']) > 60
                else f"[{r['tag']}] score={r['score']:.3f} correct={r['correct']} "
                     f"prompt=\"{r['prompt']}\""
            )

    output_text = "\n".join(output_lines) + "\n"

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text)
        print(f"Results saved to: {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output_text)

    # Print summary for batch mode
    if len(results) > 1:
        avg_score = sum(r["score"] for r in results) / len(results)
        correct_count = sum(1 for r in results if r["correct"])
        print(
            f"\nSummary: {correct_count}/{len(results)} correct, "
            f"avg_score={avg_score:.4f}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
