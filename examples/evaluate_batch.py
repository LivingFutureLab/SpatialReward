"""
Example: Batch evaluation from a JSONL file.

Usage:
    python examples/evaluate_batch.py --input examples/sample_prompts.jsonl
"""

import argparse
import json
from pathlib import Path
from spatial_reward import SpatialReward


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation example")
    parser.add_argument("--input", type=str, required=True, help="JSONL file path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    # Load data
    images = []
    prompts = []
    with open(args.input) as f:
        for line in f:
            entry = json.loads(line.strip())
            images.append(entry["image"])
            prompts.append(entry["prompt"])

    print(f"Loaded {len(images)} samples")

    # Initialize evaluator
    evaluator = SpatialReward(device=args.device)

    # Process in batches
    all_results = []
    for i in range(0, len(images), args.batch_size):
        batch_images = images[i:i + args.batch_size]
        batch_prompts = prompts[i:i + args.batch_size]

        scores, rewards, results = evaluator.evaluate(batch_images, batch_prompts)
        all_results.extend(results)

        print(f"  Batch {i // args.batch_size + 1}: "
              f"avg_score={sum(scores)/len(scores):.3f}")

    # Summary
    avg_score = sum(r["score"] for r in all_results) / len(all_results)
    correct = sum(1 for r in all_results if r["correct"])
    print(f"\nResults: {correct}/{len(all_results)} correct, avg_score={avg_score:.4f}")

    # Per-tag breakdown
    tag_groups = {}
    for r in all_results:
        tag_groups.setdefault(r["tag"], []).append(r)

    print("\nPer-tag breakdown:")
    for tag, group in sorted(tag_groups.items()):
        n = len(group)
        avg_s = sum(r["score"] for r in group) / n
        cor = sum(1 for r in group if r["correct"])
        print(f"  [{tag}] n={n}, correct={cor}/{n}, avg_score={avg_s:.3f}")


if __name__ == "__main__":
    main()
