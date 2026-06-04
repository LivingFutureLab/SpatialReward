#!/usr/bin/env python3
"""
Unified Benchmark Runner for SpatialReward.

Auto-discovers GenEval-style datasets under a root directory,
evaluates each benchmark, and prints a consolidated summary table.

Usage:
    # Evaluate all benchmarks under a directory
    python run_benchmarks.py --datasets /path/to/datasets/

    # With custom config
    python run_benchmarks.py --datasets /path/to/datasets/ --device cuda \
        --qwen-model /path/to/Qwen2.5-VL-7B-Instruct --batch-size 4

    # Evaluate specific benchmarks only
    python run_benchmarks.py --datasets /path/to/datasets/ --benchmarks geneval geneval_3d

    # JSON output
    python run_benchmarks.py --datasets /path/to/datasets/ --output results.json --json
"""

import argparse
import json
import os
import sys
import glob
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from PIL import Image


# ============================================================
# Dataset Loader
# ============================================================

def load_all_samples(dataset_dir: str) -> List[Dict[str, Any]]:
    """
    Walk subfolders in a GenEval-style dataset directory.
    Each subfolder should contain: metadata.jsonl + samples/00000.png
    """
    samples = []
    subfolders = sorted(glob.glob(os.path.join(dataset_dir, "*/")))

    for folder in subfolders:
        folder_name = os.path.basename(folder.rstrip("/"))
        meta_path = os.path.join(folder, "metadata.jsonl")

        if not os.path.exists(meta_path):
            continue

        with open(meta_path, "r") as f:
            metadata = json.loads(f.read().strip())

        prompt = metadata.get("prompt", None)
        if prompt is None:
            continue

        image_path = os.path.join(folder, "samples", "00000.png")
        if not os.path.exists(image_path):
            continue

        samples.append({
            "folder": folder_name,
            "prompt": prompt,
            "image_path": image_path,
            "metadata": metadata,
        })

    return samples


# ============================================================
# Evaluation Runner
# ============================================================

def run_benchmark(samples: List[Dict], evaluator, batch_size: int = 8) -> List[Dict]:
    """Evaluate a single benchmark dataset in batches."""
    all_results = []
    total = len(samples)
    errors = 0

    for batch_start in range(0, total, batch_size):
        batch = samples[batch_start: batch_start + batch_size]
        batch_end = min(batch_start + batch_size, total)

        print(f"  [{batch_start+1:4d}-{batch_end:4d} / {total}]", end=" ... ", flush=True)

        images = []
        prompts = []
        valid_batch = []

        for sample in batch:
            try:
                img = Image.open(sample["image_path"]).convert("RGB")
                images.append(img)
                prompts.append(sample["prompt"])
                valid_batch.append(sample)
            except Exception as e:
                print(f"Failed to load {sample['image_path']}: {e}")
                continue

        if not images:
            print("no images")
            errors += 1
            continue

        try:
            scores, rewards, results = evaluator.evaluate(images, prompts)
        except Exception as e:
            print(f"Evaluation error: {e}")
            for sample in valid_batch:
                all_results.append({
                    "folder": sample["folder"],
                    "prompt": sample["prompt"],
                    "image_path": sample["image_path"],
                    "tag": sample["metadata"].get("tag", "unknown"),
                    "score": 0.0,
                    "reward": 0.0,
                    "correct": False,
                    "error": str(e),
                })
            errors += 1
            continue

        for sample, score, reward, result in zip(valid_batch, scores, rewards, results):
            all_results.append({
                "folder": sample["folder"],
                "prompt": sample["prompt"],
                "image_path": sample["image_path"],
                "tag": result.get("tag", sample["metadata"].get("tag", "unknown")),
                "score": round(score, 4),
                "reward": round(reward, 4),
                "correct": result.get("correct", False),
            })

        # Print progress for this batch
        batch_scores = [score for score, _ in zip(scores, valid_batch)]
        batch_correct = sum(1 for r in results if r.get("correct", False))
        avg_s = sum(batch_scores) / len(batch_scores) if batch_scores else 0
        print(f"score={avg_s:.3f} correct={batch_correct}/{len(results)}")

    if errors > 0:
        print(f"  ({errors} batch(es) had errors)")

    return all_results


# ============================================================
# Summary Helpers
# ============================================================

def compute_summary(results: List[Dict]) -> Dict[str, Any]:
    """Compute overall + per-tag statistics for a benchmark's results."""
    total = len(results)
    if total == 0:
        return {"n": 0}

    avg_score = sum(r["score"] for r in results) / total
    avg_reward = sum(r["reward"] for r in results) / total
    correct = sum(1 for r in results if r["correct"])

    # Per-tag breakdown
    tag_groups = {}
    for r in results:
        tag = r["tag"]
        tag_groups.setdefault(tag, []).append(r)

    per_tag = {}
    for tag, group in tag_groups.items():
        n = len(group)
        per_tag[tag] = {
            "n": n,
            "correct": sum(1 for r in group if r["correct"]),
            "avg_score": round(sum(r["score"] for r in group) / n, 4),
            "avg_reward": round(sum(r["reward"] for r in group) / n, 4),
        }

    return {
        "n": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "avg_score": round(avg_score, 4),
        "avg_reward": round(avg_reward, 4),
        "per_tag": per_tag,
    }


def print_summary_table(benchmark_results: Dict[str, Dict]) -> None:
    """Print a consolidated comparison table for all benchmarks."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Benchmark':<25} {'N':>5} {'Accuracy':>10} {'AvgScore':>10} {'AvgReward':>10}")
    print("-" * 70)

    for name, summary in sorted(benchmark_results.items()):
        n = summary.get("n", 0)
        if n == 0:
            print(f"{name:<25} {'0':>5} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            continue
        acc = f"{summary['accuracy'] * 100:.1f}%"
        score = f"{summary['avg_score']:.4f}"
        reward = f"{summary['avg_reward']:.4f}"
        print(f"{name:<25} {n:>5} {acc:>10} {score:>10} {reward:>10}")

    print("=" * 70)

    # Overall averages across all benchmarks
    total_n = sum(s["n"] for s in benchmark_results.values())
    if total_n > 0:
        overall_correct = sum(s.get("correct", 0) for s in benchmark_results.values())
        overall_avg_score = (
            sum(s["avg_score"] * s["n"] for s in benchmark_results.values()) / total_n
        )
        overall_avg_reward = (
            sum(s["avg_reward"] * s["n"] for s in benchmark_results.values()) / total_n
        )
        print(
            f"{'OVERALL':<25} {total_n:>5} "
            f"{overall_correct/total_n*100:>9.1f}% "
            f"{overall_avg_score:>10.4f} {overall_avg_reward:>10.4f}"
        )
        print("=" * 70)

    # Per-tag breakdown for each benchmark
    print("\n--- Per-Tag Breakdown ---")
    for name, summary in sorted(benchmark_results.items()):
        per_tag = summary.get("per_tag", {})
        if not per_tag:
            continue
        print(f"\n  [{name}]")
        for tag, stats in sorted(per_tag.items()):
            n = stats["n"]
            acc = f"{stats['correct']}/{n} ({stats['correct']/n*100:.0f}%)"
            print(f"    {tag:<22} n={n:>4}  accuracy={acc:<15} score={stats['avg_score']:.4f}")


# ============================================================
# Main
# ============================================================

def discover_benchmarks(datasets_dir: str, benchmarks: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Discover benchmark directories.

    If benchmarks list is provided, only those are used (matched by name).
    Otherwise, all subdirectories with samples are included.

    Returns:
        Dict mapping benchmark name -> directory path
    """
    subdirs = sorted(glob.glob(os.path.join(datasets_dir, "*/")))
    available = {}
    for d in subdirs:
        name = os.path.basename(d.rstrip("/"))
        # Verify it has at least one sample
        if glob.glob(os.path.join(d, "*/metadata.jsonl")):
            available[name] = d

    if benchmarks:
        result = {}
        for b in benchmarks:
            if b in available:
                result[b] = available[b]
            else:
                print(f"[WARN] Benchmark '{b}' not found in {datasets_dir}, skipping")
        return result
    return available


def main():
    parser = argparse.ArgumentParser(
        description="SpatialReward: Unified Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks
  python run_benchmarks.py --datasets /path/to/datasets/

  # Run specific benchmarks
  python run_benchmarks.py --datasets /path/to/datasets/ --benchmarks geneval geneval_3d

  # Save results as JSON
  python run_benchmarks.py --datasets /path/to/datasets/ --output results.json --json
        """,
    )

    parser.add_argument(
        "--datasets", type=str, required=True,
        help="Root directory containing GenEval-style benchmark subdirectories",
    )
    parser.add_argument(
        "--benchmarks", type=str, nargs="*",
        help="Specific benchmark names to run (default: all discovered)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--qwen-model", type=str, default=None, help="QwenVL model path")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR")
    parser.add_argument("--no-color", action="store_true", help="Disable CLIP color")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--json", action="store_true", help="Output detailed results as JSON")
    parser.add_argument("--verbose", action="store_true", help="Show detailed logs")

    args = parser.parse_args()

    if not args.verbose:
        import warnings
        warnings.filterwarnings("ignore")
        import logging
        logging.disable(logging.WARNING)

    # Discover benchmarks
    bench_map = discover_benchmarks(args.datasets, args.benchmarks)
    if not bench_map:
        print(f"[ERROR] No valid benchmarks found in {args.datasets}")
        sys.exit(1)

    print(f"Discovered {len(bench_map)} benchmark(s): {', '.join(bench_map.keys())}")

    # Initialize evaluator
    from spatial_reward import SpatialReward, SpatialRewardConfig
    config = SpatialRewardConfig(
        device=args.device,
        qwen_model=args.qwen_model,
        use_ocr=not args.no_ocr,
        use_clip_color=not args.no_color,
        cache_dir=args.cache_dir,
    )
    print("Initializing SpatialReward evaluator...")
    evaluator = SpatialReward(config=config)

    # Run each benchmark
    all_benchmark_results = {}  # name -> list of result dicts
    benchmark_summaries = {}    # name -> summary dict

    for name, bench_dir in bench_map.items():
        print(f"\n{'=' * 60}")
        print(f"BENCHMARK: {name}")
        print(f"  Directory: {bench_dir}")
        print(f"{'=' * 60}")

        samples = load_all_samples(bench_dir)
        if not samples:
            print(f"[SKIP] No valid samples in {name}")
            benchmark_summaries[name] = {"n": 0}
            continue

        print(f"  Found {len(samples)} samples, evaluating...")
        t0 = time.time()
        results = run_benchmark(samples, evaluator, batch_size=args.batch_size)
        elapsed = time.time() - t0

        summary = compute_summary(results)
        summary["elapsed_seconds"] = round(elapsed, 1)
        benchmark_summaries[name] = summary
        all_benchmark_results[name] = results

        # Print per-benchmark summary
        n = summary["n"]
        if n > 0:
            print(f"\n  [{name}] DONE in {elapsed:.1f}s")
            print(f"    Samples  : {n}")
            print(f"    Correct  : {summary['correct']}/{n} ({summary['accuracy']*100:.1f}%)")
            print(f"    AvgScore : {summary['avg_score']:.4f}")
            print(f"    AvgReward: {summary['avg_reward']:.4f}")

    # Print consolidated table
    print_summary_table(benchmark_summaries)

    # Save results
    if args.output or args.json:
        output_data = {
            "summaries": {
                name: summary for name, summary in benchmark_summaries.items()
            },
            "results": {
                name: results for name, results in all_benchmark_results.items()
            },
        }
        out_file = args.output or "benchmark_results.json"
        with open(out_file, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
