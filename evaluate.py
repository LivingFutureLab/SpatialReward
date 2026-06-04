#!/usr/bin/env python3
"""
SpatialReward - Quick evaluation script.

Usage:
    python evaluate.py --image photo.png --prompt "a cat to the left of a dog"
    python evaluate.py --input batch.jsonl --output results.json --json
"""
from spatial_reward.cli import main

if __name__ == "__main__":
    main()
