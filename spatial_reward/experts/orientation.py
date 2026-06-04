import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Tuple

from ..models.orientation import get_3angle


def evaluate_orientation(
    image: Image.Image,
    objects: Dict[str, List],
    metadata: Dict[str, Any],
    orientation_model=None,
    orientation_preprocess=None,
    device: str = "cuda",
) -> Tuple[bool, float]:
    """
    Evaluate object orientation by cropping detected objects and running Orient-Anything.

    Args:
        image: Original PIL Image
        objects: Detection results {class: [(bbox, score), ...]}
        metadata: Structured metadata with 'orientation' fields in 'include'
        orientation_model: DINOv2_MLP orientation model
        orientation_preprocess: AutoImageProcessor for DINOv2
        device: torch device string

    Returns:
        (correct, reward_score)
    """
    correct = True
    rewards = []

    for req in metadata.get("include", []):
        if "orientation" not in req:
            continue

        classname = req["class"]
        found_objects = objects.get(classname, [])

        if len(found_objects) == 0:
            print(f"[WARN] {classname} not detected, skip orientation check.")
            correct = False
            rewards.append(0.0)
            continue

        # Filter by confidence > 0.5
        high_conf_objects = [
            obj for obj in found_objects
            if (obj[1].item() if hasattr(obj[1], "item") else float(obj[1])) > 0.5
        ]

        if len(high_conf_objects) == 0:
            print(
                f"[WARN] {classname} detected but no result with confidence > 0.5, "
                f"skip orientation check."
            )
            correct = False
            rewards.append(0.0)
            continue

        # Take highest-confidence detection
        found_objects = sorted(high_conf_objects, key=lambda x: x[1], reverse=True)
        bbox = found_objects[0][0]
        x1, y1, x2, y2 = map(int, bbox[:4])

        crop_img = image.crop((x1, y1, x2, y2))

        # Run Orient-Anything model
        angles = get_3angle(crop_img, orientation_model, orientation_preprocess, device)
        azimuth, polar, rotation, confidence = map(float, angles)

        expected_ori = req["orientation"]

        def angular_distance(a: float, b: float) -> float:
            diff = abs(a - b) % 360
            return min(diff, 360 - diff)

        is_match = True
        if isinstance(expected_ori, list):
            expected_ori = expected_ori[0]

        if "facing forward" in expected_ori:
            is_match = angular_distance(azimuth, 0.0) <= 45.0
        elif "facing backward" in expected_ori:
            is_match = angular_distance(azimuth, 180.0) <= 45.0
        elif "facing left" in expected_ori:
            is_match = angular_distance(azimuth, 90.0) <= 45.0
        elif "facing right" in expected_ori:
            is_match = angular_distance(azimuth, 270.0) <= 45.0

        rewards.append(confidence if is_match else 0.0)
        if not is_match:
            correct = False

        print(
            f"[orientation-check] {classname}: azimuth={azimuth:.2f}, "
            f"conf={confidence:.2f}, match={is_match}"
        )

    reward_score = sum(rewards) / len(rewards) if rewards else 0.0
    return correct, reward_score
