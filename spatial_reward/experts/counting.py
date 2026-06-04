import numpy as np
from PIL import Image
from typing import Dict, List, Any, Tuple

from ..utils.geometry import relative_position


def evaluate_reward(
    image: Image.Image,
    objects: Dict[str, List],
    metadata: Dict[str, Any],
    soft_confidence: bool = False,
    color_classifier_fn=None,
) -> Tuple[bool, float]:
    """
    Evaluate object counting, color, and basic position.

    Args:
        image: PIL Image
        objects: Detection results {class: [(bbox, score), ...]}
        metadata: Structured metadata with 'include' list
        soft_confidence: Whether to weight by detection confidence
        color_classifier_fn: Optional callable(image, bboxes, classname) -> list of colors

    Returns:
        (correct, reward_score)
    """
    correct = True
    rewards = []
    matched_groups = []

    for req in metadata.get("include", []):
        classname = req["class"]
        matched = True
        found_objects = objects.get(classname, [])

        # Count check
        score_count = int(req["count"] == len(found_objects))
        if soft_confidence:
            box_conf = np.mean([obj[1] for obj in found_objects]) if found_objects else 1.0
            score_count *= box_conf
        rewards.append(score_count)

        if len(found_objects) != req["count"]:
            correct = matched = False
            if "color" in req or "position" in req:
                rewards.append(0.0)
        else:
            # Color check
            if "color" in req:
                if color_classifier_fn is not None:
                    colors = color_classifier_fn(image, found_objects, classname)
                    score_color = int(req["count"] == colors.count(req["color"]))
                    rewards.append(score_color)
                    if colors.count(req["color"]) != req["count"]:
                        correct = matched = False
                else:
                    # Skip color check if no classifier available
                    rewards.append(1.0)

            # Position check
            if "position" in req and matched:
                expected_rel, target_group = req["position"]
                if target_group >= len(matched_groups) or matched_groups[target_group] is None:
                    correct = matched = False
                    rewards.append(0.0)
                else:
                    pos_matched = True
                    for obj in found_objects:
                        for tgt in matched_groups[target_group]:
                            true_rels = relative_position(obj, tgt)
                            if expected_rel not in true_rels:
                                pos_matched = False
                                break
                        if not pos_matched:
                            break
                    if not pos_matched:
                        correct = matched = False
                        rewards.append(0.0)
                    else:
                        rewards.append(1.0)

        matched_groups.append(found_objects if matched else None)

    reward = sum(rewards) / len(rewards) if rewards else 0
    return correct, reward
