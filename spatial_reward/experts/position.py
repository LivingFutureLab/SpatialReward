from PIL import Image
from typing import Dict, List, Any, Tuple

from ..utils.geometry import relative_position_2d


def evaluate_complex_positions(
    image: Image.Image,
    objects: Dict[str, List],
    metadata: Dict[str, Any],
) -> Tuple[bool, float]:
    """
    Evaluate multi-object 2D spatial position relations.

    Args:
        image: PIL Image
        objects: Detection results {class: [(bbox, score), ...]}
        metadata: Structured metadata with 'include' list containing 'position' fields

    Returns:
        (correct, reward_score)
    """
    correct = True
    rewards = []

    for i, req in enumerate(metadata.get("include", [])):
        classname = req["class"]
        found_objects = objects.get(classname, [])

        if len(found_objects) != req["count"]:
            correct = False
            rewards.append(0.0)
            continue

        if "position" not in req or not req["position"]:
            rewards.append(1.0)
            continue

        if len(req["position"]) < 2:
            rewards.append(1.0)
            continue

        expected_rel, target_group = req["position"]

        if target_group >= len(metadata["include"]):
            print(f"[WARN] target_group={target_group} out of range, skip.")
            rewards.append(0.0)
            correct = False
            continue

        target_class = metadata["include"][target_group]["class"]
        target_objects = objects.get(target_class, [])

        if not target_objects:
            print(f"[WARN] target class '{target_class}' not detected.")
            correct = False
            rewards.append(0.0)
            continue

        matched_rel = False
        for obj_bbox_conf in found_objects:
            obj_bbox = obj_bbox_conf[0]
            for tgt_bbox_conf in target_objects:
                tgt_bbox = tgt_bbox_conf[0]
                rels = relative_position_2d(obj_bbox, tgt_bbox)
                if expected_rel in rels:
                    matched_rel = True
                    break
            if matched_rel:
                break

        print(f"[complex-pos] {classname} '{expected_rel}' {target_class}: {matched_rel}")
        rewards.append(1.0 if matched_rel else 0.0)
        if not matched_rel:
            correct = False

    reward = sum(rewards) / len(rewards) if rewards else 0.0
    return correct, reward
