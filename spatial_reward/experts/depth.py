import numpy as np
from PIL import Image
from typing import Dict, List, Any, Tuple

from ..utils.geometry import mean_depth_in_bbox, is_inside_bbox


def evaluate_3d_relation(
    image: Image.Image,
    objects: Dict[str, List],
    metadata: Dict[str, Any],
    depth_pipe=None,
) -> Tuple[bool, float]:
    """
    Evaluate 3D spatial relations using Depth-Anything-V2.

    Note: Depth-Anything outputs disparity maps where:
        - Higher values = closer (in front of)
        - Lower values = farther (behind)

    Args:
        image: PIL Image
        objects: Detection results {class: [(bbox, score), ...]}
        metadata: Structured metadata with 'position3d' fields
        depth_pipe: HuggingFace depth estimation pipeline

    Returns:
        (correct, reward_score)
    """
    depth_output = depth_pipe(image)
    depth_map = np.array(depth_output["depth"])

    print(
        f"[3d-relation] depth_map shape={depth_map.shape}, "
        f"min={depth_map.min():.3f}, max={depth_map.max():.3f}, "
        f"mean={depth_map.mean():.3f}"
    )

    correct = True
    rewards = []

    for req_idx, req in enumerate(metadata.get("include", [])):
        classname = req["class"]
        found_objects = objects.get(classname, [])

        if len(found_objects) != req["count"]:
            print(
                f"[3d-relation] {classname}: expected {req['count']}, "
                f"found {len(found_objects)}, skip."
            )
            correct = False
            rewards.append(0.0)
            continue

        if "position3d" not in req or not req["position3d"]:
            rewards.append(1.0)
            continue

        if len(req["position3d"]) < 2:
            print(f"[WARN] position3d format error: {req['position3d']}")
            rewards.append(1.0)
            continue

        rel, target_group = req["position3d"]

        if target_group >= len(metadata["include"]):
            print(f"[WARN] target_group={target_group} out of range, skip.")
            continue

        # Take highest confidence detection
        found_objects_sorted = sorted(found_objects, key=lambda x: x[1], reverse=True)
        bbox = found_objects_sorted[0][0]

        target_class = metadata["include"][target_group]["class"]
        target_objects = objects.get(target_class, [])

        if not target_objects:
            print(f"[WARN] target class '{target_class}' not detected, skip.")
            correct = False
            rewards.append(0.0)
            continue

        target_objects_sorted = sorted(target_objects, key=lambda x: x[1], reverse=True)
        target_bbox = target_objects_sorted[0][0]

        disp_a = mean_depth_in_bbox(depth_map, bbox)
        disp_b = mean_depth_in_bbox(depth_map, target_bbox)

        print(
            f"[3d-relation] {classname}(disp={disp_a:.3f}) "
            f"{rel} {target_class}(disp={disp_b:.3f})"
        )

        if rel == "in front of":
            res = disp_a > disp_b
        elif rel == "behind":
            res = disp_a < disp_b
        elif rel == "inside":
            res = is_inside_bbox(bbox, target_bbox)
        else:
            print(f"[WARN] Unknown 3d relation: '{rel}'")
            res = False

        print(f"[3d-relation] result={res}")
        rewards.append(1.0 if res else 0.0)
        if not res:
            correct = False

    reward = sum(rewards) / len(rewards) if rewards else 0.0
    return correct, reward
