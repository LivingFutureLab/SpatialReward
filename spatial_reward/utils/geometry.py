import numpy as np
from typing import Set, Tuple

from ..config import SpatialRewardConfig


POSITION_THRESHOLD = 0.1


def relative_position(obj_a, obj_b, position_threshold: float = POSITION_THRESHOLD) -> Set[str]:
    """
    Compute 2D spatial relations between two detected objects using offset-based logic.

    Args:
        obj_a: Tuple of (bbox, score) for object A
        obj_b: Tuple of (bbox, score) for object B
        position_threshold: Threshold for offset normalization

    Returns:
        Set of relation strings (e.g. {"left of", "above"})
    """
    boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    revised_offset = np.maximum(
        np.abs(offset) - position_threshold * (dim_a + dim_b), 0
    ) * np.sign(offset)

    if np.all(np.abs(revised_offset) < 1e-3):
        return set()

    dx, dy = revised_offset / np.linalg.norm(offset)
    relations = set()
    if dx < -0.5:
        relations.add("left of")
    if dx > 0.5:
        relations.add("right of")
    if dy < -0.5:
        relations.add("above")
    if dy > 0.5:
        relations.add("below")
    return relations


def relative_position_2d(bbox_a, bbox_b) -> Set[str]:
    """
    Simple center-based 2D spatial relation between two bounding boxes.

    Args:
        bbox_a: [x1, y1, x2, y2]
        bbox_b: [x1, y1, x2, y2]

    Returns:
        Set of relation strings
    """
    xa = (bbox_a[0] + bbox_a[2]) / 2
    ya = (bbox_a[1] + bbox_a[3]) / 2
    xb = (bbox_b[0] + bbox_b[2]) / 2
    yb = (bbox_b[1] + bbox_b[3]) / 2
    rels = set()
    if xa < xb:
        rels.add("left of")
    elif xa > xb:
        rels.add("right of")
    if ya < yb:
        rels.add("above")
    elif ya > yb:
        rels.add("below")
    return rels


def mean_depth_in_bbox(depth_map: np.ndarray, bbox) -> float:
    """
    Compute the mean depth value within a bounding box region.

    Args:
        depth_map: 2D depth array (H, W)
        bbox: [x1, y1, x2, y2]

    Returns:
        Mean depth value (float('inf') if region is empty)
    """
    x1, y1, x2, y2 = map(int, bbox[:4])
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, depth_map.shape[1] - 1)
    y2 = min(y2, depth_map.shape[0] - 1)
    region = depth_map[y1:y2, x1:x2]
    return float(np.mean(region)) if region.size > 0 else float("inf")


def is_inside_bbox(bbox_a, bbox_b, iou_threshold: float = 0.8) -> bool:
    """
    Check if bbox_a is inside bbox_b by computing overlap ratio.

    Args:
        bbox_a: The potentially contained box [x1, y1, x2, y2]
        bbox_b: The container box [x1, y1, x2, y2]
        iou_threshold: Minimum overlap ratio of bbox_a within bbox_b

    Returns:
        True if bbox_a is sufficiently inside bbox_b
    """
    inter_x1 = max(bbox_a[0], bbox_b[0])
    inter_y1 = max(bbox_a[1], bbox_b[1])
    inter_x2 = min(bbox_a[2], bbox_b[2])
    inter_y2 = min(bbox_a[3], bbox_b[3])

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return False

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])

    if area_a <= 0:
        return False

    overlap_ratio = inter_area / area_a
    return overlap_ratio >= iou_threshold


def compute_iou_boxB(box_a, box_b) -> float:
    """
    Compute intersection area / box_b area.
    Used for checking if an OCR box falls within a detection box.

    Args:
        box_a: [x1, y1, x2, y2]
        box_b: [x1, y1, x2, y2]

    Returns:
        Overlap ratio relative to box_b's area
    """
    def area_fn(box):
        return max(0, box[2] - box[0] + 1) * max(0, box[3] - box[1] + 1)

    inter_area = area_fn([
        max(box_a[0], box_b[0]),
        max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]),
        min(box_a[3], box_b[3]),
    ])
    area_b = area_fn(box_b)
    return inter_area / area_b if area_b > 0 else 0.0
