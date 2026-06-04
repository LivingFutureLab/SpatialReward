import numpy as np
from PIL import Image
from typing import Dict, List, Any, Tuple

from ..utils.geometry import compute_iou_boxB


def evaluate_text(
    image: Image.Image,
    objects: Dict[str, List],
    metadata: Dict[str, Any],
    ocr_engine=None,
) -> Tuple[bool, float]:
    """
    Evaluate text content on detected objects using OCR.

    Args:
        image: PIL Image or np.ndarray
        objects: Detection results {class: [(bbox, score), ...]}
        metadata: Metadata with 'text' fields in 'include' entries
        ocr_engine: PaddleOCR instance

    Returns:
        (correct, reward_score)
    """
    from Levenshtein import distance as levenshtein_distance

    if ocr_engine is None:
        print("[WARN] OCR engine not loaded, skip text evaluation")
        return True, 1.0

    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB"))
    else:
        image_np = image

    # Run OCR on the full image
    ocr_result = ocr_engine.ocr(image_np, cls=False)
    ocr_boxes = []
    if ocr_result and ocr_result[0]:
        for res in ocr_result[0]:
            pts = res[0]
            text = res[1][0]
            score = res[1][1]
            x1 = min(p[0] for p in pts)
            y1 = min(p[1] for p in pts)
            x2 = max(p[0] for p in pts)
            y2 = max(p[1] for p in pts)
            ocr_boxes.append(([x1, y1, x2, y2], text, score))

    IOU_THRESHOLD = 0.5
    correct = True
    rewards = []

    for req in metadata.get("include", []):
        if "text" not in req or not req["text"]:
            continue

        classname = req["class"]
        found_objects = objects.get(classname, [])

        if len(found_objects) != req["count"]:
            print(
                f"[text-eval] {classname}: count mismatch "
                f"(expected {req['count']}, found {len(found_objects)})"
            )
            correct = False
            rewards.append(0.0)
            continue

        # Sort by x-position (left to right)
        found_objects_sorted = sorted(found_objects, key=lambda x: x[0][0])

        text_rewards = []
        zero_text_rewards = []

        for text_idx, expected_text in enumerate(req["text"]):
            if text_idx >= len(found_objects_sorted):
                print(f"[WARN] text index {text_idx} out of range for {classname}")
                text_rewards.append(0.0)
                continue

            target_bbox = found_objects_sorted[text_idx][0]

            # Collect OCR text within this detection box
            merged_text = ""
            for ocr_bbox, ocr_text, ocr_score in ocr_boxes:
                iou = compute_iou_boxB(target_bbox, ocr_bbox)
                if iou > IOU_THRESHOLD and ocr_score > 0:
                    merged_text += ocr_text

            recognized_text = merged_text.strip()
            expected_text = expected_text.strip()

            print(
                f"[text-eval] {classname}[{text_idx}]: "
                f"expected='{expected_text}', recognized='{recognized_text}'"
            )

            if expected_text == "":
                if recognized_text == "":
                    zero_text_rewards.append(1.0)
                else:
                    zero_text_rewards.append(0.0)
                    correct = False
            else:
                dist = min(levenshtein_distance(recognized_text, expected_text), len(expected_text))
                reward = 1.0 - dist / len(expected_text)
                text_rewards.append(reward)
                print(f"[text-eval] levenshtein dist={dist}, reward={reward:.3f}")
                if reward < 0.99:
                    correct = False

        # Combine text rewards for this requirement
        if text_rewards and zero_text_rewards:
            score_text = (
                sum(text_rewards) / len(text_rewards)
                + sum(zero_text_rewards) / len(zero_text_rewards)
            ) / 2.0
        elif text_rewards:
            score_text = sum(text_rewards) / len(text_rewards)
        elif zero_text_rewards:
            score_text = sum(zero_text_rewards) / len(zero_text_rewards)
        else:
            score_text = 1.0

        print(f"[text-eval] {classname} score_text={score_text:.3f}")
        rewards.append(score_text)

    reward = sum(rewards) / len(rewards) if rewards else 1.0
    return correct, reward


def evaluate_text_with_position(
    image: Image.Image,
    objects: Dict[str, List],
    metadata: Dict[str, Any],
    ocr_engine=None,
    position_eval_fn=None,
) -> Tuple[bool, float]:
    """
    Joint evaluation: position relations + text content.

    Args:
        image: PIL Image
        objects: Detection results
        metadata: Metadata with both position and text fields
        ocr_engine: PaddleOCR instance
        position_eval_fn: Callable for position evaluation (evaluate_reward or evaluate_complex_positions)

    Returns:
        (correct, reward_score)
    """
    has_position = any("position" in req for req in metadata.get("include", []))
    has_text = any("text" in req for req in metadata.get("include", []))

    results = []

    if has_position and position_eval_fn is not None:
        pos_correct, pos_reward = position_eval_fn(image, objects, metadata)
        results.append((pos_correct, pos_reward))
        print(f"[joint-eval] position: correct={pos_correct}, reward={pos_reward:.3f}")

    if has_text:
        txt_correct, txt_reward = evaluate_text(image, objects, metadata, ocr_engine=ocr_engine)
        results.append((txt_correct, txt_reward))
        print(f"[joint-eval] text: correct={txt_correct}, reward={txt_reward:.3f}")

    overall_correct = all(r[0] for r in results)
    overall_reward = sum(r[1] for r in results) / len(results) if results else 0.0
    return overall_correct, overall_reward
