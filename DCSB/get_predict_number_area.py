import torch as t
import numpy as np
from typing import Any, Dict, Iterable, List, Tuple


# ====== CONFIG ======
# This should be the l_min found by get_confidence_threshold.py (an integer in [1, 500]).
# Example: confidence_threshold = 237  -> actual score threshold = 0.237
confidence_threshold: int = 200
# ====================


def to_numpy(dets: Any) -> np.ndarray:
    """
    Convert dets to numpy array safely.
    dets is usually list/np.ndarray/torch.Tensor.
    """
    if dets is None:
        return np.asarray([])
    if isinstance(dets, np.ndarray):
        return dets
    if hasattr(dets, "detach"):  # torch tensor
        return dets.detach().cpu().numpy()
    return np.asarray(dets)


def extract_scores_and_boxes(dets: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract scores and (optional) boxes from a detection result.

    Expected DCSB-like det format per detection row:
        [score, x1, y1, x2, y2, ...]
    But we make it robust:
        - empty dets -> ([], empty boxes)
        - 1D dets -> treat as scores only, boxes empty
        - if dets has < 5 columns -> boxes empty

    Returns:
        scores: shape [N], float32
        boxes:  shape [N, 4], float32  (xyxy), or empty with shape [0, 4]
    """
    arr = to_numpy(dets)

    if arr.size == 0:
        return np.asarray([], dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    # 1D: scores only
    if arr.ndim == 1:
        scores = np.asarray(arr, dtype=np.float32)
        return scores, np.zeros((0, 4), dtype=np.float32)

    # 2D or higher: use first dim as N, take columns
    arr2 = np.asarray(arr, dtype=np.float32)
    scores = arr2[:, 0].reshape(-1)

    # boxes if available
    if arr2.shape[1] >= 5:
        boxes = arr2[:, 1:5]  # x1,y1,x2,y2
    else:
        boxes = np.zeros((0, 4), dtype=np.float32)

    return scores, boxes


def count_num_targets(scores: np.ndarray, thr: float) -> int:
    """
    Count detections with score >= thr.
    Keep original behavior: if count == 0, force it to 1.
    """
    cnt = int((scores >= thr).sum())
    return cnt if cnt > 0 else 1


def box_area_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Compute area of xyxy boxes with clamping to avoid negative areas.
    boxes: [N, 4] -> (x1,y1,x2,y2)

    Returns:
        areas: [N] float32
    """
    if boxes.size == 0:
        return np.asarray([], dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    w = np.maximum(0.0, x2 - x1)
    h = np.maximum(0.0, y2 - y1)
    return (w * h).astype(np.float32)


def main():
    # Load inputs
    data_small: Dict[Any, Any] = t.load("small_model_results")
    keys: List[Any] = list(t.load("image_file_name"))
    image_area: Dict[Any, Any] = t.load("image_size")  # expected: {k: W*H}

    thr = confidence_threshold / 1000.0

    # Output dicts
    image_predict_object_num: Dict[Any, int] = {}
    image_predict_object_area: Dict[Any, List[float]] = {}

    for k in keys:
        if k not in data_small:
            raise KeyError(f"Key {k} not found in small_model_results.")
        if k not in image_area:
            raise KeyError(f"Key {k} not found in image_size (image_area).")

        scores, boxes = extract_scores_and_boxes(data_small[k])

        # ---------- 1) predicted object count ----------
        num_target = count_num_targets(scores, thr)
        image_predict_object_num[k] = num_target

        # ---------- 2) predicted object area ratios ----------
        # We only compute areas for detections passing threshold AND having boxes.
        areas_ratio: List[float] = []

        # If we have boxes aligned with scores (typical case)
        if boxes.shape[0] == scores.shape[0] and boxes.shape[0] > 0:
            keep = scores >= thr  # original area code uses ">" not ">="
            kept_boxes = boxes[keep]
            areas = box_area_xyxy(kept_boxes)

            denom = float(image_area[k])
            if denom <= 0:
                # avoid division by zero; fall back to 1.0
                denom = 1.0

            areas_ratio = (areas / denom).astype(np.float32).tolist()

        # original behavior: if no kept boxes, store [0]
        if len(areas_ratio) == 0:
            areas_ratio = [0.0]

        image_predict_object_area[k] = areas_ratio

    # Save outputs (same filenames as original)
    t.save(image_predict_object_num, "image_predict_object_num")
    t.save(image_predict_object_area, "image_predict_object_area")


if __name__ == "__main__":
    main()
