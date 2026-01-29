import torch as t
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, Iterable, Tuple


def extract_scores(dets: Any) -> np.ndarray:
    """
    Extract confidence scores from a detection result.

    Original DCSB expectation:
        dets is array-like with shape [num_det, >=1] and dets[:, 0] is score.

    This function is more robust:
        - supports empty detections
        - supports 1D arrays (already scores)
        - supports list/np array/torch tensor

    Returns:
        scores: 1D numpy array of dtype float32
    """
    arr = np.asarray(dets)

    if arr.size == 0:
        return np.asarray([], dtype=np.float32)

    # If dets is a 1D array, treat it as scores directly
    if arr.ndim == 1:
        scores = arr
    else:
        scores = arr[:, 0]

    # Ensure numeric dtype
    scores = np.asarray(scores, dtype=np.float32)
    return scores


def count_targets(scores: np.ndarray, threshold: float) -> int:
    """
    Count how many scores >= threshold.
    Keeps the original repo's behavior: if count == 0, force it to 1.
    """
    cnt = int((scores >= threshold).sum())
    if cnt == 0:
        cnt = 1
    return cnt


def compute_global_difference(
    data_small: Dict[Any, Any],
    gt_area: Dict[Any, Any],
    keys: Iterable[Any],
    threshold: float,
) -> int:
    """
    Compute the global difference:
        differ = sum_over_images( num_target(threshold) - gt_count )

    NOTE: `num_target` has the "0 -> 1" forcing behavior to match original code.
    """
    differ = 0
    for k in keys:
        gt_count = len(gt_area[k])

        scores = extract_scores(data_small[k])
        pred_count = count_targets(scores, threshold)

        differ += (pred_count - gt_count)
    return differ


def search_best_threshold(
    data_small: Dict[Any, Any],
    gt_area: Dict[Any, Any],
    keys: Iterable[Any],
    l_start: int = 1,
    l_end: int = 500,
) -> Tuple[Dict[int, int], int, int]:
    """
    Search l in [l_start, l_end] (inclusive), where threshold = l/1000.

    Objective (same as original):
        pick l that minimizes abs(differ)
    Tie-breaking (same as original `<=`):
        if abs(differ) is equal, take the later l (because `<=` updates).
    """
    l_differ: Dict[int, int] = {}
    best_abs = float("inf")
    best_l = 0
    best_differ = 0

    for l in tqdm(range(l_start, l_end + 1)):
        thr = l / 1000.0
        differ = compute_global_difference(data_small, gt_area, keys, thr)

        l_differ[l] = differ

        # Same behavior as original:
        # if abs(differ) <= differ_min, update (so ties choose later l)
        abs_diff = abs(differ)
        if abs_diff <= best_abs:
            best_abs = abs_diff
            best_l = l
            best_differ = differ

    return l_differ, best_differ, best_l


def main():
    data_small = t.load("small_model_results")
    keys = t.load("image_file_name")
    gt_area = t.load("ground_truth_area")

    # Make sure keys is a concrete list (prevents surprises if it's an iterator)
    keys = list(keys)

    l_differ, differ_min, l_min = search_best_threshold(
        data_small=data_small,
        gt_area=gt_area,
        keys=keys,
        l_start=1,
        l_end=500,
    )

    print(str(l_differ))
    print("differ_min:", differ_min, "l_min", l_min)


if __name__ == "__main__":
    main()
