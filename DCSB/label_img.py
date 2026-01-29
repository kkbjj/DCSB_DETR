import torch 
import numpy as np
from typing import Dict, Any, Iterable, Tuple


def count_detections_over_threshold(dets: Any, score_thr: float = 0.5) -> int:
    """
    Count how many detections have score >= score_thr.

    Expected dets format (as used in the original repo):
        dets: array-like of shape [num_det, >=1]
              dets[:, 0] is the confidence score.

    Returns:
        int: number of detections whose score >= score_thr
    """
    arr = np.asarray(dets)

    # Handle empty detections robustly
    if arr.size == 0:
        return 0

    # Ensure 2D
    if arr.ndim == 1:
        # If someone stored only scores, allow shape [num_det]
        scores = arr
    else:
        scores = arr[:, 0]

    scores_t = torch.from_numpy(scores)
    return int((scores_t >= score_thr).sum().item())


def clip_to_gt(count: int, gt_list: Any) -> int:
    """
    Clip predicted count so it does not exceed the number of GT objects.
    gt_list is expected to be list/array; len(gt_list) is treated as GT count.
    """
    try:
        gt_num = len(gt_list)
    except TypeError:
        # If gt_list isn't sized, don't clip
        return count
    return min(count, gt_num)


def generate_image_labels(
    big_results: Dict[Any, Any],
    small_results: Dict[Any, Any],
    gt_area: Dict[Any, Any],
    keys: Iterable[Any],
    score_thr: float = 0.5,
    miss_thr: int = 1,
    clip_big_to_gt: bool = True,
    clip_small_to_gt: bool = False,  # default matches the original script's behavior
) -> Tuple[Dict[Any, int], Dict[Any, int], Dict[Any, int]]:
    """
    Re-implementation of data_process/label_img.py with better readability.

    For each image key:
      - count big detections with score >= score_thr
      - count small detections with score >= score_thr
      - optionally clip counts so they don't exceed GT count
      - miss = big_count - small_count
      - label = 1 if miss >= miss_thr else 0

    Returns:
      image_label: dict {key: 0/1}
      big_count_map: dict {key: big_count}
      small_count_map: dict {key: small_count}
    """
    image_label: Dict[Any, int] = {}
    big_count_map: Dict[Any, int] = {}
    small_count_map: Dict[Any, int] = {}

    sum_big = 0
    sum_small = 0
    difficult_keys = []

    for k in keys:
        # --- Big model count ---
        big_cnt = count_detections_over_threshold(big_results[k], score_thr=score_thr)
        if clip_big_to_gt:
            big_cnt = clip_to_gt(big_cnt, gt_area[k])

        big_count_map[k] = big_cnt
        sum_big += big_cnt

        # --- Small model count ---
        small_cnt = count_detections_over_threshold(small_results[k], score_thr=score_thr)
        if clip_small_to_gt:
            small_cnt = clip_to_gt(small_cnt, gt_area[k])

        small_count_map[k] = small_cnt
        sum_small += small_cnt

        # --- Labeling ---
        miss = big_cnt - small_cnt
        label = 1 if miss >= miss_thr else 0
        image_label[k] = label
        if label == 1:
            difficult_keys.append(k)

    # Optional debug prints (can remove if you like)
    print(f"Total images: {len(list(keys))}")
    print(f"SUM_big={sum_big}, SUM_small={sum_small}, difficult={len(difficult_keys)}")

    return image_label, big_count_map, small_count_map


def main():
    # --- Load inputs (same filenames as original) ---
    big_results = torch.load("big_model_results")
    small_results = torch.load("small_model_results")
    gt_area = torch.load("ground_truth_area")
    keys = torch.load("image_file_name")

    # --- Generate labels ---
    # Defaults reproduce original behavior:
    #   - big clipped to GT
    #   - small NOT clipped to GT (as in original due to overwrite)
    image_label, big_cnt, small_cnt = generate_image_labels(
        big_results=big_results,
        small_results=small_results,
        gt_area=gt_area,
        keys=keys,
        score_thr=0.5,
        miss_thr=1,
        clip_big_to_gt=True,
        clip_small_to_gt=False,
    )

    # --- Save outputs (same output filenames as original) ---
    torch.save(image_label, "image_label")
    torch.save(big_cnt, "image_object_num_big_model")
    torch.save(small_cnt, "image_object_num_small_model")  # keep the original typo


if __name__ == "__main__":
    main()
