import torch as t
import numpy as np
from typing import Any, Dict, List, Tuple


# ======================
# CONFIG (match original)
# ======================
SCORE_THR_FOR_B_SMALL: float = 0.5   # original code uses 0.5000 here
MAX_NUM_AREAS: int = 18              # original uses max_num=18

# New: prefer using saved small_cnt (from label_img output) as b_small
USE_SAVED_SMALL_CNT: bool = True

# New: in the original repo the filename is often misspelled as "samll"
# Set this to the exact filename you actually saved.
SMALL_CNT_FILENAME: str = "image_object_num_samll_model"  # or "image_object_num_small_model"
# ======================


def _to_numpy(x: Any) -> np.ndarray:
    """
    Convert input to numpy array safely.
    Supports: list / numpy array / torch tensor / None.
    """
    if x is None:
        return np.asarray([])
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):  # torch tensor-like
        return x.detach().cpu().numpy()
    return np.asarray(x)


def count_scores_ge_threshold(dets: Any, thr: float) -> int:
    """
    Count how many detections have score >= thr.
    Expected det format per row: [score, x1, y1, x2, y2, ...]
    """
    arr = _to_numpy(dets)
    if arr.size == 0:
        return 0

    # Allow 1D scores-only edge case
    if arr.ndim == 1:
        scores = arr
    else:
        scores = arr[:, 0]

    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    return int((scores >= thr).sum())


def get_b_small(
    k: Any,
    dets_small: Any,
    score_thr_for_b_small: float,
    saved_small_cnt: Dict[Any, Any] | None = None,
) -> int:
    """
    Get b_small for image key k.

    Preferred behavior (as you requested):
      - If USE_SAVED_SMALL_CNT and saved_small_cnt is provided, use saved_small_cnt[k].
      - Otherwise, compute from small_model_results via count_scores_ge_threshold.

    We keep the original compute path intact; just switch the *source* of b_small.
    """
    if USE_SAVED_SMALL_CNT and saved_small_cnt is not None:
        if k not in saved_small_cnt:
            raise KeyError(f"Key {k} missing in saved small_cnt dict ({SMALL_CNT_FILENAME})")
        return int(saved_small_cnt[k])

    # Fallback: compute from raw detections
    return count_scores_ge_threshold(dets_small, score_thr_for_b_small)


def build_feature_vector(
    k: Any,
    dets_small: Any,
    num_target_p_1: int,
    area_ratios: List[float],
    saved_small_cnt: Dict[Any, Any] | None = None,
    max_num_areas: int = MAX_NUM_AREAS,
    score_thr_for_b_small: float = SCORE_THR_FOR_B_SMALL,
) -> List[float]:
    """
    Reproduce original per-image feature construction:

      temp = [
        b_small,                  # count of scores >= 0.5 from small_model_results
                                 # (or loaded from saved small_cnt if USE_SAVED_SMALL_CNT=True)
        num_target_p_1[i],         # loaded from image_predict_object_num
        sorted(area_ratios)...,    # loaded from image_predict_object_area
        padding zeros to length max_num_areas
      ]

    Output feature dimension = 2 + max_num_areas (original: 20)
    """
    b_small = get_b_small(
        k=k,
        dets_small=dets_small,
        score_thr_for_b_small=score_thr_for_b_small,
        saved_small_cnt=saved_small_cnt,
    )

    feats: List[float] = [float(b_small), float(num_target_p_1)]

    # IMPORTANT: original sorts in-place ascending
    areas_sorted = list(area_ratios)  # copy to avoid mutating loaded dict (safer)
    areas_sorted.sort()

    feats.extend([float(a) for a in areas_sorted])

    # Pad with zeros if needed (original behavior)
    if len(areas_sorted) < max_num_areas:
        feats.extend([0.0] * (max_num_areas - len(areas_sorted)))

    # NOTE: original code does NOT truncate if > max_num_areas
    # Keeping the exact behavior could make feature lengths inconsistent if that happens.
    # In the original dataset/pipeline they likely ensure len(area_ratios) <= 18.
    return feats


def build_training_tensors(
    k_list: List[Any],
    data_small: Dict[Any, Any],
    num_target_p_1: Dict[Any, Any],
    target_s_p_1: Dict[Any, Any],
    image_tag: Dict[Any, Any],
    saved_small_cnt: Dict[Any, Any] | None = None,
) -> Tuple[t.Tensor, t.Tensor]:
    """
    Build x,y tensors exactly like the original script:
      x: Tensor[num_images, 2 + MAX_NUM_AREAS]
      y: Tensor[num_images]

    Now supports using pre-saved small_cnt as b_small if provided.
    """
    train_x: List[List[float]] = []
    train_y: List[float] = []

    for k in k_list:
        if k not in data_small:
            raise KeyError(f"Key {k} missing in small_model_results")
        if k not in num_target_p_1:
            raise KeyError(f"Key {k} missing in image_predict_object_num")
        if k not in target_s_p_1:
            raise KeyError(f"Key {k} missing in image_predict_object_area")
        if k not in image_tag:
            raise KeyError(f"Key {k} missing in image_label")

        feats = build_feature_vector(
            k=k,
            dets_small=data_small[k],
            num_target_p_1=int(num_target_p_1[k]),
            area_ratios=list(target_s_p_1[k]),
            saved_small_cnt=saved_small_cnt,
        )

        train_x.append(feats)
        train_y.append(float(image_tag[k]))

    x = t.tensor(train_x, dtype=t.float32)
    y = t.tensor(train_y, dtype=t.float32)
    return x, y


def dict_to_ordered_list(k_list: List[Any], per_image_dict: Dict[Any, Any], name: str) -> List[int]:
    """
    Convert {k: value} dict to list aligned with k_list order.
    """
    out: List[int] = []
    for k in k_list:
        if k not in per_image_dict:
            raise KeyError(f"Key {k} missing in {name}")
        out.append(int(per_image_dict[k]))
    return out


def main():
    # -------- Load inputs (same filenames as original) --------
    target_s_p_1: Dict[Any, Any] = t.load("image_predict_object_area")
    num_target_p_1: Dict[Any, Any] = t.load("image_predict_object_num")
    image_tag: Dict[Any, Any] = t.load("image_label")
    k_list: List[Any] = list(t.load("image_file_name"))
    data_small: Dict[Any, Any] = t.load("small_model_results")

    # These two are loaded for the later stats/save stage (same as original)
    target_num_big: Dict[Any, Any] = t.load("image_object_num_big_model")
    target_num_small: Dict[Any, Any] = t.load("image_object_num_small_model")

    # New: load saved small_cnt if enabled
    saved_small_cnt: Dict[Any, Any] | None = None
    if USE_SAVED_SMALL_CNT:
        saved_small_cnt = t.load(SMALL_CNT_FILENAME)

    # -------- Build discriminator training data --------
    x, y = build_training_tensors(
        k_list=k_list,
        data_small=data_small,
        num_target_p_1=num_target_p_1,
        target_s_p_1=target_s_p_1,
        image_tag=image_tag,
        saved_small_cnt=saved_small_cnt,
    )

    # Save outputs (same filenames as original)
    t.save(x, "data_for_discriminator")
    t.save(y, "data_label_for_discriminator")

    # -------- Stats / ordered lists (same outputs as original) --------
    target_num_small_data = dict_to_ordered_list(k_list, target_num_small, "image_object_num_small_model")
    target_num_big_data = dict_to_ordered_list(k_list, target_num_big, "image_object_num_big_model")

    print(max(target_num_small_data) if target_num_small_data else 0)
    print(max(target_num_big_data) if target_num_big_data else 0)
    print(sum(target_num_small_data))
    print(sum(target_num_big_data))

    t.save(target_num_big_data, "object_num_data_test_big_model")
    t.save(target_num_small_data, "object_num_data_test_small_model")


if __name__ == "__main__":
    main()
