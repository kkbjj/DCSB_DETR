# -*- coding: utf-8 -*-
"""
Export D-FINE postprocessor results for DFINE-N and DFINE-X (DCSB-ready, minimal & DFINE-aligned).

Key points:
- Strictly reuse D-FINE evaluate() pipeline:
    samples = samples.to(device)
    targets = [{k: v.to(device) if tensor else v for k,v in t.items()} for t in targets]
    outputs = model(samples)
    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    results = postprocessor(outputs, orig_target_sizes)

- For exporting train2017 with val pipeline: rebuild YAMLConfig with overrides
  (DO NOT mutate existing dataset fields; it may not reload COCO annotations.)

Saved under out_dir:
- image_ids.pt               : List[int]
- file_names.pt (optional)   : Dict[int, str]  (image_id -> file_name) if present in targets
- dfinen_post_results.pt     : Dict[int, Dict[str, Tensor/Any]]
- dfinex_post_results.pt     : Dict[int, Dict[str, Tensor/Any]]
- targets_by_id.pt           : Dict[int, Dict[str, Any/Tensor]]   (for later labeling / stats)
- image_size.pt              : Dict[int, int]  (H*W, from orig_size)
- ground_truth_object_num.pt : Dict[int, int]  (#GT boxes)
"""

import os
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import pprint
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# D-FINE
from src.core import YAMLConfig, yaml_utils


# -------------------------
# helpers
# -------------------------
def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def safe_cpu_detach(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach().cpu()
    return x


def extract_img_id(t: Dict[str, Any]) -> int:
    v = t.get("image_id", None)
    if v is None:
        raise KeyError("target dict missing 'image_id'")
    if torch.is_tensor(v):
        return int(v.item())
    return int(v)


def extract_orig_size_hw(t: Dict[str, Any]) -> Tuple[int, int]:
    """
    D-FINE uses t['orig_size'] as Tensor([H,W]) or similar.
    """
    if "orig_size" not in t:
        raise KeyError("target dict missing 'orig_size'")
    osz = t["orig_size"]
    if torch.is_tensor(osz):
        h = int(osz[0].item())
        w = int(osz[1].item())
        return h, w
    return int(osz[0]), int(osz[1])


# -------------------------
# config / build
# -------------------------
def get_args():
    ap = argparse.ArgumentParser("Export DFINE-N & DFINE-X postprocessed outputs for DCSB (minimal)")

    ap.add_argument("--config_n", type=str, required=True)
    ap.add_argument("--ckpt_n", type=str, required=True)
    ap.add_argument("--config_x", type=str, required=True)
    ap.add_argument("--ckpt_x", type=str, required=True)

    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument(
        "--export_split",
        type=str,
        choices=["val", "train"],
        default="val",
        help="val: use YAML val paths; train: override val_dataloader.dataset paths to train2017",
    )
    ap.add_argument("--train_img_folder", type=str, default=None)
    ap.add_argument("--train_ann_file", type=str, default=None)

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--max_images", type=int, default=-1)

    ap.add_argument("--update_n", type=str, nargs="*", default=None, help="YAML overrides dotlist for N")
    ap.add_argument("--update_x", type=str, nargs="*", default=None, help="YAML overrides dotlist for X")

    ap.add_argument("--assert_fixed_size", action="store_true")
    ap.add_argument("--expected_hw", type=int, nargs=2, default=[640, 640])

    return ap.parse_args()

from src.solver import TASKS
from src.misc import dist_utils



def build_cfg_and_model(
    cfg_path: str,
    ckpt_path: str,
    device: str,
    updates: Dict[str, Any],
) -> Tuple[torch.nn.Module, YAMLConfig]:
    cfg = YAMLConfig(cfg_path, **updates)
    # === 关键：和 DFINE train.py 对齐 ===
    # 只要你是从完整ckpt加载，就不要再走HGNetv2 stage1 pretrained
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    model = cfg.model
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Load] {Path(cfg_path).name} missing:{len(missing)} unexpected:{len(unexpected)}")

    model.eval().to(device)
    return model, cfg
    # 让 cfg 走 DFINE 的 resume 逻辑（对齐 train.py）
    # updates = dict(updates)
    # updates.update({
    #     "device": device,
    #     "resume": ckpt_path,
    #     "test_only": True,
    # })

    # cfg = YAMLConfig(cfg_path, **updates)

    # # 对齐 train.py：resume/tuning 时关闭 HGNetv2 stage1 pretrained
    # if "HGNetv2" in cfg.yaml_cfg:
    #     cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    # # 走 solver.eval() 的加载链路：_setup + load_resume_state
    # solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    # solver.eval()  # 注意：这是 solver 的 eval（会 setup + load_resume_state），不会跑 COCO evaluate

    # # DFINE 验证时用 EMA 模型（你 train.py 也是这么做的）
    # module = solver.ema.module if getattr(solver, "ema", None) else solver.model
    # module.eval().to(device)

    # return module, cfg


def build_overrides_for_train_split(train_img_folder: str, train_ann_file: str) -> Dict[str, Any]:
    """
    Override val_dataloader.dataset.* so that we still use val pipeline transforms/collate,
    but dataset points to train2017.
    """
    return {
        "val_dataloader.dataset.img_folder": train_img_folder,
        "val_dataloader.dataset.ann_file": train_ann_file,
    }

def to_dcsb_dets(result: Dict[str, Any]) -> List[List[float]]:
    """
    Convert DFINE postprocessor output (dict) to DCSB-compatible dets:
    dets: List[List], shape [N, >=5], each row: [score, x1, y1, x2, y2]
    """
    if ("scores" not in result) or ("boxes" not in result):
        return []

    scores = result["scores"]
    boxes = result["boxes"]

    if (not torch.is_tensor(scores)) or (not torch.is_tensor(boxes)):
        return []

    if scores.numel() == 0 or boxes.numel() == 0:
        return []

    # ensure shapes: scores [N], boxes [N,4]
    scores = scores.view(-1).to(torch.float32).cpu()
    boxes = boxes.view(-1, 4).to(torch.float32).cpu()

    dets = torch.cat([scores[:, None], boxes], dim=1)  # [N,5]
    return dets.tolist()

def compute_gt_area_ratios(
    target: Dict[str, Any],
    input_hw: Tuple[int, int],
) -> List[float]:
    """
    Compute per-GT-box area ratio in original image pixels.

    Assumptions:
    - target["boxes"] is Tensor[N,4] in cxcywh format, normalized or input-scale
    - input_hw = (input_h, input_w)
    - target["orig_size"] = (H, W) original image size
    """
    if "boxes" not in target or not torch.is_tensor(target["boxes"]):
        return []

    boxes = target["boxes"]  # [N,4], cxcywh
    # optional sanity check
    mx = float(boxes.max().detach().cpu())
    if mx > 2.0:  # 明显不是 normalized
        raise ValueError(f"GT boxes seem not normalized (max={mx}). Please remove normalization scaling in compute_gt_area_ratios.")

    if boxes.numel() == 0:
        return []

    input_h, input_w = input_hw
    orig_h, orig_w = extract_orig_size_hw(target)

    # cxcywh (input scale) -> xyxy (input scale)
    cx, cy, w, h = boxes.unbind(dim=1)

    # === 新增：normalized cxcywh -> input-scale cxcywh ===
    cx = cx * float(input_w)
    w  = w  * float(input_w)
    cy = cy * float(input_h)
    h  = h  * float(input_h)

    # cxcywh (input scale) -> xyxy (input scale)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h


    # scale to original image size
    scale_x = orig_w / float(input_w)
    scale_y = orig_h / float(input_h)

    x1 = x1 * scale_x
    x2 = x2 * scale_x
    y1 = y1 * scale_y
    y2 = y2 * scale_y

    # area in original pixels
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    img_area = float(orig_h * orig_w)
    ratios = (areas / img_area).detach().cpu().tolist()

    return ratios


# -------------------------
# main
# -------------------------
def main():
    args = get_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("⚠ CUDA not available, switching to CPU")
        args.device = "cpu"

    if args.export_split == "train":
        if not args.train_img_folder or not args.train_ann_file:
            raise ValueError("export_split=train requires --train_img_folder and --train_ann_file")

    # parse CLI dotlist overrides
    update_n = yaml_utils.parse_cli(args.update_n) if args.update_n else {}
    update_x = yaml_utils.parse_cli(args.update_x) if args.update_x else {}

    # IMPORTANT: for train split, rebuild cfg with explicit overrides (no in-place mutation)
    if args.export_split == "train":
        train_over = build_overrides_for_train_split(args.train_img_folder, args.train_ann_file)
        update_n = {**train_over, **update_n}  # CLI overrides win
        update_x = {**train_over, **update_x}

    model_n, cfg_n = build_cfg_and_model(args.config_n, args.ckpt_n, args.device, update_n)
    model_x, cfg_x = build_cfg_and_model(args.config_x, args.ckpt_x, args.device, update_x)
    print("\n========== Config 对象属性 ==========")
    pprint.pprint(cfg_n.yaml_cfg, width=120, sort_dicts=False)
    print("========================================\n")
    print("\n========== Config 对象属性 ==========")
    pprint.pprint(cfg_x.yaml_cfg, width=120, sort_dicts=False)
    print("========================================\n")

    # Reuse dataloader pipeline (like evaluate())
    loader = cfg_n.val_dataloader

    post_n = cfg_n.postprocessor
    post_x = cfg_x.postprocessor
    if hasattr(post_n, "remap_mscoco_category"):
        post_n.remap_mscoco_category = True
    if hasattr(post_x, "remap_mscoco_category"):
        post_x.remap_mscoco_category = True

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # outputs
    image_ids: List[int] = []
    file_names: Dict[int, str] = {}

    preds_n: Dict[int, Dict[str, Any]] = {}
    preds_x: Dict[int, Dict[str, Any]] = {}

    targets_by_id: Dict[int, Dict[str, Any]] = {}

    image_size: Dict[int, int] = {}
    gt_obj_num: Dict[int, int] = {}

    dfinen_dcsb: Dict[int, List[List[float]]] = {}
    dfinex_dcsb: Dict[int, List[List[float]]] = {}

    ground_truth_area: Dict[int, List[float]] = {}


    max_images = None if args.max_images is None or args.max_images < 0 else int(args.max_images)

    # fixed-size checks
    expected_h, expected_w = int(args.expected_hw[0]), int(args.expected_hw[1])
    first_hw: Optional[Tuple[int, int]] = None

    saved = 0
    print(f"[Export] split={args.export_split} device={args.device}")
    print(f"[Export] assert_fixed_size={args.assert_fixed_size} expected_hw={(expected_h, expected_w)}")
    print(f"[Export] saving to: {out_dir.resolve()}")

    with torch.no_grad():
        for batch_idx, (samples, targets) in enumerate(loader):
            # === STRICTLY mimic D-FINE evaluate() ===
            samples = samples.to(args.device)
            targets = [
                {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()}
                for t in targets
            ]

            # fixed-size assertion
            hw = (int(samples.shape[-2]), int(samples.shape[-1]))
            if args.assert_fixed_size:
                if first_hw is None:
                    first_hw = hw
                    if hw != (expected_h, expected_w):
                        raise AssertionError(f"First batch input size {hw} != expected {(expected_h, expected_w)}")
                else:
                    if hw != first_hw:
                        raise AssertionError(f"Input size changed across batches: {hw} vs {first_hw}")

            # forward
            out_n = model_n(samples)
            out_x = model_x(samples)

            # orig_target_sizes like evaluate()
            if "orig_size" not in targets[0]:
                raise KeyError("targets must contain 'orig_size' for postprocessor scaling.")
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

            # === STRICTLY mimic D-FINE evaluate(): postprocessor(outputs, orig_target_sizes) ===
            results_n = post_n(out_n, orig_target_sizes)
            results_x = post_x(out_x, orig_target_sizes)

            if not isinstance(results_n, (list, tuple)) or not isinstance(results_x, (list, tuple)):
                raise TypeError("postprocessor must return list[dict] per image")
            
            input_hw = (samples.shape[-2], samples.shape[-1])
            for tgt, rn, rx in zip(targets, results_n, results_x):
                img_id = extract_img_id(tgt)
                image_ids.append(img_id)
                # === 新增：计算 ground truth area ratio ===
                ground_truth_area[img_id] = compute_gt_area_ratios(tgt, input_hw)


                # optional file_name
                if "file_name" in tgt:
                    try:
                        file_names[img_id] = str(tgt["file_name"])
                    except Exception:
                        pass

                # post results -> cpu
                preds_n[img_id] = {k: safe_cpu_detach(v) for k, v in rn.items()}
                preds_x[img_id] = {k: safe_cpu_detach(v) for k, v in rx.items()}
                # DCSB-compatible flat format
                dfinen_dcsb[img_id] = to_dcsb_dets(rn)
                dfinex_dcsb[img_id] = to_dcsb_dets(rx)

                # targets -> cpu
                targets_by_id[img_id] = {k: safe_cpu_detach(v) for k, v in tgt.items()}

                # image size & gt count (for later labeling / stats)
                orig_h, orig_w = extract_orig_size_hw(tgt)
                image_size[img_id] = int(orig_h * orig_w)

                if "boxes" in tgt and torch.is_tensor(tgt["boxes"]):
                    gt_obj_num[img_id] = int(tgt["boxes"].shape[0])
                else:
                    gt_obj_num[img_id] = 0

                saved += 1
                if max_images is not None and saved >= max_images:
                    break

            if max_images is not None and saved >= max_images:
                break

            if (batch_idx + 1) % 20 == 0:
                print(f"[Progress] batches={batch_idx+1}, images_saved={saved}")

    # dump
    torch.save(image_ids, out_dir / "image_ids.pt")
    if len(file_names) > 0:
        torch.save(file_names, out_dir / "file_names.pt")

    torch.save(preds_n, out_dir / "dfinen_post_results.pt")
    torch.save(preds_x, out_dir / "dfinex_post_results.pt")
    torch.save(dfinen_dcsb, out_dir / "dfinen_dcsb.pt")
    torch.save(dfinex_dcsb, out_dir / "dfinex_dcsb.pt")

    torch.save(targets_by_id, out_dir / "targets_by_id.pt")

    torch.save(image_size, out_dir / "image_size.pt")
    torch.save(gt_obj_num, out_dir / "ground_truth_object_num.pt")
    torch.save(ground_truth_area, out_dir / "ground_truth_area.pt")


    print(f"[Done] split={args.export_split} images_saved={saved}")
    print("Saved files:")
    for p in sorted(out_dir.glob("*.pt")):
        print("  -", p.name)


if __name__ == "__main__":
    main()
