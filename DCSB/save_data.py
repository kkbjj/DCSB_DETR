# -*- coding: utf-8 -*-
"""
Export D-FINE postprocessor results for DFINE-N and DFINE-X using *val pipeline*.

Key design (based on our discussion):
- Always reuse val_dataloader pipeline (no train-time aug, no multiscale).
- For exporting "train set inference outputs", ONLY swap val_dataloader.dataset paths to train2017.
- Run DFINE-N and DFINE-X on the same batch and save postprocessed outputs per image_id.
- Do NOT threshold here: conf_thr=0.0 (thresholding/statistics are done later).

Outputs (saved under out_dir):
- image_file_name.pt              : List[int]
- dfinen_post_results.pt          : Dict[img_id, dict]  (boxes/scores/labels/(index...))
- dfinex_post_results.pt          : Dict[img_id, dict]
- targets.pt                      : Dict[img_id, dict]  (GT copied from dataloader)

Example (export val2017):
python tools/data_process/export_dfine_post_results_nx.py \
  --config_n configs/dfine/dfine_hgnetv2_n_coco.yml --ckpt_n ./model/dfine_n_coco.pth \
  --config_x configs/dfine/dfine_hgnetv2_x_coco.yml --ckpt_x ./model/dfine_x_coco.pth \
  --out_dir ./export/val2017_post --export_split val --device cuda:0 --conf_thr 0.0 --return_index

Example (export train2017 but using val pipeline):
python tools/data_process/export_dfine_post_results_nx.py \
  --config_n configs/dfine/dfine_hgnetv2_n_coco.yml --ckpt_n ./model/dfine_n_coco.pth \
  --config_x configs/dfine/dfine_hgnetv2_x_coco.yml --ckpt_x ./model/dfine_x_coco.pth \
  --out_dir ./export/train2017_post --export_split train --device cuda:0 --conf_thr 0.0 --return_index \
  --train_img_folder /root/autodl-tmp/datasets/coco2017/train2017/ \
  --train_ann_file /root/autodl-tmp/datasets/coco2017/annotations/instances_train2017.json
"""

import os
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch

# D-FINE
from src.core import YAMLConfig, yaml_utils


def get_args():
    ap = argparse.ArgumentParser("Export D-FINE postprocessed outputs (DFINE-N & DFINE-X) using val pipeline")

    ap.add_argument("--config_n", type=str, required=True)
    ap.add_argument("--ckpt_n", type=str, required=True)
    ap.add_argument("--config_x", type=str, required=True)
    ap.add_argument("--ckpt_x", type=str, required=True)

    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--export_split", type=str, choices=["val", "train"], default="val",
                    help="val: use val2017 paths; train: swap val_dataloader.dataset paths to train2017")
    ap.add_argument("--train_img_folder", type=str, default=None,
                    help="required if export_split=train (train2017 img folder)")
    ap.add_argument("--train_ann_file", type=str, default=None,
                    help="required if export_split=train (instances_train2017.json)")

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--conf_thr", type=float, default=0.0, help="postprocessor conf_thr; 0.0 means no filtering")
    ap.add_argument("--return_index", action="store_true", help="ask postprocessor to return selected query indices")
    ap.add_argument("--max_images", type=int, default=-1, help="limit images for debugging; -1 means all")

    ap.add_argument("--update_n", type=str, nargs="*", default=None, help="YAML overrides for N (dotlist)")
    ap.add_argument("--update_x", type=str, nargs="*", default=None, help="YAML overrides for X (dotlist)")

    ap.add_argument("--assert_fixed_size", action="store_true",
                    help="assert input H,W stay fixed across all batches (recommended)")
    ap.add_argument("--expected_hw", type=int, nargs=2, default=[640, 640],
                    help="expected input size H W when assert_fixed_size is enabled")

    return ap.parse_args()


def build_model_and_cfg(cfg_path: str, ckpt_path: str, device: str, updates: dict | None):
    updates = updates or {}
    cfg = YAMLConfig(cfg_path, **updates)

    model = cfg.model
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Load] {Path(cfg_path).name} missing:{len(missing)} unexpected:{len(unexpected)}")

    model.eval().to(device)
    return model, cfg


def safe_cpu_detach(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    return x


def call_postprocessor(post, pred_logits, pred_boxes, orig_target_sizes, conf_thr: float, return_index: bool):
    """Be robust to slightly different signatures."""
    inp = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
    try:
        return post(inp, orig_target_sizes=orig_target_sizes, conf_thr=conf_thr, return_index=return_index)
    except TypeError:
        # fallback: post(inp, orig_target_sizes, conf_thr)
        return post(inp, orig_target_sizes, conf_thr)


def main():
    args = get_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("⚠ CUDA not available, switching to CPU")
        args.device = "cpu"

    if args.export_split == "train":
        if not args.train_img_folder or not args.train_ann_file:
            raise ValueError("export_split=train requires --train_img_folder and --train_ann_file")

    update_n = yaml_utils.parse_cli(args.update_n) if args.update_n else {}
    update_x = yaml_utils.parse_cli(args.update_x) if args.update_x else {}

    model_n, cfg_n = build_model_and_cfg(args.config_n, args.ckpt_n, args.device, update_n)
    model_x, cfg_x = build_model_and_cfg(args.config_x, args.ckpt_x, args.device, update_x)

    # Always start from val_dataloader pipeline (inference-like)
    loader = cfg_n.val_dataloader

    # Swap dataset paths to train2017 if requested (still using val pipeline)
    if args.export_split == "train":
        # Most YAMLConfig objects expose underlying dataset object at loader.dataset
        ds = loader.dataset
        # These attributes match your YAML keys
        if hasattr(ds, "img_folder"):
            ds.img_folder = args.train_img_folder
        if hasattr(ds, "ann_file"):
            ds.ann_file = args.train_ann_file
        print("[Dataset Swap] val_dataloader.dataset -> train2017 paths")
        print(f"  img_folder: {args.train_img_folder}")
        print(f"  ann_file  : {args.train_ann_file}")

    post_n = cfg_n.postprocessor
    post_x = cfg_x.postprocessor
    # keep COCO mapping consistent (optional but common)
    if hasattr(post_n, "remap_mscoco_category"):
        post_n.remap_mscoco_category = True
    if hasattr(post_x, "remap_mscoco_category"):
        post_x.remap_mscoco_category = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_ids: List[int] = []
    preds_n: Dict[int, Dict[str, Any]] = {}
    preds_x: Dict[int, Dict[str, Any]] = {}
    targets_by_id: Dict[int, Dict[str, Any]] = {}

    max_images = None if args.max_images is None or args.max_images < 0 else int(args.max_images)
    saved = 0

    # fixed-size checks (recommended to guarantee no multiscale slipped in)
    expected_h, expected_w = int(args.expected_hw[0]), int(args.expected_hw[1])
    first_hw: Tuple[int, int] | None = None

    print(f"[Export] split={args.export_split}, conf_thr={args.conf_thr}, return_index={args.return_index}")
    print(f"[Export] device={args.device}, assert_fixed_size={args.assert_fixed_size}, expected_hw={(expected_h, expected_w)}")
    print(f"[Export] saving to: {out_dir.resolve()}")

    with torch.no_grad():
        for batch_idx, (samples, targets) in enumerate(loader):
            images = samples.to(args.device)

            if args.assert_fixed_size:
                hw = (int(images.shape[-2]), int(images.shape[-1]))
                if first_hw is None:
                    first_hw = hw
                    if hw != (expected_h, expected_w):
                        raise AssertionError(f"First batch input size {hw} != expected {(expected_h, expected_w)}")
                else:
                    if hw != first_hw:
                        raise AssertionError(f"Input size changed across batches: {hw} vs {first_hw}")

            # forward both models on the same images
            out_n = model_n(images)
            out_x = model_x(images)

            for k in ["pred_logits", "pred_boxes"]:
                if k not in out_n or k not in out_x:
                    raise KeyError(f"Missing key '{k}' in model outputs. N keys={list(out_n.keys())}, X keys={list(out_x.keys())}")

            # orig_target_sizes: [B,2] (h,w) in pixels
            if not isinstance(targets, (list, tuple)):
                raise TypeError(f"targets should be list[dict], got {type(targets)}")
            if "orig_size" not in targets[0]:
                raise KeyError("targets must contain 'orig_size' for postprocessor scaling (D-FINE).")
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(images.device)

            results_n = call_postprocessor(
                post_n, out_n["pred_logits"], out_n["pred_boxes"],
                orig_target_sizes, conf_thr=float(args.conf_thr), return_index=bool(args.return_index)
            )
            results_x = call_postprocessor(
                post_x, out_x["pred_logits"], out_x["pred_boxes"],
                orig_target_sizes, conf_thr=float(args.conf_thr), return_index=bool(args.return_index)
            )

            if not isinstance(results_n, (list, tuple)) or not isinstance(results_x, (list, tuple)):
                raise TypeError("postprocessor must return list[dict] per image.")

            # save per image
            for t, rn, rx in zip(targets, results_n, results_x):
                img_id = int(t["image_id"].item() if torch.is_tensor(t["image_id"]) else t["image_id"])
                image_ids.append(img_id)

                # N
                en: Dict[str, Any] = {
                    "boxes": safe_cpu_detach(rn.get("boxes")),
                    "scores": safe_cpu_detach(rn.get("scores")),
                    "labels": safe_cpu_detach(rn.get("labels")),
                }
                if "index" in rn:
                    en["index"] = safe_cpu_detach(rn["index"])
                for kk, vv in rn.items():
                    if kk in en:
                        continue
                    en[kk] = safe_cpu_detach(vv)
                preds_n[img_id] = en

                # X
                ex: Dict[str, Any] = {
                    "boxes": safe_cpu_detach(rx.get("boxes")),
                    "scores": safe_cpu_detach(rx.get("scores")),
                    "labels": safe_cpu_detach(rx.get("labels")),
                }
                if "index" in rx:
                    ex["index"] = safe_cpu_detach(rx["index"])
                for kk, vv in rx.items():
                    if kk in ex:
                        continue
                    ex[kk] = safe_cpu_detach(vv)
                preds_x[img_id] = ex

                # targets (cpu copy)
                tgt_cpu: Dict[str, Any] = {}
                for kk, vv in t.items():
                    tgt_cpu[kk] = safe_cpu_detach(vv)
                targets_by_id[img_id] = tgt_cpu

                saved += 1
                if max_images is not None and saved >= max_images:
                    break

            if max_images is not None and saved >= max_images:
                break

            if (batch_idx + 1) % 20 == 0:
                print(f"[Progress] batches={batch_idx+1}, images_saved={saved}")

    # dump
    torch.save(image_ids, out_dir / "image_file_name.pt")
    torch.save(preds_n, out_dir / "dfinen_post_results.pt")
    torch.save(preds_x, out_dir / "dfinex_post_results.pt")
    torch.save(targets_by_id, out_dir / "targets.pt")

    print(f"[Done] images={len(image_ids)} saved={saved}")
    print(f"  - {(out_dir / 'image_file_name.pt').name}")
    print(f"  - {(out_dir / 'dfinen_post_results.pt').name}")
    print(f"  - {(out_dir / 'dfinex_post_results.pt').name}")
    print(f"  - {(out_dir / 'targets.pt').name}")


if __name__ == "__main__":
    main()
