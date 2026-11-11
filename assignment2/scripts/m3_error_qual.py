"""M3: error analysis + data-driven qualitative selection for the selected final model.

Final model = run with the highest best-validation mIoU (selection rule fixed before looking at test).
Every number written to error_stats.json is computed here from model predictions on the 500-image
local test split (official Cityscapes val, evaluated at 1024x512, fp16 autocast as in src/evaluate.py).

Error taxonomy (mutually exclusive, every wrong pixel of the final model gets exactly one group):
  1. boundary  : wrong pixel within k=3 px of a ground-truth class transition
  2. small     : interior wrong pixel belonging to a GT connected component < 32x32 = 1024 px
  3. related   : remaining wrong pixel whose (GT, pred) pair lies in the same semantic family
  4. other     : everything else (unrelated-class confusion inside large regions)
Class-imbalance effects are measured separately on the confusion matrix (rare-10 vs frequent-3).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from src.common import get_logits  # noqa: E402
from src.data import get_loaders  # noqa: E402
from src.error_analysis import compute_boundary_mask, plot_confusion_matrix  # noqa: E402
from src.visualize import UNLABELED_SAMPLES, load_model, process_labeled_group, process_unlabeled_group  # noqa: E402

CLASS_NAMES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
               "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
               "motorcycle", "bicycle"]
NC = 19
RARE_10 = [17, 12, 16, 6, 15, 14, 18, 3, 7, 4]
FREQ_3 = [0, 2, 8]
FAMILIES = [[0, 1, 9], [2, 3, 4], [5, 6, 7], [8, 9], [11, 12], [13, 14, 15, 16], [12, 17, 18]]
RELATED = np.zeros((NC, NC), dtype=bool)
for fam in FAMILIES:
    for i in fam:
        for j in fam:
            if i != j:
                RELATED[i, j] = True
GROUPS = ["boundary", "small", "related", "other"]


def metrics_from_cm(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    iou = np.where(tp + fp + fn > 0, tp / np.maximum(tp + fp + fn, 1), np.nan)
    rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), np.nan)
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), np.nan)
    dice = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)
    return iou, rec, prec, dice


def image_miou(cm):
    iou = metrics_from_cm(cm)[0]
    return float(np.nanmean(iou))


def area_map_of(gt):
    amap = np.zeros(gt.shape, dtype=np.int64)
    n_small = 0
    for c in np.unique(gt[gt < NC]):
        n, lab, st, _ = cv2.connectedComponentsWithStats((gt == c).astype(np.uint8), connectivity=8)
        areas = st[:, cv2.CC_STAT_AREA].copy()
        areas[0] = 0
        sel = lab > 0
        amap[sel] = areas[lab[sel]]
        n_small += int((areas[1:] < 1024).sum())
    return amap, n_small


def scale_bins(amap, valid):
    return {
        "small": valid & (amap > 0) & (amap < 1024),
        "medium": valid & (amap >= 1024) & (amap <= 9216),
        "large": valid & (amap > 9216),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_config", default="configs/dlv3_cedice.yaml")
    ap.add_argument("--final_ckpt", default="checkpoints/dlv3_cedice_1024x512_s666/best.pt")
    ap.add_argument("--unet_config", default="configs/unet_final.yaml")
    ap.add_argument("--unet_ckpt", default="checkpoints/unet_final_1024x512_s42/best.pt")
    ap.add_argument("--out_dir", default="docs/figures/m3")
    args = ap.parse_args()

    out = BASE_DIR / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    cfg = yaml.safe_load(open(BASE_DIR / args.final_config, encoding="utf-8"))
    final = load_model(BASE_DIR / args.final_config, BASE_DIR / args.final_ckpt, dev)
    unet = load_model(BASE_DIR / args.unet_config, BASE_DIR / args.unet_ckpt, dev)
    loader = get_loaders(cfg)[2]
    samples = loader.dataset.samples

    cm = {"final": np.zeros((NC, NC), np.int64), "unet": np.zeros((NC, NC), np.int64)}
    bnd = {m: {"band": 0, "band_err": 0, "int": 0, "int_err": 0} for m in cm}
    scl = {m: {b: [0, 0] for b in ("small", "medium", "large")} for m in cm}
    grp_total = dict.fromkeys(GROUPS, 0)
    err_rare_gt = 0
    per_image = []
    pair_best = {}

    k = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for images, masks in loader:
            images = images.to(dev)
            preds = {"final": get_logits(final(images)).argmax(1).cpu().numpy(),
                     "unet": get_logits(unet(images)).argmax(1).cpu().numpy()}
            gts = masks.numpy()
            for b in range(gts.shape[0]):
                gt = gts[b]
                name = samples[k]; k += 1
                valid = gt != 255
                band, interior = compute_boundary_mask(gt, k=3)
                amap, n_small = area_map_of(gt)
                bins = scale_bins(amap, valid)
                row = {"name": name, "n_small_components": n_small}
                for m, P in preds.items():
                    p = P[b]
                    c_img = np.bincount(gt[valid].astype(np.int64) * NC + p[valid], minlength=NC * NC).reshape(NC, NC)
                    cm[m] += c_img
                    err = (p != gt) & valid
                    bnd[m]["band"] += int(band.sum()); bnd[m]["band_err"] += int((err & band).sum())
                    bnd[m]["int"] += int(interior.sum()); bnd[m]["int_err"] += int((err & interior).sum())
                    for bn, bm in bins.items():
                        scl[m][bn][0] += int(((p == gt) & bm).sum()); scl[m][bn][1] += int(bm.sum())
                    row[f"miou_{m}"] = image_miou(c_img)
                    row[f"acc_{m}"] = float(np.diag(c_img).sum() / max(c_img.sum(), 1))
                    if m == "final":
                        g_b = err & band
                        g_s = err & interior & (amap < 1024)
                        rest = err & interior & (amap >= 1024)
                        rel = rest & RELATED[np.where(valid, gt, 0), p]
                        g_o = rest & ~rel
                        counts = {"boundary": int(g_b.sum()), "small": int(g_s.sum()),
                                  "related": int(rel.sum()), "other": int(g_o.sum())}
                        for g in GROUPS:
                            grp_total[g] += counts[g]
                            row[f"err_{g}"] = counts[g]
                        err_rare_gt += int((err & np.isin(gt, RARE_10)).sum())
                        off = c_img.copy(); np.fill_diagonal(off, 0)
                        for i, j in zip(*np.nonzero(off)):
                            if off[i, j] > pair_best.get((i, j), ("", 0))[1]:
                                pair_best[(i, j)] = (name, int(off[i, j]))
                per_image.append(row)

    res = {"final_config": args.final_config, "final_ckpt": args.final_ckpt,
           "unet_ckpt": args.unet_ckpt, "num_images": k, "models": {}}
    for m in cm:
        iou, rec, prec, dice = metrics_from_cm(cm[m])
        b = bnd[m]
        be, ie = b["band_err"] / b["band"] * 100, b["int_err"] / b["int"] * 100
        res["models"][m] = {
            "miou": float(np.nanmean(iou) * 100), "mdice": float(np.nanmean(dice) * 100),
            "pixel_acc": float(np.diag(cm[m]).sum() / cm[m].sum() * 100),
            "per_class": {CLASS_NAMES[c]: {"iou": iou[c] * 100, "recall": rec[c] * 100, "precision": prec[c] * 100}
                          for c in range(NC)},
            "boundary_err_rate": be, "interior_err_rate": ie, "boundary_density_ratio": be / ie,
            "boundary_share_of_errors": b["band_err"] / (b["band_err"] + b["int_err"]) * 100,
            "band_share_of_valid_pixels": b["band"] / (b["band"] + b["int"]) * 100,
            "scale_recall": {bn: v[0] / v[1] * 100 for bn, v in scl[m].items()},
            "scale_pixels": {bn: v[1] for bn, v in scl[m].items()},
            "rare10_iou": float(np.mean(iou[RARE_10]) * 100), "freq3_iou": float(np.mean(iou[FREQ_3]) * 100),
            "rare10_recall": float(np.mean(rec[RARE_10]) * 100),
            "thin3_recall": float(np.mean(rec[[5, 6, 7]]) * 100),
        }
    tot_err = sum(grp_total.values())
    res["taxonomy"] = {g: {"pixels": grp_total[g], "share_pct": grp_total[g] / tot_err * 100} for g in GROUPS}
    res["taxonomy"]["total_error_pixels"] = tot_err
    res["errors_with_rare10_gt_pct"] = err_rare_gt / tot_err * 100
    off = cm["final"].copy(); np.fill_diagonal(off, 0)
    rows_sum = cm["final"].sum(1)
    top = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1][:10], off.shape))[0]
    res["top10_confusions"] = [
        {"true": CLASS_NAMES[i], "pred": CLASS_NAMES[j], "pixels": int(off[i, j]),
         "pct_of_true": float(off[i, j] / rows_sum[i] * 100),
         "pct_of_all_errors": float(off[i, j] / tot_err * 100),
         "worst_image": pair_best[(i, j)][0]} for i, j in top]
    for g in GROUPS:
        res["taxonomy"][g]["top_images"] = [r["name"] for r in sorted(per_image, key=lambda r: -r[f"err_{g}"])[:3]]

    # ---- data-driven qualitative selection ----
    mi = np.array([r["miou_final"] for r in per_image])
    ns = np.array([r["n_small_components"] for r in per_image])
    order_good = np.argsort(-mi)[:4]
    hard_pool = [i for i in range(k) if ns[i] >= np.percentile(ns, 75) and mi[i] >= np.median(mi)]
    order_hard = sorted(hard_pool, key=lambda i: -(per_image[i]["miou_final"] - per_image[i]["miou_unet"]))[:4]
    order_fail = np.argsort([r["acc_final"] for r in per_image])[:4]
    sel = {"good": [int(i) for i in order_good], "hard": [int(i) for i in order_hard], "fail": [int(i) for i in order_fail]}
    res["qualitative"] = {
        "rules": {
            "good": "4 highest per-image mIoU of final model",
            "hard": "images in top quartile of #GT components <1024 px AND final per-image mIoU >= median; 4 largest (final - U-Net) per-image mIoU gap",
            "fail": "4 lowest per-image pixel accuracy of final model",
        },
        "small_components_p75": float(np.percentile(ns, 75)), "final_miou_median": float(np.median(mi)),
    }
    for g, idxs in sel.items():
        res["qualitative"][g] = [per_image[i] for i in idxs]
        meta = [(per_image[i]["name"], "") for i in idxs]
        process_labeled_group(meta, unet, final, dev, out / f"a2_m3_qual_{g}.png",
                              f"{g}: RGB | GT | U-Net | DeepLabV3 (final) | error map of final", BASE_DIR / "dataset")
    process_unlabeled_group(UNLABELED_SAMPLES, unet, final, dev, out / "a2_m3_qual_unlabeled.png",
                            "Unlabeled official test images: U-Net vs final DeepLabV3 (red = disagreement)")

    # ---- figures ----
    plot_confusion_matrix(cm["final"], out / "a2_m3_confusion_matrix.png")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    xs = np.arange(2); w = 0.36
    for o, (m, lab) in enumerate([("unet", "U-Net"), ("final", "DeepLabV3 final")]):
        r = res["models"][m]
        bars = a1.bar(xs + (o - 0.5) * w, [r["boundary_err_rate"], r["interior_err_rate"]], w, label=lab)
        a1.bar_label(bars, fmt="%.1f", fontsize=8)
        bars = a2.bar(np.arange(3) + (o - 0.5) * w, [r["scale_recall"][s] for s in ("small", "medium", "large")], w, label=lab)
        a2.bar_label(bars, fmt="%.1f", fontsize=8)
    a1.set_xticks(xs, ["boundary band (k=3 px)", "interior"]); a1.set_ylabel("pixel error rate (%)"); a1.legend()
    a1.set_title("Error rate: boundary band vs interior")
    a2.set_xticks(np.arange(3), ["small (<1024 px)", "medium (1024-9216)", "large (>9216)"])
    a2.set_ylabel("pixel recall (%)"); a2.set_ylim(0, 105); a2.legend(); a2.set_title("Recall by GT component size")
    plt.tight_layout(); plt.savefig(out / "a2_m3_boundary_scale.png", dpi=200); plt.close()

    with open(out / "per_image_test.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(per_image[0].keys())); wr.writeheader(); wr.writerows(per_image)
    with open(out / "error_stats.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps({m: {k2: v for k2, v in res["models"][m].items() if k2 != "per_class"} for m in res["models"]}, indent=1, default=float))
    print(json.dumps(res["taxonomy"], indent=1), "rare-gt errors %.2f" % res["errors_with_rare10_gt_pct"])
    print(json.dumps(res["top10_confusions"], indent=1))
    print(json.dumps(res["qualitative"], indent=1, default=float))


if __name__ == "__main__":
    main()
