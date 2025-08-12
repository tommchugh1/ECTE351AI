#!/usr/bin/env python3
"""
Bolt defect detection (multi-label) trainer
- Expects pre-cropped, preprocessed bolt images on disk
- Uses a CSV manifest for each split with columns:
    path,label_cross,label_ding,label_bend,bolt_id,frame_id,lot_id,datetime
  Only 'path' and the three 'label_*' columns are required.

Outputs in --out-dir:
  best.pt                  Best model weights by macro-AP on val
  final.pt                 Last-epoch weights
  thresholds.json          Per-class decision thresholds at target precision
  temperatures.json        Per-class temperature scaling factors
  metrics_val.json         Validation metrics at best checkpoint
  metrics_test.json        Test metrics (after calibration)
  model.onnx               ONNX export (opset 17)
  model.ts                 TorchScript export

Example:
  python bolt_defect_training.py \
    --csv-train data/train.csv \
    --csv-val data/val.csv \
    --csv-test data/test.csv \
    --image-size 320 \
    --arch resnet18 \
    --epochs 50 --batch-size 64 --precision-target 0.995 \
    --out-dir runs/bolts_v1
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from torchvision.models import resnet18, mobilenet_v3_small, efficientnet_b0


# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Args:
    csv_train: str
    csv_val: str
    csv_test: str | None
    out_dir: str
    image_size: int = 320
    arch: str = "resnet18"  # resnet18, mobilenet_v3_small, efficientnet_b0
    pretrained: bool = False
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    precision_target: float = 0.995  # target precision for thresholding
    early_stop: int = 8
    mixed_precision: bool = True
    resume: str | None = None
    seed: int = 42


CLASS_NAMES = ["cross", "ding", "bend"]
NUM_CLASSES = 3


# ----------------------------
# Dataset
# ----------------------------
class BoltDataset(Dataset):
    def __init__(self, csv_path: str, image_size: int, augment: bool):
        df = pd.read_csv(csv_path)
        missing = {"path", "label_cross", "label_ding", "label_bend"} - set(df.columns)
        if missing:
            raise ValueError(f"CSV {csv_path} missing columns: {sorted(list(missing))}")
        self.paths = df["path"].astype(str).tolist()
        labels = df[["label_cross", "label_ding", "label_bend"]].astype(float).values
        self.labels = labels.astype(np.float32)
        self.augment = augment
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if augment:
            self.tf = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.95, 1.05), shear=0.0),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.1),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        with Image.open(p) as im:
            im = im.convert("RGB")
            x = self.tf(im)
        y = torch.from_numpy(self.labels[idx])
        return x, y


# ----------------------------
# Model factory
# ----------------------------

def build_model(arch: str, pretrained: bool, num_classes: int = NUM_CLASSES) -> nn.Module:
    if arch == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        m = resnet18(weights=weights)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        return m
    if arch == "mobilenet_v3_small":
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        m = mobilenet_v3_small(weights=weights)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    if arch == "efficientnet_b0":
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = efficientnet_b0(weights=weights)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    raise ValueError(f"Unknown arch: {arch}")


# ----------------------------
# Metrics: PR curve and AP (no sklearn dependency)
# ----------------------------

def precision_recall_curve_binary(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return precision, recall, thresholds similar to sklearn, y_true in {0,1}."""
    # Sort by decreasing score
    desc_score_indices = np.argsort(-y_score)
    y_true = y_true[desc_score_indices]
    y_score = y_score[desc_score_indices]

    # True positives cumulative
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    # Avoid division by zero
    precision = tp / np.maximum(tp + fp, 1)
    # Total positives
    total_pos = tp[-1] if tp.size > 0 else 0
    recall = tp / max(total_pos, 1)

    # Thresholds are the distinct scores
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    precision = precision[threshold_idxs]
    recall = recall[threshold_idxs]
    thresholds = y_score[threshold_idxs]

    # Add (0,1) start point per convention
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    thresholds = np.r_[thresholds[0] if thresholds.size else 1.0, thresholds]
    return precision, recall, thresholds


def average_precision_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    # Make precision non-increasing (envelope)
    for i in range(precision.size - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    # Area under curve using recall as x-axis
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def find_threshold_for_precision(y_true: np.ndarray, y_score: np.ndarray, precision_target: float) -> Tuple[float, float, float]:
    """Return threshold achieving >= precision_target with max recall.
    Returns: threshold, achieved_precision, achieved_recall
    """
    p, r, t = precision_recall_curve_binary(y_true, y_score)
    mask = p >= precision_target
    if not np.any(mask):
        # If we cannot reach the precision target, pick the point with max precision
        idx = int(np.argmax(p))
        return float(t[idx]), float(p[idx]), float(r[idx])
    # Among points meeting precision, pick max recall
    idxs = np.where(mask)[0]
    idx = int(idxs[np.argmax(r[idxs])])
    return float(t[idx]), float(p[idx]), float(r[idx])


# ----------------------------
# Temperature scaling (one T per head)
# ----------------------------
class TemperatureScaler(nn.Module):
    def __init__(self, init_log_t: float = 0.0):
        super().__init__()
        # We store log-temperature to ensure positivity
        self.log_t = nn.Parameter(torch.tensor([init_log_t], dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # logits shape: (N,) for a single head
        T = torch.exp(self.log_t) + 1e-6
        return logits / T


def fit_temperature_per_head(logits: torch.Tensor, labels: torch.Tensor, max_steps: int = 500, lr: float = 0.01) -> float:
    device = logits.device
    scaler = TemperatureScaler(0.0).to(device)
    opt = torch.optim.Adam([scaler.log_t], lr=lr)
    for _ in range(max_steps):
        opt.zero_grad(set_to_none=True)
        scaled = scaler(logits)
        loss = F.binary_cross_entropy_with_logits(scaled, labels)
        loss.backward()
        opt.step()
    with torch.no_grad():
        T = float(torch.exp(scaler.log_t).item())
    return T


# ----------------------------
# Training and evaluation
# ----------------------------

def compute_class_pos_weights(loader: DataLoader) -> torch.Tensor:
    total = 0
    pos = torch.zeros(NUM_CLASSES)
    for _, y in loader:
        total += y.shape[0]
        pos += y.sum(dim=0)
    neg = total - pos
    # pos_weight = neg / pos for BCEWithLogitsLoss
    pos_weight = neg / torch.clamp(pos, min=1.0)
    pos_weight = torch.clamp(pos_weight, 1.0, 10.0)
    return pos_weight


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, return_logits: bool = False):
    model.eval()
    ys = []
    ps = []
    logits_all = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x)
            logits = out
            prob = torch.sigmoid(logits)
            ys.append(y.cpu().numpy())
            ps.append(prob.cpu().numpy())
            if return_logits:
                logits_all.append(logits.cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    y_prob = np.concatenate(ps, axis=0)
    metrics = {}
    ap_per_class = []
    for k, name in enumerate(CLASS_NAMES):
        yk = y_true[:, k]
        pk = y_prob[:, k]
        if yk.sum() == 0:
            ap = float("nan")
            p = np.array([1.0])
            r = np.array([0.0])
        else:
            p, r, _ = precision_recall_curve_binary(yk, pk)
            ap = average_precision_from_pr(p.copy(), r.copy())
        metrics[f"AP_{name}"] = ap
        ap_per_class.append(0.0 if np.isnan(ap) else ap)
    metrics["AP_macro"] = float(np.mean(ap_per_class))
    if return_logits:
        logits_np = np.concatenate(logits_all, axis=0)
        return metrics, y_true, y_prob, logits_np
    return metrics, y_true, y_prob


def train(args: Args):
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    ds_train = BoltDataset(args.csv_train, args.image_size, augment=True)
    ds_val = BoltDataset(args.csv_val, args.image_size, augment=False)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    model = build_model(args.arch, args.pretrained, NUM_CLASSES)
    model.to(device)

    # Loss with class imbalance handling
    pos_weight = compute_class_pos_weights(dl_train).to(device)
    print(f"pos_weight: {pos_weight}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Cosine schedule with warmup: simple implementation
    total_steps = args.epochs * max(1, len(dl_train))
    warmup_steps = int(0.1 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    best_ap = -1.0
    best_path = os.path.join(args.out_dir, "best.pt")
    epochs_no_improve = 0

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        for xb, yb in dl_train:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += float(loss.item())
            global_step += 1
        train_loss = loss_sum / max(1, len(dl_train))

        # Validate
        metrics_val, _, _ = evaluate(model, dl_val, device)
        ap_macro = metrics_val["AP_macro"]
        dt = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:03d} | loss {train_loss:.4f} | AP_macro {ap_macro:.4f} | lr {lr_now:.3e} | {dt:.1f}s")

        # Save last
        torch.save({"model": model.state_dict(), "epoch": epoch}, os.path.join(args.out_dir, "final.pt"))

        # Early stopping on macro-AP
        if ap_macro > best_ap:
            best_ap = ap_macro
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)
            with open(os.path.join(args.out_dir, "metrics_val.json"), "w") as f:
                json.dump(metrics_val, f, indent=2)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop:
                print("Early stopping.")
                break

    # Load best and run calibration + thresholding
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])  # type: ignore

    # Collect logits on val for temperature scaling
    metrics_val, y_true_val, y_prob_val, logits_val = evaluate(model, dl_val, device, return_logits=True)

    # Fit one temperature per head
    logits_t = torch.from_numpy(logits_val).to(device)
    labels_t = torch.from_numpy(y_true_val).to(device)
    temperatures: List[float] = []
    for k in range(NUM_CLASSES):
        T_k = fit_temperature_per_head(logits_t[:, k], labels_t[:, k])
        temperatures.append(T_k)
    with open(os.path.join(args.out_dir, "temperatures.json"), "w") as f:
        json.dump({CLASS_NAMES[i]: temperatures[i] for i in range(NUM_CLASSES)}, f, indent=2)

    # Apply temperature and sweep thresholds on val
    logits_scaled = logits_val / np.array(temperatures)[None, :]
    y_prob_cal = 1.0 / (1.0 + np.exp(-logits_scaled))

    thresholds: Dict[str, Dict[str, float]] = {}
    for k, name in enumerate(CLASS_NAMES):
        thr, p_at_thr, r_at_thr = find_threshold_for_precision(y_true_val[:, k], y_prob_cal[:, k], args.precision_target)
        thresholds[name] = {"threshold": thr, "precision": p_at_thr, "recall": r_at_thr}
    with open(os.path.join(args.out_dir, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    # Optional test evaluation
    if args.csv_test:
        ds_test = BoltDataset(args.csv_test, args.image_size, augment=False)
        dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        # Evaluate with calibrated logits
        model.eval()
        ys = []
        logits_all = []
        with torch.no_grad():
            for xb, yb in dl_test:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                ys.append(yb.cpu().numpy())
                logits_all.append(logits.cpu().numpy())
        y_true_test = np.concatenate(ys, axis=0)
        logits_test = np.concatenate(logits_all, axis=0)
        logits_test_scaled = logits_test / np.array(temperatures)[None, :]
        y_prob_test = 1.0 / (1.0 + np.exp(-logits_test_scaled))
        # Compute macro AP
        metrics_test = {}
        ap_per_class = []
        for k, name in enumerate(CLASS_NAMES):
            yk = y_true_test[:, k]
            pk = y_prob_test[:, k]
            if yk.sum() == 0:
                ap = float("nan")
            else:
                p, r, _ = precision_recall_curve_binary(yk, pk)
                ap = average_precision_from_pr(p.copy(), r.copy())
            metrics_test[f"AP_{name}"] = ap
            ap_per_class.append(0.0 if np.isnan(ap) else ap)
        metrics_test["AP_macro"] = float(np.mean(ap_per_class))
        # Also compute bolt-level decisions with thresholds
        recalls = {}
        precisions = {}
        for k, name in enumerate(CLASS_NAMES):
            thr = thresholds[name]["threshold"]
            yhat = (y_prob_test[:, k] >= thr).astype(np.int32)
            yk = y_true_test[:, k].astype(np.int32)
            tp = int(((yhat == 1) & (yk == 1)).sum())
            fp = int(((yhat == 1) & (yk == 0)).sum())
            fn = int(((yhat == 0) & (yk == 1)).sum())
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            precisions[name] = prec
            recalls[name] = rec
        metrics_test["precision_at_thresholds"] = precisions
        metrics_test["recall_at_thresholds"] = recalls
        with open(os.path.join(args.out_dir, "metrics_test.json"), "w") as f:
            json.dump(metrics_test, f, indent=2)
        print("Test metrics:", json.dumps(metrics_test, indent=2))

    # Exports
    model.eval()
    dummy = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    # TorchScript
    try:
        ts = torch.jit.trace(model, dummy)
        ts_path = os.path.join(args.out_dir, "model.ts")
        ts.save(ts_path)
        print(f"Saved TorchScript to {ts_path}")
    except Exception as e:
        print("TorchScript export failed:", e)
    # ONNX
    try:
        onnx_path = os.path.join(args.out_dir, "model.onnx")
        torch.onnx.export(
            model, dummy, onnx_path, opset_version=17, input_names=["images"], output_names=["logits"],
            dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        )
        print(f"Saved ONNX to {onnx_path}")
    except Exception as e:
        print("ONNX export failed:", e)


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Train multi-label bolt defect classifier")
    p.add_argument("--csv-train", required=True)
    p.add_argument("--csv-val", required=True)
    p.add_argument("--csv-test", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--image-size", type=int, default=320)
    p.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "mobilenet_v3_small", "efficientnet_b0"])
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--precision-target", type=float, default=0.995)
    p.add_argument("--early-stop", type=int, default=8)
    p.add_argument("--no-amp", dest="mixed_precision", action="store_false", help="Disable mixed precision")
    p.add_argument("--seed", type=int, default=42)
    args_ns = p.parse_args()
    return Args(
        csv_train=args_ns.csv_train,
        csv_val=args_ns.csv_val,
        csv_test=args_ns.csv_test,
        out_dir=args_ns.out_dir,
        image_size=args_ns.image_size,
        arch=args_ns.arch,
        pretrained=args_ns.pretrained,
        epochs=args_ns.epochs,
        batch_size=args_ns.batch_size,
        lr=args_ns.lr,
        weight_decay=args_ns.weight_decay,
        num_workers=args_ns.num_workers,
        precision_target=args_ns.precision_target,
        early_stop=args_ns.early_stop,
        mixed_precision=args_ns.mixed_precision,
        seed=args_ns.seed,
    )


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2, default=str))
    train(args)
