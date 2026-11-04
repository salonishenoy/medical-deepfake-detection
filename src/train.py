"""Train + evaluate one detector run on the BTD splits from Team A's loaders.

Examples (run from the repo root, with BTD_DATA_ROOT set):
  python -m src.train --task removal   --train-method CTGAN --seed 0
  python -m src.train --task injection --train-method CTGAN --seed 0
  python -m src.train --task removal   --train-method SD    --seed 0
  python -m src.train --task removal   --train-method CTGAN --no-pretrained --seed 0
  python -m src.train --modality MRI --task removal --seed 0
  python -m src.train --task removal --epochs 2 --limit-batches 3   # quick smoke test

Each run writes runs/<run_name>/:
  best.pt              best checkpoint (by val AUC) + args + threshold
  history.csv          per-epoch train loss / val loss / val AUC
  metrics.json         metrics for every evaluation split
  preds_<split>.csv    per-sample predictions + metadata (for Team C)
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src.models import ARCHS, backbone_modules, build_model, set_backbone_frozen

META_KEEP = ("path", "scan_id", "scanner", "cur_slice", "x", "y", "source_x", "source_y")


# ----------------------------------------------------------------------------- setup
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modality", choices=["CT", "MRI"], default="CT")
    p.add_argument("--task", choices=["injection", "removal"], required=True)
    p.add_argument("--train-method", choices=["CTGAN", "SD"], default="CTGAN", help="CT only; MRI is SD only")
    p.add_argument("--arch", choices=ARCHS, default="efficientnet_b0")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--freeze-epochs", type=int, default=3, help="head-only epochs at the start")
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=1e-4, help="full-network lr after unfreezing")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--patience", type=int, default=7, help="early stopping on val AUC")
    p.add_argument("--recall-target", type=float, default=0.95,
                   help="second threshold: highest threshold reaching this val recall")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0, help="0 on Windows; 2-4 on Mac/Linux")
    p.add_argument("--root", default=None, help="dataset root; defaults to $BTD_DATA_ROOT")
    p.add_argument("--out", default="runs")
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto if omitted)")
    p.add_argument("--limit-batches", type=int, default=0, help="debug: cap batches per pass")
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loaders(args):
    root = args.root or os.environ.get("BTD_DATA_ROOT")
    if args.modality == "CT":
        from src.data_loader import get_dataloaders
        return get_dataloaders(args.task, train_method=args.train_method, root=root,
                               batch_size=args.batch_size, num_workers=args.num_workers)
    from src.mri_data_loader import get_mri_dataloaders
    return get_mri_dataloaders(args.task, root=root, batch_size=args.batch_size,
                               num_workers=args.num_workers)


def run_name(args):
    method = args.train_method if args.modality == "CT" else "SD"
    pre = "pre" if args.pretrained else "scratch"
    return f"{args.arch}_{args.modality}_{args.task}_{method}_{pre}_seed{args.seed}"


# ----------------------------------------------------------------------------- train / predict
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, frozen, arch, limit):
    model.train()
    if frozen:  # keep BatchNorm running stats fixed while the backbone is frozen
        for m in backbone_modules(model, arch):
            m.eval()
    total, n = 0.0, 0
    for i, (images, labels, _meta) in enumerate(loader):
        if limit and i >= limit:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=scaler is not None):
            loss = criterion(model(images).squeeze(1), labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += loss.item() * len(labels)
        n += len(labels)
    return total / max(n, 1)


@torch.inference_mode()
def predict(model, loader, criterion, device, limit):
    """Return (mean loss, DataFrame with label, prob and metadata per sample)."""
    model.eval()
    rows, total, n = [], 0.0, 0
    for i, (images, labels, meta) in enumerate(loader):
        if limit and i >= limit:
            break
        logits = model(images.to(device)).squeeze(1).float().cpu()
        total += criterion(logits, labels.float()).item() * len(labels)
        n += len(labels)
        batch = {"label": labels.tolist(), "prob": torch.sigmoid(logits).tolist()}
        for k in META_KEEP:
            if k in meta:
                v = meta[k]
                batch[k] = v.tolist() if torch.is_tensor(v) else list(v)
        rows.append(pd.DataFrame(batch))
    return total / max(n, 1), pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------- metrics
def metrics_at(y, prob, thr):
    y, prob = np.asarray(y), np.asarray(prob)
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0.0
    return {
        "threshold": float(thr),
        "accuracy": float((tp + tn) / len(y)),
        "balanced_accuracy": float((recall + spec) / 2),
        "precision": float(prec), "recall": float(recall),
        "specificity": float(spec), "f1": float(f1),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def auc(y, prob):
    return float(roc_auc_score(y, prob)) if len(set(y)) == 2 else float("nan")


def choose_thresholds(y_val, p_val, recall_target):
    """Thresholds picked on VALIDATION only, then reused unchanged on test."""
    fpr, tpr, thr = roc_curve(y_val, p_val)
    thr = np.clip(thr, 0.0, 1.0)  # roc_curve's first threshold is inf
    youden = float(thr[np.argmax(tpr - fpr)])
    reach = np.where(tpr >= recall_target)[0]
    high_recall = float(thr[reach[0]]) if len(reach) else 0.0
    return {"default_0.5": 0.5, "youden_val": youden, f"recall{recall_target:g}_val": high_recall}


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    name = run_name(args)
    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"run {name} on {device}")

    loaders = make_loaders(args)
    train_labels = loaders["train"].dataset.df["label"]
    n_fake, n_real = int((train_labels == 1).sum()), int((train_labels == 0).sum())
    print("split sizes:", {k: len(v.dataset) for k, v in loaders.items()},
          f"| train authentic={n_real} fake={n_fake}")

    model = build_model(args.arch, args.pretrained, args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_real / n_fake, device=device))
    eval_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_real / n_fake))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    freeze = args.freeze_epochs if args.pretrained else 0  # nothing to protect when training from scratch
    best_auc, best_epoch, bad, history = -math.inf, -1, 0, []
    optimizer = scheduler = None

    for epoch in range(args.epochs):
        frozen = epoch < freeze
        if epoch == 0 and frozen:
            set_backbone_frozen(model, args.arch, True)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                          lr=args.head_lr, weight_decay=args.weight_decay)
        if epoch == freeze:  # start (or switch to) full fine-tuning
            set_backbone_frozen(model, args.arch, False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - freeze, 1))
            bad = 0  # give the unfrozen phase a fresh patience window

        t0 = time.time()
        tr_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device, scaler,
                                  frozen, args.arch, args.limit_batches)
        if scheduler is not None and not frozen:
            scheduler.step()
        va_loss, va = predict(model, loaders["val"], eval_criterion, device, args.limit_batches)
        va_auc = auc(va["label"], va["prob"])
        history.append({"epoch": epoch, "phase": "head" if frozen else "full", "train_loss": tr_loss,
                        "val_loss": va_loss, "val_auc": va_auc, "lr": optimizer.param_groups[0]["lr"],
                        "seconds": round(time.time() - t0, 1)})
        print(f"epoch {epoch:2d} [{history[-1]['phase']}] train_loss {tr_loss:.4f} "
              f"val_loss {va_loss:.4f} val_auc {va_auc:.4f} ({history[-1]['seconds']}s)")
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)

        if va_auc > best_auc:
            best_auc, best_epoch, bad = va_auc, epoch, 0
            torch.save({"state_dict": model.state_dict(), "args": vars(args), "epoch": epoch,
                        "val_auc": va_auc}, out / "best.pt")
        else:
            bad += 1
            if not frozen and bad >= args.patience:
                print(f"early stop: no val AUC gain for {args.patience} epochs")
                break

    # ---- final evaluation with the best checkpoint
    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    _, va = predict(model, loaders["val"], eval_criterion, device, args.limit_batches)
    thresholds = choose_thresholds(va["label"], va["prob"], args.recall_target)
    ckpt["thresholds"] = thresholds
    torch.save(ckpt, out / "best.pt")

    results = {"run": name, "args": vars(args), "best_epoch": best_epoch, "thresholds": thresholds, "splits": {}}
    for split, loader in loaders.items():
        if split == "train":
            continue
        _, df = (None, va) if split == "val" else predict(model, loader, eval_criterion, device, args.limit_batches)
        df["pred_youden"] = (df["prob"] >= thresholds["youden_val"]).astype(int)
        df.to_csv(out / f"preds_{split}.csv", index=False)
        results["splits"][split] = {
            "n": len(df), "auc": auc(df["label"], df["prob"]),
            **{k: metrics_at(df["label"], df["prob"], t) for k, t in thresholds.items()},
        }
        r = results["splits"][split]
        y = r["youden_val"]
        print(f"{split:10s} n={r['n']:4d} AUC {r['auc']:.3f} | @youden: bal_acc {y['balanced_accuracy']:.3f} "
              f"recall {y['recall']:.3f} spec {y['specificity']:.3f} F1 {y['f1']:.3f} FN {y['fn']}")

    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
