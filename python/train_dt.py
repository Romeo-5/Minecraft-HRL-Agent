#!/usr/bin/env python3
"""
train_dt.py

Offline training of a Decision Transformer on dataset_final.json.

No Minecraft server needed — pure PyTorch training on the 605-sample dataset.

Each sample is converted to a sequence of (RTG, state, action) triplets.
The model learns to predict the next skill given current game state + RTG.

Usage:
    python train_dt.py
    python train_dt.py --epochs 200 --hidden-dim 128 --n-layers 4
    python train_dt.py --checkpoint checkpoints/dt_best.pt  # resume
"""

import argparse
import json
import os
import sys
import math
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Path setup
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "data"))

from models.decision_transformer import DecisionTransformer
from models.state_encoder import StateEncoder, encode_state, SKILL_IDX, PAD_ACTION, ACTION_DIM
from models.rtg_utils import load_reward_table, compute_rtg, total_return


# ── Dataset ──────────────────────────────────────────────────────────────────

class MinecraftDTDataset(Dataset):
    """
    Converts dataset_final.json samples into (states, actions, rtgs, mask) tensors.

    Each sample → one sequence of up to max_len (RTG, state, action) triplets.
    Shorter sequences are right-padded with zeros (states/RTGs) and PAD_ACTION.
    """

    def __init__(self, samples: list, reward_table: dict, max_len: int = 15):
        self.max_len      = max_len
        self.reward_table = reward_table
        self.data         = [self._process(s) for s in samples]

    def _process(self, sample: dict) -> dict:
        path   = sample["reasoning_path"]
        T      = min(len(path), self.max_len)

        state_vec = encode_state(sample)                         # (41,)
        rtg_vals  = compute_rtg(path, self.reward_table)[:T]    # list of T floats
        act_ids   = [SKILL_IDX.get(s, PAD_ACTION) for s in path[:T]]  # list of T ints

        # Pad to max_len
        pad_len = self.max_len - T
        states  = np.stack([state_vec] * self.max_len)                        # (max_len, 41)
        rtgs    = np.array(rtg_vals + [0.0] * pad_len, dtype=np.float32)     # (max_len,)
        actions = np.array(act_ids  + [PAD_ACTION] * pad_len, dtype=np.int64)  # (max_len,)
        mask    = np.array([1] * T  + [0] * pad_len, dtype=np.bool_)         # (max_len,)

        return {
            "states":  states,    # (max_len, 41)
            "rtgs":    rtgs,      # (max_len,)
            "actions": actions,   # (max_len,)
            "mask":    mask,      # (max_len,) — True where valid
            "task":    sample.get("task", ""),
        }

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return {
            "states":  torch.tensor(d["states"],  dtype=torch.float32),
            "rtgs":    torch.tensor(d["rtgs"],    dtype=torch.float32).unsqueeze(-1),  # (T, 1)
            "actions": torch.tensor(d["actions"], dtype=torch.long),
            "mask":    torch.tensor(d["mask"],    dtype=torch.bool),
        }


# ── Splitting ─────────────────────────────────────────────────────────────────

def stratified_split(samples: list, val_frac=0.1, test_frac=0.1, seed=42):
    """Split samples by task to preserve task distribution across splits."""
    rng = random.Random(seed)
    by_task = defaultdict(list)
    for s in samples:
        by_task[s.get("task", "unknown")].append(s)

    train, val, test = [], [], []
    for task_samples in by_task.values():
        rng.shuffle(task_samples)
        n = len(task_samples)
        n_test = max(1, int(n * test_frac))
        n_val  = max(1, int(n * val_frac))
        test  += task_samples[:n_test]
        val   += task_samples[n_test:n_test + n_val]
        train += task_samples[n_test + n_val:]

    return train, val, test


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss(reduction="none")

    for batch in loader:
        states  = batch["states"].to(device)   # (B, T, 41)
        rtgs    = batch["rtgs"].to(device)     # (B, T, 1)
        actions = batch["actions"].to(device)  # (B, T)
        mask    = batch["mask"].to(device)     # (B, T) bool

        logits = model(states, actions, rtgs)  # (B, T, act_dim)

        B, T, A = logits.shape
        loss_flat = criterion(logits.reshape(B * T, A), actions.reshape(B * T))
        loss_flat = loss_flat * mask.reshape(B * T).float()
        loss = loss_flat.sum() / mask.float().sum().clamp(min=1)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accuracy (only on valid tokens)
        preds = logits.argmax(dim=-1)  # (B, T)
        correct = ((preds == actions) & mask).sum().item()

        total_loss    += loss.item() * mask.float().sum().item()
        total_correct += correct
        total_tokens  += mask.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    acc      = total_correct / max(total_tokens, 1)
    return avg_loss, acc


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss, total_correct, total_top3, total_tokens = 0.0, 0, 0, 0
    criterion = nn.CrossEntropyLoss(reduction="none")

    for batch in loader:
        states  = batch["states"].to(device)
        rtgs    = batch["rtgs"].to(device)
        actions = batch["actions"].to(device)
        mask    = batch["mask"].to(device)

        logits = model(states, actions, rtgs)

        B, T, A = logits.shape
        loss_flat = criterion(logits.reshape(B * T, A), actions.reshape(B * T))
        loss_flat = loss_flat * mask.reshape(B * T).float()
        loss = loss_flat.sum() / mask.float().sum().clamp(min=1)

        preds   = logits.argmax(dim=-1)
        top3    = logits.topk(3, dim=-1).indices           # (B, T, 3)
        correct = ((preds == actions) & mask).sum().item()
        top3_correct = ((top3 == actions.unsqueeze(-1)).any(dim=-1) & mask).sum().item()

        total_loss    += loss.item() * mask.float().sum().item()
        total_correct += correct
        total_top3    += top3_correct
        total_tokens  += mask.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    top1_acc = total_correct / max(total_tokens, 1)
    top3_acc = total_top3    / max(total_tokens, 1)
    return avg_loss, top1_acc, top3_acc


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Decision Transformer on dataset_final.json")
    p.add_argument("--dataset",     default=str(ROOT.parent / "data" / "processed" / "dataset_final.json"))
    p.add_argument("--tech-tree",   default=os.path.join(os.path.expanduser("~"), "Documents", "GitHub", "MC_Tech_Tree", "training_config.json"))
    p.add_argument("--checkpoint",  default=None,  help="Path to resume from checkpoint")
    p.add_argument("--out-dir",     default=str(ROOT / "checkpoints"), help="Output directory for checkpoints")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight-decay",type=float, default=1e-4)
    p.add_argument("--hidden-dim",  type=int,   default=64)
    p.add_argument("--n-layers",    type=int,   default=2)
    p.add_argument("--n-heads",     type=int,   default=4)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--max-len",     type=int,   default=15)
    p.add_argument("--patience",    type=int,   default=15, help="Early stopping patience")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--device",      default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return p.parse_args()


def main():
    args = parse_args()

    # ── Device ──────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset}")
    with open(args.dataset) as f:
        samples = json.load(f)
    print(f"  {len(samples)} samples loaded")

    print(f"Loading reward table: {args.tech_tree}")
    reward_table = load_reward_table(args.tech_tree)

    train_samples, val_samples, test_samples = stratified_split(
        samples, val_frac=0.1, test_frac=0.1, seed=args.seed
    )
    print(f"  Train: {len(train_samples)}  Val: {len(val_samples)}  Test: {len(test_samples)}")

    train_ds = MinecraftDTDataset(train_samples, reward_table, args.max_len)
    val_ds   = MinecraftDTDataset(val_samples,   reward_table, args.max_len)
    test_ds  = MinecraftDTDataset(test_samples,  reward_table, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Model ────────────────────────────────────────────────────────────────
    model = DecisionTransformer(
        hidden_dim = args.hidden_dim,
        n_layers   = args.n_layers,
        n_heads    = args.n_heads,
        dropout    = args.dropout,
        max_len    = args.max_len,
    ).to(device)
    print(f"Model parameters: {model.num_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    start_epoch    = 0
    best_val_loss  = float("inf")
    patience_count = 0
    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, "dt_best.pt")

    # ── Resume ───────────────────────────────────────────────────────────────
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\n{'Epoch':>6}  {'TrainLoss':>10}  {'TrainAcc':>9}  {'ValLoss':>8}  {'ValTop1':>8}  {'ValTop3':>8}  {'LR':>8}")
    print("-" * 75)

    for epoch in range(start_epoch, args.epochs):
        tr_loss, tr_acc           = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_top1, val_top3 = eval_epoch(model, val_loader, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch+1:>6}  {tr_loss:>10.4f}  {tr_acc:>8.1%}  {val_loss:>8.4f}  {val_top1:>8.1%}  {val_top3:>8.1%}  {lr:>8.2e}")

        # ── Checkpoint best model ─────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save({
                "epoch":         epoch,
                "model":         model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "scheduler":     scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "args":          vars(args),
            }, best_path)
            print(f"         ✓ saved best model (val_loss={val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1} (patience={args.patience})")
                break

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_top1, test_top3 = eval_epoch(model, test_loader, device)
    print(f"Test  — loss: {test_loss:.4f}  top-1: {test_top1:.1%}  top-3: {test_top3:.1%}")
    print(f"\nBest checkpoint: {best_path}")


if __name__ == "__main__":
    main()
