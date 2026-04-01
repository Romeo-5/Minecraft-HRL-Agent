#!/usr/bin/env python3
"""
eval_dt.py

Evaluate a trained Decision Transformer on the dataset_final.json samples.

Generates skill sequences autoregressively and scores them against ground truth
using the same three metrics as the LLM baseline in evaluate_results.py:
  - step_coverage:      fraction of ground truth steps matched (fuzzy)
  - shortcut_detection: did model predict a structure loot skill when one was available?
  - efficiency:         ground_truth_length / predicted_length

Results are printed as a summary table matching Table 1 in the report.

Usage:
    python eval_dt.py --checkpoint checkpoints/dt_best.pt
    python eval_dt.py --checkpoint checkpoints/dt_best.pt --split test
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "data"))

from models.decision_transformer import DecisionTransformer
from models.state_encoder import (
    StateEncoder, encode_state, SKILL_IDX, ACTION_DIM, PAD_ACTION
)
from models.rtg_utils import load_reward_table, total_return

# Reuse evaluate_results.py step coverage logic
sys.path.insert(0, str(ROOT.parent / "data"))
try:
    from evaluate_results import compute_step_coverage, STEP_SYNONYMS
    _HAVE_EVALUATOR = True
except ImportError:
    _HAVE_EVALUATOR = False


# ── Simple step coverage fallback ────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split("_")), set(b.split("_"))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def simple_step_coverage(ground_truth: list, predicted: list, threshold: float = 0.6) -> float:
    """Greedy bipartite matching coverage (simplified version of evaluate_results.py)."""
    if not ground_truth:
        return 1.0
    matched = set()
    for gt in ground_truth:
        for i, pred in enumerate(predicted):
            if i in matched:
                continue
            score = 1.0 if gt == pred else _jaccard(gt, pred)
            if score >= threshold:
                matched.add(i)
                break
    return len(matched) / len(ground_truth)


# ── Structure shortcut keywords (from evaluate_results.py) ───────────────────

STRUCTURE_KEYWORDS = {
    "blacksmith", "loot_blacksmith", "village", "mineshaft", "desert_temple",
    "jungle_temple", "shipwreck", "igloo", "ruined_portal", "dungeon",
    "loot_supply", "search_mineshaft", "loot_portal",
}


def shortcut_detected(sample: dict, predicted_skills: list) -> bool | None:
    """
    Returns True if model predicted a structure shortcut skill when one was available.
    Returns None if context_matters is False or no structures nearby.
    """
    if not sample.get("context_matters", False):
        return None
    structs = sample.get("nearby_structures", [])
    if not structs or structs == ["none"]:
        return None
    predicted_text = " ".join(predicted_skills)
    return any(kw in predicted_text for kw in STRUCTURE_KEYWORDS)


# ── Inference ─────────────────────────────────────────────────────────────────

def generate_sequence(model, sample, reward_table, device, max_steps=15):
    """
    Autoregressively generate a skill sequence for a single sample.
    Returns list of canonical skill strings.
    """
    from models.state_encoder import SKILL_VOCAB

    state_vec  = torch.tensor(encode_state(sample), dtype=torch.float32)
    target_rtg = total_return(sample["reasoning_path"], reward_table)
    target_rtg = max(target_rtg, 1.0)  # ensure positive conditioning

    history_states  = []
    history_actions = []
    predicted_skills = []

    for step in range(max_steps):
        action_idx = model.predict(
            state       = state_vec,
            rtg         = target_rtg,
            history_states  = history_states,
            history_actions = history_actions,
            temperature = 1.0,
            device      = device,
        )
        if action_idx == PAD_ACTION:
            break  # model predicts padding → stop

        skill = SKILL_VOCAB[action_idx] if action_idx < len(SKILL_VOCAB) else None
        if skill is None:
            break

        predicted_skills.append(skill)
        history_states.append(state_vec)
        history_actions.append(action_idx)

        # Stop if we've repeated the same skill 3 times in a row (degenerate loop)
        if len(predicted_skills) >= 3 and predicted_skills[-3:] == [skill] * 3:
            break

    return predicted_skills


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Decision Transformer")
    p.add_argument("--checkpoint", required=True, help="Path to dt_best.pt")
    p.add_argument("--dataset",    default=str(ROOT.parent / "data" / "processed" / "dataset_final.json"))
    p.add_argument("--tech-tree",  default=os.path.join(os.path.expanduser("~"), "Documents", "GitHub", "MC_Tech_Tree", "training_config.json"))
    p.add_argument("--split",      default="all", choices=["all", "test", "val"],
                   help="Which split to evaluate (all = full dataset)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--out",        default=None, help="Save JSONL results to this path")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Device ──────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    print(f"Device: {device}")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})

    model = DecisionTransformer(
        hidden_dim = saved_args.get("hidden_dim", 64),
        n_layers   = saved_args.get("n_layers",   2),
        n_heads    = saved_args.get("n_heads",     4),
        dropout    = 0.0,   # no dropout at eval
        max_len    = saved_args.get("max_len",     15),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Parameters: {model.num_parameters():,}")

    # ── Load data ────────────────────────────────────────────────────────────
    with open(args.dataset) as f:
        samples = json.load(f)
    reward_table = load_reward_table(args.tech_tree)

    # Optionally filter to test split (same seed/logic as train_dt.py)
    if args.split != "all":
        from collections import defaultdict
        rng = random.Random(args.seed)
        by_task = defaultdict(list)
        for s in samples:
            by_task[s.get("task", "unknown")].append(s)
        test_samples = []
        for task_samples in by_task.values():
            rng.shuffle(task_samples)
            n = len(task_samples)
            n_test = max(1, int(n * 0.1))
            n_val  = max(1, int(n * 0.1))
            if args.split == "test":
                test_samples += task_samples[:n_test]
            else:
                test_samples += task_samples[n_test:n_test + n_val]
        samples = test_samples
        print(f"Using {args.split} split: {len(samples)} samples")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print(f"\nEvaluating {len(samples)} samples...\n")

    coverages   = []
    shortcuts   = []   # True/False only (excluding None)
    efficiencies = []
    results     = []

    for i, sample in enumerate(samples):
        predicted = generate_sequence(model, sample, reward_table, device)
        ground_truth = sample["reasoning_path"]

        cov = simple_step_coverage(ground_truth, predicted)
        sc  = shortcut_detected(sample, predicted)
        eff = len(ground_truth) / max(len(predicted), len(ground_truth)) if predicted else 0.0

        coverages.append(cov)
        if sc is not None:
            shortcuts.append(float(sc))
        efficiencies.append(eff)

        results.append({
            "sample_id":          sample.get("id"),
            "task":               sample.get("task"),
            "biome":              sample.get("biome"),
            "ground_truth":       ground_truth,
            "predicted":          predicted,
            "step_coverage":      round(cov, 4),
            "shortcut_detected":  sc,
            "efficiency":         round(eff, 4),
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(samples)} done...")

    # ── Print results table ───────────────────────────────────────────────────
    mean_cov  = np.mean(coverages) * 100
    mean_sc   = np.mean(shortcuts) * 100 if shortcuts else float("nan")
    mean_eff  = np.mean(efficiencies) * 100

    print("\n" + "=" * 65)
    print("Decision Transformer — Evaluation Results")
    print("=" * 65)
    print(f"{'Model':<25} {'Condition':<12} {'Coverage':>10} {'Shortcut':>10} {'Efficiency':>10}")
    print("-" * 65)
    print(f"{'decision_transformer':<25} {'env_aware':<12} {mean_cov:>9.1f}% {mean_sc:>9.1f}% {mean_eff:>9.1f}%")
    print("=" * 65)
    print("\nBaseline (from Table 1):")
    print(f"  llama3.2:3b  env-aware   16.6%    96%    65%")
    print(f"  mistral:7b   env-aware   18.8%    86%    34%")
    print()

    # Per-task breakdown
    from collections import defaultdict as _dd
    by_task = _dd(lambda: {"cov": [], "eff": []})
    for r in results:
        by_task[r["task"]]["cov"].append(r["step_coverage"])
        by_task[r["task"]]["eff"].append(r["efficiency"])
    print("Per-task coverage:")
    for task, vals in sorted(by_task.items()):
        print(f"  {task:<30} {np.mean(vals['cov'])*100:6.1f}%")

    # ── Save JSONL ───────────────────────────────────────────────────────────
    if args.out:
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
