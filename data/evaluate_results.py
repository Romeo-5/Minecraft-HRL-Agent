#!/usr/bin/env python3
"""
Evaluate Benchmark Results for Minecraft Reasoning Path Dataset.

Computes metrics:
  - Step coverage: fraction of ground truth steps mentioned by model
  - Shortcut detection: did the model identify structure-based shortcuts?
  - Efficiency score: ratio of ground truth steps to predicted steps
  - Context benefit: improvement from with_context vs without_context

Usage:
    python evaluate_results.py
    python evaluate_results.py --results benchmark_results/results.jsonl
    python evaluate_results.py --threshold 0.5  # lower fuzzy match threshold
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# STEP SYNONYM DICTIONARY
# =============================================================================

# Maps canonical step names to alternative phrasings LLMs might use
STEP_SYNONYMS: Dict[str, List[str]] = {
    # Wood & basic crafting
    "harvest_wood": [
        "chop tree", "punch tree", "collect wood", "gather logs", "get wood",
        "find tree", "break tree", "cut tree", "obtain log", "chop down",
        "harvest log", "collect log", "mine tree", "gather wood",
    ],
    "craft_planks": [
        "make planks", "convert logs", "create planks", "wooden planks",
        "turn logs into planks", "craft wooden planks",
    ],
    "craft_planks_and_sticks": [
        "make planks and sticks", "craft planks", "create sticks",
        "convert logs to planks", "make sticks", "craft sticks",
    ],
    "craft_crafting_table": [
        "make crafting table", "create workbench", "build crafting table",
        "craft workbench", "place crafting table",
    ],
    "craft_wooden_pickaxe": [
        "make wooden pickaxe", "craft wood pickaxe", "create wooden pick",
        "wooden pickaxe", "make wood pick",
    ],
    # Mining & stone
    "mine_stone": [
        "mine cobblestone", "dig stone", "collect cobblestone", "gather stone",
        "break stone", "mine cobble", "get cobblestone",
    ],
    "craft_stone_pickaxe": [
        "make stone pickaxe", "craft stone pick", "create stone pickaxe",
        "stone pickaxe", "upgrade to stone",
    ],
    # Iron
    "mine_iron_ore": [
        "mine iron", "find iron", "collect iron ore", "gather iron",
        "dig for iron", "obtain iron ore", "locate iron",
    ],
    "craft_furnace": [
        "make furnace", "build furnace", "create furnace", "craft a furnace",
    ],
    "smelt_iron": [
        "smelt iron ore", "cook iron", "process iron", "turn iron ore into ingots",
        "make iron ingots", "create iron ingot",
    ],
    "craft_iron_pickaxe": [
        "make iron pickaxe", "craft iron pick", "create iron pickaxe",
        "iron pickaxe",
    ],
    # Diamond
    "mine_diamonds": [
        "mine diamond", "find diamonds", "collect diamonds", "gather diamonds",
        "dig for diamonds", "obtain diamonds", "locate diamonds",
    ],
    "craft_diamond_pickaxe": [
        "make diamond pickaxe", "craft diamond pick", "create diamond pickaxe",
        "diamond pickaxe",
    ],
    "dig_to_diamond_level": [
        "dig down to y -59", "dig to y=-59", "mine down to diamond level",
        "go to diamond level", "reach y -59", "descend to diamond",
        "dig to bedrock", "strip mine", "branch mine at diamond level",
    ],
    # Gold
    "mine_gold_ore": [
        "mine gold", "find gold", "collect gold ore", "dig for gold",
    ],
    "smelt_gold": [
        "smelt gold ore", "cook gold", "make gold ingots", "process gold",
    ],
    "dig_to_gold_level": [
        "dig down to y -16", "mine down to gold level", "go to gold level",
    ],
    # Food
    "search_for_animals": [
        "find animals", "look for animals", "locate livestock",
        "hunt for animals", "find cows", "find pigs", "find sheep",
    ],
    "kill_animals_for_meat": [
        "kill animals", "hunt animals", "slaughter animals", "get meat",
        "obtain raw meat", "kill cow", "kill pig", "kill chicken",
    ],
    "cook_meat": [
        "cook food", "smelt meat", "roast meat", "grill meat",
        "cook in furnace", "use campfire",
    ],
    "craft_furnace_or_campfire": [
        "make furnace", "build campfire", "craft campfire", "create furnace",
    ],
    # Shelter
    "build_walls_and_roof": [
        "build walls", "construct shelter", "place blocks", "build house",
        "make walls", "create structure",
    ],
    "craft_and_place_door": [
        "make door", "craft door", "place door", "create door",
    ],
    "place_torches": [
        "make torches", "craft torches", "light up", "add lighting",
        "illuminate", "place torches inside",
    ],
    # Structure interactions
    "go_to_village": [
        "travel to village", "walk to village", "head to village",
        "navigate to village", "find village", "reach village",
    ],
    "find_blacksmith": [
        "locate blacksmith", "find weaponsmith", "find toolsmith",
        "search for blacksmith",
    ],
    "loot_blacksmith_chest": [
        "open blacksmith chest", "check blacksmith", "search blacksmith chest",
        "loot chest", "take items from blacksmith",
    ],
    "navigate_to_mineshaft": [
        "go to mineshaft", "find mineshaft", "enter mineshaft",
        "explore mineshaft", "locate mineshaft",
    ],
    "search_mineshaft_chests": [
        "loot mineshaft chest", "check mineshaft chests",
        "open chest in mineshaft",
    ],
    "go_to_ruined_portal": [
        "find ruined portal", "travel to portal", "locate portal ruins",
    ],
    "loot_portal_chest": [
        "open portal chest", "check portal chest", "search portal loot",
    ],
    "go_to_desert_temple": [
        "find desert temple", "travel to temple", "locate pyramid",
        "go to pyramid",
    ],
    "avoid_tnt_trap": [
        "avoid trap", "watch for tnt", "disarm trap", "break pressure plate",
        "avoid pressure plate",
    ],
    "go_to_jungle_temple": [
        "find jungle temple", "travel to jungle temple", "locate jungle temple",
    ],
    "swim_to_shipwreck": [
        "find shipwreck", "go to shipwreck", "locate shipwreck",
        "swim to wreck", "dive to shipwreck",
    ],
    "loot_supply_chest": [
        "open supply chest", "check supply chest", "search shipwreck food",
    ],
    "go_to_igloo": [
        "find igloo", "travel to igloo", "locate igloo",
    ],
    # Biome-specific
    "harvest_melons_from_ground": [
        "collect melons", "pick melons", "gather melons", "find melons",
    ],
    "harvest_sweet_berries_from_bushes": [
        "collect sweet berries", "pick berries", "gather berries",
        "harvest berries", "find sweet berries",
    ],
    "harvest_village_crops": [
        "harvest crops", "collect farm food", "take village food",
        "harvest wheat", "harvest carrots", "harvest potatoes",
    ],
    "milk_mooshroom_with_bowl": [
        "milk mooshroom", "get mushroom stew", "use bowl on mooshroom",
    ],
    # Armor
    "craft_iron_armor_set": [
        "craft iron armor", "make iron armor", "create armor",
        "craft helmet", "craft chestplate", "craft leggings", "craft boots",
        "make iron helmet", "make iron chestplate",
    ],
}


# =============================================================================
# FUZZY MATCHING
# =============================================================================

def normalize_step(text: str) -> str:
    """Normalize a step string for comparison."""
    text = text.lower().strip()
    # Remove numbering
    text = re.sub(r"^\d+[\.\)\:]\s*", "", text)
    # Remove common filler words
    fillers = [
        "then", "next", "now", "first", "finally", "afterwards",
        "after that", "subsequently", "once done", "you should",
        "you need to", "you can", "make sure to", "be sure to",
        "remember to", "don't forget to",
    ]
    for filler in fillers:
        text = text.replace(filler, "")
    # Remove articles
    text = re.sub(r"\b(a|an|the|some|any)\b", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove punctuation except hyphens
    text = re.sub(r"[^\w\s-]", "", text)
    return text


def tokenize(text: str) -> set:
    """Split text into word tokens."""
    return set(normalize_step(text).split())


def fuzzy_match_score(predicted: str, ground_truth: str) -> float:
    """Score how well a predicted step matches a ground truth step.

    Returns score in [0.0, 1.0]:
      1.0 = exact match after normalization
      0.9 = synonym match
      0.0-0.8 = token overlap (Jaccard similarity)
    """
    pred_norm = normalize_step(predicted)
    gt_norm = normalize_step(ground_truth)

    # Exact match
    if pred_norm == gt_norm:
        return 1.0

    # Check synonyms: does the predicted text contain a known synonym?
    gt_key = ground_truth.lower().strip()
    if gt_key in STEP_SYNONYMS:
        for synonym in STEP_SYNONYMS[gt_key]:
            if synonym.lower() in pred_norm:
                return 0.9

    # Also check if the ground truth key appears as substring
    gt_key_clean = gt_key.replace("_", " ")
    if gt_key_clean in pred_norm:
        return 0.85

    # Token overlap (Jaccard similarity)
    pred_tokens = tokenize(predicted)
    gt_tokens = tokenize(ground_truth)

    if not pred_tokens or not gt_tokens:
        return 0.0

    intersection = pred_tokens & gt_tokens
    union = pred_tokens | gt_tokens

    jaccard = len(intersection) / len(union) if union else 0.0
    return jaccard


def compute_step_coverage(
    predicted_steps: List[str],
    ground_truth_steps: List[str],
    threshold: float = 0.6,
) -> Tuple[float, List[str], List[str], List[str]]:
    """Compute fraction of ground truth steps covered by prediction.

    Uses greedy bipartite matching: for each ground truth step,
    find the best matching predicted step above the threshold.

    Returns (coverage, matched, missed, extra).
    """
    if not ground_truth_steps:
        return 1.0, [], [], list(predicted_steps)

    used_predictions = set()
    matched = []
    missed = []

    for gt_step in ground_truth_steps:
        best_score = 0.0
        best_idx = -1

        for i, pred_step in enumerate(predicted_steps):
            if i in used_predictions:
                continue
            score = fuzzy_match_score(pred_step, gt_step)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score >= threshold and best_idx >= 0:
            matched.append(gt_step)
            used_predictions.add(best_idx)
        else:
            missed.append(gt_step)

    extra = [predicted_steps[i] for i in range(len(predicted_steps)) if i not in used_predictions]
    coverage = len(matched) / len(ground_truth_steps) if ground_truth_steps else 1.0

    return coverage, matched, missed, extra


# =============================================================================
# SHORTCUT DETECTION
# =============================================================================

# Structure keywords that indicate shortcut awareness
STRUCTURE_KEYWORDS: Dict[str, List[str]] = {
    "village": ["village", "villager", "farm", "crops", "trading"],
    "blacksmith": ["blacksmith", "weaponsmith", "toolsmith", "smithing"],
    "mineshaft": ["mineshaft", "mine shaft", "abandoned mine", "rail"],
    "ruined_portal": ["ruined portal", "portal", "gilded", "obsidian"],
    "desert_temple": ["desert temple", "pyramid", "temple", "tnt trap"],
    "jungle_temple": ["jungle temple", "puzzle", "lever", "arrow trap"],
    "igloo": ["igloo", "basement", "trapdoor"],
    "shipwreck": ["shipwreck", "ship", "wreck", "supply chest", "treasure chest"],
    "pillager_outpost": ["pillager", "outpost", "crossbow"],
    "stronghold": ["stronghold", "end portal", "library"],
}


def detect_shortcut(predicted_steps: List[str], sample: dict) -> Optional[bool]:
    """Check if model response shows awareness of nearby structures.

    Returns:
      True  = model used the shortcut
      False = model did NOT use the shortcut
      None  = not applicable (no structures or context_matters=False)
    """
    structures = sample.get("nearby_structures", [])
    if not structures or structures == ["none"]:
        return None
    if not sample.get("context_matters", False):
        return None

    response_text = " ".join(predicted_steps).lower()

    for struct in structures:
        if struct == "none":
            continue
        keywords = STRUCTURE_KEYWORDS.get(struct, [struct.replace("_", " ")])
        for kw in keywords:
            if kw in response_text:
                return True

    return False


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_efficiency(predicted_steps: List[str], ground_truth_steps: List[str]) -> float:
    """Compute efficiency score.

    efficiency = len(ground_truth) / max(len(predicted), len(ground_truth))

    1.0 = same length, < 1.0 = model used more steps (less efficient).
    """
    if not ground_truth_steps and not predicted_steps:
        return 1.0
    if not predicted_steps:
        return 0.0

    gt_len = len(ground_truth_steps)
    pred_len = len(predicted_steps)

    return gt_len / max(pred_len, gt_len)


def evaluate_sample(
    result: dict,
    sample: dict,
    threshold: float = 0.6,
) -> dict:
    """Evaluate a single benchmark result against ground truth."""
    predicted = result.get("parsed_steps", [])
    ground_truth = sample.get("reasoning_path", [])

    coverage, matched, missed, extra = compute_step_coverage(
        predicted, ground_truth, threshold
    )
    shortcut = detect_shortcut(predicted, sample)
    efficiency = compute_efficiency(predicted, ground_truth)

    return {
        "sample_id": result["sample_id"],
        "model": result["model"],
        "condition": result["condition"],
        "step_coverage": round(coverage, 4),
        "shortcut_detected": shortcut,
        "efficiency_score": round(efficiency, 4),
        "num_predicted_steps": len(predicted),
        "num_ground_truth_steps": len(ground_truth),
        "matched_steps": matched,
        "missed_steps": missed,
        "extra_steps": extra,
    }


def aggregate_metrics(
    per_sample: List[dict],
) -> List[dict]:
    """Aggregate per-sample metrics by (model, condition)."""
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for m in per_sample:
        key = (m["model"], m["condition"])
        groups[key].append(m)

    aggregated = []
    for (model, condition), metrics in sorted(groups.items()):
        coverages = [m["step_coverage"] for m in metrics]
        efficiencies = [m["efficiency_score"] for m in metrics]
        shortcuts = [m["shortcut_detected"] for m in metrics if m["shortcut_detected"] is not None]

        shortcut_rate = sum(shortcuts) / len(shortcuts) if shortcuts else None

        aggregated.append({
            "model": model,
            "condition": condition,
            "mean_step_coverage": round(np.mean(coverages), 4),
            "std_step_coverage": round(np.std(coverages), 4),
            "shortcut_detection_rate": round(shortcut_rate, 4) if shortcut_rate is not None else None,
            "mean_efficiency_score": round(np.mean(efficiencies), 4),
            "std_efficiency_score": round(np.std(efficiencies), 4),
            "n_samples": len(metrics),
        })

    return aggregated


def compute_context_benefit(per_sample: List[dict]) -> List[dict]:
    """Compute context benefit (with_context - without_context) per model.

    Uses paired comparisons on the same samples.
    """
    # Group by model -> sample_id -> condition -> metrics
    by_model: Dict[str, Dict[int, Dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for m in per_sample:
        by_model[m["model"]][m["sample_id"]][m["condition"]] = m

    results = []
    for model, samples in sorted(by_model.items()):
        with_coverages = []
        without_coverages = []
        with_efficiencies = []
        without_efficiencies = []
        with_shortcuts = []
        without_shortcuts = []

        for sample_id, conditions in samples.items():
            if "with_context" in conditions and "without_context" in conditions:
                wc = conditions["with_context"]
                woc = conditions["without_context"]

                with_coverages.append(wc["step_coverage"])
                without_coverages.append(woc["step_coverage"])
                with_efficiencies.append(wc["efficiency_score"])
                without_efficiencies.append(woc["efficiency_score"])

                if wc["shortcut_detected"] is not None:
                    with_shortcuts.append(1 if wc["shortcut_detected"] else 0)
                    without_shortcuts.append(1 if woc["shortcut_detected"] else 0)

        if not with_coverages:
            continue

        # Paired t-test for coverage
        p_coverage = _paired_ttest(with_coverages, without_coverages)
        p_efficiency = _paired_ttest(with_efficiencies, without_efficiencies)
        p_shortcut = _mcnemar_test(with_shortcuts, without_shortcuts) if with_shortcuts else None

        results.append({
            "model": model,
            "n_paired_samples": len(with_coverages),
            "context_benefit_coverage": round(
                np.mean(with_coverages) - np.mean(without_coverages), 4
            ),
            "context_benefit_efficiency": round(
                np.mean(with_efficiencies) - np.mean(without_efficiencies), 4
            ),
            "context_benefit_shortcut": round(
                np.mean(with_shortcuts) - np.mean(without_shortcuts), 4
            ) if with_shortcuts else None,
            "p_value_coverage": round(p_coverage, 6) if p_coverage is not None else None,
            "p_value_efficiency": round(p_efficiency, 6) if p_efficiency is not None else None,
            "p_value_shortcut": round(p_shortcut, 6) if p_shortcut is not None else None,
        })

    return results


def _paired_ttest(a: List[float], b: List[float]) -> Optional[float]:
    """Paired t-test. Returns p-value or None if not enough data."""
    if len(a) < 2:
        return None
    try:
        from scipy.stats import ttest_rel
        _, p = ttest_rel(a, b)
        return p
    except ImportError:
        # Fallback: manual computation
        diffs = [x - y for x, y in zip(a, b)]
        n = len(diffs)
        mean_d = sum(diffs) / n
        var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
        if var_d == 0:
            return 1.0
        se = (var_d / n) ** 0.5
        t_stat = mean_d / se
        # Approximate p-value using normal distribution for large n
        from math import erfc
        p = erfc(abs(t_stat) / (2 ** 0.5))
        return p


def _mcnemar_test(a: List[int], b: List[int]) -> Optional[float]:
    """McNemar's test for paired binary data. Returns p-value."""
    if len(a) < 2:
        return None

    # Count discordant pairs
    b_c = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)  # with=1, without=0
    c_b = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)  # with=0, without=1

    n = b_c + c_b
    if n == 0:
        return 1.0

    try:
        from scipy.stats import binom_test
        p = binom_test(b_c, n, 0.5)
        return p
    except ImportError:
        # Approximate with chi-squared
        chi2 = (abs(b_c - c_b) - 1) ** 2 / n if n > 0 else 0
        from math import exp
        p = exp(-chi2 / 2)  # very rough approximation
        return p


# =============================================================================
# OUTPUT
# =============================================================================

def print_aggregate_table(aggregated: List[dict]):
    """Print aggregated metrics as a formatted table."""
    print(f"\n{'='*90}")
    print(f"{'AGGREGATE METRICS':^90}")
    print(f"{'='*90}")
    print(f"{'Model':<20} {'Condition':<16} {'Coverage':>10} {'Shortcut%':>10} {'Efficiency':>10} {'N':>5}")
    print(f"{'-'*90}")
    for a in aggregated:
        shortcut = f"{a['shortcut_detection_rate']:.2%}" if a['shortcut_detection_rate'] is not None else "N/A"
        print(
            f"{a['model']:<20} "
            f"{a['condition']:<16} "
            f"{a['mean_step_coverage']:.2%} ±{a['std_step_coverage']:.2f}  "
            f"{shortcut:>7}  "
            f"{a['mean_efficiency_score']:.2%} ±{a['std_efficiency_score']:.2f}  "
            f"{a['n_samples']:>5}"
        )
    print(f"{'='*90}")


def print_context_benefit_table(benefits: List[dict]):
    """Print context benefit analysis."""
    print(f"\n{'='*90}")
    print(f"{'CONTEXT BENEFIT ANALYSIS':^90}")
    print(f"{'='*90}")
    print(f"{'Model':<20} {'Coverage Δ':>12} {'Efficiency Δ':>14} {'Shortcut Δ':>12} {'p(cov)':>10} {'Sig?':>5}")
    print(f"{'-'*90}")
    for b in benefits:
        sig = "YES" if b['p_value_coverage'] is not None and b['p_value_coverage'] < 0.05 else "no"
        p_str = f"{b['p_value_coverage']:.4f}" if b['p_value_coverage'] is not None else "N/A"
        sc_str = f"{b['context_benefit_shortcut']:+.4f}" if b['context_benefit_shortcut'] is not None else "N/A"
        print(
            f"{b['model']:<20} "
            f"{b['context_benefit_coverage']:>+12.4f} "
            f"{b['context_benefit_efficiency']:>+14.4f} "
            f"{sc_str:>12} "
            f"{p_str:>10} "
            f"{sig:>5}"
        )
    print(f"{'='*90}")


def save_csv(data: List[dict], filepath: str):
    """Save a list of dicts as CSV."""
    if not data:
        return
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    keys = data[0].keys()
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)


def save_summary_markdown(
    aggregated: List[dict],
    benefits: List[dict],
    output_dir: str,
):
    """Save a human-readable markdown summary."""
    filepath = os.path.join(output_dir, "summary.md")
    with open(filepath, "w") as f:
        f.write("# Benchmark Results Summary\n\n")
        f.write(f"Generated: {__import__('datetime').datetime.now().isoformat()}\n\n")

        f.write("## Aggregate Metrics\n\n")
        f.write("| Model | Condition | Step Coverage | Shortcut Detection | Efficiency | N |\n")
        f.write("|-------|-----------|--------------|-------------------|------------|---|\n")
        for a in aggregated:
            sc = f"{a['shortcut_detection_rate']:.1%}" if a['shortcut_detection_rate'] is not None else "N/A"
            f.write(
                f"| {a['model']} | {a['condition']} | "
                f"{a['mean_step_coverage']:.1%} (±{a['std_step_coverage']:.2f}) | "
                f"{sc} | "
                f"{a['mean_efficiency_score']:.1%} (±{a['std_efficiency_score']:.2f}) | "
                f"{a['n_samples']} |\n"
            )

        f.write("\n## Context Benefit Analysis\n\n")
        f.write("| Model | Coverage Δ | Efficiency Δ | Shortcut Δ | p-value (cov) | Significant? |\n")
        f.write("|-------|-----------|-------------|-----------|--------------|-------------|\n")
        for b in benefits:
            sig = "Yes" if b['p_value_coverage'] is not None and b['p_value_coverage'] < 0.05 else "No"
            p = f"{b['p_value_coverage']:.4f}" if b['p_value_coverage'] is not None else "N/A"
            sc = f"{b['context_benefit_shortcut']:+.4f}" if b['context_benefit_shortcut'] is not None else "N/A"
            f.write(
                f"| {b['model']} | {b['context_benefit_coverage']:+.4f} | "
                f"{b['context_benefit_efficiency']:+.4f} | {sc} | {p} | {sig} |\n"
            )

    print(f"\nSummary saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate benchmark results")
    parser.add_argument(
        "--results", default=None,
        help="Path to results.jsonl (default: benchmark_results/results.jsonl)",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Path to reasoning_paths.json",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for metrics (default: benchmark_results/)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.6,
        help="Fuzzy match threshold (default: 0.6)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = args.results or os.path.join(script_dir, "benchmark_results", "results.jsonl")
    dataset_path = args.dataset or os.path.join(script_dir, "reasoning_paths.json")
    output_dir = args.output_dir or os.path.join(script_dir, "benchmark_results")

    # Load dataset
    with open(dataset_path) as f:
        samples = json.load(f)
    sample_map = {s["id"]: s for s in samples}
    print(f"Loaded {len(samples)} ground truth samples")

    # Load results
    results = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    print(f"Loaded {len(results)} benchmark results")

    # Compute per-sample metrics
    per_sample_metrics = []
    for result in results:
        sample = sample_map.get(result["sample_id"])
        if sample is None:
            print(f"WARNING: No ground truth for sample_id={result['sample_id']}")
            continue
        metrics = evaluate_sample(result, sample, threshold=args.threshold)
        per_sample_metrics.append(metrics)

    print(f"Computed metrics for {len(per_sample_metrics)} evaluations")

    # Aggregate
    aggregated = aggregate_metrics(per_sample_metrics)
    print_aggregate_table(aggregated)

    # Context benefit
    benefits = compute_context_benefit(per_sample_metrics)
    if benefits:
        print_context_benefit_table(benefits)

    # Save outputs
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Per-sample metrics (exclude lists for CSV)
    csv_metrics = []
    for m in per_sample_metrics:
        row = {k: v for k, v in m.items() if not isinstance(v, list)}
        csv_metrics.append(row)
    save_csv(csv_metrics, os.path.join(metrics_dir, "per_sample_metrics.csv"))
    save_csv(aggregated, os.path.join(metrics_dir, "aggregate_metrics.csv"))
    save_csv(benefits, os.path.join(metrics_dir, "context_benefit.csv"))

    save_summary_markdown(aggregated, benefits, output_dir)

    # Also save full per-sample metrics as JSON for analysis notebook
    with open(os.path.join(metrics_dir, "per_sample_metrics.json"), "w") as f:
        json.dump(per_sample_metrics, f, indent=2)

    print(f"\nAll metrics saved to: {metrics_dir}/")


if __name__ == "__main__":
    main()
