"""
validate_dataset.py

Validates dataset_final.json against Sombaudy's tech tree DAG (training_config.json).

For each sample's reasoning_path, checks that skills appear in a tier-consistent
order according to the DAG — i.e., no skill produces an item that requires
something only available at a higher tier later in the same path.

Structure-shortcut paths (where loot skills bypass normal prerequisites) are
handled by marking loot/navigate skills as tier-agnostic.

Usage:
    python data/validate_dataset.py
    python data/validate_dataset.py --dataset data/processed/dataset_final.json
                                    --tech-tree /path/to/training_config.json
                                    --strict
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Skill → tech tree node mapping
# Maps our action-based skill vocab to Sombaudy's item/node names.
# Skills not listed here are "tier-agnostic" (structure shortcuts, navigation,
# biome-specific foraging) and are skipped in tier validation.
# ---------------------------------------------------------------------------
SKILL_TO_NODE = {
    "harvest_wood":           "wood_log",          # tier 0
    "craft_planks_and_sticks": "planks",            # tier 1
    "craft_crafting_table":   "crafting_table",     # tier 2
    "craft_wooden_pickaxe":   "wooden_pickaxe",     # tier 3
    "craft_torch":            "torch",              # tier 4
    "mine_stone":             "stone",              # tier 4
    "craft_furnace":          "furnace",            # tier 5
    "craft_stone_pickaxe":    "stone_pickaxe",      # tier 5
    "mine_iron_ore":          "iron_ore",           # tier 6
    "smelt_iron":             "iron_ingot",         # tier 6
    "craft_iron_pickaxe":     "iron_pickaxe",       # tier 7
    "craft_iron_armor_set":   "full_iron",          # tier 7
    "mine_diamonds":          "diamond",            # tier 8
    "craft_diamond_pickaxe":  "diamond_pickaxe",    # tier 9
    # gold/coal not in tech tree — skipped
}

# ---------------------------------------------------------------------------
# Skills that are inherently order-agnostic (structure shortcuts, navigation,
# foraging). These can appear anywhere in a path without violating DAG order.
# ---------------------------------------------------------------------------
AGNOSTIC_SKILLS = {
    "go_to_village", "find_blacksmith", "loot_blacksmith_chest",
    "navigate_to_mineshaft", "search_mineshaft_chests",
    "go_to_ruined_portal", "loot_portal_chest",
    "go_to_desert_temple", "avoid_tnt_trap", "loot_supply_chest",
    "go_to_jungle_temple", "swim_to_shipwreck", "go_to_igloo",
    "navigate_to_structure", "return_to_surface", "explore_cave",
    "combat_mob", "eat_food",
    "search_for_animals", "kill_animals_for_meat", "cook_meat",
    "harvest_village_crops", "harvest_melons_from_ground",
    "harvest_sweet_berries_from_bushes", "milk_mooshroom_with_bowl",
    "build_walls_and_roof", "craft_and_place_door", "place_torches",
    "dig_to_diamond_level", "dig_to_gold_level",
    "mine_gold_ore", "smelt_gold",
    "mine_stone",   # already mapped but can appear in structure paths safely
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_tier_map(tech_tree):
    """Return {node_id: tier} from tech tree tiers dict."""
    tier_map = {}
    for tier_str, nodes in tech_tree["tiers"].items():
        for node in nodes:
            tier_map[node] = int(tier_str)
    return tier_map


def validate_sample(sample, tier_map, strict=False):
    """
    Check that mapped skills appear in non-decreasing tier order.
    Returns list of violation dicts (empty = valid).
    """
    path = sample.get("reasoning_path", [])
    violations = []
    last_tier = -1
    last_skill = None

    for skill in path:
        node = SKILL_TO_NODE.get(skill)
        if node is None:
            continue  # tier-agnostic or unmapped — skip

        tier = tier_map.get(node)
        if tier is None:
            continue  # node not in tech tree — skip

        if tier < last_tier:
            violations.append({
                "skill": skill,
                "node": node,
                "tier": tier,
                "previous_skill": last_skill,
                "previous_tier": last_tier,
                "message": (
                    f"'{skill}' (tier {tier}) appears after "
                    f"'{last_skill}' (tier {last_tier}) — prerequisite violated"
                ),
            })
            if not strict:
                # Don't update last_tier on violation so we keep tracking from
                # the highest tier seen, catching all further regressions.
                continue

        last_tier = tier
        last_skill = skill

    return violations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=os.path.join(os.path.dirname(__file__), "processed", "dataset_final.json"),
    )
    parser.add_argument(
        "--tech-tree",
        default="/Users/romeonickel/Documents/GitHub/MC_Tech_Tree/training_config.json",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Stop updating tier baseline on violation (catches cascading errors)"
    )
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    if not os.path.exists(args.dataset):
        print(f"ERROR: dataset not found at {args.dataset}")
        sys.exit(1)
    if not os.path.exists(args.tech_tree):
        print(f"ERROR: tech tree not found at {args.tech_tree}")
        sys.exit(1)

    dataset = load_json(args.dataset)
    tech_tree = load_json(args.tech_tree)
    tier_map = build_tier_map(tech_tree)

    print(f"Loaded {len(dataset)} samples")
    print(f"Tech tree nodes: {len(tier_map)} with tier assignments")
    print(f"Skills mapped to tech tree: {len(SKILL_TO_NODE)}\n")

    # ── Validate ──────────────────────────────────────────────────────────────
    invalid_samples = []
    total_violations = 0

    for sample in dataset:
        violations = validate_sample(sample, tier_map, strict=args.strict)
        if violations:
            invalid_samples.append((sample, violations))
            total_violations += len(violations)

    # ── Report ────────────────────────────────────────────────────────────────
    valid = len(dataset) - len(invalid_samples)
    print(f"{'='*55}")
    print(f"Valid samples:    {valid} / {len(dataset)}")
    print(f"Invalid samples:  {len(invalid_samples)}")
    print(f"Total violations: {total_violations}")
    print(f"{'='*55}")

    if invalid_samples:
        print(f"\nInvalid samples detail:\n")
        for sample, violations in invalid_samples[:20]:  # cap at 20
            print(f"  Sample {sample['id']} | biome={sample['biome']} "
                  f"task={sample['task']} source={sample.get('source','?')}")
            print(f"  Path: {sample['reasoning_path']}")
            for v in violations:
                print(f"    ✗ {v['message']}")
            print()
        if len(invalid_samples) > 20:
            print(f"  ... and {len(invalid_samples)-20} more.")

        # Summarize by violation type
        print(f"\nMost common violations:")
        counts = {}
        for _, violations in invalid_samples:
            for v in violations:
                key = f"{v['previous_skill']} → {v['skill']}"
                counts[key] = counts.get(key, 0) + 1
        for pair, count in sorted(counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {count:3}x  {pair}")
    else:
        print("\nAll samples pass DAG tier ordering.")

    # ── mine_coal check ───────────────────────────────────────────────────────
    # Check whether any path uses craft_torch / craft_furnace without first
    # having a way to get coal (mine_coal is not in our vocab).
    print(f"\n{'='*55}")
    print("mine_coal gap check:")
    torch_without_coal = []
    for s in dataset:
        path = s["reasoning_path"]
        if "craft_torch" in path:
            # Check if any coal source exists: structure loot or explicit mine_coal
            has_coal_source = any(skill in path for skill in [
                "loot_blacksmith_chest", "search_mineshaft_chests",
                "loot_supply_chest", "loot_portal_chest",
            ])
            if not has_coal_source:
                torch_without_coal.append(s["id"])

    if torch_without_coal:
        print(f"  {len(torch_without_coal)} samples use craft_torch with no coal source "
              f"in path (mine_coal missing from vocab):")
        print(f"  Sample IDs: {torch_without_coal[:20]}")
    else:
        print("  No samples affected — craft_torch always paired with a loot skill.")


if __name__ == "__main__":
    main()
