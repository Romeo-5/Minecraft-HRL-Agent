"""
validate_dataset.py

Validates dataset_final.json against Sombaudy's tech tree DAG (training_config.json).

For each sample's reasoning_path, checks that skills appear in a prerequisite-valid
order — i.e., no skill produces an item that requires something that only appears
LATER in the same path.

Uses transitive prerequisite closure rather than tier ordering, eliminating false
positives from parallel prerequisite relationships (e.g. craft_furnace → mine_iron_ore
is valid since iron_ore is NOT a prerequisite of furnace).

Structure-shortcut paths (where loot skills bypass normal prerequisites) are
handled by marking loot/navigate skills as tier-agnostic.

Usage:
    python data/validate_dataset.py
    python data/validate_dataset.py --dataset data/processed/dataset_final.json
                                    --tech-tree /path/to/training_config.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Skill → tech tree node mapping
# Maps our action-based skill vocab to Sombaudy's item/node names.
# Skills not listed here are "agnostic" (structure shortcuts, navigation,
# biome-specific foraging) and are skipped in prerequisite validation.
# ---------------------------------------------------------------------------
SKILL_TO_NODE = {
    "harvest_wood":            "wood_log",        # tier 0
    "craft_planks_and_sticks": "planks",           # tier 1
    "craft_crafting_table":    "crafting_table",   # tier 2
    "craft_wooden_pickaxe":    "wooden_pickaxe",   # tier 3
    "mine_stone":              "stone",            # tier 4
    "mine_coal":               "coal",             # tier 4
    "craft_torch":             "torch",            # tier 4
    "craft_furnace":           "furnace",          # tier 5
    "craft_stone_pickaxe":     "stone_pickaxe",    # tier 5
    "mine_iron_ore":           "iron_ore",         # tier 6
    "smelt_iron":              "iron_ingot",       # tier 6
    "craft_iron_pickaxe":      "iron_pickaxe",     # tier 7
    "craft_iron_armor_set":    "full_iron",        # tier 7
    "mine_diamonds":           "diamond",          # tier 8
    "craft_diamond_pickaxe":   "diamond_pickaxe",  # tier 9
    # gold/shelter/food — not in tech tree, treated as agnostic
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
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_prereq_closure(tech_tree):
    """
    Build transitive prerequisite closure from tech tree DAG.

    Returns ancestors: {node_id: set of node_ids that must be obtained before it}

    A node's ancestors = its direct requires + all of their ancestors (transitive).
    """
    # nodes is a dict {node_id: node_data}
    nodes = tech_tree["nodes"]

    # Direct prerequisites: what each node immediately requires
    direct_requires = {nid: set(node.get("requires", [])) for nid, node in nodes.items()}

    # Compute transitive closure via memoised DFS
    ancestors = {}

    def get_ancestors(node_id, visited=None):
        if node_id in ancestors:
            return ancestors[node_id]
        if visited is None:
            visited = set()
        if node_id in visited:
            return set()  # cycle guard (shouldn't exist in a DAG)
        visited.add(node_id)

        result = set()
        for req in direct_requires.get(node_id, []):
            result.add(req)
            result |= get_ancestors(req, visited)

        ancestors[node_id] = result
        return result

    for nid in nodes:
        get_ancestors(nid)

    return ancestors


def validate_sample(sample, ancestors):
    """
    Check that no skill appears before another skill whose node is a prerequisite
    of the first skill's node.

    For every pair (i < j) of mapped skills in the path:
      Violation if node(skill[j]) ∈ ancestors[node(skill[i])]
      (skill[j]'s output is needed to produce skill[i]'s output, but skill[i] comes first)

    Returns list of violation dicts (empty = valid).
    """
    path = sample.get("reasoning_path", [])
    violations = []

    # Extract only the mapped skills with their positions
    mapped = []
    for idx, skill in enumerate(path):
        node = SKILL_TO_NODE.get(skill)
        if node is not None and skill not in AGNOSTIC_SKILLS:
            mapped.append((idx, skill, node))

    # Check all pairs (i, j) where i < j
    for i in range(len(mapped)):
        for j in range(i + 1, len(mapped)):
            idx_i, skill_i, node_i = mapped[i]
            idx_j, skill_j, node_j = mapped[j]

            # skill_i appears before skill_j in the path.
            # Violation: node_j is a prerequisite OF node_i
            # (meaning node_j should come first, but skill_i appears at position idx_i < idx_j)
            if node_j in ancestors.get(node_i, set()):
                violations.append({
                    "skill": skill_i,
                    "node": node_i,
                    "position": idx_i,
                    "prereq_skill": skill_j,
                    "prereq_node": node_j,
                    "prereq_position": idx_j,
                    "message": (
                        f"'{skill_i}' (pos {idx_i}) requires '{skill_j}' (pos {idx_j}) "
                        f"as a prerequisite, but '{skill_j}' appears later in the path"
                    ),
                })

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
        "--verbose", action="store_true",
        help="Print all invalid samples (default: cap at 20)"
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
    ancestors = build_prereq_closure(tech_tree)

    print(f"Loaded {len(dataset)} samples")
    print(f"Tech tree nodes: {len(ancestors)} with prerequisite closure")
    print(f"Skills mapped to tech tree: {len(SKILL_TO_NODE)}\n")

    # ── Validate ──────────────────────────────────────────────────────────────
    invalid_samples = []
    total_violations = 0

    for sample in dataset:
        violations = validate_sample(sample, ancestors)
        if violations:
            invalid_samples.append((sample, violations))
            total_violations += len(violations)

    # ── Report ────────────────────────────────────────────────────────────────
    valid = len(dataset) - len(invalid_samples)
    print(f"{'='*60}")
    print(f"Valid samples:    {valid} / {len(dataset)}")
    print(f"Invalid samples:  {len(invalid_samples)}")
    print(f"Total violations: {total_violations}")
    print(f"{'='*60}")

    cap = len(invalid_samples) if args.verbose else 20
    if invalid_samples:
        print(f"\nInvalid samples detail (showing up to {cap}):\n")
        for sample, violations in invalid_samples[:cap]:
            print(f"  Sample {sample['id']} | biome={sample['biome']} "
                  f"task={sample['task']} source={sample.get('source','?')}")
            print(f"  Path: {sample['reasoning_path']}")
            for v in violations:
                print(f"    ✗ {v['message']}")
            print()
        if len(invalid_samples) > cap:
            print(f"  ... and {len(invalid_samples) - cap} more. Use --verbose to see all.")

        # Summarize by violation type
        print(f"\nMost common violations:")
        counts = {}
        for _, violations in invalid_samples:
            for v in violations:
                key = f"{v['skill']} before {v['prereq_skill']}"
                counts[key] = counts.get(key, 0) + 1
        for pair, count in sorted(counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {count:3}x  {pair}")
    else:
        print("\nAll samples pass DAG prerequisite ordering.")

    # ── mine_coal check ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("mine_coal gap check:")
    torch_without_coal = []
    COAL_SOURCES = {
        "mine_coal", "loot_blacksmith_chest", "search_mineshaft_chests",
        "loot_supply_chest", "loot_portal_chest",
    }
    for s in dataset:
        path = s["reasoning_path"]
        if "craft_torch" in path:
            has_coal_source = any(skill in path for skill in COAL_SOURCES)
            if not has_coal_source:
                torch_without_coal.append(s["id"])

    if torch_without_coal:
        print(f"  {len(torch_without_coal)} samples use craft_torch with no coal source in path:")
        print(f"  IDs: {torch_without_coal[:20]}")
        if len(torch_without_coal) > 20:
            print(f"  ... and {len(torch_without_coal) - 20} more.")
    else:
        print("  No gaps — every craft_torch is paired with a coal source.")


if __name__ == "__main__":
    main()
