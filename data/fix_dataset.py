"""
fix_dataset.py

Auto-fixes two categories of errors in dataset_final.json:

Pass 1 — DAG ordering violations:
  For samples where a skill appears before one of its prerequisites, reorder
  the mapped skills into a topological order consistent with the tech tree DAG,
  then reinsert agnostic skills (navigation, loot, food) at the end of the path
  in their original relative order.

Pass 2 — mine_coal gap:
  For samples that use craft_torch but have no coal source in the path
  (mine_coal or a structure loot skill), insert mine_coal immediately before
  craft_torch.

Output overwrites data/processed/dataset_final.json in place.

Usage:
    python data/fix_dataset.py
    python data/fix_dataset.py --dataset data/processed/dataset_final.json
                               --tech-tree /path/to/training_config.json
                               --dry-run
"""

import argparse
import json
import os
import sys
from collections import defaultdict, deque

# Re-use the same skill→node mapping and agnostic set as validate_dataset.py
SKILL_TO_NODE = {
    "harvest_wood":            "wood_log",
    "craft_planks_and_sticks": "planks",
    "craft_crafting_table":    "crafting_table",
    "craft_wooden_pickaxe":    "wooden_pickaxe",
    "mine_stone":              "stone",
    "mine_coal":               "coal",
    "craft_torch":             "torch",
    "craft_furnace":           "furnace",
    "craft_stone_pickaxe":     "stone_pickaxe",
    "mine_iron_ore":           "iron_ore",
    "smelt_iron":              "iron_ingot",
    "craft_iron_pickaxe":      "iron_pickaxe",
    "craft_iron_armor_set":    "full_iron",
    "mine_diamonds":           "diamond",
    "craft_diamond_pickaxe":   "diamond_pickaxe",
}

NODE_TO_SKILL = {v: k for k, v in SKILL_TO_NODE.items()}

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

COAL_SOURCES = {
    "mine_coal", "loot_blacksmith_chest", "search_mineshaft_chests",
    "loot_supply_chest", "loot_portal_chest",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def build_prereq_closure(tech_tree):
    """Build {node_id: set of all ancestor node_ids} from tech tree DAG."""
    # nodes is a dict {node_id: node_data}
    nodes = tech_tree["nodes"]
    direct_requires = {nid: set(node.get("requires", [])) for nid, node in nodes.items()}

    ancestors = {}

    def get_ancestors(node_id, visited=None):
        if node_id in ancestors:
            return ancestors[node_id]
        if visited is None:
            visited = set()
        if node_id in visited:
            return set()
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


def has_violation(sample, ancestors):
    """Return True if the sample has any DAG prerequisite violation."""
    path = sample.get("reasoning_path", [])
    mapped = [
        (i, skill, SKILL_TO_NODE[skill])
        for i, skill in enumerate(path)
        if skill in SKILL_TO_NODE and skill not in AGNOSTIC_SKILLS
    ]
    for i in range(len(mapped)):
        for j in range(i + 1, len(mapped)):
            _, skill_i, node_i = mapped[i]
            _, skill_j, node_j = mapped[j]
            if node_j in ancestors.get(node_i, set()):
                return True
    return False


def topo_sort_skills(skill_nodes, ancestors):
    """
    Topologically sort a list of (skill, node) pairs using the prerequisite
    closure. Returns a sorted list of skills where prerequisites come first.

    Uses Kahn's algorithm on the subgraph induced by the given skills.
    """
    nodes = [node for _, node in skill_nodes]
    node_set = set(nodes)
    skill_of = {node: skill for skill, node in skill_nodes}

    # Build in-degree within the subgraph
    in_degree = defaultdict(int)
    adj = defaultdict(set)  # node → set of nodes that depend on it

    for node in nodes:
        in_degree[node]  # ensure key exists
        for ancestor in ancestors.get(node, set()):
            if ancestor in node_set:
                # ancestor must come before node
                adj[ancestor].add(node)
                in_degree[node] += 1

    queue = deque([n for n in nodes if in_degree[n] == 0])
    sorted_nodes = []

    while queue:
        n = queue.popleft()
        sorted_nodes.append(n)
        for dependent in adj[n]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    # If we couldn't sort all (cycle?), append the rest unchanged
    remaining = [n for n in nodes if n not in sorted_nodes]
    sorted_nodes.extend(remaining)

    return [skill_of[n] for n in sorted_nodes]


def fix_ordering(sample, ancestors):
    """
    Fix DAG ordering violations in a sample's reasoning_path.

    Strategy:
    - Separate the path into mapped skills and agnostic skills
    - Topologically sort the mapped skills by the DAG
    - Append agnostic skills at the end in their original relative order
    - Return (new_path, was_changed)
    """
    path = sample["reasoning_path"]
    original = list(path)

    mapped_skills = []   # (skill, node) — in original order
    agnostic_skills = [] # skills not in SKILL_TO_NODE or in AGNOSTIC_SKILLS

    for skill in path:
        node = SKILL_TO_NODE.get(skill)
        if node is not None and skill not in AGNOSTIC_SKILLS:
            mapped_skills.append((skill, node))
        else:
            agnostic_skills.append(skill)

    if not mapped_skills:
        return original, False

    sorted_mapped = topo_sort_skills(mapped_skills, ancestors)
    new_path = sorted_mapped + agnostic_skills

    changed = new_path != original
    return new_path, changed


def fix_coal_gap(sample):
    """
    If the sample uses craft_torch but has no coal source, insert mine_coal
    immediately before the first occurrence of craft_torch.
    Returns (new_path, was_changed).
    """
    path = list(sample["reasoning_path"])

    if "craft_torch" not in path:
        return path, False

    has_coal = any(s in path for s in COAL_SOURCES)
    if has_coal:
        return path, False

    idx = path.index("craft_torch")
    path.insert(idx, "mine_coal")
    return path, True


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
        "--dry-run", action="store_true",
        help="Report fixes without writing to disk"
    )
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    for path, label in [(args.dataset, "dataset"), (args.tech_tree, "tech tree")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found at {path}")
            sys.exit(1)

    dataset = load_json(args.dataset)
    tech_tree = load_json(args.tech_tree)
    ancestors = build_prereq_closure(tech_tree)

    print(f"Loaded {len(dataset)} samples")
    print(f"Dry run: {args.dry_run}\n")

    # ── Pass 1: Fix DAG ordering violations ───────────────────────────────────
    print("=" * 60)
    print("Pass 1: Fix DAG ordering violations")
    print("=" * 60)

    pass1_fixed = 0
    for sample in dataset:
        if not has_violation(sample, ancestors):
            continue
        old_path = list(sample["reasoning_path"])
        new_path, changed = fix_ordering(sample, ancestors)
        if changed:
            print(f"  Sample {sample['id']} ({sample['task']}, {sample['biome']}):")
            print(f"    Before: {old_path}")
            print(f"    After:  {new_path}")
            if not args.dry_run:
                sample["reasoning_path"] = new_path
            pass1_fixed += 1

    print(f"\nPass 1 complete: fixed {pass1_fixed} samples\n")

    # ── Pass 2: Fix mine_coal gaps ─────────────────────────────────────────────
    print("=" * 60)
    print("Pass 2: Fix mine_coal gaps (craft_torch without coal source)")
    print("=" * 60)

    pass2_fixed = 0
    for sample in dataset:
        new_path, changed = fix_coal_gap(sample)
        if changed:
            print(f"  Sample {sample['id']}: inserted mine_coal before craft_torch")
            print(f"    Path: {new_path}")
            if not args.dry_run:
                sample["reasoning_path"] = new_path
            pass2_fixed += 1

    print(f"\nPass 2 complete: fixed {pass2_fixed} samples\n")

    # ── Write ──────────────────────────────────────────────────────────────────
    if not args.dry_run:
        save_json(args.dataset, dataset)
        print(f"Wrote {len(dataset)} samples to {args.dataset}")
    else:
        print("(dry-run — no changes written)")

    print(f"\nSummary: {pass1_fixed} ordering fixes + {pass2_fixed} coal-gap fixes "
          f"= {pass1_fixed + pass2_fixed} total changes")
    print("Run validate_dataset.py to confirm 0 remaining violations.")


if __name__ == "__main__":
    main()
