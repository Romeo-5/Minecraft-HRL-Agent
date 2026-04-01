"""
rtg_utils.py

Utilities for computing Return-to-Go (RTG) values from dataset reasoning paths
using the tech tree DAG reward table.

RTG at step t = sum of rewards for skills at positions t, t+1, ..., T-1
(undiscounted by default). This is the key quantity the Decision Transformer
conditions on — a high RTG at step 0 tells the DT "generate a high-return path".

Skill → tech tree node mapping reused from validate_dataset.py.
Skills not in this map (navigation, food, shelter) get reward 0.0.
"""

import json
import os

# ---------------------------------------------------------------------------
# Skill → tech tree node ID (same as validate_dataset.py)
# ---------------------------------------------------------------------------
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
    # Gold path
    "mine_gold_ore":           "iron_ore",   # closest proxy in reward table
    "smelt_gold":              "iron_ingot",
    # dig skills
    "dig_to_diamond_level":    "diamond",
    "dig_to_gold_level":       "iron_ore",
}

_DEFAULT_TECH_TREE = os.path.join(
    os.path.expanduser("~"),
    "Documents", "GitHub", "MC_Tech_Tree", "training_config.json"
)


def load_reward_table(tech_tree_path: str = _DEFAULT_TECH_TREE) -> dict:
    """
    Load the reward_table dict from training_config.json.
    Returns {node_id: float reward}.
    """
    with open(tech_tree_path) as f:
        data = json.load(f)
    return data["reward_table"]


def skill_reward(skill: str, reward_table: dict) -> float:
    """
    Return the tech-tree reward for executing a given skill.
    Returns 0.0 for skills not mapped to the tech tree.
    """
    node = SKILL_TO_NODE.get(skill)
    if node is None:
        return 0.0
    return reward_table.get(node, 0.0)


def compute_rtg(reasoning_path: list, reward_table: dict, discount: float = 1.0) -> list:
    """
    Compute Return-to-Go values for each step in a reasoning path.

    Args:
        reasoning_path: List of canonical skill strings (e.g. ["harvest_wood", ...])
        reward_table:   Dict {node_id: float} from training_config.json
        discount:       Discount factor γ (1.0 = undiscounted, matching DT paper)

    Returns:
        List of RTG floats, same length as reasoning_path.
        rtg[t] = Σ_{i=t}^{T-1} γ^(i-t) * reward[i]
    """
    T = len(reasoning_path)
    rewards = [skill_reward(skill, reward_table) for skill in reasoning_path]

    rtg = [0.0] * T
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t] + discount * running
        rtg[t] = running

    return rtg


def total_return(reasoning_path: list, reward_table: dict) -> float:
    """
    Compute total undiscounted return for a reasoning path.
    This is the RTG value used as the conditioning target at inference time.
    """
    return sum(skill_reward(s, reward_table) for s in reasoning_path)
