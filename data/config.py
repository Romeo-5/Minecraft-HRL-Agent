# data/config.py
# Single source of truth for all dataset vocabulary.
# Import from here in generate_dataset.py, normalize_dataset.py, benchmark_models.py, evaluate_results.py.

# ── Tasks ─────────────────────────────────────────────────────────────────────
# Keep simple per team agreement — stone/iron pickaxe level is ambitious for RL.
TASKS = [
    "obtain_wooden_pickaxe",
    "obtain_stone_pickaxe",
    "obtain_iron_pickaxe",
    "obtain_diamond_pickaxe",
    "obtain_gold",
    "obtain_food",
    "build_shelter",
    "obtain_armor",
    "obtain_iron_armor",
    "explore_cave",
    "craft_furnace",
    "smelt_iron",
]

# ── Skill vocabulary ───────────────────────────────────────────────────────────
# All valid reasoning_path step names. Only these strings should appear in any sample.
# Specific names preferred over generic (mine_iron_ore > mine_ore) — maps to Mineflayer skills.
SKILL_VOCAB = [
    # Wood & basic crafting
    "harvest_wood",
    "craft_planks_and_sticks",
    "craft_crafting_table",
    "craft_wooden_pickaxe",
    "craft_torch",
    # Stone
    "mine_stone",
    "mine_coal",
    "craft_stone_pickaxe",
    # Iron path
    "mine_iron_ore",
    "craft_furnace",
    "smelt_iron",
    "craft_iron_pickaxe",
    "craft_iron_armor_set",
    # Diamond path
    "dig_to_diamond_level",
    "mine_diamonds",
    "craft_diamond_pickaxe",
    # Gold path
    "dig_to_gold_level",
    "mine_gold_ore",
    "smelt_gold",
    # Food
    "search_for_animals",
    "kill_animals_for_meat",
    "cook_meat",
    "harvest_village_crops",
    "harvest_melons_from_ground",
    "harvest_sweet_berries_from_bushes",
    "milk_mooshroom_with_bowl",
    # Shelter
    "build_walls_and_roof",
    "craft_and_place_door",
    "place_torches",
    # Structure interactions
    "go_to_village",
    "find_blacksmith",
    "loot_blacksmith_chest",
    "navigate_to_mineshaft",
    "search_mineshaft_chests",
    "go_to_ruined_portal",
    "loot_portal_chest",
    "go_to_desert_temple",
    "avoid_tnt_trap",
    "go_to_jungle_temple",
    "swim_to_shipwreck",
    "loot_supply_chest",
    "go_to_igloo",
    # General / context-driven
    "navigate_to_structure",
    "return_to_surface",
    "explore_cave",
    "combat_mob",
    "eat_food",
]

SKILL_VOCAB_SET = set(SKILL_VOCAB)

# ── Biomes ─────────────────────────────────────────────────────────────────────
# Overworld only — no nether or end biomes.
BIOMES = [
    "plains",
    "forest",
    "desert",
    "mesa",
    "ice_spikes",
    "jungle",
    "swamp",
    "savanna",
    "taiga",
    "mountains",
    "ocean",
    "mushroom_island",
    "dark_forest",
    "lush_caves",
    "dripstone_caves",
    "deep_dark",
]

# ── Structures ─────────────────────────────────────────────────────────────────
# Overworld only — no nether_fortress.
STRUCTURES = [
    "none",
    "village",
    "blacksmith",
    "mineshaft",
    "ruined_portal",
    "desert_temple",
    "jungle_temple",
    "igloo",
    "shipwreck",
    "pillager_outpost",
    "stronghold",
    "dungeon",
]

# ── Which structures can appear in which biomes ────────────────────────────────
VALID_BIOME_STRUCTURES = {
    "plains":         ["none", "village", "blacksmith", "mineshaft", "ruined_portal", "dungeon"],
    "forest":         ["none", "mineshaft", "ruined_portal", "dungeon"],
    "desert":         ["none", "village", "blacksmith", "desert_temple", "mineshaft", "ruined_portal"],
    "mesa":           ["none", "mineshaft", "ruined_portal"],
    "ice_spikes":     ["none", "igloo", "mineshaft"],
    "jungle":         ["none", "jungle_temple", "mineshaft", "ruined_portal"],
    "swamp":          ["none", "mineshaft", "ruined_portal", "dungeon"],
    "savanna":        ["none", "village", "blacksmith", "mineshaft", "ruined_portal"],
    "taiga":          ["none", "village", "mineshaft", "ruined_portal", "igloo"],
    "mountains":      ["none", "mineshaft", "ruined_portal", "dungeon"],
    "ocean":          ["none", "shipwreck", "ruined_portal"],
    "mushroom_island":["none", "shipwreck"],
    "dark_forest":    ["none", "mineshaft", "ruined_portal", "dungeon"],
    "lush_caves":     ["none", "mineshaft", "dungeon"],
    "dripstone_caves":["none", "mineshaft", "dungeon"],
    "deep_dark":      ["none", "mineshaft", "stronghold"],
}

# ── Y-level ore strategies (Minecraft 1.18+ / 1.21 distribution) ──────────────
Y_LEVEL_STRATEGIES = {
    "diamond": {"optimal": -59, "range": (-64, -1)},
    "iron":    {"optimal": 16,  "range": (-64, 320)},
    "gold":    {"optimal": -16, "range": (-64, 32)},
    "coal":    {"optimal": 96,  "range": (0, 320)},
    "copper":  {"optimal": 48,  "range": (-16, 112)},
}
