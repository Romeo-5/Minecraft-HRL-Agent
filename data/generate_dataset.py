#!/usr/bin/env python3
"""
Generate Minecraft HRL Reasoning Path Benchmark Dataset.

Produces reasoning_paths.json with 150+ samples showing how an intelligent
Minecraft agent should plan based on environmental context (biome, structures, y-level).

Targeting Minecraft 1.20.1 (1.18+ ore distribution).

Usage:
    python generate_dataset.py
    python generate_dataset.py --output custom_output.json
    python generate_dataset.py --stats-only  # just print stats for existing dataset
"""

import argparse
import json
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

# Seed for reproducibility
random.seed(42)

# =============================================================================
# MINECRAFT KNOWLEDGE BASE (1.20.1)
# =============================================================================

BIOMES = [
    "plains", "forest", "desert", "mesa", "ice_spikes", "jungle",
    "swamp", "savanna", "taiga", "mountains", "ocean",
    "mushroom_island", "dark_forest"
]

STRUCTURES = [
    "none", "village", "blacksmith", "mineshaft", "ruined_portal",
    "desert_temple", "jungle_temple", "igloo", "shipwreck",
    "pillager_outpost", "stronghold"
]

TASKS = [
    "obtain_wooden_pickaxe", "obtain_stone_pickaxe", "obtain_iron_pickaxe",
    "obtain_diamond_pickaxe", "obtain_gold", "obtain_food",
    "build_shelter", "obtain_armor"
]

# Which structures can realistically appear in which biomes
VALID_BIOME_STRUCTURES: Dict[str, List[str]] = {
    "plains":           ["none", "village", "blacksmith", "mineshaft", "ruined_portal", "pillager_outpost"],
    "forest":           ["none", "mineshaft", "ruined_portal"],
    "desert":           ["none", "village", "blacksmith", "mineshaft", "ruined_portal", "desert_temple", "pillager_outpost"],
    "mesa":             ["none", "mineshaft", "ruined_portal"],
    "ice_spikes":       ["none", "igloo", "ruined_portal"],
    "jungle":           ["none", "jungle_temple", "mineshaft", "ruined_portal"],
    "swamp":            ["none", "mineshaft", "ruined_portal"],
    "savanna":          ["none", "village", "blacksmith", "mineshaft", "ruined_portal", "pillager_outpost"],
    "taiga":            ["none", "village", "blacksmith", "mineshaft", "ruined_portal", "pillager_outpost", "igloo"],
    "mountains":        ["none", "mineshaft", "ruined_portal", "pillager_outpost"],
    "ocean":            ["none", "shipwreck", "ruined_portal", "stronghold"],
    "mushroom_island":  ["none", "shipwreck"],
    "dark_forest":      ["none", "mineshaft", "ruined_portal"],
}

# Base task progressions: the "vanilla" path with no environmental shortcuts
# Uses key decision points (not every crafting sub-step)
BASE_TASK_PROGRESSIONS: Dict[str, List[str]] = {
    "obtain_wooden_pickaxe": [
        "harvest_wood",
        "craft_planks_and_sticks",
        "craft_crafting_table",
        "craft_wooden_pickaxe",
    ],
    "obtain_stone_pickaxe": [
        "harvest_wood",
        "craft_planks_and_sticks",
        "craft_crafting_table",
        "craft_wooden_pickaxe",
        "mine_stone",
        "craft_stone_pickaxe",
    ],
    "obtain_iron_pickaxe": [
        "harvest_wood",
        "craft_planks_and_sticks",
        "craft_crafting_table",
        "craft_wooden_pickaxe",
        "mine_stone",
        "craft_stone_pickaxe",
        "mine_iron_ore",
        "craft_furnace",
        "smelt_iron",
        "craft_iron_pickaxe",
    ],
    "obtain_diamond_pickaxe": [
        "harvest_wood",
        "craft_planks_and_sticks",
        "craft_crafting_table",
        "craft_wooden_pickaxe",
        "mine_stone",
        "craft_stone_pickaxe",
        "mine_iron_ore",
        "craft_furnace",
        "smelt_iron",
        "craft_iron_pickaxe",
        "dig_to_diamond_level",
        "mine_diamonds",
        "craft_diamond_pickaxe",
    ],
    "obtain_gold": [
        "harvest_wood",
        "craft_planks_and_sticks",
        "craft_crafting_table",
        "craft_wooden_pickaxe",
        "mine_stone",
        "craft_stone_pickaxe",
        "mine_iron_ore",
        "craft_furnace",
        "smelt_iron",
        "craft_iron_pickaxe",
        "dig_to_gold_level",
        "mine_gold_ore",
        "smelt_gold",
    ],
    "obtain_food": [
        "search_for_animals",
        "kill_animals_for_meat",
        "craft_furnace_or_campfire",
        "cook_meat",
    ],
    "build_shelter": [
        "harvest_wood",
        "craft_planks",
        "build_walls_and_roof",
        "craft_and_place_door",
        "place_torches_inside",
    ],
    "obtain_armor": [
        "harvest_wood",
        "craft_planks_and_sticks",
        "craft_crafting_table",
        "craft_wooden_pickaxe",
        "mine_stone",
        "craft_stone_pickaxe",
        "mine_iron_ore",
        "craft_furnace",
        "smelt_iron",
        "craft_iron_armor_set",
    ],
}

# Structure shortcuts: what each structure can provide for which tasks
STRUCTURE_SHORTCUTS: Dict[str, Dict] = {
    "village": {
        "provides": {
            "obtain_food": {
                "steps": ["go_to_village", "harvest_village_crops"],
                "explanation": "Villages have farms with wheat, carrots, potatoes, and beetroot for easy food",
            },
            "build_shelter": {
                "steps": ["go_to_village", "use_existing_village_house"],
                "explanation": "Village houses provide ready-made shelter with beds and lighting",
            },
            "obtain_wooden_pickaxe": {
                "steps": ["go_to_village", "check_village_chests", "if_no_tools_then_standard_progression"],
                "explanation": "Village chests occasionally contain tools, saving crafting time",
            },
        },
    },
    "blacksmith": {
        "provides": {
            "obtain_iron_pickaxe": {
                "steps": ["go_to_village", "find_blacksmith", "loot_blacksmith_chest"],
                "fallback": "if_no_iron_pickaxe_then_standard_progression",
                "explanation": "Blacksmith chests often contain iron pickaxes, iron ingots, and obsidian",
            },
            "obtain_armor": {
                "steps": ["go_to_village", "find_blacksmith", "loot_blacksmith_chest"],
                "fallback": "if_no_armor_then_standard_progression",
                "explanation": "Blacksmith chests can contain iron armor pieces (helmet, chestplate, leggings, boots)",
            },
            "obtain_stone_pickaxe": {
                "steps": ["go_to_village", "find_blacksmith", "loot_blacksmith_chest"],
                "fallback": "if_no_pickaxe_then_standard_progression",
                "explanation": "Blacksmith chests may contain stone or iron pickaxes",
            },
            "obtain_diamond_pickaxe": {
                "steps": ["go_to_village", "find_blacksmith", "loot_blacksmith_chest_for_iron_tools",
                          "skip_to_diamond_mining_if_iron_pickaxe_found"],
                "explanation": "Blacksmith iron pickaxe lets you skip straight to diamond mining",
            },
            "obtain_food": {
                "steps": ["go_to_village", "harvest_village_crops", "trade_with_butcher"],
                "explanation": "Villages provide crops and butcher villagers trade cooked meat",
            },
            "obtain_gold": {
                "steps": ["go_to_village", "find_blacksmith", "loot_blacksmith_chest_for_iron_tools",
                          "use_iron_pickaxe_to_mine_gold"],
                "explanation": "Blacksmith iron pickaxe allows direct gold mining, skipping tool progression",
            },
        },
    },
    "mineshaft": {
        "provides": {
            "obtain_iron_pickaxe": {
                "steps": ["navigate_to_mineshaft", "collect_iron_from_exposed_veins",
                          "search_mineshaft_chests", "smelt_and_craft"],
                "explanation": "Mineshafts expose ore veins in walls and contain chest loot with iron ingots and rails",
            },
            "obtain_diamond_pickaxe": {
                "steps": ["navigate_to_mineshaft", "follow_tunnels_to_deeper_levels",
                          "mine_exposed_ores_along_way", "reach_diamond_level"],
                "explanation": "Mineshaft tunnels provide pre-dug paths to deeper levels with exposed ores",
            },
            "obtain_gold": {
                "steps": ["navigate_to_mineshaft", "mine_exposed_gold_ore",
                          "search_mineshaft_chests", "smelt_gold"],
                "explanation": "Mineshafts at lower levels expose gold ore veins in tunnel walls",
            },
            "obtain_stone_pickaxe": {
                "steps": ["navigate_to_mineshaft", "mine_exposed_stone",
                          "craft_stone_pickaxe"],
                "explanation": "Mineshafts provide immediate access to stone without digging down",
            },
        },
    },
    "ruined_portal": {
        "provides": {
            "obtain_gold": {
                "steps": ["go_to_ruined_portal", "loot_portal_chest",
                          "mine_gilded_blackstone_blocks"],
                "explanation": "Ruined portal chests contain gold ingots, nuggets, and golden items; gilded blackstone drops gold nuggets",
            },
            "obtain_armor": {
                "steps": ["go_to_ruined_portal", "loot_portal_chest_for_golden_armor"],
                "explanation": "Ruined portal chests can contain golden armor pieces and enchanted golden items",
            },
        },
    },
    "desert_temple": {
        "provides": {
            "obtain_gold": {
                "steps": ["go_to_desert_temple", "avoid_tnt_trap",
                          "loot_temple_chests"],
                "explanation": "Desert temple chests contain gold ingots, golden apples, and enchanted items; watch for TNT trap at bottom",
            },
            "obtain_iron_pickaxe": {
                "steps": ["go_to_desert_temple", "avoid_tnt_trap",
                          "loot_temple_chests_for_iron"],
                "explanation": "Desert temple chests can contain iron ingots and iron tools",
            },
            "obtain_diamond_pickaxe": {
                "steps": ["go_to_desert_temple", "avoid_tnt_trap",
                          "loot_temple_chests_for_diamonds"],
                "fallback": "if_no_diamonds_then_standard_mining",
                "explanation": "Desert temple chests rarely contain diamonds; more likely iron tools to skip progression",
            },
            "obtain_armor": {
                "steps": ["go_to_desert_temple", "avoid_tnt_trap",
                          "loot_temple_chests_for_armor"],
                "explanation": "Desert temple chests can contain enchanted armor pieces",
            },
        },
    },
    "jungle_temple": {
        "provides": {
            "obtain_gold": {
                "steps": ["go_to_jungle_temple", "navigate_puzzles_and_traps",
                          "loot_temple_chests"],
                "explanation": "Jungle temple chests contain gold, diamonds, and enchanted items; navigate lever puzzles and arrow traps",
            },
            "obtain_diamond_pickaxe": {
                "steps": ["go_to_jungle_temple", "navigate_puzzles_and_traps",
                          "loot_temple_chests_for_diamonds_or_iron"],
                "explanation": "Jungle temple chests can contain diamonds for direct pickaxe crafting",
            },
            "obtain_armor": {
                "steps": ["go_to_jungle_temple", "navigate_puzzles_and_traps",
                          "loot_temple_chests_for_armor_or_materials"],
                "explanation": "Jungle temple chests may contain iron or diamond armor pieces",
            },
        },
    },
    "igloo": {
        "provides": {
            "build_shelter": {
                "steps": ["go_to_igloo", "use_igloo_as_shelter"],
                "explanation": "Igloos provide immediate shelter with a bed, furnace, and crafting table",
            },
            "obtain_food": {
                "steps": ["go_to_igloo", "check_igloo_chest"],
                "explanation": "Igloo chests sometimes contain food items",
            },
            "obtain_gold": {
                "steps": ["go_to_igloo", "find_basement_trapdoor",
                          "descend_to_basement", "take_golden_apple"],
                "explanation": "Igloo basements (50% chance) contain a golden apple and brewing stand",
            },
        },
    },
    "shipwreck": {
        "provides": {
            "obtain_food": {
                "steps": ["swim_to_shipwreck", "loot_supply_chest"],
                "explanation": "Shipwreck supply chests contain wheat, carrots, potatoes, and suspicious stew",
            },
            "obtain_iron_pickaxe": {
                "steps": ["swim_to_shipwreck", "loot_treasure_chest_for_iron"],
                "explanation": "Shipwreck treasure chests can contain iron ingots and iron nuggets",
            },
            "obtain_gold": {
                "steps": ["swim_to_shipwreck", "loot_treasure_chest_for_gold"],
                "explanation": "Shipwreck treasure chests can contain gold nuggets and emeralds",
            },
        },
    },
    "pillager_outpost": {
        "provides": {
            "obtain_iron_pickaxe": {
                "steps": ["approach_pillager_outpost_carefully", "clear_pillagers",
                          "loot_outpost_chests_for_iron"],
                "explanation": "Pillager outpost chests contain iron ingots and crossbows; must defeat pillagers first",
            },
            "obtain_food": {
                "steps": ["approach_pillager_outpost_carefully", "clear_pillagers",
                          "take_hay_bales_and_pumpkins"],
                "explanation": "Pillager outposts have hay bale decorations (craft to wheat) and pumpkins nearby",
            },
        },
    },
    "stronghold": {
        "provides": {
            "obtain_iron_pickaxe": {
                "steps": ["locate_stronghold_entrance", "navigate_stronghold_corridors",
                          "loot_stronghold_chests"],
                "explanation": "Stronghold chests contain iron ingots, armor, and tools; also has a library with books",
            },
            "obtain_armor": {
                "steps": ["locate_stronghold_entrance", "navigate_stronghold_corridors",
                          "loot_stronghold_chests_for_armor"],
                "explanation": "Stronghold storerooms and corridor chests can contain iron armor and enchanted items",
            },
            "obtain_diamond_pickaxe": {
                "steps": ["locate_stronghold_entrance", "navigate_stronghold_corridors",
                          "loot_chests_for_iron_or_diamond"],
                "explanation": "Stronghold chests occasionally contain diamonds and iron tools for progression skip",
            },
        },
    },
}

# Biome-specific modifiers: how each biome affects strategies
BIOME_MODIFIERS: Dict[str, Dict] = {
    "plains": {
        "wood": "normal",
        "food": "abundant_animals",
        "shelter_note": "flat_terrain_easy_building",
        "special": None,
        "food_detail": "Plains have abundant passive mobs (cows, pigs, sheep, chickens) for easy food",
        "shelter_detail": "Flat terrain makes building straightforward; tall grass for seeds",
    },
    "forest": {
        "wood": "abundant",
        "food": "moderate_animals",
        "shelter_note": "abundant_wood",
        "special": None,
        "food_detail": "Moderate animal spawns; can also find apples from oak leaves",
        "shelter_detail": "Abundant wood makes shelter building fast; can hollow out trees",
    },
    "desert": {
        "wood": "scarce",
        "food": "very_scarce",
        "shelter_note": "sandstone_or_dig_shelter",
        "special": "no_passive_mobs_except_rabbits",
        "wood_steps": ["search_for_dead_bushes", "find_village_for_wood"],
        "food_steps": ["hunt_rabbits", "find_village_crops", "find_cactus"],
        "shelter_steps": ["dig_into_sand_carefully", "use_sandstone_blocks"],
        "food_detail": "No passive mobs except rabbits; cacti are plentiful but hurt; villages are the best food source",
        "shelter_detail": "No trees for wood; use sandstone or dig into hillside; sand falls so be careful",
    },
    "mesa": {
        "wood": "scarce",
        "food": "scarce",
        "shelter_note": "terracotta_caves",
        "special": "gold_at_all_y_levels",
        "gold_note": "Gold ore generates at all Y levels in mesa biome, not just deep underground",
        "food_detail": "Very few animals; mine shafts at surface may have food in chests",
        "shelter_detail": "Terracotta and hardened clay available; cave openings common in cliff faces",
    },
    "ice_spikes": {
        "wood": "none",
        "food": "scarce",
        "shelter_note": "urgent_dig_underground",
        "special": "packed_ice_available",
        "wood_steps": ["search_for_spruce_at_biome_edge"],
        "food_steps": ["hunt_polar_bears_risky", "fish_in_frozen_rivers"],
        "shelter_steps": ["dig_underground_immediately", "use_packed_ice_blocks"],
        "food_detail": "Very few animals; polar bears are hostile if provoked; freezing is a concern",
        "shelter_detail": "No trees; must dig underground or travel to biome edge for wood; packed ice available for building",
    },
    "jungle": {
        "wood": "very_abundant",
        "food": "abundant",
        "shelter_note": "jungle_tree_platforms",
        "special": "melons_cocoa_bamboo",
        "food_steps": ["harvest_melons", "harvest_cocoa_beans", "hunt_parrots_or_ocelots_avoid"],
        "food_detail": "Melons grow naturally; cocoa beans on jungle trees; bamboo for scaffolding; very dense vegetation",
        "shelter_detail": "Massive jungle trees can serve as shelter platforms; abundant wood varieties",
    },
    "swamp": {
        "wood": "moderate",
        "food": "moderate",
        "shelter_note": "elevated_build",
        "special": "slimes_and_witch_huts",
        "food_detail": "Some animals; lily pads for navigation; blue orchids for dye",
        "shelter_detail": "Build elevated to avoid water; swamp oaks available; watch for slimes at night",
    },
    "savanna": {
        "wood": "moderate_acacia",
        "food": "moderate_animals",
        "shelter_note": "acacia_building",
        "special": None,
        "food_detail": "Horses, cows, and sheep spawn; acacia trees for wood",
        "shelter_detail": "Flat terrain with acacia wood; similar to plains building",
    },
    "taiga": {
        "wood": "abundant_spruce",
        "food": "moderate",
        "shelter_note": "spruce_cabin",
        "special": "sweet_berries",
        "food_steps": ["harvest_sweet_berries", "hunt_foxes_or_rabbits"],
        "food_detail": "Sweet berry bushes are abundant and easy food; foxes and rabbits spawn",
        "shelter_detail": "Spruce wood is abundant; village igloos nearby in snowy variants",
    },
    "mountains": {
        "wood": "moderate_at_lower_elevations",
        "food": "scarce_at_peaks",
        "shelter_note": "cave_shelter",
        "special": "emerald_ore",
        "food_detail": "Goats at peaks; fewer animals at high elevation; llamas in meadows",
        "shelter_detail": "Natural caves are abundant; stone is immediately accessible at the surface",
    },
    "ocean": {
        "wood": "none_unless_island",
        "food": "fish_abundant",
        "shelter_note": "island_or_underwater",
        "special": "limited_land_resources",
        "wood_steps": ["find_nearby_island", "break_shipwreck_wood"],
        "food_steps": ["fish_with_rod", "kill_fish_in_shallow_water"],
        "shelter_steps": ["find_island", "build_on_island", "or_find_shipwreck_shelter"],
        "food_detail": "Fish are abundant; kelp is edible when smelted; drowned may drop items",
        "shelter_detail": "Must find an island or build on water; shipwrecks can serve as temporary shelter",
    },
    "mushroom_island": {
        "wood": "none",
        "food": "mooshrooms_infinite",
        "shelter_note": "mycelium_surface",
        "special": "no_hostile_mobs",
        "food_steps": ["milk_mooshroom_with_bowl_for_stew"],
        "wood_steps": ["no_trees_must_import_or_find_shipwreck"],
        "food_detail": "Mooshrooms provide infinite mushroom stew with a bowl; no hostile mobs spawn at all",
        "shelter_detail": "No hostile mobs means shelter is less urgent; no trees available; must bring or find wood",
    },
    "dark_forest": {
        "wood": "very_abundant_dark_oak",
        "food": "moderate",
        "shelter_note": "dense_canopy",
        "special": "woodland_mansions_rarely",
        "food_detail": "Mushrooms grow on surface due to low light; some animals in clearings",
        "shelter_detail": "Dense dark oak canopy creates natural cover; giant mushrooms provide building material",
    },
}

# Y-level strategies for 1.20.1 (1.18+ ore distribution)
Y_LEVEL_STRATEGIES = {
    "diamond": {"optimal": -59, "range": (-64, -1), "note": "Diamond ore peaks at Y=-59 in 1.18+"},
    "iron": {"optimal": 16, "range": (-64, 320), "note": "Iron ore peaks at Y=16 (also Y=232) in 1.18+"},
    "gold": {"optimal": -16, "range": (-64, 32), "note": "Gold ore peaks at Y=-16 in 1.18+; any Y in mesa"},
    "coal": {"optimal": 96, "range": (0, 320), "note": "Coal ore peaks at Y=96, abundant at most surface levels"},
    "copper": {"optimal": 48, "range": (-16, 112), "note": "Copper ore peaks at Y=48"},
}


# =============================================================================
# SAMPLE GENERATION
# =============================================================================

def make_sample(
    sample_id: int,
    biome: str,
    structures: List[str],
    y_level: int,
    task: str,
    reasoning_path: List[str],
    reasoning_text: str,
    context_matters: bool,
    context_explanation: str,
) -> dict:
    """Create a single dataset sample."""
    return {
        "id": sample_id,
        "biome": biome,
        "nearby_structures": structures,
        "y_level": y_level,
        "task": task,
        "reasoning_path": reasoning_path,
        "reasoning_text": reasoning_text,
        "context_matters": context_matters,
        "context_explanation": context_explanation,
    }


def generate_vanilla_baselines() -> List[dict]:
    """Generate baseline samples with no environmental shortcuts (plains, no structures, surface)."""
    samples = []
    for task in TASKS:
        base_steps = BASE_TASK_PROGRESSIONS[task]
        task_readable = task.replace("_", " ")

        reasoning_text = (
            f"In a plains biome at the surface with no nearby structures, "
            f"the standard approach to {task_readable} is required. "
            f"Follow the default progression: {' -> '.join(s.replace('_', ' ') for s in base_steps)}. "
            f"No environmental shortcuts are available."
        )

        samples.append(make_sample(
            sample_id=0,  # will be reassigned
            biome="plains",
            structures=["none"],
            y_level=72,
            task=task,
            reasoning_path=base_steps,
            reasoning_text=reasoning_text,
            context_matters=False,
            context_explanation="No structures nearby and plains biome offers no special advantages; standard progression is optimal.",
        ))
    return samples


def generate_structure_shortcut_samples() -> List[dict]:
    """Generate samples where structures provide meaningful shortcuts."""
    samples = []

    for structure, info in STRUCTURE_SHORTCUTS.items():
        for task, shortcut_data in info["provides"].items():
            # Pick a valid biome for this structure
            valid_biomes = [b for b, structs in VALID_BIOME_STRUCTURES.items() if structure in structs]
            if not valid_biomes:
                continue

            biome = random.choice(valid_biomes)
            y_level = 72 if "mine" not in task and "diamond" not in task else 64

            steps = list(shortcut_data["steps"])
            fallback = shortcut_data.get("fallback")
            if fallback:
                steps.append(fallback)

            explanation = shortcut_data["explanation"]

            task_readable = task.replace("_", " ")
            struct_readable = structure.replace("_", " ")

            reasoning_text = (
                f"In a {biome.replace('_', ' ')} biome with a {struct_readable} nearby, "
                f"the optimal approach to {task_readable} leverages the structure. "
                f"{explanation}. "
            )
            if fallback:
                reasoning_text += f"If the desired items are not found, fall back to standard progression."

            structures_list = [structure]
            # Blacksmith implies village
            if structure == "blacksmith":
                structures_list = ["village", "blacksmith"]

            samples.append(make_sample(
                sample_id=0,
                biome=biome,
                structures=structures_list,
                y_level=y_level,
                task=task,
                reasoning_path=steps,
                reasoning_text=reasoning_text,
                context_matters=True,
                context_explanation=explanation,
            ))

    return samples


def generate_biome_specific_samples() -> List[dict]:
    """Generate samples where biome significantly affects strategy."""
    samples = []

    # Desert - food scarcity
    samples.append(make_sample(
        sample_id=0, biome="desert", structures=["none"], y_level=72,
        task="obtain_food",
        reasoning_path=[
            "scan_for_rabbits", "hunt_rabbits", "search_for_village",
            "harvest_village_crops_if_found", "cook_rabbit_meat"
        ],
        reasoning_text=(
            "In a desert biome with no structures, food is extremely scarce. "
            "No passive mobs spawn except rabbits. Hunt rabbits for meat, "
            "search for a nearby village with farms, or harvest cactus as emergency food (damages you). "
            "Cooking rabbit meat provides the best sustenance available."
        ),
        context_matters=True,
        context_explanation="Desert has no passive mobs except rabbits, making food much harder to obtain than in most biomes.",
    ))

    # Desert - shelter
    samples.append(make_sample(
        sample_id=0, biome="desert", structures=["none"], y_level=72,
        task="build_shelter",
        reasoning_path=[
            "dig_into_sandstone_hillside", "create_room_underground",
            "seal_entrance_with_sandstone", "place_torches"
        ],
        reasoning_text=(
            "In a desert biome with no structures, building shelter requires adapting to the lack of wood. "
            "Dig into a sandstone hillside to create an underground room. "
            "Sand blocks fall due to gravity, so use sandstone which is stable. "
            "Seal the entrance with sandstone blocks and place torches for lighting."
        ),
        context_matters=True,
        context_explanation="Desert lacks trees for wood, requiring sandstone-based or underground shelter instead of standard wood construction.",
    ))

    # Desert - wood scarcity affects tool crafting
    samples.append(make_sample(
        sample_id=0, biome="desert", structures=["none"], y_level=72,
        task="obtain_wooden_pickaxe",
        reasoning_path=[
            "search_for_dead_bushes_for_sticks", "search_for_biome_edge_trees",
            "if_no_trees_then_dig_for_mineshaft_wood",
            "harvest_wood", "craft_planks_and_sticks",
            "craft_crafting_table", "craft_wooden_pickaxe"
        ],
        reasoning_text=(
            "In a desert biome with no structures, obtaining a wooden pickaxe is challenging due to scarce wood. "
            "Search for dead bushes (drop sticks) and biome-edge trees. "
            "If no surface wood is found, dig down to find mineshaft planks. "
            "Once wood is acquired, follow standard crafting progression."
        ),
        context_matters=True,
        context_explanation="Desert biome has no natural trees, making the first step of the tech tree significantly harder.",
    ))

    # Mesa - gold at all Y levels
    samples.append(make_sample(
        sample_id=0, biome="mesa", structures=["none"], y_level=72,
        task="obtain_gold",
        reasoning_path=[
            "harvest_wood_from_biome_edge", "craft_basic_tools",
            "mine_surface_gold_ore", "craft_furnace", "smelt_gold"
        ],
        reasoning_text=(
            "In a mesa/badlands biome, gold ore generates at all Y levels, not just deep underground. "
            "This means surface-level gold mining is possible. "
            "Craft basic stone tools, then mine gold ore from exposed cliff faces and surface caves. "
            "No need to dig down to Y=-16 like in other biomes."
        ),
        context_matters=True,
        context_explanation="Mesa biome has gold ore at all Y levels, eliminating the need to dig deep underground (normally Y=-16 optimal).",
    ))

    # Mesa - gold underground (combined with y-level)
    samples.append(make_sample(
        sample_id=0, biome="mesa", structures=["none"], y_level=32,
        task="obtain_gold",
        reasoning_path=[
            "mine_gold_at_current_level", "craft_furnace_if_needed", "smelt_gold"
        ],
        reasoning_text=(
            "In a mesa biome at Y=32 underground, gold ore is available at this level "
            "(mesa generates gold at all Y levels). "
            "Mine gold directly at the current depth without needing to dig further. "
            "Craft a furnace and smelt the gold ore."
        ),
        context_matters=True,
        context_explanation="Being underground in mesa means gold is available right here, unlike other biomes where you'd need to go to Y=-16.",
    ))

    # Ice spikes - urgent shelter
    samples.append(make_sample(
        sample_id=0, biome="ice_spikes", structures=["none"], y_level=72,
        task="build_shelter",
        reasoning_path=[
            "dig_underground_immediately", "create_underground_room",
            "place_torches_for_warmth_and_light", "seal_entrance"
        ],
        reasoning_text=(
            "In an ice spikes biome, shelter is urgently needed. "
            "There are no trees for wood, so dig straight underground to create an emergency shelter. "
            "Packed ice blocks can be used for building but provide no warmth. "
            "Place torches immediately for light and to prevent mob spawns."
        ),
        context_matters=True,
        context_explanation="Ice spikes has no trees and freezing conditions, making underground shelter the only immediate option.",
    ))

    # Ice spikes - tool crafting challenge
    samples.append(make_sample(
        sample_id=0, biome="ice_spikes", structures=["none"], y_level=72,
        task="obtain_wooden_pickaxe",
        reasoning_path=[
            "travel_toward_biome_edge_for_spruce", "harvest_spruce_wood",
            "craft_planks_and_sticks", "craft_crafting_table", "craft_wooden_pickaxe"
        ],
        reasoning_text=(
            "In an ice spikes biome with no structures, there are no trees. "
            "Travel toward the biome edge to find spruce trees in adjacent snowy taiga. "
            "This is the primary challenge - once wood is obtained, standard crafting applies."
        ),
        context_matters=True,
        context_explanation="Ice spikes has no trees, requiring travel to biome edge before standard tool progression can begin.",
    ))

    # Jungle - food abundance
    samples.append(make_sample(
        sample_id=0, biome="jungle", structures=["none"], y_level=72,
        task="obtain_food",
        reasoning_path=[
            "harvest_melons_from_ground", "harvest_cocoa_beans_from_trees",
            "craft_melon_slices"
        ],
        reasoning_text=(
            "In a jungle biome, food is extremely abundant. "
            "Melons generate naturally on the ground and provide instant food. "
            "Cocoa beans grow on jungle tree trunks. "
            "No need to hunt animals or cook - melons provide immediate sustenance."
        ),
        context_matters=True,
        context_explanation="Jungle has natural melon spawns and cocoa beans, providing immediate food without hunting or cooking.",
    ))

    # Jungle - shelter
    samples.append(make_sample(
        sample_id=0, biome="jungle", structures=["none"], y_level=72,
        task="build_shelter",
        reasoning_path=[
            "harvest_jungle_wood", "build_platform_on_large_tree",
            "craft_planks", "build_elevated_shelter", "add_ladder_access"
        ],
        reasoning_text=(
            "In a jungle biome, massive jungle trees provide natural building platforms. "
            "Harvest jungle wood and build an elevated shelter on or between large trees. "
            "The dense canopy provides partial cover from rain and some mob protection. "
            "An elevated shelter avoids ground-level mobs."
        ),
        context_matters=True,
        context_explanation="Jungle's massive trees enable elevated platform shelters, which are safer than ground-level builds.",
    ))

    # Mushroom island - no hostile mobs
    samples.append(make_sample(
        sample_id=0, biome="mushroom_island", structures=["none"], y_level=72,
        task="build_shelter",
        reasoning_path=[
            "no_urgent_shelter_needed", "harvest_giant_mushroom_blocks",
            "build_mushroom_shelter_when_convenient"
        ],
        reasoning_text=(
            "On a mushroom island, no hostile mobs spawn at all, making shelter much less urgent. "
            "Build shelter at your convenience using giant mushroom blocks or mycelium. "
            "The primary concern is not mobs but eventually obtaining wood for further progression."
        ),
        context_matters=True,
        context_explanation="Mushroom island spawns no hostile mobs, making shelter a low-priority comfort rather than a survival necessity.",
    ))

    # Mushroom island - food
    samples.append(make_sample(
        sample_id=0, biome="mushroom_island", structures=["none"], y_level=72,
        task="obtain_food",
        reasoning_path=[
            "craft_bowl_from_planks", "milk_mooshroom_with_bowl",
            "obtain_unlimited_mushroom_stew"
        ],
        reasoning_text=(
            "On a mushroom island, mooshrooms provide infinite food via mushroom stew. "
            "Craft a wooden bowl (requires wood from elsewhere or a shipwreck) and milk a mooshroom for stew. "
            "Each bowl of mushroom stew restores 6 hunger. This is an infinite food source."
        ),
        context_matters=True,
        context_explanation="Mooshrooms provide unlimited mushroom stew, the easiest infinite food source in the game.",
    ))

    # Mushroom island - tool challenge (no wood)
    samples.append(make_sample(
        sample_id=0, biome="mushroom_island", structures=["none"], y_level=72,
        task="obtain_wooden_pickaxe",
        reasoning_path=[
            "search_for_nearby_shipwreck", "break_shipwreck_for_wood",
            "or_travel_to_mainland_for_trees",
            "craft_planks_and_sticks", "craft_crafting_table", "craft_wooden_pickaxe"
        ],
        reasoning_text=(
            "On a mushroom island, there are no trees at all. "
            "Search for a nearby shipwreck to break for wood planks, or travel to the mainland. "
            "This makes the first tech tree step the biggest challenge. "
            "Once wood is obtained, standard crafting progression applies."
        ),
        context_matters=True,
        context_explanation="Mushroom island has zero trees, making wood acquisition the critical first challenge.",
    ))

    # Ocean - limited resources
    samples.append(make_sample(
        sample_id=0, biome="ocean", structures=["none"], y_level=62,
        task="obtain_food",
        reasoning_path=[
            "craft_fishing_rod_if_wood_available", "fish_for_food",
            "cook_fish", "or_eat_dried_kelp"
        ],
        reasoning_text=(
            "In an ocean biome, traditional hunting is impossible. "
            "Fish using a crafting rod (requires sticks and string from spider) or kill fish in shallow water. "
            "Kelp can be smelted into dried kelp for emergency food. "
            "Fish is the most reliable food source in ocean biomes."
        ),
        context_matters=True,
        context_explanation="Ocean biome has no land animals; fishing and kelp are the only food options.",
    ))

    # Ocean - shelter
    samples.append(make_sample(
        sample_id=0, biome="ocean", structures=["none"], y_level=62,
        task="build_shelter",
        reasoning_path=[
            "find_small_island_or_sandbar", "gather_sand_and_gravel",
            "build_basic_shelter_on_island", "or_build_underwater_shelter_with_doors"
        ],
        reasoning_text=(
            "In an ocean biome, shelter options are limited. "
            "Find a small island or sandbar for a land-based shelter, or build an underwater shelter using doors "
            "(which create air pockets). Land resources are very scarce."
        ),
        context_matters=True,
        context_explanation="Ocean biome has minimal land, requiring island search or underwater building techniques.",
    ))

    # Dark forest - abundant wood
    samples.append(make_sample(
        sample_id=0, biome="dark_forest", structures=["none"], y_level=72,
        task="build_shelter",
        reasoning_path=[
            "harvest_dark_oak_wood", "clear_small_area",
            "build_shelter_under_canopy", "place_torches_to_prevent_mob_spawns"
        ],
        reasoning_text=(
            "In a dark forest, the dense canopy creates low light levels even during the day, "
            "causing hostile mob spawns on the surface. Build shelter quickly using abundant dark oak wood. "
            "Heavily torch the surrounding area to prevent daytime mob spawns. "
            "The thick canopy provides natural rain cover."
        ),
        context_matters=True,
        context_explanation="Dark forest's low light level causes surface hostile mob spawns during the day, making shelter and torches critical.",
    ))

    # Swamp - slime advantage
    samples.append(make_sample(
        sample_id=0, biome="swamp", structures=["none"], y_level=72,
        task="obtain_food",
        reasoning_path=[
            "hunt_animals_in_clearings", "harvest_mushrooms_from_surface",
            "craft_mushroom_stew", "fish_in_swamp_water"
        ],
        reasoning_text=(
            "In a swamp biome, food is moderately available. "
            "Hunt animals in clearings between water patches. "
            "Mushrooms grow on the surface due to lower light under swamp oaks. "
            "Craft mushroom stew (red + brown mushroom + bowl) for a good food source."
        ),
        context_matters=True,
        context_explanation="Swamp has surface mushrooms due to low light, enabling mushroom stew crafting as an alternative food source.",
    ))

    # Mountains - easy stone access
    samples.append(make_sample(
        sample_id=0, biome="mountains", structures=["none"], y_level=95,
        task="obtain_stone_pickaxe",
        reasoning_path=[
            "harvest_wood_at_lower_elevation", "craft_wooden_pickaxe",
            "mine_exposed_stone_from_cliff_face", "craft_stone_pickaxe"
        ],
        reasoning_text=(
            "In a mountains biome, stone is exposed at the surface in cliff faces. "
            "After crafting a wooden pickaxe, mine stone directly from exposed rock faces "
            "without needing to dig underground. This saves time compared to flat biomes."
        ),
        context_matters=True,
        context_explanation="Mountains have surface-exposed stone, skipping the digging step needed in flat biomes like plains.",
    ))

    # Mountains - cave shelter
    samples.append(make_sample(
        sample_id=0, biome="mountains", structures=["none"], y_level=95,
        task="build_shelter",
        reasoning_path=[
            "find_natural_cave_opening", "clear_cave_interior",
            "place_torches", "seal_entrance_with_stone"
        ],
        reasoning_text=(
            "In a mountains biome, natural caves are extremely common in cliff faces. "
            "Find a cave opening, clear any hostile mobs inside, place torches, "
            "and seal the entrance with stone blocks. This is faster than building from scratch."
        ),
        context_matters=True,
        context_explanation="Mountains have abundant natural caves, providing instant shelter with minimal construction needed.",
    ))

    # Taiga - sweet berries
    samples.append(make_sample(
        sample_id=0, biome="taiga", structures=["none"], y_level=72,
        task="obtain_food",
        reasoning_path=[
            "harvest_sweet_berries_from_bushes", "eat_berries_directly"
        ],
        reasoning_text=(
            "In a taiga biome, sweet berry bushes are abundant and provide instant food. "
            "Simply walk up to bushes and harvest berries. No hunting, crafting, or cooking needed. "
            "Each berry restores 2 hunger points. Collect extras for later."
        ),
        context_matters=True,
        context_explanation="Taiga has sweet berry bushes as instant no-craft food, faster than hunting and cooking in other biomes.",
    ))

    # Savanna - normal with village potential
    samples.append(make_sample(
        sample_id=0, biome="savanna", structures=["none"], y_level=72,
        task="obtain_food",
        reasoning_path=[
            "search_for_animals", "kill_animals_for_meat",
            "craft_furnace_or_campfire", "cook_meat"
        ],
        reasoning_text=(
            "In a savanna biome with no structures, food acquisition follows the standard approach. "
            "Horses, cows, and sheep spawn regularly. Hunt for meat and cook it. "
            "Savanna is similar to plains for food gathering."
        ),
        context_matters=False,
        context_explanation="Savanna food gathering is essentially the same as plains - standard hunting and cooking applies.",
    ))

    return samples


def generate_y_level_samples() -> List[dict]:
    """Generate samples where Y-level significantly affects mining strategy."""
    samples = []

    # Diamond mining at different Y levels
    for y_level, desc, steps_prefix, matters, expl in [
        (-59, "already at optimal diamond level",
         ["mine_diamonds_at_current_level", "craft_diamond_pickaxe"],
         True, "Already at Y=-59 (optimal diamond level), can mine immediately without digging down."),
        (72, "need to dig down from surface",
         ["dig_down_to_y_minus_59", "create_strip_mine", "mine_diamonds", "craft_diamond_pickaxe"],
         True, "Starting at surface requires digging ~131 blocks down to reach diamond level at Y=-59."),
        (16, "moderately deep, need to dig further",
         ["dig_down_from_y16_to_y_minus_59", "mine_diamonds", "craft_diamond_pickaxe"],
         True, "At Y=16 (iron level), still need to dig ~75 blocks deeper to reach diamond level at Y=-59."),
    ]:
        base = BASE_TASK_PROGRESSIONS["obtain_diamond_pickaxe"][:9]  # up to iron pickaxe
        full_steps = base + steps_prefix

        samples.append(make_sample(
            sample_id=0, biome="plains", structures=["none"], y_level=y_level,
            task="obtain_diamond_pickaxe",
            reasoning_path=full_steps if y_level != -59 else steps_prefix,
            reasoning_text=(
                f"At Y-level {y_level} in a plains biome, {desc}. "
                f"{'Standard tool progression needed first, then ' if y_level != -59 else ''}"
                f"{'mine diamonds directly at this depth.' if y_level == -59 else 'dig to Y=-59 for optimal diamond finding.'}"
            ),
            context_matters=matters,
            context_explanation=expl,
        ))

    # Iron mining at different Y levels
    for y_level, desc, matters, expl in [
        (16, "already at optimal iron level", True,
         "At Y=16 (optimal iron level in 1.18+), iron ore is abundant and can be mined directly."),
        (72, "surface level, need to dig or find cave", True,
         "At surface level, need to find a cave or dig down to ~Y=16 for best iron density."),
        (-50, "deep underground, iron still available but not optimal", True,
         "At Y=-50, iron ore spawns but is less dense than at Y=16; still mineable nearby though."),
    ]:
        samples.append(make_sample(
            sample_id=0, biome="forest", structures=["none"], y_level=y_level,
            task="obtain_iron_pickaxe",
            reasoning_path=(
                ["mine_iron_ore_at_current_level", "craft_furnace", "smelt_iron", "craft_iron_pickaxe"]
                if y_level == 16 else
                BASE_TASK_PROGRESSIONS["obtain_iron_pickaxe"]
            ),
            reasoning_text=(
                f"At Y-level {y_level} in a forest biome, {desc}. "
                f"{'Mine iron directly at this level.' if y_level == 16 else 'Follow standard tool progression and mine iron.'}"
            ),
            context_matters=matters,
            context_explanation=expl,
        ))

    # Gold mining at different Y levels (non-mesa)
    for y_level, desc, matters, expl in [
        (-16, "at optimal gold level", True,
         "At Y=-16 (optimal gold level in 1.18+), gold ore is most dense here."),
        (72, "surface level, need to dig deep", True,
         "At surface, must dig down ~88 blocks to Y=-16 for optimal gold mining."),
    ]:
        samples.append(make_sample(
            sample_id=0, biome="plains", structures=["none"], y_level=y_level,
            task="obtain_gold",
            reasoning_path=(
                ["mine_gold_ore_at_current_level", "craft_furnace", "smelt_gold"]
                if y_level == -16 else
                BASE_TASK_PROGRESSIONS["obtain_gold"]
            ),
            reasoning_text=(
                f"At Y-level {y_level} in a plains biome, {desc}. "
                f"{'Mine gold directly.' if y_level == -16 else 'Full tool progression and deep mining required.'}"
            ),
            context_matters=matters,
            context_explanation=expl,
        ))

    # Gold in mesa at surface (contrasting with non-mesa)
    samples.append(make_sample(
        sample_id=0, biome="mesa", structures=["none"], y_level=72,
        task="obtain_gold",
        reasoning_path=[
            "craft_basic_tools", "mine_exposed_gold_ore_at_surface",
            "craft_furnace", "smelt_gold"
        ],
        reasoning_text=(
            "In a mesa biome at surface level, gold ore generates at all Y levels. "
            "No need to dig to Y=-16 like other biomes. "
            "Mine gold directly from surface cliff faces and cave openings."
        ),
        context_matters=True,
        context_explanation="Mesa generates gold at all Y levels; surface mining saves huge time vs digging to Y=-16 in other biomes.",
    ))

    return samples


def generate_combined_samples() -> List[dict]:
    """Generate samples with multiple environmental factors interacting."""
    samples = []

    # Mesa + mineshaft + underground = gold paradise
    samples.append(make_sample(
        sample_id=0, biome="mesa", structures=["mineshaft"], y_level=-20,
        task="obtain_gold",
        reasoning_path=[
            "navigate_to_mineshaft", "mine_exposed_gold_from_tunnel_walls",
            "search_mineshaft_chests", "smelt_gold"
        ],
        reasoning_text=(
            "In a mesa biome at Y=-20 with a mineshaft nearby, gold is everywhere. "
            "Mesa generates gold at all Y levels, and the mineshaft exposes ore veins in tunnel walls. "
            "Additionally, mineshaft chests may contain gold ingots. "
            "This is the ideal gold-gathering scenario."
        ),
        context_matters=True,
        context_explanation="Mesa's universal gold generation + mineshaft exposed veins = maximum gold efficiency without any tool progression.",
    ))

    # Desert + village + blacksmith = oasis of resources
    samples.append(make_sample(
        sample_id=0, biome="desert", structures=["village", "blacksmith"], y_level=72,
        task="obtain_iron_pickaxe",
        reasoning_path=[
            "go_to_village", "find_blacksmith", "loot_blacksmith_chest",
            "if_no_iron_pickaxe_then_use_village_wood_for_standard_progression"
        ],
        reasoning_text=(
            "In a desert biome, a village with a blacksmith is an oasis of resources. "
            "The desert normally lacks trees, making tool crafting very difficult. "
            "The blacksmith chest likely contains iron tools, completely bypassing the wood scarcity problem. "
            "If not, village structures provide wooden planks for crafting."
        ),
        context_matters=True,
        context_explanation="Village+blacksmith in desert is critical: bypasses desert's wood scarcity AND provides iron tools directly.",
    ))

    # Desert + village for food
    samples.append(make_sample(
        sample_id=0, biome="desert", structures=["village"], y_level=72,
        task="obtain_food",
        reasoning_path=[
            "go_to_village", "harvest_village_farm_crops",
            "trade_with_farmer_villager", "store_excess_food"
        ],
        reasoning_text=(
            "In a desert biome with a village nearby, the village farms solve the desert's food scarcity. "
            "Desert normally has no passive mobs except rabbits. "
            "Village farms provide wheat, carrots, potatoes, and beetroot. "
            "Farmer villagers also trade food items for emeralds."
        ),
        context_matters=True,
        context_explanation="Village in desert is critical for food since desert has almost no natural food sources.",
    ))

    # Ice spikes + igloo = survival lifeline
    samples.append(make_sample(
        sample_id=0, biome="ice_spikes", structures=["igloo"], y_level=72,
        task="build_shelter",
        reasoning_path=[
            "go_to_igloo", "use_igloo_as_shelter",
            "check_for_basement_with_resources"
        ],
        reasoning_text=(
            "In an ice spikes biome with an igloo nearby, the igloo is a survival lifeline. "
            "Ice spikes has no trees for building, and shelter is urgently needed. "
            "The igloo provides a bed, furnace, and crafting table. "
            "Check for a hidden basement (trapdoor under carpet) with a brewing stand and golden apple."
        ),
        context_matters=True,
        context_explanation="Igloo in ice spikes provides critical shelter, bed, and crafting resources where none are otherwise available.",
    ))

    # Ocean + shipwreck = primary resource source
    samples.append(make_sample(
        sample_id=0, biome="ocean", structures=["shipwreck"], y_level=62,
        task="obtain_food",
        reasoning_path=[
            "swim_to_shipwreck", "loot_supply_chest_for_food",
            "also_check_treasure_chest"
        ],
        reasoning_text=(
            "In an ocean biome with a shipwreck nearby, the shipwreck supply chest is the best food source. "
            "Ocean biome has no land animals; fishing requires a rod (needs sticks and string). "
            "Shipwreck supply chests contain wheat, carrots, potatoes, and suspicious stew - immediate food."
        ),
        context_matters=True,
        context_explanation="Shipwreck in ocean provides immediate food without needing to craft a fishing rod.",
    ))

    # Ocean + shipwreck for tools
    samples.append(make_sample(
        sample_id=0, biome="ocean", structures=["shipwreck"], y_level=62,
        task="obtain_iron_pickaxe",
        reasoning_path=[
            "swim_to_shipwreck", "loot_treasure_chest_for_iron",
            "break_shipwreck_wood_for_planks",
            "craft_tools_using_shipwreck_materials"
        ],
        reasoning_text=(
            "In an ocean biome with a shipwreck, the wreck provides both iron and wood. "
            "Loot the treasure chest for iron ingots and nuggets. "
            "Break the shipwreck's wooden planks for crafting material. "
            "This solves ocean's lack of land-based resources."
        ),
        context_matters=True,
        context_explanation="Shipwreck provides both iron and wood in a biome that otherwise has neither.",
    ))

    # Jungle + jungle temple = treasure hunting
    samples.append(make_sample(
        sample_id=0, biome="jungle", structures=["jungle_temple"], y_level=72,
        task="obtain_diamond_pickaxe",
        reasoning_path=[
            "go_to_jungle_temple", "navigate_arrow_traps",
            "solve_lever_puzzle", "loot_temple_chests",
            "use_looted_materials_to_skip_progression"
        ],
        reasoning_text=(
            "In a jungle biome with a jungle temple, the temple chests can contain diamonds. "
            "Navigate the arrow trap hallway carefully, solve the lever puzzle for the hidden chest. "
            "Temple chests may contain diamonds directly, or iron to skip tool progression. "
            "Jungle provides abundant wood for any crafting needed."
        ),
        context_matters=True,
        context_explanation="Jungle temple can contain diamonds, potentially skipping the entire mining progression.",
    ))

    # Taiga + village + blacksmith for armor
    samples.append(make_sample(
        sample_id=0, biome="taiga", structures=["village", "blacksmith"], y_level=72,
        task="obtain_armor",
        reasoning_path=[
            "go_to_village", "find_blacksmith", "loot_blacksmith_chest_for_armor",
            "trade_with_armorer_villager_if_present"
        ],
        reasoning_text=(
            "In a taiga biome with a village and blacksmith, armor is readily available. "
            "Blacksmith chests often contain iron armor pieces. "
            "If an armorer villager is present, trade for additional armor pieces. "
            "Taiga villages also provide sweet berries for trade currency."
        ),
        context_matters=True,
        context_explanation="Blacksmith chests and armorer villagers provide armor without mining or smelting.",
    ))

    # Plains + ruined portal for gold
    samples.append(make_sample(
        sample_id=0, biome="plains", structures=["ruined_portal"], y_level=72,
        task="obtain_gold",
        reasoning_path=[
            "go_to_ruined_portal", "loot_portal_chest",
            "mine_gilded_blackstone_for_gold_nuggets",
            "combine_gold_nuggets_to_ingots"
        ],
        reasoning_text=(
            "In a plains biome with a ruined portal, gold is obtainable without deep mining. "
            "The portal chest contains gold ingots, nuggets, and golden items. "
            "Gilded blackstone blocks around the portal drop gold nuggets when mined. "
            "9 gold nuggets craft into 1 gold ingot."
        ),
        context_matters=True,
        context_explanation="Ruined portal provides gold through chest loot and gilded blackstone, no deep mining needed.",
    ))

    # Forest + mineshaft underground for iron
    samples.append(make_sample(
        sample_id=0, biome="forest", structures=["mineshaft"], y_level=25,
        task="obtain_iron_pickaxe",
        reasoning_path=[
            "navigate_to_mineshaft", "mine_iron_from_exposed_veins",
            "search_mineshaft_chests_for_iron_ingots",
            "craft_furnace_and_smelt_if_needed", "craft_iron_pickaxe"
        ],
        reasoning_text=(
            "In a forest biome at Y=25 near a mineshaft, iron ore is accessible through exposed tunnel walls. "
            "At Y=25, iron ore density is still good (near optimal Y=16). "
            "The mineshaft provides pre-dug tunnels revealing ore veins and chests with iron. "
            "Wood is abundant in forest for crafting tools and furnace."
        ),
        context_matters=True,
        context_explanation="Mineshaft at near-optimal iron level provides exposed ore + chest loot, skipping the dig-down step.",
    ))

    # Dark forest - obtain armor standard (no shortcut)
    samples.append(make_sample(
        sample_id=0, biome="dark_forest", structures=["none"], y_level=72,
        task="obtain_armor",
        reasoning_path=[
            "harvest_dark_oak_wood", "craft_planks_and_sticks",
            "craft_crafting_table", "craft_wooden_pickaxe",
            "mine_stone", "craft_stone_pickaxe",
            "mine_iron_ore", "craft_furnace", "smelt_iron",
            "craft_iron_armor_set"
        ],
        reasoning_text=(
            "In a dark forest with no structures, armor follows standard progression. "
            "Abundant dark oak wood accelerates initial crafting. "
            "Watch for hostile mobs that spawn under the dense canopy during the day. "
            "Standard iron armor progression applies."
        ),
        context_matters=False,
        context_explanation="Dark forest provides fast wood but doesn't change the armor crafting progression.",
    ))

    # Plains + pillager outpost for iron (risky)
    samples.append(make_sample(
        sample_id=0, biome="plains", structures=["pillager_outpost"], y_level=72,
        task="obtain_iron_pickaxe",
        reasoning_path=[
            "approach_outpost_cautiously", "assess_pillager_numbers",
            "clear_pillagers_or_sneak_around",
            "loot_outpost_chests_for_iron",
            "if_no_iron_then_standard_progression"
        ],
        reasoning_text=(
            "In a plains biome with a pillager outpost, the outpost chests may contain iron. "
            "However, this is risky - pillagers are hostile and attack with crossbows. "
            "Must defeat or evade them first. If iron is found in chests, it saves mining time. "
            "If not, standard tool progression is the fallback."
        ),
        context_matters=True,
        context_explanation="Pillager outpost offers potential iron shortcut but requires combat, adding risk-reward decision.",
    ))

    # Stronghold - deep underground
    samples.append(make_sample(
        sample_id=0, biome="plains", structures=["stronghold"], y_level=-10,
        task="obtain_armor",
        reasoning_path=[
            "navigate_stronghold_corridors", "loot_stronghold_chests",
            "check_storeroom_for_iron_armor",
            "if_no_armor_then_mine_iron_nearby_and_craft"
        ],
        reasoning_text=(
            "Near a stronghold at Y=-10, the stronghold's corridor and storeroom chests can contain iron armor. "
            "Navigate the maze-like structure, clearing silverfish spawners. "
            "Storeroom chests have a good chance of iron armor pieces. "
            "If not found, mine iron from nearby cave walls and craft armor."
        ),
        context_matters=True,
        context_explanation="Stronghold storeroom chests can contain iron armor, bypassing the entire smelting progression.",
    ))

    # Desert temple for armor
    samples.append(make_sample(
        sample_id=0, biome="desert", structures=["desert_temple"], y_level=72,
        task="obtain_armor",
        reasoning_path=[
            "go_to_desert_temple", "dig_to_treasure_room_avoiding_tnt",
            "loot_four_chests", "equip_any_armor_found"
        ],
        reasoning_text=(
            "In a desert with a desert temple, the four treasure chests can contain iron and golden armor. "
            "Carefully access the treasure room - a pressure plate triggers TNT that destroys loot. "
            "Dig around the sides or break the pressure plate first. "
            "Desert temples are one of the best early armor sources."
        ),
        context_matters=True,
        context_explanation="Desert temple's four chests provide high chance of armor pieces, avoiding the full crafting progression.",
    ))

    # Savanna + village for diamond pickaxe (shortcut to iron then mine)
    samples.append(make_sample(
        sample_id=0, biome="savanna", structures=["village", "blacksmith"], y_level=72,
        task="obtain_diamond_pickaxe",
        reasoning_path=[
            "go_to_village", "find_blacksmith", "loot_iron_pickaxe_from_chest",
            "dig_down_to_diamond_level", "mine_diamonds", "craft_diamond_pickaxe"
        ],
        reasoning_text=(
            "In a savanna village with a blacksmith, skip straight to diamond mining. "
            "Blacksmith chests often contain iron pickaxes, bypassing the entire wood-to-iron progression. "
            "With the iron pickaxe, dig directly to Y=-59 for diamond mining. "
            "This saves significant time over standard progression."
        ),
        context_matters=True,
        context_explanation="Blacksmith iron pickaxe lets you skip 9 tech tree steps and go straight to diamond mining.",
    ))

    return samples


def generate_negative_examples() -> List[dict]:
    """Generate samples where context doesn't change the optimal path."""
    samples = []

    # Forest - obtain wooden pickaxe (forest wood is nice but doesn't change path)
    samples.append(make_sample(
        sample_id=0, biome="forest", structures=["none"], y_level=72,
        task="obtain_wooden_pickaxe",
        reasoning_path=BASE_TASK_PROGRESSIONS["obtain_wooden_pickaxe"],
        reasoning_text=(
            "In a forest biome with no structures, obtaining a wooden pickaxe follows the standard path. "
            "Forest has abundant wood, but the crafting progression is the same as any biome with trees."
        ),
        context_matters=False,
        context_explanation="Forest has abundant wood but the wooden pickaxe crafting path is identical to any biome with trees.",
    ))

    # Taiga - obtain stone pickaxe (normal)
    samples.append(make_sample(
        sample_id=0, biome="taiga", structures=["none"], y_level=72,
        task="obtain_stone_pickaxe",
        reasoning_path=BASE_TASK_PROGRESSIONS["obtain_stone_pickaxe"],
        reasoning_text=(
            "In a taiga biome with no structures, stone pickaxe progression is standard. "
            "Spruce wood is available for crafting. The path doesn't change from the default."
        ),
        context_matters=False,
        context_explanation="Taiga doesn't offer any shortcuts for stone pickaxe crafting; standard progression applies.",
    ))

    # Savanna - obtain wooden pickaxe (same as plains)
    samples.append(make_sample(
        sample_id=0, biome="savanna", structures=["none"], y_level=72,
        task="obtain_wooden_pickaxe",
        reasoning_path=BASE_TASK_PROGRESSIONS["obtain_wooden_pickaxe"],
        reasoning_text=(
            "In a savanna biome with no structures, wooden pickaxe crafting is standard. "
            "Acacia trees provide wood, and the progression is the same as in any tree-bearing biome."
        ),
        context_matters=False,
        context_explanation="Savanna has normal tree availability; wooden pickaxe path is unchanged from default.",
    ))

    # Dark forest - obtain iron pickaxe (no structural help)
    samples.append(make_sample(
        sample_id=0, biome="dark_forest", structures=["none"], y_level=72,
        task="obtain_iron_pickaxe",
        reasoning_path=BASE_TASK_PROGRESSIONS["obtain_iron_pickaxe"],
        reasoning_text=(
            "In a dark forest with no structures, iron pickaxe follows standard progression. "
            "While dark oak provides abundant wood, the mining and smelting steps remain the same."
        ),
        context_matters=False,
        context_explanation="Dark forest has abundant wood but no structural shortcuts for iron pickaxe progression.",
    ))

    # Swamp - obtain stone pickaxe (normal)
    samples.append(make_sample(
        sample_id=0, biome="swamp", structures=["none"], y_level=72,
        task="obtain_stone_pickaxe",
        reasoning_path=BASE_TASK_PROGRESSIONS["obtain_stone_pickaxe"],
        reasoning_text=(
            "In a swamp biome with no structures, stone pickaxe crafting is standard. "
            "Swamp oak trees provide wood. Nothing in the swamp environment changes the tool progression."
        ),
        context_matters=False,
        context_explanation="Swamp doesn't affect tool crafting progression; standard path applies.",
    ))

    # Desert temple for wooden pickaxe (temple doesn't help)
    samples.append(make_sample(
        sample_id=0, biome="desert", structures=["desert_temple"], y_level=72,
        task="obtain_wooden_pickaxe",
        reasoning_path=[
            "search_for_dead_bushes_for_sticks", "search_for_biome_edge_trees",
            "harvest_wood", "craft_planks_and_sticks",
            "craft_crafting_table", "craft_wooden_pickaxe"
        ],
        reasoning_text=(
            "In a desert with a desert temple, the temple doesn't help with obtaining a wooden pickaxe. "
            "Temple chests contain gold and enchanted items, not wood or basic tools. "
            "Still need to find trees at the biome edge. "
            "The desert's wood scarcity is the real challenge here, and the temple doesn't solve it."
        ),
        context_matters=False,
        context_explanation="Desert temple contains treasure but no wood; the wooden pickaxe challenge remains the same.",
    ))

    # Ruined portal for food (portal doesn't help)
    samples.append(make_sample(
        sample_id=0, biome="plains", structures=["ruined_portal"], y_level=72,
        task="obtain_food",
        reasoning_path=BASE_TASK_PROGRESSIONS["obtain_food"],
        reasoning_text=(
            "In a plains biome with a ruined portal, the portal doesn't provide food. "
            "Portal chests contain gold, obsidian, and flint, not food items. "
            "Standard food gathering applies: hunt animals and cook meat."
        ),
        context_matters=False,
        context_explanation="Ruined portal chests don't contain food; the food gathering strategy is unchanged.",
    ))

    # Mineshaft for food (mineshaft doesn't help)
    samples.append(make_sample(
        sample_id=0, biome="forest", structures=["mineshaft"], y_level=30,
        task="obtain_food",
        reasoning_path=[
            "return_to_surface", "search_for_animals",
            "kill_animals_for_meat", "cook_meat"
        ],
        reasoning_text=(
            "Near a forest mineshaft at Y=30, the mineshaft doesn't help with food. "
            "Mineshaft chests contain rails, torches, and mining resources, not food. "
            "Return to the surface to hunt animals in the forest biome."
        ),
        context_matters=False,
        context_explanation="Mineshaft chests don't contain food; must return to surface for standard food gathering.",
    ))

    # Jungle temple for shelter (temple doesn't help)
    samples.append(make_sample(
        sample_id=0, biome="jungle", structures=["jungle_temple"], y_level=72,
        task="build_shelter",
        reasoning_path=[
            "harvest_jungle_wood", "build_platform_on_large_tree",
            "craft_planks", "build_elevated_shelter", "add_ladder_access"
        ],
        reasoning_text=(
            "In a jungle with a jungle temple, the temple isn't ideal as shelter. "
            "Jungle temples have arrow traps inside making them dangerous to inhabit. "
            "Better to build a custom tree platform shelter using abundant jungle wood."
        ),
        context_matters=False,
        context_explanation="Jungle temple traps make it unsuitable as shelter; custom building from jungle wood is safer.",
    ))

    # Stronghold for food (stronghold doesn't help with food)
    samples.append(make_sample(
        sample_id=0, biome="plains", structures=["stronghold"], y_level=-10,
        task="obtain_food",
        reasoning_path=[
            "return_to_surface", "search_for_animals",
            "kill_animals_for_meat", "cook_meat"
        ],
        reasoning_text=(
            "Near a stronghold at Y=-10, the stronghold provides no food. "
            "Stronghold chests contain weapons, armor, and ender items. "
            "Return to the surface to find animals for food."
        ),
        context_matters=False,
        context_explanation="Stronghold chests don't contain food; surface hunting is still required.",
    ))

    # Pillager outpost for shelter (too dangerous)
    samples.append(make_sample(
        sample_id=0, biome="plains", structures=["pillager_outpost"], y_level=72,
        task="build_shelter",
        reasoning_path=[
            "harvest_wood_away_from_outpost",
            "build_shelter_at_safe_distance", "place_torches"
        ],
        reasoning_text=(
            "In plains with a pillager outpost, the outpost is NOT suitable as shelter. "
            "Pillagers continuously spawn near the outpost. "
            "Build a standard shelter at a safe distance using plains wood."
        ),
        context_matters=False,
        context_explanation="Pillager outpost is actively hostile and unsuitable as shelter; build away from it.",
    ))

    # Village for diamond pickaxe without blacksmith (limited help)
    samples.append(make_sample(
        sample_id=0, biome="plains", structures=["village"], y_level=72,
        task="obtain_diamond_pickaxe",
        reasoning_path=BASE_TASK_PROGRESSIONS["obtain_diamond_pickaxe"],
        reasoning_text=(
            "In plains with a village (no blacksmith), the village provides limited help for diamond pickaxe. "
            "Regular village chests may have basic items but rarely iron pickaxes. "
            "Standard full tech tree progression is needed: wood to iron, then diamond mining."
        ),
        context_matters=False,
        context_explanation="Village without blacksmith doesn't significantly shortcut the diamond pickaxe progression.",
    ))

    return samples


def generate_edge_case_samples() -> List[dict]:
    """Generate interesting edge cases."""
    samples = []

    # Mushroom island - diamond pickaxe (extreme challenge)
    samples.append(make_sample(
        sample_id=0, biome="mushroom_island", structures=["none"], y_level=72,
        task="obtain_diamond_pickaxe",
        reasoning_path=[
            "find_shipwreck_or_travel_to_mainland_for_wood",
            "standard_tool_progression_once_wood_obtained",
            "dig_down_to_diamond_level",
            "mine_diamonds", "craft_diamond_pickaxe"
        ],
        reasoning_text=(
            "On a mushroom island, the diamond pickaxe requires solving the wood problem first. "
            "No trees exist on the island. Find a nearby shipwreck for wood or swim to the mainland. "
            "Once wood is obtained, standard progression applies but takes longer due to initial delay. "
            "The advantage: no hostile mobs means safe surface travel and peaceful mining."
        ),
        context_matters=True,
        context_explanation="No trees on mushroom island delays the start, but no hostile mobs makes all subsequent steps safer.",
    ))

    # Mushroom island + shipwreck
    samples.append(make_sample(
        sample_id=0, biome="mushroom_island", structures=["shipwreck"], y_level=62,
        task="obtain_iron_pickaxe",
        reasoning_path=[
            "swim_to_shipwreck", "break_planks_for_wood",
            "loot_treasure_chest_for_iron",
            "craft_tools_from_shipwreck_materials"
        ],
        reasoning_text=(
            "On a mushroom island with a shipwreck, the wreck solves the no-trees problem. "
            "Break the shipwreck for wood planks, and loot the treasure chest for iron. "
            "No hostile mobs make the entire process safe. "
            "The shipwreck may provide enough materials for iron tools without any mining."
        ),
        context_matters=True,
        context_explanation="Shipwreck on treeless mushroom island provides both wood and iron; no hostile mobs makes it safe.",
    ))

    # Deep underground with no context
    samples.append(make_sample(
        sample_id=0, biome="plains", structures=["none"], y_level=-58,
        task="obtain_food",
        reasoning_path=[
            "return_to_surface", "search_for_animals",
            "kill_animals_for_meat", "cook_meat"
        ],
        reasoning_text=(
            "Deep underground at Y=-58, there is no food available. "
            "Must return to the surface (~122 blocks up) to find animals. "
            "Consider leaving markers to find the way back if mining nearby."
        ),
        context_matters=True,
        context_explanation="Being deep underground means food requires a long return trip to the surface.",
    ))

    # Surface level asking for something that needs depth
    samples.append(make_sample(
        sample_id=0, biome="plains", structures=["none"], y_level=100,
        task="obtain_diamond_pickaxe",
        reasoning_path=[
            "harvest_wood", "craft_planks_and_sticks",
            "craft_crafting_table", "craft_wooden_pickaxe",
            "mine_stone", "craft_stone_pickaxe",
            "mine_iron_ore", "craft_furnace", "smelt_iron",
            "craft_iron_pickaxe",
            "dig_down_159_blocks_to_y_minus_59",
            "mine_diamonds", "craft_diamond_pickaxe"
        ],
        reasoning_text=(
            "At Y=100 in plains with no structures, the full progression is needed. "
            "Standard tool progression from wood to iron pickaxe. "
            "Then dig down approximately 159 blocks to reach diamond level at Y=-59. "
            "This is one of the longest possible progressions."
        ),
        context_matters=True,
        context_explanation="Starting at Y=100 means 159 blocks of digging to reach diamond level, much more than starting lower.",
    ))

    # Multiple structures interact
    samples.append(make_sample(
        sample_id=0, biome="desert", structures=["village", "blacksmith", "desert_temple"], y_level=72,
        task="obtain_diamond_pickaxe",
        reasoning_path=[
            "go_to_blacksmith_first", "loot_iron_pickaxe",
            "then_go_to_desert_temple", "avoid_tnt_trap",
            "loot_temple_chests_for_diamonds_or_more_iron",
            "if_diamonds_found_craft_diamond_pickaxe",
            "if_not_dig_to_diamond_level_with_iron_pickaxe"
        ],
        reasoning_text=(
            "In a desert with a village, blacksmith, AND desert temple, combine shortcuts. "
            "First, loot the blacksmith for an iron pickaxe (likely find one). "
            "Then explore the desert temple for potential diamonds in the four treasure chests. "
            "If diamonds are found, craft the diamond pickaxe immediately. "
            "If not, the iron pickaxe still skips most of the tech tree for diamond mining."
        ),
        context_matters=True,
        context_explanation="Multiple structures chain together: blacksmith provides tools, temple provides potential diamonds.",
    ))

    # Ice spikes + igloo for iron (creative)
    samples.append(make_sample(
        sample_id=0, biome="ice_spikes", structures=["igloo"], y_level=72,
        task="obtain_iron_pickaxe",
        reasoning_path=[
            "go_to_igloo_for_shelter_and_crafting",
            "use_igloo_furnace", "search_igloo_basement_if_present",
            "mine_stone_from_underground", "craft_stone_pickaxe",
            "mine_iron_in_igloo_basement_area",
            "smelt_iron_using_igloo_furnace", "craft_iron_pickaxe"
        ],
        reasoning_text=(
            "In ice spikes with an igloo, use the igloo's built-in furnace and crafting table. "
            "The igloo provides immediate access to crafting infrastructure without needing wood. "
            "Mine stone from the underground area, then mine iron. "
            "Use the igloo's furnace to smelt iron. The crafting table lets you craft the pickaxe."
        ),
        context_matters=True,
        context_explanation="Igloo's furnace and crafting table solve ice spikes' no-wood problem for tool infrastructure.",
    ))

    # Swamp - obtain gold (no special advantage)
    samples.append(make_sample(
        sample_id=0, biome="swamp", structures=["none"], y_level=72,
        task="obtain_gold",
        reasoning_path=BASE_TASK_PROGRESSIONS["obtain_gold"],
        reasoning_text=(
            "In a swamp with no structures, gold requires full standard progression. "
            "Swamp offers no special gold mining advantages. "
            "Follow complete tech tree to iron pickaxe, then dig to Y=-16 for gold."
        ),
        context_matters=False,
        context_explanation="Swamp provides no gold-related advantages; standard deep mining progression required.",
    ))

    # Mountains at high elevation for iron
    samples.append(make_sample(
        sample_id=0, biome="mountains", structures=["none"], y_level=120,
        task="obtain_iron_pickaxe",
        reasoning_path=[
            "harvest_wood_at_treeline", "craft_wooden_pickaxe",
            "mine_exposed_stone_from_cliff", "craft_stone_pickaxe",
            "find_cave_entrance_in_mountainside",
            "mine_iron_in_cave", "craft_furnace", "smelt_iron",
            "craft_iron_pickaxe"
        ],
        reasoning_text=(
            "In mountains at Y=120, exposed cliff faces provide stone without digging. "
            "Natural caves in mountainsides reveal iron ore deposits. "
            "While Y=16 is optimal for iron, mountain caves at Y=120 still contain iron ore "
            "(iron generates up to Y=320). Cave access is faster than digging a new shaft."
        ),
        context_matters=True,
        context_explanation="Mountain caves at high elevation still contain iron and provide exposed stone, faster than digging from flat terrain.",
    ))

    return samples


def generate_all_samples() -> List[dict]:
    """Generate the complete dataset."""
    all_samples = []

    all_samples.extend(generate_vanilla_baselines())
    all_samples.extend(generate_structure_shortcut_samples())
    all_samples.extend(generate_biome_specific_samples())
    all_samples.extend(generate_y_level_samples())
    all_samples.extend(generate_combined_samples())
    all_samples.extend(generate_negative_examples())
    all_samples.extend(generate_edge_case_samples())

    # Assign sequential IDs
    for i, sample in enumerate(all_samples, start=1):
        sample["id"] = i

    return all_samples


def print_dataset_statistics(samples: List[dict]):
    """Print coverage statistics for the dataset."""
    print(f"\n{'='*60}")
    print(f"DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples: {len(samples)}")

    # Biome distribution
    biome_counts = Counter(s["biome"] for s in samples)
    print(f"\nBiome coverage ({len(biome_counts)} biomes):")
    for biome, count in sorted(biome_counts.items(), key=lambda x: -x[1]):
        print(f"  {biome:20s}: {count:3d}")

    # Structure distribution
    struct_counts: Counter = Counter()
    for s in samples:
        for st in s["nearby_structures"]:
            struct_counts[st] += 1
    print(f"\nStructure coverage ({len(struct_counts)} structures):")
    for struct, count in sorted(struct_counts.items(), key=lambda x: -x[1]):
        print(f"  {struct:20s}: {count:3d}")

    # Task distribution
    task_counts = Counter(s["task"] for s in samples)
    print(f"\nTask coverage ({len(task_counts)} tasks):")
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        print(f"  {task:25s}: {count:3d}")

    # Y-level distribution
    y_ranges = Counter()
    for s in samples:
        y = s["y_level"]
        if y >= 64:
            y_ranges["surface (64+)"] += 1
        elif y >= 0:
            y_ranges["underground (0-63)"] += 1
        else:
            y_ranges["deep (-64 to -1)"] += 1
    print(f"\nY-level ranges:")
    for r, count in sorted(y_ranges.items()):
        print(f"  {r:25s}: {count:3d}")

    # Context matters distribution
    context_counts = Counter(s["context_matters"] for s in samples)
    print(f"\nContext matters:")
    print(f"  True  (context helps):  {context_counts[True]:3d}")
    print(f"  False (no difference):  {context_counts[False]:3d}")

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Generate Minecraft HRL reasoning path dataset")
    parser.add_argument("--output", default=None,
                        help="Output JSON file path (default: reasoning_paths.json in same directory)")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only print stats for existing dataset")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = args.output or os.path.join(script_dir, "reasoning_paths.json")

    if args.stats_only:
        with open(output_path) as f:
            samples = json.load(f)
        print_dataset_statistics(samples)
        return

    samples = generate_all_samples()
    print_dataset_statistics(samples)

    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"\nDataset saved to: {output_path}")


if __name__ == "__main__":
    main()
