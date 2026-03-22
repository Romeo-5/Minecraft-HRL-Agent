"""
normalize_dataset.py

Merges all dataset sources into a single, schema-consistent file:
  data/processed/dataset_final.json

What it does:
  1. Loads reasoning_paths.json (original 105) + minecraft_hrl_dataset_full.json (605 merged)
  2. Deduplicates on id
  3. Maps non-canonical skill names → canonical names via SKILL_MAP
  4. Backfills missing fields (inventory, health, time_of_day, source) for original samples
  5. Removes any nether samples
  6. Validates all skills against SKILL_VOCAB from config.py
  7. Writes data/processed/dataset_final.json

Usage:
  python data/normalize_dataset.py [--dataset PATH] [--output PATH] [--strict]
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Canonical vocab
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from config import SKILL_VOCAB_SET, BIOMES

NETHER_BIOMES = {
    "nether", "nether_wastes", "crimson_forest", "warped_forest",
    "soul_sand_valley", "basalt_deltas",
}
NETHER_STRUCTURES = {"nether_fortress", "bastion_remnant"}

# ---------------------------------------------------------------------------
# Skill name normalization map
# Covers divergences between generate_dataset.py vocab and notebook SYSTEM prompt vocab.
# ---------------------------------------------------------------------------
SKILL_MAP = {
    # ── Conditional pseudo-steps → real skills ──────────────────────────────
    "if_no_iron_pickaxe_then_standard_progression":              "craft_iron_pickaxe",
    "if_no_iron_pickaxe_then_standard_craft":                    "craft_iron_pickaxe",
    "if_no_iron_pickaxe_then_use_village_wood_for_standard_progression": "craft_iron_pickaxe",
    "if_not_found_then_craft":                                   "craft_iron_pickaxe",
    "if_diamonds_found_craft_diamond_pickaxe":                   "craft_diamond_pickaxe",
    "if_not_dig_to_diamond_level_with_iron_pickaxe":             "dig_to_diamond_level",
    "if_no_armor_then_mine_iron_nearby_and_craft":               "mine_iron_ore",
    "if_no_armor_then_standard_progression":                     "craft_iron_armor_set",
    "if_no_diamonds_then_standard_mining":                       "mine_diamonds",
    "if_no_iron_then_standard_progression":                      "mine_iron_ore",
    "if_no_pickaxe_then_standard_progression":                   "craft_wooden_pickaxe",
    "if_no_tools_then_standard_progression":                     "craft_wooden_pickaxe",
    "if_no_trees_then_dig_for_mineshaft_wood":                   "navigate_to_mineshaft",
    "skip_to_diamond_mining_if_iron_pickaxe_found":              "mine_diamonds",
    "fallback_to_crafting":                                      "craft_iron_pickaxe",
    "standard_tool_progression_once_wood_obtained":              "craft_wooden_pickaxe",
    # ── Shelter sub-steps ───────────────────────────────────────────────────
    "place_torches_inside":             "place_torches",
    "place_torches_for_warmth_and_light": "place_torches",
    "place_torches_to_prevent_mob_spawns": "place_torches",
    "craft_planks":                     "craft_planks_and_sticks",
    "add_ladder_access":                "craft_and_place_door",
    "seal_entrance":                    "craft_and_place_door",
    "seal_entrance_with_sandstone":     "craft_and_place_door",
    "seal_entrance_with_stone":         "craft_and_place_door",
    "or_build_underwater_shelter_with_doors": "craft_and_place_door",
    "place_bed_if_not_present":         "craft_and_place_door",
    "fortify_door":                     "craft_and_place_door",
    "build_shelter":                    "build_walls_and_roof",
    "build_basic_shelter_on_island":    "build_walls_and_roof",
    "build_elevated_shelter":           "build_walls_and_roof",
    "build_platform_on_large_tree":     "build_walls_and_roof",
    "build_shelter_at_safe_distance":   "build_walls_and_roof",
    "build_shelter_under_canopy":       "build_walls_and_roof",
    "no_urgent_shelter_needed":         "build_walls_and_roof",
    "harvest_giant_mushroom_blocks":    "build_walls_and_roof",
    "build_mushroom_shelter_when_convenient": "build_walls_and_roof",
    "use_existing_village_house":       "build_walls_and_roof",
    "clear_cave_interior":              "mine_stone",
    "clear_small_area":                 "mine_stone",
    "create_room_underground":          "mine_stone",
    "create_underground_room":          "mine_stone",
    "dig_into_sandstone_hillside":      "mine_stone",
    "dig_underground_immediately":      "mine_stone",
    "gather_sand_and_gravel":           "mine_stone",
    # ── Cooking / smelting ──────────────────────────────────────────────────
    "craft_furnace_or_campfire":        "craft_furnace",
    "craft_furnace_if_needed":          "craft_furnace",
    "craft_furnace_and_smelt_if_needed": "smelt_iron",
    "smelt_and_craft":                  "smelt_iron",
    "smelt_iron_using_igloo_furnace":   "smelt_iron",
    "smelt_food":                       "cook_meat",
    "cook_fish":                        "cook_meat",
    "cook_rabbit_meat":                 "cook_meat",
    "craft_mushroom_stew":              "cook_meat",
    "eat_berries_directly":             "eat_food",
    "store_excess_food":                "eat_food",
    # ── Wood harvesting ─────────────────────────────────────────────────────
    "harvest_dark_oak_wood":            "harvest_wood",
    "harvest_jungle_wood":              "harvest_wood",
    "harvest_spruce_wood":              "harvest_wood",
    "harvest_wood_at_lower_elevation":  "harvest_wood",
    "harvest_wood_at_treeline":         "harvest_wood",
    "harvest_wood_away_from_outpost":   "harvest_wood",
    "harvest_wood_from_biome_edge":     "harvest_wood",
    "break_planks_for_wood":            "harvest_wood",
    "break_shipwreck_for_wood":         "harvest_wood",
    "break_shipwreck_wood_for_planks":  "harvest_wood",
    "search_for_biome_edge_trees":      "harvest_wood",
    "or_travel_to_mainland_for_trees":  "harvest_wood",
    "travel_toward_biome_edge_for_spruce": "harvest_wood",
    "craft_basic_tools":                "craft_wooden_pickaxe",
    "craft_tools_from_shipwreck_materials": "craft_wooden_pickaxe",
    "craft_tools_using_shipwreck_materials": "craft_wooden_pickaxe",
    # ── Mining variations ───────────────────────────────────────────────────
    "mine_ore":                         "mine_iron_ore",
    "harvest_stone":                    "mine_stone",
    "mine_deepslate":                   "mine_stone",
    "mine_exposed_stone":               "mine_stone",
    "mine_exposed_stone_from_cliff":    "mine_stone",
    "mine_exposed_stone_from_cliff_face": "mine_stone",
    "mine_stone_from_underground":      "mine_stone",
    "mine_iron_from_exposed_veins":     "mine_iron_ore",
    "mine_iron_in_cave":                "mine_iron_ore",
    "mine_iron_in_igloo_basement_area": "mine_iron_ore",
    "collect_iron_from_exposed_veins":  "mine_iron_ore",
    "mine_exposed_ores_along_way":      "mine_iron_ore",
    "mine_iron_ore_at_current_level":   "mine_iron_ore",
    "mine_exposed_gold_from_tunnel_walls": "mine_gold_ore",
    "mine_exposed_gold_ore":            "mine_gold_ore",
    "mine_exposed_gold_ore_at_surface": "mine_gold_ore",
    "mine_surface_gold_ore":            "mine_gold_ore",
    "mine_gold_at_current_level":       "mine_gold_ore",
    "mine_gold_ore_at_current_level":   "mine_gold_ore",
    "use_iron_pickaxe_to_mine_gold":    "mine_gold_ore",
    "mine_gilded_blackstone_blocks":    "mine_gold_ore",
    "mine_gilded_blackstone_for_gold_nuggets": "mine_gold_ore",
    "combine_gold_nuggets_to_ingots":   "smelt_gold",
    "mine_diamonds_at_current_level":   "mine_diamonds",
    "search_diamond_vein":              "mine_diamonds",
    "obtain_diamond_tools_or_swift_sneak_enchant": "mine_diamonds",
    "dig_to_diamond_level":             "dig_to_diamond_level",  # already canonical
    "dig_down_to_diamond_level":        "dig_to_diamond_level",
    "dig_down_to_y_minus_59":           "dig_to_diamond_level",
    "dig_down_from_y16_to_y_minus_59":  "dig_to_diamond_level",
    "dig_down_159_blocks_to_y_minus_59": "dig_to_diamond_level",
    "reach_diamond_level":              "dig_to_diamond_level",
    "create_strip_mine":                "mine_iron_ore",
    # ── Navigation / exploration ────────────────────────────────────────────
    "navigate_to_village":              "go_to_village",
    "search_for_village":               "go_to_village",
    "navigate_underground":             "explore_cave",
    "find_cave_entrance_in_mountainside": "explore_cave",
    "find_natural_cave_opening":        "explore_cave",
    "follow_tunnels_to_deeper_levels":  "explore_cave",
    "find_small_island_or_sandbar":     "navigate_to_structure",
    "locate_stronghold_entrance":       "navigate_to_structure",
    "navigate_stronghold_corridors":    "navigate_to_structure",
    "navigate_arrow_traps":             "navigate_to_structure",
    "navigate_puzzles_and_traps":       "navigate_to_structure",
    "navigate_to_ancient_city_silently": "navigate_to_structure",
    "avoid_triggering_sculk_shriekers": "navigate_to_structure",
    "solve_lever_puzzle":               "navigate_to_structure",
    "approach_outpost_cautiously":      "navigate_to_structure",
    "approach_pillager_outpost_carefully": "navigate_to_structure",
    "assess_pillager_numbers":          "navigate_to_structure",
    "find_shipwreck_or_travel_to_mainland_for_wood": "swim_to_shipwreck",
    "search_for_nearby_shipwreck":      "swim_to_shipwreck",
    # ── Structure interactions ──────────────────────────────────────────────
    "check_village_blacksmith":         "find_blacksmith",
    "go_to_blacksmith_first":           "find_blacksmith",
    "loot_village_chests":              "loot_blacksmith_chest",
    "check_village_chests":             "loot_blacksmith_chest",
    "take_golden_apple":                "loot_blacksmith_chest",
    "loot_blacksmith_chest_for_armor":  "loot_blacksmith_chest",
    "loot_blacksmith_chest_for_iron_tools": "loot_blacksmith_chest",
    "loot_iron_pickaxe":                "loot_blacksmith_chest",
    "loot_iron_pickaxe_from_chest":     "loot_blacksmith_chest",
    "loot_dungeon_chest":               "search_mineshaft_chests",
    "search_mineshaft_chests_for_iron_ingots": "search_mineshaft_chests",
    "loot_outpost_chests_for_iron":     "search_mineshaft_chests",
    "loot_stronghold_chests":           "search_mineshaft_chests",
    "loot_stronghold_chests_for_armor": "search_mineshaft_chests",
    "check_storeroom_for_iron_armor":   "search_mineshaft_chests",
    "check_for_basement_with_resources": "search_mineshaft_chests",
    "loot_chests_for_iron_or_diamond":  "search_mineshaft_chests",
    "loot_ancient_city_chests":         "search_mineshaft_chests",
    "loot_temple_chests":               "loot_portal_chest",
    "loot_temple_chests_for_armor":     "loot_portal_chest",
    "loot_temple_chests_for_armor_or_materials": "loot_portal_chest",
    "loot_temple_chests_for_diamonds":  "loot_portal_chest",
    "loot_temple_chests_for_diamonds_or_iron": "loot_portal_chest",
    "loot_temple_chests_for_diamonds_or_more_iron": "loot_portal_chest",
    "loot_temple_chests_for_iron":      "loot_portal_chest",
    "loot_four_chests":                 "loot_portal_chest",
    "dig_to_treasure_room_avoiding_tnt": "avoid_tnt_trap",
    "loot_portal_chest_for_golden_armor": "loot_portal_chest",
    "loot_supply_chest_for_food":       "loot_supply_chest",
    "loot_treasure_chest_for_gold":     "loot_supply_chest",
    "loot_treasure_chest_for_iron":     "loot_supply_chest",
    "also_check_treasure_chest":        "loot_supply_chest",
    "check_igloo_chest":                "go_to_igloo",
    "descend_to_basement":              "go_to_igloo",
    "find_basement_trapdoor":           "go_to_igloo",
    "search_igloo_basement_if_present": "go_to_igloo",
    "use_igloo_as_shelter":             "go_to_igloo",
    "go_to_igloo_for_shelter_and_crafting": "go_to_igloo",
    "use_igloo_furnace":                "craft_furnace",
    "then_go_to_desert_temple":         "go_to_desert_temple",
    # ── Combat ─────────────────────────────────────────────────────────────
    "clear_pillagers":                  "combat_mob",
    "clear_pillagers_or_sneak_around":  "combat_mob",
    "hunt_rabbits":                     "kill_animals_for_meat",
    "hunt_animals_in_clearings":        "kill_animals_for_meat",
    # ── Food / biome-specific ───────────────────────────────────────────────
    "fish_for_food":                    "search_for_animals",
    "fish_in_swamp_water":              "search_for_animals",
    "craft_fishing_rod_if_wood_available": "craft_wooden_pickaxe",
    "scan_for_rabbits":                 "search_for_animals",
    "harvest_mushrooms_from_surface":   "search_for_animals",
    "harvest_village_crops_if_found":   "harvest_village_crops",
    "harvest_village_farm_crops":       "harvest_village_crops",
    "take_hay_bales_and_pumpkins":      "harvest_village_crops",
    "harvest_cocoa_beans_from_trees":   "harvest_village_crops",
    "craft_melon_slices":               "harvest_melons_from_ground",
    "obtain_unlimited_mushroom_stew":   "milk_mooshroom_with_bowl",
    "craft_bowl_from_planks":           "craft_planks_and_sticks",
    # ── Iron/armor ──────────────────────────────────────────────────────────
    "craft_iron_armor":                 "craft_iron_armor_set",
    "equip_any_armor_found":            "craft_iron_armor_set",
    "loot_blacksmith_chest_for_armor":  "loot_blacksmith_chest",
    "trade_with_armorer_villager_if_present": "go_to_village",
    "trade_with_butcher":               "go_to_village",
    "trade_with_farmer_villager":       "go_to_village",
    "use_looted_materials_to_skip_progression": "loot_blacksmith_chest",
    # ── Task names used as steps (wrong — map to first step) ────────────────
    "obtain_food":                      "search_for_animals",
    "or_eat_dried_kelp":                "eat_food",
    "search_for_dead_bushes_for_sticks": "harvest_wood",
    # ── Misc ────────────────────────────────────────────────────────────────
    "smelt_and_craft":                  "smelt_iron",
    "take_hay_bales_and_pumpkins":      "harvest_village_crops",
}

# Schema defaults for original samples missing synthetic fields
SCHEMA_DEFAULTS = {
    "inventory":    {},
    "health":       20,
    "time_of_day":  "day",
    "source":       "original",
}


def normalize_skills(path: list[str], strict: bool = False) -> tuple[list[str], list[str]]:
    """Map each skill in path to canonical name. Returns (normalized, unknowns)."""
    normalized = []
    unknowns = []
    for skill in path:
        mapped = SKILL_MAP.get(skill, skill)
        if mapped not in SKILL_VOCAB_SET:
            unknowns.append(skill)
            if not strict:
                normalized.append(mapped)  # keep anyway, will be reported
        else:
            normalized.append(mapped)
    return normalized, unknowns


def is_nether(sample: dict) -> bool:
    biome = sample.get("biome", "")
    structures = sample.get("nearby_structures", [])
    if biome in NETHER_BIOMES:
        return True
    if any(s in NETHER_STRUCTURES for s in structures):
        return True
    return False


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None,
                        help="Path to merged dataset JSON (default: auto-detect)")
    parser.add_argument("--output", default="data/processed/dataset_final.json")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with error if any unknown skills remain after mapping")
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    base_dir = os.path.dirname(__file__)

    # Try to load the full merged dataset first, fall back to original
    candidates = [
        args.dataset,
        os.path.join(os.path.expanduser("~"), "Downloads", "minecraft_hrl_dataset_full.json"),
        os.path.join(base_dir, "reasoning_paths.json"),
    ]
    dataset = None
    for path in candidates:
        if path and os.path.exists(path):
            print(f"Loading dataset from: {path}")
            dataset = load_dataset(path)
            break

    if dataset is None:
        print("ERROR: Could not find a dataset file. Pass --dataset PATH.")
        sys.exit(1)

    print(f"  Loaded {len(dataset)} samples")

    # ── Deduplicate on id ─────────────────────────────────────────────────────
    seen_ids = {}
    deduped = []
    for s in dataset:
        sid = s.get("id")
        if sid not in seen_ids:
            seen_ids[sid] = True
            deduped.append(s)
    removed_dupes = len(dataset) - len(deduped)
    if removed_dupes:
        print(f"  Removed {removed_dupes} duplicate ids")
    dataset = deduped

    # ── Remove nether ─────────────────────────────────────────────────────────
    pre = len(dataset)
    dataset = [s for s in dataset if not is_nether(s)]
    removed_nether = pre - len(dataset)
    print(f"  Removed {removed_nether} nether samples")

    # ── Normalize + backfill ──────────────────────────────────────────────────
    all_unknowns = []
    for s in dataset:
        # Backfill missing schema fields
        for field, default in SCHEMA_DEFAULTS.items():
            if field not in s:
                s[field] = default

        # Normalize skill names
        normalized, unknowns = normalize_skills(s["reasoning_path"], strict=args.strict)
        s["reasoning_path"] = normalized
        if unknowns:
            all_unknowns.extend([(s["id"], u) for u in unknowns])

    # ── Report unknowns ───────────────────────────────────────────────────────
    if all_unknowns:
        print(f"\n  WARNING: {len(all_unknowns)} unknown skill(s) after normalization:")
        by_skill = {}
        for sid, skill in all_unknowns:
            by_skill.setdefault(skill, []).append(sid)
        for skill, ids in sorted(by_skill.items()):
            print(f"    '{skill}' in samples: {ids[:5]}{'...' if len(ids)>5 else ''}")
        if args.strict:
            print("Exiting due to --strict mode.")
            sys.exit(1)
    else:
        print("  All skills normalized successfully — 0 unknowns")

    # ── Re-assign sequential ids ──────────────────────────────────────────────
    for i, s in enumerate(dataset, start=1):
        s["id"] = i

    # ── Write output ──────────────────────────────────────────────────────────
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.path.dirname(base_dir), output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    biome_counts = {}
    task_counts = {}
    ctx_true = sum(1 for s in dataset if s.get("context_matters"))
    for s in dataset:
        biome_counts[s["biome"]] = biome_counts.get(s["biome"], 0) + 1
        task_counts[s["task"]] = task_counts.get(s["task"], 0) + 1
    avg_path = sum(len(s["reasoning_path"]) for s in dataset) / len(dataset)
    sources = {}
    for s in dataset:
        src = s.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    print(f"\n{'='*50}")
    print(f"Output: {output_path}")
    print(f"Total samples:    {len(dataset)}")
    print(f"Biomes covered:   {len(biome_counts)}")
    print(f"Tasks covered:    {len(task_counts)}")
    print(f"Avg path length:  {avg_path:.1f}")
    print(f"Context matters:  {ctx_true} yes / {len(dataset)-ctx_true} no")
    print(f"Sources:          {sources}")
    print(f"Unknown skills:   {len(all_unknowns)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
