"""
context_reward.py

Context-sensitive bonus reward shaper for the Minecraft HRL environment.

Addresses the key ablation finding: env-aware conditioning failed to outperform
env-blind conditioning because the reward function did not differentially reward
context-sensitive behavior. This module provides an additive bonus layer on top
of the base tech tree rewards that explicitly incentivizes:
  1. Structure shortcut detection — large one-shot bonuses for using nearby
     structures to bypass normal crafting progressions
  2. Biome-adaptive strategies — bonuses for playing to the biome's strengths
     (e.g. mining surface gold in mesa, looting shipwrecks in ocean)
  3. Context-ignorant penalties — small per-step penalties for playing the
     default strategy when a better biome/structure option is available

Usage:
    shaper = ContextRewardShaper()

    # In your training loop, after each step:
    bonus = shaper.get_bonus(obs, skill_name)
    reward = base_reward + bonus

    # At the start of each episode:
    shaper.reset()

obs dict expected keys:
    biome           (str)  — current biome name, e.g. "plains"
    nearby_structures (list[str]) — structure names in render distance
"""


class ContextRewardShaper:
    """
    Computes context-sensitive bonus rewards based on biome and nearby structures.

    One-shot bonuses (awarded at most once per episode, tracked in _awarded):
        These fire the first time the agent takes a context-appropriate action
        that demonstrates it is leveraging environmental information.

    Per-step penalties (fire every time the condition is met):
        These penalize context-ignorant behavior that ignores available shortcuts.
    """

    # ── One-shot structure shortcut bonuses ──────────────────────────────────
    # Format: (required_structure, skill_name) → bonus
    # These reward the agent for using a nearby structure to bypass the normal
    # crafting progression (e.g. looting blacksmith instead of smelting iron).
    STRUCTURE_SHORTCUT_BONUSES = {
        ("blacksmith",     "loot_blacksmith_chest"):    2.0,
        ("mineshaft",      "search_mineshaft_chests"):  1.5,
        ("desert_temple",  "loot_supply_chest"):        1.5,
        ("shipwreck",      "swim_to_shipwreck"):        1.5,
        ("jungle_temple",  "go_to_jungle_temple"):      1.0,
        ("dungeon",        "search_mineshaft_chests"):  1.0,
        ("ruined_portal",  "loot_portal_chest"):        1.0,
        ("igloo",          "go_to_igloo"):              0.8,
    }

    # ── One-shot biome-adaptive bonuses ─────────────────────────────────────
    # Format: (biome, skill_name) → bonus
    # These reward biome-specific optimal play (e.g. mining surface gold in mesa).
    BIOME_ADAPTIVE_BONUSES = {
        ("mesa",             "mine_gold_ore"):                 0.5,
        ("mushroom_island",  "milk_mooshroom_with_bowl"):      0.5,
        ("swamp",            "harvest_sweet_berries_from_bushes"): 0.3,
        ("taiga",            "harvest_sweet_berries_from_bushes"): 0.3,
        ("jungle",           "harvest_melons_from_ground"):    0.3,
        ("plains",           "harvest_village_crops"):         0.3,
        ("savanna",          "harvest_village_crops"):         0.3,
    }

    # ── Biomes where wood is scarce (no trees → harvesting wood is suboptimal) ─
    WOOD_SCARCE_BIOMES = {"desert", "ocean", "mushroom_island", "mesa"}

    # ── Structure sets (for checking what's nearby) ──────────────────────────
    LOOT_STRUCTURES = {
        "blacksmith", "mineshaft", "desert_temple", "jungle_temple",
        "shipwreck", "dungeon", "ruined_portal", "igloo",
    }

    def __init__(self):
        # Tracks which one-shot bonuses have already been awarded this episode
        self._awarded: set = set()

    def reset(self):
        """Call at the start of each episode to reset one-shot tracking."""
        self._awarded.clear()

    def get_bonus(self, obs: dict, skill_name: str) -> float:
        """
        Compute the context-sensitive bonus reward for executing skill_name
        in the given observation state.

        Args:
            obs:        Observation dict. Must contain at minimum:
                        - "biome" (str)
                        - "nearby_structures" (list[str])
            skill_name: The canonical skill string executed this step.

        Returns:
            float: Additive bonus (can be negative for penalties). Zero if no
                   context applies.
        """
        biome = obs.get("biome", "")
        nearby = set(obs.get("nearby_structures", []))
        bonus = 0.0

        # ── 1. Structure shortcut bonuses (one-shot) ─────────────────────────
        for (req_structure, req_skill), reward in self.STRUCTURE_SHORTCUT_BONUSES.items():
            key = f"struct_{req_structure}_{req_skill}"
            if (
                req_skill == skill_name
                and req_structure in nearby
                and key not in self._awarded
            ):
                bonus += reward
                self._awarded.add(key)
                break  # only one structure bonus per step

        # ── 2. Biome-adaptive bonuses (one-shot) ─────────────────────────────
        biome_key = f"biome_{biome}_{skill_name}"
        biome_reward = self.BIOME_ADAPTIVE_BONUSES.get((biome, skill_name), 0.0)
        if biome_reward > 0.0 and biome_key not in self._awarded:
            bonus += biome_reward
            self._awarded.add(biome_key)

        # ── 3. Context-ignorant penalties (per-step) ─────────────────────────
        # Penalise harvesting wood in biomes where it's scarce.
        # Base penalty -0.2 for any WOOD_SCARCE_BIOME; extra -0.1 if a loot
        # structure is also available (agent should use that shortcut instead).
        if skill_name == "harvest_wood" and biome in self.WOOD_SCARCE_BIOMES:
            bonus -= 0.2
            if bool(nearby & self.LOOT_STRUCTURES):
                bonus -= 0.1  # extra penalty: structure shortcut being ignored

        return bonus

    # ── Introspection helpers (useful for logging / unit tests) ──────────────

    def awarded_bonuses(self) -> set:
        """Return the set of one-shot bonus keys awarded so far this episode."""
        return set(self._awarded)

    def total_possible_bonus(self, biome: str, nearby_structures: list) -> float:
        """
        Return the maximum possible one-shot bonus achievable in this context
        (useful for normalising reward curves in analysis).
        """
        nearby = set(nearby_structures)
        total = 0.0
        for (req_struct, _), reward in self.STRUCTURE_SHORTCUT_BONUSES.items():
            if req_struct in nearby:
                total += reward
                break  # only one structure bonus counts
        for (b, _), reward in self.BIOME_ADAPTIVE_BONUSES.items():
            if b == biome:
                total += reward
        return total
