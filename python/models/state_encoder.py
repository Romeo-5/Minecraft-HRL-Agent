"""
state_encoder.py

Encodes a dataset_final.json sample's game state into a fixed-size float vector
suitable for input to the Decision Transformer.

No environment connection needed — works directly on JSON fields.

State vector (41-dim):
    biome:      16-dim one-hot  (config.py BIOMES list)
    structures: 12-dim multi-hot (config.py STRUCTURES list — multiple allowed)
    task:       12-dim one-hot  (config.py TASKS list)
    y_level:     1-dim normalized (y_level / 256.0, clipped to [0, 1])

Total: 16 + 12 + 12 + 1 = 41 dimensions
"""

import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# Import canonical vocab from data/config.py
# ---------------------------------------------------------------------------
_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data"
)
sys.path.insert(0, _DATA_DIR)
from config import BIOMES, STRUCTURES, TASKS, SKILL_VOCAB  # noqa: E402

# Pre-built index lookups
BIOME_IDX    = {b: i for i, b in enumerate(BIOMES)}
STRUCT_IDX   = {s: i for i, s in enumerate(STRUCTURES)}
TASK_IDX     = {t: i for i, t in enumerate(TASKS)}
SKILL_IDX    = {s: i for i, s in enumerate(SKILL_VOCAB)}

STATE_DIM    = len(BIOMES) + len(STRUCTURES) + len(TASKS) + 1   # 41
ACTION_DIM   = len(SKILL_VOCAB)                                   # 47
PAD_ACTION   = ACTION_DIM                                         # index used for padding


class StateEncoder:
    """Stateless encoder: converts dataset sample dicts → numpy state vectors."""

    state_dim  = STATE_DIM
    action_dim = ACTION_DIM

    @staticmethod
    def encode(sample: dict) -> np.ndarray:
        return encode_state(sample)

    @staticmethod
    def encode_action(skill: str) -> int:
        return SKILL_IDX.get(skill, PAD_ACTION)

    @staticmethod
    def decode_action(idx: int) -> str:
        if 0 <= idx < len(SKILL_VOCAB):
            return SKILL_VOCAB[idx]
        return "<pad>"


def encode_state(sample: dict) -> np.ndarray:
    """
    Encode a dataset sample's game state into a 41-dim float32 vector.

    Args:
        sample: A dataset_final.json sample dict. Required keys:
                biome (str), nearby_structures (list[str]),
                task (str), y_level (int, optional — defaults to 64)

    Returns:
        np.ndarray of shape (41,), dtype float32
    """
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    offset = 0

    # Biome one-hot (16-dim)
    biome_idx = BIOME_IDX.get(sample.get("biome", ""), -1)
    if biome_idx >= 0:
        vec[offset + biome_idx] = 1.0
    offset += len(BIOMES)

    # Structure multi-hot (12-dim)
    for struct in sample.get("nearby_structures", []):
        sidx = STRUCT_IDX.get(struct, -1)
        if sidx >= 0:
            vec[offset + sidx] = 1.0
    offset += len(STRUCTURES)

    # Task one-hot (12-dim)
    task_idx = TASK_IDX.get(sample.get("task", ""), -1)
    if task_idx >= 0:
        vec[offset + task_idx] = 1.0
    offset += len(TASKS)

    # Y-level normalized (1-dim)
    y = sample.get("y_level", 64)
    vec[offset] = float(np.clip(y / 256.0, 0.0, 1.0))

    return vec
