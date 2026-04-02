# Minecraft Hierarchical Reinforcement Learning Agent

A research-grade implementation of Hierarchical Reinforcement Learning (HRL) for Minecraft, developed for USC CSCI 566 Deep Learning (Spring 2026).

**Team:** Romeo Nickel, Ved Chadderwala, Vishnu Gamini, Gavin Jiang, Adam Lehavi, Jonah Ji, Akash Gandi

---

## Project Overview

Instead of learning raw motor controls, this agent operates at a higher level of abstraction using a discrete **skill library** as its action space. The project explores whether **environment-aware conditioning** (biome, nearby structures, y-level) meaningfully improves policy performance over environment-blind baselines — and whether context-sensitive reward shaping is the key to making that work.

Three parallel model families are trained and evaluated against the same interface:

| Model | Type | Status |
|-------|------|--------|
| **DQN** w/ ContextRewardShaper | Online RL (env-aware) | Training |
| **Decision Transformer** | Offline RL on expert dataset | 70.4% step coverage |
| **T5 Planner** | Seq2Seq SFT + RL fine-tuning | In progress |

All three reduce to the same runtime interface: output a skill index `0–46` → send over WebSocket to `bridge.js` → Mineflayer executes it in the world.

---

## Architecture

```
Dataset (605 samples, 47-skill vocab)
      ↓ trains              ↓ trains            ↓ validates
Decision Transformer    T5 Planner (SFT)    RL Environment (DQN)
  ↓ offline SR           ↓ planned path        ↓ online SR
  70.4% coverage    high-level goal → skills   22–30% success
                              ↓ combined (HRL)
                    Mineflayer Bot (47 skills)
                              ↓
                      Minecraft Server

┌─────────────────────────────────────────────────────────────────┐
│                        Python Backend                           │
│  ┌──────────────┐    ┌─────────────────┐    ┌───────────────┐  │
│  │  Gymnasium   │───▶│  DQN / DT / T5  │───▶│ContextReward  │  │
│  │  Env Wrapper │    │     Planner     │    │    Shaper     │  │
│  └──────────────┘    └─────────────────┘    └───────────────┘  │
│         │                    │                                  │
│         │              Skill ID (0–46)                          │
│         ▼                    │                                  │
│  ┌──────────────┐            │                                  │
│  │   WebSocket  │◀───────────┘                                  │
│  │    Client    │                                               │
│  └──────────────┘                                               │
└─────────│───────────────────────────────────────────────────────┘
          │ JSON over WebSocket
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Mineflayer Bot (Node.js)                    │
│  ┌──────────────┐    ┌─────────────────┐    ┌───────────────┐  │
│  │   WebSocket  │───▶│  Skill Manager  │───▶│  Mineflayer   │  │
│  │    Server    │    │  (47-skill vocab│    │     API       │  │
│  │   bridge.js  │    │   13 live)      │    │  + pathfinder │  │
│  └──────────────┘    └─────────────────┘    └───────────────┘  │
│                             │                        │          │
│                      StuckDetector              RCON Client     │
└─────────────────────────────────────────────────────────────────┘
                                                       ▓
                                              ┌──────────────┐
                                              │  Minecraft   │
                                              │ Server 1.20.1│
                                              └──────────────┘
```

---

## Project Structure

```
minecraft-hrl-agent/
├── mineflayer/                     # JavaScript bot
│   ├── package.json
│   └── src/
│       ├── index.js                # Bot entry point & config
│       ├── skillManager.js         # Skill definitions & execution
│       └── bridge.js               # WebSocket server + StuckDetector
│
├── python/                         # Python RL backend
│   ├── requirements.txt
│   ├── main.py                     # Training entry point
│   ├── env/
│   │   └── minecraft_env.py        # Gymnasium environment wrapper
│   ├── agent/
│   │   ├── planner.py              # DQN agent
│   │   └── context_reward_shaper.py # Biome/structure bonus rewards
│   ├── models/
│   │   └── decision_transformer.py # Offline RL model
│   ├── data/
│   │   └── dataset_final.json      # 605-sample benchmark dataset
│   ├── scripts/
│   │   └── run_context_ablation.sh # Launches 2-condition ablation
│   └── checkpoints/                # Saved model weights
│
├── MC_Tech_Tree/                   # Tech tree DAG + visual editor
│   ├── tech_tree.json
│   └── editor.html
│
├── minecraft-server/               # Vanilla 1.20.1 server
│   └── server.properties
│
└── README.md
```

---

## Quick Start

### Prerequisites

- **Minecraft Server** 1.20.1 (Vanilla or Paper)
- **Node.js** 18+
- **Python** 3.10+
- **Java** 17+ (for Minecraft server)
- **CUDA** (optional, for GPU training)

### Installation

```bash
# Clone the repository
git clone https://github.com/Romeo-5/minecraft-hrl-agent.git
cd minecraft-hrl-agent

# Install Mineflayer dependencies
cd mineflayer
npm install

# Install Python dependencies
cd ../python
pip install -r requirements.txt
```

### Running

**Step 1: Start Minecraft Server**
```bash
cd minecraft-server
# Windows
java -Xmx4G -Xms4G -jar paper.jar nogui
```

**Step 2: Start Mineflayer Bot**
```bash
cd mineflayer
npm start
# Recommended: pipe to log file for monitoring
npm start 2>&1 | tee training.log
```

**Step 3: Start DQN Training**
```bash
cd python

# Activate virtual environment
source venv/bin/activate        # macOS/Linux
# or: venv\Scripts\activate     # Windows

# Env-aware DQN with context reward shaping (primary experiment)
python main.py --mode dqn --env-aware --timesteps 200000

# Env-blind baseline (ablation condition 2)
python main.py --mode dqn --no-context-reward --timesteps 200000

# Run both conditions sequentially (full ablation)
bash scripts/run_context_ablation.sh

# Offline Decision Transformer inference
python main.py --mode dt
```

**Step 4: Monitor Training**

TensorBoard logs live reward, success rate, and context bonus curves:
```bash
tensorboard --logdir python/logs
# Open http://localhost:6006
```

Prismarine 3D viewer (real-time bot perspective):
```
http://localhost:3007
```

---

## Skill Library

The canonical skill vocabulary contains **47 skills** across the full tech tree. Currently **28 are live** in the Mineflayer bot (IDs 0–27); the remainder are planned for expansion.

| ID | Skill | Description | Preconditions |
|----|-------|-------------|---------------|
| 0  | `idle` | No-op | None |
| 1  | `harvest_wood` | Find and chop trees | None |
| 2  | `mine_stone` | Mine cobblestone | Wooden pickaxe |
| 3  | `craft_planks` | Craft wooden planks | Has logs |
| 4  | `craft_sticks` | Craft sticks | Has planks |
| 5  | `craft_crafting_table` | Craft crafting table | 4+ planks, none placed nearby |
| 6  | `craft_wooden_pickaxe` | Craft wooden pickaxe | 3 planks + 2 sticks, no pickaxe held |
| 7  | `craft_stone_pickaxe` | Craft stone pickaxe | 3 cobblestone + 2 sticks, no iron/diamond pickaxe |
| 8  | `eat_food` | Consume food | Has food + not full |
| 9  | `explore` | Move to random location | None |
| 10 | `place_crafting_table` | Place crafting table | Has crafting table |
| 11 | `mine_iron` | Mine iron ore | Stone or iron pickaxe |
| 12 | `smelt_iron` | Smelt raw iron (auto-crafts/places furnace) | Has raw iron + furnace or 8 cobblestone |
| 13 | `craft_furnace` | Craft a furnace | 8 cobblestone, no furnace in world |
| 14 | `craft_iron_pickaxe` | Craft iron pickaxe | 3 iron ingots + 2 sticks, no iron/diamond pickaxe |
| 15 | `craft_iron_helmet` | Craft iron helmet | 5 iron ingots |
| 16 | `craft_iron_chestplate` | Craft iron chestplate | 8 iron ingots |
| 17 | `craft_iron_leggings` | Craft iron leggings | 7 iron ingots |
| 18 | `craft_iron_boots` | Craft iron boots | 4 iron ingots |
| 19 | `dig_to_diamond_level` | Dig down to Y=−59 | Iron pickaxe + currently above Y=−50 |
| 20 | `return_to_surface` | Navigate back to surface (Y≥64) | Currently below Y=0 |
| 21 | `mine_diamond` | Mine deepslate diamond ore | Iron pickaxe + at Y≤−50 |
| 22 | `craft_diamond_pickaxe` | Craft diamond pickaxe | 3 diamonds + 2 sticks |
| 23 | `craft_diamond_helmet` | Craft diamond helmet | 5 diamonds |
| 24 | `craft_diamond_chestplate` | Craft diamond chestplate | 8 diamonds |
| 25 | `craft_diamond_leggings` | Craft diamond leggings | 7 diamonds |
| 26 | `craft_diamond_boots` | Craft diamond boots | 4 diamonds |
| 27 | `clear_junk` | Drop low-value blocks to free inventory | ≥27 inventory slots occupied |
| 28–46 | *(planned)* | Combat, shelter, loot, coal, navigation, food | — |

The full 47-skill vocabulary is defined in `python/data/skill_vocab.json` and mirrors the Tech Tree DAG.

---

## Tech Tree

The tech tree is a directed acyclic graph (DAG) with 37 nodes representing Minecraft items and milestones. It serves as the ground truth for dataset construction, reward shaping, and dataset validation.

| Tier | Key Nodes | Reward | Prerequisites |
|------|-----------|--------|---------------|
| 0 | `wood_log` | 0.3 | None |
| 1–2 | `planks`, `sticks`, `crafting_table` | 0.2–1.0 | — |
| 3 | `wooden_pickaxe`, `gate_combat` | 1.0, 2.5 | crafting_table |
| 4–5 | `coal`, `stone`, `furnace`, `stone_pickaxe` | 0.2–1.5 | wooden_pickaxe |
| 6 | `iron_ore`, `gate_mine_iron`, `gate_smelt` | 0.5, 3.0, 2.0 | stone_pickaxe / furnace |
| 7 | `iron_pickaxe`, full iron armor set | 3.0, 6.0 | crafting_table |
| 8 | `diamond`, `gate_mine_diamond` | 2.5, 5.0 | iron_pickaxe |
| 9 | `diamond_pickaxe`, full diamond armor | 6.0, 12.0 | crafting_table |

Gate nodes (`gate_mine_iron`, `gate_mine_diamond`, etc.) give large one-time milestone rewards. One-shot IDs ensure crafting table, furnace, and diamond gear are only rewarded once per episode.

---

## Dataset

**605 environment-aware reasoning path samples** — the ground truth for all model training and evaluation.

- **105** hand-crafted originals + **500** LLM-augmented (Claude Sonnet via few-shot prompting)
- **16 biomes**, **12 structure types**, **8 tasks** covered
- **78%** of samples have context-dependent optimal paths (biome or structure changes best strategy)
- **47-skill vocabulary**, validated against the Tech Tree DAG via transitive prerequisite closure
- **0 ordering violations** after automated fix with Kahn's topological sort

Each sample has 13 fields: `id`, `biome`, `nearby_structures`, `y_level`, `task`, `reasoning_path`, `reasoning_text`, `context_matters`, `context_explanation`, `inventory`, `health`, `time_of_day`, `source`.

---

## Observation Space

### Standard (env-blind)

| Feature | Shape | Description |
|---------|-------|-------------|
| `health` | (1,) | Normalized health [0, 1] |
| `food` | (1,) | Normalized hunger [0, 1] |
| `position` | (3,) | Normalized x, y, z |
| `inventory` | (20,) | Count of tracked items |
| `nearby_blocks` | (12,) | Proximity to block types |
| `available_skills` | (13,) | Binary mask of valid skills |
| `time_of_day` | (1,) | Game time [0, 1] |
| `is_day` | (1,) | Daytime flag |

### Extended (--env-aware)

Adds two additional observation components:

| Feature | Shape | Description |
|---------|-------|-------------|
| `biome_vec` | (16,) | One-hot biome encoding |
| `structure_vec` | (12,) | Multi-hot nearby structure encoding |

---

## Context-Aware Reward Shaping

The `ContextRewardShaper` addresses the key finding from the RL ablation: env-aware conditioning failed to beat env-blind *because the reward function didn't incentivize using biome/structure context*. Three bonus signal types are added on top of the base Mineflayer reward:

| Signal | Condition | Bonus | One-Shot? |
|--------|-----------|-------|-----------|
| Structure Shortcut | Execute loot skill near matching structure (e.g. `loot_blacksmith_chest` near a blacksmith) | +0.8 to +2.0 | Yes |
| Biome Adaptive | Use biome-optimal skill (e.g. `mine_gold_ore` in mesa biome) | +0.5 | No |
| Wood-Scarce Penalty | `harvest_wood` in desert/ocean/mesa when loot structure available | -0.2 to -0.3/step | No |

Controlled via `--env-aware` (enables context reward) and `--no-context-reward` (disables for ablation baseline).

---

## Experimental Results

### Ablation 1 — LLM Zero-Shot Baseline (105 samples, 2 conditions)

| Model | Condition | Step Coverage | Shortcut Detection | Efficiency |
|-------|-----------|---------------|--------------------|------------|
| llama3.2:3b | env-aware | 16.6% | **96%** | 65% |
| llama3.2:3b | env-blind | 12.3% | 2% | 65% |
| mistral:7b | env-aware | 18.8% | **86%** | 34% |
| mistral:7b | env-blind | 16.7% | 12% | 33% |

Both models leverage structural shortcuts dramatically when given context (p < 0.001). Low step coverage reflects a vocabulary alignment problem — models produce semantically correct plans in natural language that fail to match discrete skill tokens.

### Ablation 2 — 2×2×2 RL Ablation (200K steps per condition)

Eight conditions trained varying algorithm (PPO vs. DQN), encoder (MLP vs. Transformer), and conditioning (env-blind vs. env-aware):

- **DQN consistently outperforms PPO**: 22–30% success rate vs. 0–16%
- **Env-aware did not beat env-blind** in this run — attributed to the reward function not incentivizing biome/structure use (motivates ContextRewardShaper)

### Decision Transformer — Offline RL (112K parameters)

Trained purely on `dataset_final.json` with no live Minecraft connection needed.

| Top-1 Accuracy | Top-3 Accuracy | Step Coverage | Shortcut Detection |
|----------------|----------------|---------------|--------------------|
| 59.6% | 83.3% | **70.4%** | 38.6% |

The 70.4% step coverage vastly outperforms zero-shot LLM baselines (16–18%), validating that learning from demonstrations beats zero-shot language reasoning for this planning task.

### Currently Running — DQN Context Reward Ablation

| Condition | Flags | Status |
|-----------|-------|--------|
| Env-aware + Context Reward | `--env-aware` | **Training** |
| Env-blind Baseline | `--no-context-reward` | Queued |

Hypothesis: env-aware + context reward ≥ env-blind by ≥5% success rate. If confirmed, context-aware reward shaping is validated as a meaningful architectural choice for the final paper.

---

## StuckDetector

The Mineflayer bot includes a 4-level automatic recovery system for when the agent gets physically stuck:

| Level | Trigger | Action |
|-------|---------|--------|
| L1 | 15s without movement | Jump + walk in random direction |
| L2 | 30s without movement | Mine surrounding blocks |
| L3 | 60s without movement | Place block underfoot to escape |
| L4 | 90s without movement | RCON `/kill` → force respawn |

All levels include a `_skillRunning` guard to avoid interrupting active skills.

---

## Configuration

### Training Arguments

```bash
python main.py \
  --mode dqn \              # Agent mode: dqn, dt, hybrid
  --env-aware \             # Enable biome/structure observation + context reward
  --no-context-reward \     # Disable context reward (ablation baseline)
  --timesteps 200000 \      # Total training steps
  --save-freq 10000 \       # Checkpoint save frequency
  --device cuda             # GPU training
```

### Server Properties (key settings)

```properties
difficulty=peaceful
allow-flight=true          # Required — pathfinder triggers anti-cheat otherwise
enable-rcon=true
rcon.password=hrltraining
spawn-monsters=false
online-mode=false          # Allows bot accounts without premium login
```

---

## In-Game Debug Commands

When the bot is connected, chat commands are available:

- `!skills` — List available skills and their IDs
- `!exec <id>` — Manually execute a skill
- `!state` — Print current game state
- `!inventory` — Show current inventory

---

## Adding New Skills

1. Add the skill definition in `mineflayer/src/skillManager.js`:
```javascript
this.register({
    id: 13,
    name: 'my_new_skill',
    description: 'Description here',
    preconditions: () => this._hasItem('required_item'),
    execute: async () => {
        // Skill implementation using mineflayer API
        return { success: true, message: 'Done!' };
    }
});
```

2. Add the skill to `SKILL_VOCAB` in `python/data/skill_vocab.json`

3. Update `available_skills` mask logic in `python/env/minecraft_env.py`

4. Add the node and edges to `MC_Tech_Tree/tech_tree.json`

---

## References

1. Sutton, R. S., Precup, D., & Singh, S. (1999). Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning.
2. Kaelbling, L. P., Littman, M. L., & Moore, A. W. (1996). Reinforcement Learning: A Survey.
3. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*.
4. Chen, L., et al. (2021). Decision Transformer: Reinforcement Learning via Sequence Modeling. *NeurIPS*.
5. Wang, G., et al. (2023). Voyager: An Open-Ended Embodied Agent with Large Language Models. *arXiv:2305.16291*.
6. Wang, Z., et al. (2023). Describe, Explain, Plan and Select: Interactive Planning with LLMs. *arXiv:2302.01560*.
7. Yuan, H., et al. (2023). Plan4MC: Skill Reinforcement Learning and Planning for Open-World Minecraft Tasks. *arXiv:2303.16563*.
8. Jiang, H., et al. (2024). Reinforcement Learning Friendly Vision-Language Model for Minecraft. *ECCV*.
9. Lifshitz, S., et al. (2023). Steve-1: A Generative Model for Text-to-Behavior in Minecraft. *NeurIPS*.
10. Li, Z., et al. (2024). Optimus-1: Hybrid Multimodal Memory Empowered Agents Excel in Long-Horizon Tasks. *NeurIPS*.
11. Li, Z., et al. (2025). Optimus-2: Multimodal Minecraft Agent with Goal-Observation-Action Conditioned Policy. *CVPR*.
12. Li, Z., et al. (2025). Optimus-3: Towards Generalist Multimodal Minecraft Agents with Scalable Task Experts. *arXiv:2506.10357*.

---

## License

MIT License — See LICENSE file for details.

## Acknowledgments

- Mineflayer community for the excellent bot framework
- Stable-Baselines3 team for the RL implementations
- Ollama for local LLM inference used in baseline benchmarking
