# Minecraft Hierarchical Reinforcement Learning Agent

A research-grade implementation of Hierarchical Reinforcement Learning (HRL) for Minecraft. This project implements the **Options Framework** to enable sample-efficient learning of complex tasks.

## 🎯 Project Overview

Instead of learning raw motor controls (move, jump, attack), this agent operates at a higher level of abstraction:
- **Action Space**: A library of reusable "skills" (harvest_wood, craft_pickaxe, explore)
- **State Space**: High-level metadata (inventory, nearby blocks, health) rather than pixels
- **Learning Objective**: Navigate the Minecraft "tech tree" efficiently

### Research Inspiration

This implementation draws from three influential papers:

1. **Voyager** (Wang et al., 2023): Code-as-action paradigm, skill library persistence
2. **DEPS** (Wang et al., 2023): Describe-Explain-Plan-Select loop for error correction  
3. **Plan4MC** (Yuan et al., 2023): Graph-based skill decomposition and planning

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python Backend                           │
│  ┌──────────────┐    ┌─────────────────┐    ┌───────────────┐  │
│  │  Gymnasium   │───▶│   RL Planner    │───▶│   Novelty     │  │
│  │  Env Wrapper │    │  (PPO/DQN/Heur) │    │   Tracker     │  │
│  └──────────────┘    └─────────────────┘    └───────────────┘  │
│         │                    │                                  │
│         │              Skill ID                                 │
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
│  │    Server    │    │  (13+ skills)   │    │     API       │  │
│  └──────────────┘    └─────────────────┘    └───────────────┘  │
│                                                      │          │
└──────────────────────────────────────────────────────│──────────┘
                                                       ▼
                                              ┌──────────────┐
                                              │  Minecraft   │
                                              │    Server    │
                                              └──────────────┘
```

## 📁 Project Structure

```
minecraft-hrl-agent/
├── mineflayer/                 # JavaScript bot
│   ├── package.json
│   └── src/
│       ├── index.js            # Bot entry point
│       ├── skillManager.js     # Skill definitions & execution
│       └── bridge.js           # WebSocket server
│
├── python/                     # Python RL backend
│   ├── requirements.txt
│   ├── main.py                 # Training script
│   ├── env/
│   │   └── minecraft_env.py    # Gymnasium environment
│   └── agent/
│       └── planner.py          # HRL agent & novelty tracking
│
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- **Minecraft Server** (1.20.1 recommended) - Can use Paper/Spigot/Vanilla
- **Node.js** 18+ 
- **Python** 3.10+
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
# In online-mode=false for local testing
java -Xmx2G -jar paper-1.20.1.jar
```

**Step 2: Start Mineflayer Bot**
```bash
cd mineflayer
npm start
# Bot will connect and wait for Python agent
```

**Step 3: Start Training**
```bash
cd python

# Hybrid mode (RL + novelty exploration)
python main.py --mode hybrid --timesteps 100000

# Pure RL
python main.py --mode pure_rl --policy PPO --timesteps 50000

# Heuristic demo (no learning)
python main.py --mode heuristic
```

## 🎮 Skill Library

The current implementation includes 13 primitive skills:

| ID | Skill | Description | Preconditions |
|----|-------|-------------|---------------|
| 0 | `idle` | No operation | None |
| 1 | `harvest_wood` | Find and chop trees | None |
| 2 | `mine_stone` | Mine cobblestone | Wooden pickaxe |
| 3 | `craft_planks` | Craft wooden planks | Has logs |
| 4 | `craft_sticks` | Craft sticks | Has planks |
| 5 | `craft_crafting_table` | Craft crafting table | 4+ planks |
| 6 | `craft_wooden_pickaxe` | Craft wooden pickaxe | Sticks + planks |
| 7 | `craft_stone_pickaxe` | Craft stone pickaxe | Cobblestone + sticks |
| 8 | `eat_food` | Consume food | Has food + hungry |
| 9 | `explore` | Move to random location | None |
| 10 | `place_crafting_table` | Place crafting table | Has crafting table |
| 11 | `mine_iron` | Mine iron ore | Stone pickaxe |
| 12 | `smelt_iron` | Smelt raw iron | Raw iron + furnace |

## 🧠 Research Focus: Action Novelty Heuristics

The core research contribution is the **Action Novelty Tracker**, which provides intrinsic motivation for exploration:

### UCB-Style Skill Exploration
```python
bonus = c * sqrt(log(N) / n_k)
```
Where `N` = total steps, `n_k` = visits to skill k, `c` = exploration constant.

### Tech Tree Awareness
The **Skill Graph Planner** encodes Minecraft's tech tree as a directed graph, enabling:
- Curriculum learning (order skills by prerequisites)
- Unlock potential scoring (prefer skills that open more options)

### Hybrid Decision Making
In `hybrid` mode, the agent combines:
- **Learned Policy** (PPO/DQN output probabilities)
- **Novelty Bonus** (encourages unexplored skills)
- **Success Rate** (prefers reliable skills)

## 📊 Observation Space

The environment provides high-level state features:

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

## 🔧 Configuration

### Environment Variables

```bash
# Mineflayer bot
export MC_HOST=localhost
export MC_PORT=25565
export MC_USERNAME=HRL_Agent
export BRIDGE_PORT=8765
```

### Training Arguments

```bash
python main.py \
  --mode hybrid \              # Agent mode
  --policy PPO \               # RL algorithm
  --timesteps 100000 \         # Training steps
  --novelty-weight 0.1 \       # Exploration bonus weight
  --save-freq 10000 \          # Checkpoint frequency
  --device cuda                # GPU training
```

## 📈 Extending the Project

### Adding New Skills

1. Add skill definition in `mineflayer/src/skillManager.js`:
```javascript
this.register({
    id: 13,
    name: 'my_new_skill',
    description: 'Description here',
    preconditions: () => this._hasItem('required_item'),
    execute: async () => {
        // Skill implementation
        return { success: true, message: 'Done!' };
    }
});
```

2. Update skill graph in `python/agent/planner.py`:
```python
self.dependencies[13] = [(prereq_skill, 'required_item')]
```

### Custom Reward Shaping

Modify the reward calculation in `skillManager.js`:
```javascript
_calculateInventoryReward(before, after) {
    const techTreeValues = {
        'diamond_pickaxe': 5.0,  // High reward for advanced items
        // ...
    };
}
```

## 🐛 Debugging

### In-Game Commands

When the bot is connected, you can send chat commands:
- `!skills` - List available skills
- `!exec <id>` - Execute a skill manually
- `!state` - Print current state
- `!inventory` - Show inventory

### Logging

The WebSocket bridge logs all communication:
```
[Bridge] Python client connected
[Bridge] Received: {"type": "step", "action": 1}
[SkillManager] Executing skill: harvest_wood
```

## 📚 References

1. Sutton, R. S., Precup, D., & Singh, S. (1999). Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning.
2. Wang, G., et al. (2023). Voyager: An Open-Ended Embodied Agent with Large Language Models.
3. Wang, Z., et al. (2023). Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents.
4. Yuan, H., et al. (2023). Plan4MC: Skill Reinforcement Learning and Planning for Open-World Minecraft Tasks.

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- Mineflayer community for the excellent bot framework
- Stable-Baselines3 team for the RL implementations
- USC CSCI 566 course staff
