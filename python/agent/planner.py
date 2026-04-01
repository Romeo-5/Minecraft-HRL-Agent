"""
Planner Module - High-Level Goal Selection with Action Novelty

This module implements the high-level planner that selects which skill to execute.
The core research contribution focuses on "Action Novelty Heuristics" - 
mathematically optimizing which forced state (skill) to pursue based on:

1. Visit counts (UCB-style exploration)
2. Skill graph structure (reachability analysis)
3. Tech tree progress (curriculum learning)

Inspired by:
- Voyager: Skill persistence and discovery
- DEPS: Error feedback and replanning
- Plan4MC: Graph-based skill decomposition
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import gymnasium as gym


class ActionNoveltyTracker:
    """
    Tracks skill execution history for novelty-based exploration.
    
    The key insight is that in HRL, we want to encourage the agent to:
    1. Try skills it hasn't used recently (temporal novelty)
    2. Try skills that lead to unexplored state regions (state novelty)
    3. Try skills that unlock new capabilities (tech tree novelty)
    """
    
    def __init__(self, num_skills: int, decay_rate: float = 0.99):
        self.num_skills = num_skills
        self.decay_rate = decay_rate
        
        # Visit counts per skill
        self.skill_counts = np.zeros(num_skills, dtype=np.float32)
        
        # Success rate per skill (exponential moving average)
        self.skill_success = np.ones(num_skills, dtype=np.float32) * 0.5
        
        # State visitation pseudo-counts (for intrinsic motivation)
        self.state_visits = defaultdict(int)
        
        # Skill transition counts: P(s' | s, skill)
        self.skill_transitions = defaultdict(lambda: defaultdict(int))
        
        # Total steps
        self.total_steps = 0
        
    def update(
        self,
        skill_id: int,
        success: bool,
        prev_state_hash: str,
        next_state_hash: str
    ):
        """Update tracking after skill execution."""
        self.total_steps += 1
        
        # Update skill counts with decay
        self.skill_counts *= self.decay_rate
        self.skill_counts[skill_id] += 1
        
        # Update success rate (EMA)
        alpha = 0.1
        self.skill_success[skill_id] = (
            (1 - alpha) * self.skill_success[skill_id] + 
            alpha * (1.0 if success else 0.0)
        )
        
        # Update state visitation
        self.state_visits[next_state_hash] += 1
        
        # Update transition model
        self.skill_transitions[prev_state_hash][(skill_id, next_state_hash)] += 1
    
    def get_novelty_bonus(self, skill_id: int) -> float:
        """
        Calculate exploration bonus for a skill using UCB-style formula.
        
        bonus = c * sqrt(log(N) / n_k)
        
        where:
        - N = total steps
        - n_k = visits to skill k
        - c = exploration constant
        """
        c = 2.0  # Exploration constant (tunable)
        
        if self.total_steps == 0:
            return 0.0
            
        n_k = self.skill_counts[skill_id] + 1e-6  # Avoid division by zero
        return c * np.sqrt(np.log(self.total_steps + 1) / n_k)
    
    def get_state_novelty(self, state_hash: str) -> float:
        """
        Calculate state novelty bonus (count-based intrinsic motivation).
        
        bonus = 1 / sqrt(n(s) + 1)
        """
        count = self.state_visits[state_hash] + 1
        return 1.0 / np.sqrt(count)
    
    def get_all_novelty_bonuses(self) -> np.ndarray:
        """Get novelty bonus for all skills."""
        return np.array([
            self.get_novelty_bonus(i) 
            for i in range(self.num_skills)
        ])
    
    def save(self, path: str):
        """Save tracker state to file."""
        data = {
            'skill_counts': self.skill_counts.tolist(),
            'skill_success': self.skill_success.tolist(),
            'total_steps': self.total_steps,
            'state_visits': dict(self.state_visits)
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str):
        """Load tracker state from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.skill_counts = np.array(data['skill_counts'])
        self.skill_success = np.array(data['skill_success'])
        self.total_steps = data['total_steps']
        self.state_visits = defaultdict(int, data['state_visits'])


class SkillGraphPlanner:
    """
    Graph-based planner for skill selection (Plan4MC style).
    
    Maintains a skill dependency graph encoding which skills
    enable or require other skills. Uses this for:
    1. Curriculum learning (order skills by prerequisites)
    2. Subgoal planning (decompose goals into skill sequences)
    3. Novelty weighting (prefer skills that unlock more options)
    """
    
    def __init__(self):
        # Skill dependencies: skill_id -> list of (prerequisite_skill, required_item)
        # This encodes the Minecraft tech tree
        self.dependencies = {
            0:  [],                                          # idle
            1:  [],                                          # harvest_wood
            2:  [(6,  'wooden_pickaxe')],                   # mine_stone
            3:  [(1,  '_log')],                             # craft_planks
            4:  [(3,  '_planks')],                          # craft_sticks
            5:  [(3,  '_planks')],                          # craft_crafting_table
            6:  [(4,  'stick'), (5,  'crafting_table')],    # craft_wooden_pickaxe
            7:  [(2,  'cobblestone'), (4, 'stick')],        # craft_stone_pickaxe
            8:  [],                                          # eat_food
            9:  [],                                          # explore
            10: [(5,  'crafting_table')],                   # place_crafting_table
            11: [(7,  'stone_pickaxe')],                    # mine_iron
            12: [(11, 'raw_iron')],                         # smelt_iron
            13: [(2,  'cobblestone')],                      # craft_furnace
            14: [(12, 'iron_ingot'), (4, 'stick')],         # craft_iron_pickaxe
            15: [(12, 'iron_ingot')],                       # craft_iron_helmet
            16: [(12, 'iron_ingot')],                       # craft_iron_chestplate
            17: [(12, 'iron_ingot')],                       # craft_iron_leggings
            18: [(12, 'iron_ingot')],                       # craft_iron_boots
            19: [(14, 'iron_pickaxe')],                     # dig_to_diamond_level
            20: [],                                          # return_to_surface (positional)
            21: [(19, 'iron_pickaxe')],                     # mine_diamond
            22: [(21, 'diamond'), (4, 'stick')],            # craft_diamond_pickaxe
            23: [(21, 'diamond')],                          # craft_diamond_helmet
            24: [(21, 'diamond')],                          # craft_diamond_chestplate
            25: [(21, 'diamond')],                          # craft_diamond_leggings
            26: [(21, 'diamond')],                          # craft_diamond_boots
            27: [],                                          # clear_junk (always available)
        }
        
        # Build reverse graph (what does each skill unlock?)
        self.unlocks = defaultdict(list)
        for skill, deps in self.dependencies.items():
            for prereq, _ in deps:
                self.unlocks[prereq].append(skill)
    
    def get_available_skills(self, inventory: Dict[str, int]) -> List[int]:
        """Get list of skills whose prerequisites are satisfied."""
        available = []
        
        for skill_id, deps in self.dependencies.items():
            satisfied = True
            for prereq_skill, required_item in deps:
                # Check if required item exists in inventory
                has_item = any(
                    required_item in item_name or item_name == required_item
                    for item_name in inventory.keys()
                )
                if not has_item:
                    satisfied = False
                    break
            
            if satisfied:
                available.append(skill_id)
        
        return available
    
    def get_unlock_potential(self, skill_id: int) -> int:
        """Count how many skills this skill can help unlock."""
        # Simple: direct unlocks
        direct = len(self.unlocks[skill_id])
        
        # Recursive: transitive closure (skills unlocked by unlocked skills)
        visited = set()
        queue = list(self.unlocks[skill_id])
        while queue:
            s = queue.pop()
            if s not in visited:
                visited.add(s)
                queue.extend(self.unlocks[s])
        
        return len(visited)
    
    def get_recommended_skill(
        self,
        inventory: Dict[str, int],
        novelty_tracker: ActionNoveltyTracker
    ) -> int:
        """
        Recommend next skill based on tech tree progress and novelty.
        
        Uses a weighted combination of:
        1. Unlock potential (prefer skills that open more options)
        2. Novelty bonus (prefer less-visited skills)
        3. Success rate (prefer skills more likely to succeed)
        """
        available = self.get_available_skills(inventory)
        
        if not available:
            return 0  # Default to idle
        
        scores = []
        for skill_id in available:
            unlock_score = self.get_unlock_potential(skill_id) / 10.0
            novelty_score = novelty_tracker.get_novelty_bonus(skill_id)
            success_score = novelty_tracker.skill_success[skill_id]
            
            # Weighted combination
            total = (
                0.3 * unlock_score +
                0.5 * novelty_score +
                0.2 * success_score
            )
            scores.append((skill_id, total))
        
        # Return skill with highest score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]


class StepLoggerCallback(BaseCallback):
    """
    Logs detailed per-step metrics to TensorBoard every N steps.

    SB3's built-in logger only dumps episode-level stats (ep_rew_mean, etc.)
    at the end of each episode. This callback forces a TensorBoard write every
    `log_every` steps so training progress is visible in near-real-time.

    Logged metrics:
      custom/step_reward       – raw reward for this step
      custom/skill_success     – 1.0 if skill succeeded, 0.0 if not
      custom/inventory_count   – number of distinct item types held
      custom/rolling_reward    – mean reward over the last log_every steps
      custom/rolling_success   – mean skill success over the last log_every steps
      custom/context_bonus     – context reward shaper bonus (if available)
    """

    def __init__(self, log_every: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_every = log_every
        self._reward_buf: List[float] = []
        self._success_buf: List[float] = []

    def _on_step(self) -> bool:
        rewards = self.locals.get('rewards', [0.0])
        infos  = self.locals.get('infos',   [{}])

        reward = float(rewards[0]) if len(rewards) > 0 else 0.0
        info   = infos[0] if infos else {}

        success = float(info.get('skill_success', False))
        self._reward_buf.append(reward)
        self._success_buf.append(success)

        if self.num_timesteps % self.log_every == 0:
            self.logger.record('custom/step_reward',    reward)
            self.logger.record('custom/skill_success',  success)
            self.logger.record('custom/inventory_count', info.get('inventory_count', 0))
            self.logger.record('custom/rolling_reward',
                               float(np.mean(self._reward_buf)) if self._reward_buf else 0.0)
            self.logger.record('custom/rolling_success',
                               float(np.mean(self._success_buf)) if self._success_buf else 0.0)
            if 'context_bonus' in info:
                self.logger.record('custom/context_bonus', float(info['context_bonus']))
            self.logger.dump(self.num_timesteps)
            # Keep only recent window to avoid memory growth
            self._reward_buf  = self._reward_buf[-self.log_every:]
            self._success_buf = self._success_buf[-self.log_every:]

        return True


class NoveltyExplorationCallback(BaseCallback):
    """
    Callback for injecting novelty bonuses into training.

    Modifies rewards during training to include intrinsic
    motivation based on action novelty.
    """

    def __init__(
        self,
        novelty_tracker: ActionNoveltyTracker,
        novelty_weight: float = 0.1,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.novelty_tracker = novelty_tracker
        self.novelty_weight = novelty_weight

    def _on_step(self) -> bool:
        # Get info from the latest step
        infos = self.locals.get('infos', [])
        actions = self.locals.get('actions', [])

        for i, (info, action) in enumerate(zip(infos, actions)):
            if info:
                skill_id = int(action)
                success = info.get('skill_success', True)

                # Create state hash from raw state
                raw_state = info.get('raw_state', {})
                state_hash = self._hash_state(raw_state)

                # Update tracker
                self.novelty_tracker.update(
                    skill_id=skill_id,
                    success=success,
                    prev_state_hash=getattr(self, '_prev_hash', ''),
                    next_state_hash=state_hash
                )
                self._prev_hash = state_hash

        return True

    def _hash_state(self, state: dict) -> str:
        """Create a hashable representation of state."""
        # Focus on inventory for state abstraction
        inventory = state.get('inventory', {})
        key_items = sorted(inventory.keys())
        return ','.join(key_items)


class HRLAgent:
    """
    High-level agent that combines learned policy with novelty heuristics.
    
    The agent can operate in different modes:
    1. Pure RL: Learn policy end-to-end with SB3
    2. Hybrid: RL policy + novelty bonus for exploration
    3. Heuristic: Use skill graph planner (no learning)
    """
    
    def __init__(
        self,
        env: gym.Env,
        mode: str = "hybrid",
        policy_type: str = "PPO",
        novelty_weight: float = 0.1,
        device: str = "auto"
    ):
        self.env = env
        self.mode = mode
        self.novelty_weight = novelty_weight
        
        # Get number of skills
        self.num_skills = env.action_space.n
        
        # Initialize components
        self.novelty_tracker = ActionNoveltyTracker(self.num_skills)
        self.skill_graph = SkillGraphPlanner()
        
        # Initialize RL policy
        if policy_type == "PPO":
            self.policy = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log="./logs/",
                device=device,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01  # Entropy bonus for exploration
            )
        elif policy_type == "DQN":
            self.policy = DQN(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log="./logs/",
                device=device,
                learning_rate=1e-4,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=32,
                gamma=0.99,
                exploration_fraction=0.2,
                exploration_final_eps=0.05
            )
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
    
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action based on current mode.
        
        In hybrid mode, combines policy output with novelty bonus.
        """
        if self.mode == "heuristic":
            # Use skill graph planner
            raw_state = getattr(self.env.unwrapped, '_current_state', {})
            inventory = raw_state.get('inventory', {})
            return self.skill_graph.get_recommended_skill(
                inventory, self.novelty_tracker
            )
        
        elif self.mode == "pure_rl":
            # Pure RL policy
            action, _ = self.policy.predict(observation, deterministic=deterministic)
            return int(action)
        
        elif self.mode == "hybrid":
            # Get policy action probabilities
            if hasattr(self.policy, 'policy'):
                obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)
                with torch.no_grad():
                    # Get action distribution
                    dist = self.policy.policy.get_distribution(obs_tensor)
                    probs = dist.distribution.probs.numpy().flatten()
            else:
                # Fallback: use policy prediction
                action, _ = self.policy.predict(observation, deterministic=False)
                return int(action)
            
            # Add novelty bonus
            novelty = self.novelty_tracker.get_all_novelty_bonuses()
            combined = probs + self.novelty_weight * novelty
            
            # Normalize and sample
            combined = combined / combined.sum()
            
            if deterministic:
                return int(np.argmax(combined))
            else:
                return int(np.random.choice(len(combined), p=combined))
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def train(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_every: int = 100
    ):
        """Train the agent."""
        from stable_baselines3.common.callbacks import CallbackList

        # Step-level TensorBoard logger (every `log_every` steps)
        step_logger = StepLoggerCallback(log_every=log_every)

        # Novelty exploration callback
        novelty_callback = NoveltyExplorationCallback(
            self.novelty_tracker,
            novelty_weight=self.novelty_weight
        )

        # Combine all callbacks
        all_cbs = [step_logger, novelty_callback]
        if callback:
            all_cbs.append(callback)
        callbacks = CallbackList(all_cbs)

        # Train — log_interval=1 makes SB3 dump episode stats after every episode
        self.policy.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=1,
            progress_bar=True
        )
    
    def save(self, path: str):
        """Save agent state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save policy
        self.policy.save(path / "policy")
        
        # Save novelty tracker
        self.novelty_tracker.save(str(path / "novelty.json"))
    
    def load(self, path: str):
        """Load agent state."""
        path = Path(path)
        
        # Load policy
        if (path / "policy.zip").exists():
            self.policy = type(self.policy).load(path / "policy", self.env)
        
        # Load novelty tracker
        if (path / "novelty.json").exists():
            self.novelty_tracker.load(str(path / "novelty.json"))


# Utility functions

def create_state_hash(state: dict) -> str:
    """Create a compact hash of the state for novelty tracking."""
    inventory = state.get('inventory', {})
    pos = state.get('position', {})
    
    # Discretize position to grid
    grid_x = int(pos.get('x', 0) // 16)
    grid_z = int(pos.get('z', 0) // 16)
    
    # Key items
    key_items = [
        'wooden_pickaxe', 'stone_pickaxe', 'iron_pickaxe',
        'crafting_table', 'furnace'
    ]
    has_items = [k for k in key_items if inventory.get(k, 0) > 0]
    
    return f"{grid_x},{grid_z}:{','.join(has_items)}"


def evaluate_agent(
    agent: HRLAgent,
    env: gym.Env,
    num_episodes: int = 10,
    max_steps: int = 500
) -> Dict[str, float]:
    """Evaluate agent performance."""
    episode_rewards = []
    episode_lengths = []
    successes = []
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < max_steps:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        successes.append(1 if terminated and not truncated else 0)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': np.mean(successes)
    }
