"""
MinecraftHRLEnv - Gymnasium Environment Wrapper

This environment wraps the Minecraft world for Hierarchical RL.
Following the Options Framework (Sutton, Precup, Singh 1999), the action space
consists of discrete "options" (skills) rather than primitive actions.

Key Design Decisions:
1. Action Space: Discrete(N) where N = number of skills in the library
2. Observation Space: Dict space with high-level state features
3. Reward: Combination of intrinsic (skill success) and extrinsic (goal progress)

Reference Papers:
- Voyager: Code-as-action paradigm
- Plan4MC: Skill decomposition
- DEPS: Error correction loop
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import asyncio
import websockets
from typing import Dict, Any, Tuple, Optional
import time
import threading


class MinecraftHRLEnv(gym.Env):
    """
    Gymnasium environment for Hierarchical RL in Minecraft.
    
    The environment communicates with a Mineflayer bot via WebSocket.
    Actions are skill IDs that trigger temporally-extended behaviors.
    
    Observation Space (Dict):
        - health: float [0, 20]
        - food: float [0, 20]
        - position: Box(3,) - x, y, z coordinates
        - inventory: MultiBinary(N) - presence of key items
        - nearby_blocks: Box(M,) - counts of nearby block types
        - available_skills: MultiBinary(K) - which skills can be executed
        
    Action Space:
        - Discrete(K) where K = number of skills
        
    Reward:
        - Intrinsic: Based on skill success/failure
        - Extrinsic: Based on tech tree progress
    """
    
    metadata = {"render_modes": ["human", "none"], "render_fps": 1}
    
    # Key items to track in observation (order matters for encoding)
    TRACKED_ITEMS = [
        'oak_log', 'birch_log', 'spruce_log',
        'oak_planks', 'birch_planks', 'spruce_planks',
        'stick', 'crafting_table',
        'wooden_pickaxe', 'stone_pickaxe', 'iron_pickaxe',
        'cobblestone', 'stone',
        'coal', 'raw_iron', 'iron_ingot',
        'furnace', 'chest',
        'apple', 'bread', 'cooked_beef'
    ]
    
    # Block types to track (order matters for encoding)
    TRACKED_BLOCKS = [
        'oak_log', 'birch_log', 'spruce_log',
        'stone', 'cobblestone',
        'iron_ore', 'coal_ore', 'diamond_ore',
        'crafting_table', 'furnace',
        'water', 'lava'
    ]
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 1000
    ):
        super().__init__()
        
        self.host = host
        self.port = port
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        
        # WebSocket connection
        self.ws = None
        self._loop = None
        self._ws_thread = None
        
        # State tracking
        self._current_state = None
        self._step_count = 0
        self._connected = False
        
        # Will be set after connecting to get actual skill count
        self._num_skills = 13  # Default, updated on connect
        self._skill_info = []
        
        # Define spaces (will be updated after connection)
        self._define_spaces()
        
    def _define_spaces(self):
        """Define observation and action spaces."""
        
        # Observation space as a Dict
        self.observation_space = spaces.Dict({
            # Player stats (normalized)
            'health': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'food': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            
            # Position (normalized to reasonable range)
            'position': spaces.Box(
                low=-1, high=1, 
                shape=(3,), 
                dtype=np.float32
            ),
            
            # Inventory: count of each tracked item (capped)
            'inventory': spaces.Box(
                low=0, high=64,
                shape=(len(self.TRACKED_ITEMS),),
                dtype=np.float32
            ),
            
            # Nearby blocks: distance to nearest (normalized)
            'nearby_blocks': spaces.Box(
                low=0, high=1,
                shape=(len(self.TRACKED_BLOCKS),),
                dtype=np.float32
            ),
            
            # Available skills: binary mask
            'available_skills': spaces.MultiBinary(self._num_skills),
            
            # Time of day (normalized)
            'time_of_day': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            
            # Is daytime
            'is_day': spaces.MultiBinary(1)
        })
        
        # Action space: discrete skill selection
        self.action_space = spaces.Discrete(self._num_skills)
        
    def connect(self, timeout: float = 30.0) -> bool:
        """
        Establish WebSocket connection to the Mineflayer bot.
        
        Args:
            timeout: Maximum time to wait for connection
            
        Returns:
            True if connected successfully
        """
        async def _connect():
            uri = f"ws://{self.host}:{self.port}"
            try:
                self.ws = await asyncio.wait_for(
                    websockets.connect(uri),
                    timeout=timeout
                )
                
                # Get action space info
                await self.ws.send(json.dumps({'type': 'get_action_space'}))
                response = json.loads(await self.ws.recv())
                
                if response['type'] == 'action_space':
                    self._num_skills = response['n']
                    self._skill_info = response['skills']
                    self._define_spaces()  # Redefine with correct skill count
                    
                self._connected = True
                print(f"[Env] Connected to Mineflayer bot")
                print(f"[Env] Action space: {self._num_skills} skills")
                return True
                
            except Exception as e:
                print(f"[Env] Connection failed: {e}")
                return False
        
        # Run in event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(_connect())
    
    def _send_and_receive(self, message: dict) -> dict:
        """Send a message and wait for response (sync wrapper)."""
        async def _async_send():
            await self.ws.send(json.dumps(message))
            return json.loads(await self.ws.recv())
            
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_async_send())
    
    def _encode_observation(self, state: dict) -> dict:
        """
        Convert raw state dict to observation space format.
        
        This is where we transform the rich Minecraft state into
        a fixed-size numerical representation suitable for RL.
        """
        obs = {}
        
        # Health and food (normalized to [0, 1])
        obs['health'] = np.array([state.get('health', 20) / 20.0], dtype=np.float32)
        obs['food'] = np.array([state.get('food', 20) / 20.0], dtype=np.float32)
        
        # Position (normalize to [-1, 1] assuming ±10000 range)
        pos = state.get('position', {'x': 0, 'y': 64, 'z': 0})
        obs['position'] = np.array([
            pos['x'] / 10000.0,
            (pos['y'] - 64) / 64.0,  # Center around sea level
            pos['z'] / 10000.0
        ], dtype=np.float32).clip(-1, 1)
        
        # Inventory encoding
        inventory = state.get('inventory', {})
        inv_vec = np.zeros(len(self.TRACKED_ITEMS), dtype=np.float32)
        for i, item in enumerate(self.TRACKED_ITEMS):
            inv_vec[i] = min(inventory.get(item, 0), 64)
        obs['inventory'] = inv_vec
        
        # Nearby blocks (inverse distance, normalized)
        nearby = state.get('nearby_blocks', {})
        blocks_vec = np.zeros(len(self.TRACKED_BLOCKS), dtype=np.float32)
        for i, block in enumerate(self.TRACKED_BLOCKS):
            if block in nearby:
                # Convert distance to proximity (closer = higher value)
                dist = nearby[block].get('nearest_distance', 64)
                blocks_vec[i] = max(0, 1 - dist / 64)
        obs['nearby_blocks'] = blocks_vec
        
        # Available skills mask
        available = state.get('available_skills', list(range(self._num_skills)))
        skills_mask = np.zeros(self._num_skills, dtype=np.int8)
        for skill_id in available:
            if skill_id < self._num_skills:
                skills_mask[skill_id] = 1
        obs['available_skills'] = skills_mask
        
        # Time of day
        time_of_day = state.get('time_of_day', 0)
        obs['time_of_day'] = np.array([time_of_day / 24000.0], dtype=np.float32)
        obs['is_day'] = np.array([1 if state.get('is_day', True) else 0], dtype=np.int8)
        
        return obs
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[dict, dict]:
        """
        Reset the environment.
        
        Note: True reset requires server-side commands. This performs
        a "soft reset" that continues from current bot state.
        """
        super().reset(seed=seed)
        
        if not self._connected:
            if not self.connect():
                raise RuntimeError("Failed to connect to Mineflayer bot")
        
        self._step_count = 0
        
        # Send reset command
        response = self._send_and_receive({'type': 'reset'})
        
        if response['type'] != 'reset_result':
            raise RuntimeError(f"Unexpected response: {response}")
        
        self._current_state = response['state']
        obs = self._encode_observation(self._current_state)
        
        info = response.get('info', {})
        info['raw_state'] = self._current_state
        
        return obs, info
    
    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        """
        Execute a skill and return the result.
        
        Args:
            action: Skill ID to execute
            
        Returns:
            observation: Encoded state after skill execution
            reward: Reward signal
            terminated: True if episode ended (death or goal reached)
            truncated: True if max steps exceeded
            info: Additional information (skill result, raw state)
        """
        if not self._connected:
            raise RuntimeError("Not connected to Mineflayer bot")
        
        self._step_count += 1
        
        # Execute the skill
        response = self._send_and_receive({
            'type': 'step',
            'action': int(action)
        })
        
        if response['type'] != 'step_result':
            raise RuntimeError(f"Unexpected response: {response}")
        
        self._current_state = response['state']
        obs = self._encode_observation(self._current_state)
        
        reward = response['reward']
        terminated = response['done']
        truncated = response.get('truncated', self._step_count >= self.max_episode_steps)
        
        info = response.get('info', {})
        info['raw_state'] = self._current_state
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment (print state summary)."""
        if self.render_mode == "human" and self._current_state:
            state = self._current_state
            print(f"\n{'='*40}")
            print(f"Step: {self._step_count}")
            print(f"Health: {state.get('health', '?')}/20")
            print(f"Food: {state.get('food', '?')}/20")
            print(f"Position: {state.get('position', '?')}")
            print(f"Inventory: {len(state.get('inventory', {}))} item types")
            print(f"Available skills: {state.get('available_skills', [])}")
            print(f"{'='*40}")
    
    def close(self):
        """Clean up resources."""
        if self.ws:
            asyncio.get_event_loop().run_until_complete(self.ws.close())
            self.ws = None
            self._connected = False
            print("[Env] Disconnected from Mineflayer bot")
    
    def get_skill_info(self) -> list:
        """Return information about available skills."""
        return self._skill_info
    
    def get_skill_name(self, action: int) -> str:
        """Get the name of a skill by ID."""
        for skill in self._skill_info:
            if skill['id'] == action:
                return skill['name']
        return f"skill_{action}"


# Wrapper for flattened observation space (for standard SB3 algorithms)
class FlatMinecraftEnv(gym.ObservationWrapper):
    """
    Flattens the Dict observation space into a single Box space.
    Useful for algorithms that don't support Dict observations.
    """
    
    def __init__(self, env: MinecraftHRLEnv):
        super().__init__(env)
        
        # Calculate flattened size
        self._obs_keys = list(env.observation_space.spaces.keys())
        self._obs_sizes = {}
        total_size = 0
        
        for key in self._obs_keys:
            space = env.observation_space.spaces[key]
            if isinstance(space, spaces.Box):
                size = int(np.prod(space.shape))
            elif isinstance(space, spaces.MultiBinary):
                size = space.n
            else:
                raise ValueError(f"Unsupported space type: {type(space)}")
            self._obs_sizes[key] = size
            total_size += size
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(total_size,),
            dtype=np.float32
        )
    
    def observation(self, obs: dict) -> np.ndarray:
        """Flatten the observation dict into a single array."""
        parts = []
        for key in self._obs_keys:
            arr = np.asarray(obs[key], dtype=np.float32).flatten()
            parts.append(arr)
        return np.concatenate(parts)


# Factory function for easy environment creation
def make_minecraft_env(
    host: str = "localhost",
    port: int = 8765,
    flatten: bool = True,
    **kwargs
) -> gym.Env:
    """
    Create a Minecraft HRL environment.
    
    Args:
        host: Mineflayer bot host
        port: Mineflayer bot WebSocket port
        flatten: If True, return flattened observation space
        **kwargs: Additional arguments for MinecraftHRLEnv
        
    Returns:
        Gymnasium environment
    """
    env = MinecraftHRLEnv(host=host, port=port, **kwargs)
    
    if flatten:
        env = FlatMinecraftEnv(env)
    
    return env
