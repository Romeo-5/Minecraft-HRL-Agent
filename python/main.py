#!/usr/bin/env python3
"""
Minecraft HRL Agent - Main Training Script

This script orchestrates the training loop for the Hierarchical RL agent.

Usage:
    python main.py --mode hybrid --timesteps 100000
    python main.py --mode heuristic --episodes 100
    python main.py --eval --checkpoint ./checkpoints/agent
    
Requirements:
    1. Minecraft server running (localhost:25565)
    2. Mineflayer bot connected (npm start in mineflayer/)
    3. Python dependencies installed (pip install -r requirements.txt)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from env import make_minecraft_env, MinecraftHRLEnv
from agent import HRLAgent, evaluate_agent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Hierarchical RL agent for Minecraft"
    )
    
    # Environment settings
    parser.add_argument(
        "--host", type=str, default="localhost",
        help="Mineflayer bot WebSocket host"
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="Mineflayer bot WebSocket port"
    )
    
    # Training settings
    parser.add_argument(
        "--mode", type=str, default="hybrid",
        choices=["pure_rl", "hybrid", "heuristic"],
        help="Agent mode: pure_rl, hybrid (RL + novelty), or heuristic"
    )
    parser.add_argument(
        "--policy", type=str, default="PPO",
        choices=["PPO", "DQN"],
        help="RL algorithm to use"
    )
    parser.add_argument(
        "--timesteps", type=int, default=100000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--novelty-weight", type=float, default=0.1,
        help="Weight for novelty bonus in hybrid mode"
    )
    
    # Evaluation settings
    parser.add_argument(
        "--eval", action="store_true",
        help="Run evaluation instead of training"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=10,
        help="Number of evaluation episodes"
    )
    
    # Checkpointing
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to load/save agent checkpoint"
    )
    parser.add_argument(
        "--save-freq", type=int, default=10000,
        help="Save checkpoint every N timesteps"
    )
    
    # Misc
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device for training (cpu, cuda, auto)"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render environment during training"
    )
    
    return parser.parse_args()


def train(args):
    """Main training loop."""
    print("=" * 60)
    print("Minecraft HRL Agent - Training")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Policy: {args.policy}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Novelty weight: {args.novelty_weight}")
    print("=" * 60)
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create environment
    print("\n[Main] Creating environment...")
    env = make_minecraft_env(
        host=args.host,
        port=args.port,
        flatten=True,
        render_mode="human" if args.render else None
    )
    
    # Connect to Mineflayer bot
    print("[Main] Connecting to Mineflayer bot...")
    if not env.unwrapped.connect(timeout=30):
        print("[Main] ERROR: Could not connect to Mineflayer bot!")
        print("[Main] Make sure the bot is running: cd mineflayer && npm start")
        sys.exit(1)
    
    print(f"[Main] Connected! Action space: {env.action_space.n} skills")
    
    # Create agent
    print("\n[Main] Creating agent...")
    agent = HRLAgent(
        env=env,
        mode=args.mode,
        policy_type=args.policy,
        novelty_weight=args.novelty_weight,
        device=args.device
    )
    
    # Load checkpoint if specified
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"[Main] Loading checkpoint from {args.checkpoint}")
        agent.load(args.checkpoint)
    
    # Create checkpoint callback
    from stable_baselines3.common.callbacks import CheckpointCallback
    
    checkpoint_dir = args.checkpoint or f"./checkpoints/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=checkpoint_dir,
        name_prefix="minecraft_hrl"
    )
    
    # Train
    print("\n[Main] Starting training...")
    try:
        agent.train(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback
        )
    except KeyboardInterrupt:
        print("\n[Main] Training interrupted by user")
    
    # Save final checkpoint
    final_path = Path(checkpoint_dir) / "final"
    print(f"\n[Main] Saving final checkpoint to {final_path}")
    agent.save(str(final_path))
    
    # Final evaluation
    print("\n[Main] Running final evaluation...")
    results = evaluate_agent(agent, env, num_episodes=5)
    print(f"Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean episode length: {results['mean_length']:.1f}")
    print(f"Success rate: {results['success_rate']*100:.1f}%")
    
    # Cleanup
    env.close()
    print("\n[Main] Training complete!")


def evaluate(args):
    """Evaluation mode."""
    print("=" * 60)
    print("Minecraft HRL Agent - Evaluation")
    print("=" * 60)
    
    if not args.checkpoint:
        print("[Main] ERROR: --checkpoint required for evaluation")
        sys.exit(1)
    
    # Create environment
    env = make_minecraft_env(
        host=args.host,
        port=args.port,
        flatten=True,
        render_mode="human"
    )
    
    if not env.unwrapped.connect(timeout=30):
        print("[Main] ERROR: Could not connect to Mineflayer bot!")
        sys.exit(1)
    
    # Create and load agent
    agent = HRLAgent(
        env=env,
        mode=args.mode,
        policy_type=args.policy,
        device=args.device
    )
    agent.load(args.checkpoint)
    
    # Evaluate
    print(f"\n[Main] Evaluating for {args.eval_episodes} episodes...")
    results = evaluate_agent(
        agent, env,
        num_episodes=args.eval_episodes,
        max_steps=500
    )
    
    print("\n" + "=" * 40)
    print("Evaluation Results")
    print("=" * 40)
    print(f"Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean episode length: {results['mean_length']:.1f}")
    print(f"Success rate: {results['success_rate']*100:.1f}%")
    
    # Cleanup
    env.close()


def demo_heuristic(args):
    """Demo the heuristic planner without RL."""
    print("=" * 60)
    print("Minecraft HRL Agent - Heuristic Demo")
    print("=" * 60)
    
    env = make_minecraft_env(
        host=args.host,
        port=args.port,
        flatten=True,
        render_mode="human"
    )
    
    if not env.unwrapped.connect(timeout=30):
        print("[Main] ERROR: Could not connect!")
        sys.exit(1)
    
    # Use heuristic agent
    agent = HRLAgent(env=env, mode="heuristic")
    
    print("\n[Main] Running heuristic planner demo...")
    print("Press Ctrl+C to stop\n")
    
    obs, info = env.reset()
    total_reward = 0
    step = 0
    
    try:
        while True:
            action = agent.select_action(obs)
            skill_name = env.unwrapped.get_skill_name(action)
            
            print(f"Step {step}: Executing skill {action} ({skill_name})")
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            print(f"  Result: {info.get('skill_message', 'N/A')}")
            print(f"  Reward: {reward:.3f} (Total: {total_reward:.3f})")
            
            if terminated:
                print("\n[Main] Episode terminated (goal reached or death)")
                break
            if truncated:
                print("\n[Main] Episode truncated (max steps)")
                break
                
    except KeyboardInterrupt:
        print("\n[Main] Demo stopped by user")
    
    env.close()


def main():
    args = parse_args()
    
    if args.eval:
        evaluate(args)
    elif args.mode == "heuristic":
        demo_heuristic(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
