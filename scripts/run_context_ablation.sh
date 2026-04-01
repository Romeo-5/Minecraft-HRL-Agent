#!/bin/bash
# run_context_ablation.sh
#
# Re-runs the DQN ablation with the context reward shaper and env-aware observation.
# Directly validates the hypothesis: env-aware failed in the initial ablation because
# the reward didn't incentivize using biome/structure context, not because context
# is uninformative.
#
# Prerequisites: Minecraft server + Mineflayer bot running on localhost:8765
#   cd mineflayer && npm start
#
# Usage: bash scripts/run_context_ablation.sh
#
# Results land in python/checkpoints/. Compare success rate to prior ablation
# (dqn_mlp_aware: 22-30% SR) — expect env-aware+context to exceed env-blind+no-context.

set -e
cd "$(dirname "$0")/../python"
PYTHON="$(pwd)/venv/bin/python"

echo "======================================================"
echo " Minecraft HRL — DQN Context Reward Ablation"
echo "======================================================"
echo ""

# Condition 1: env-aware + context reward (the fix)
# Policy sees biome/structure in obs AND reward incentivises using them
echo "[1/2] DQN  env-aware  + context reward (200K steps)"
$PYTHON main.py \
    --policy DQN \
    --mode pure_rl \
    --timesteps 200000 \
    --env-aware \
    --save-freq 10000 \
    --checkpoint checkpoints/dqn_envaware_contextreward \
    --seed 42

echo ""

# Condition 2: env-blind + no context reward (baseline — matches prior ablation)
# Policy has no biome/structure in obs AND reward has no context bonus
echo "[2/2] DQN  env-blind  + no context reward (200K steps)"
$PYTHON main.py \
    --policy DQN \
    --mode pure_rl \
    --timesteps 200000 \
    --no-context-reward \
    --save-freq 10000 \
    --checkpoint checkpoints/dqn_envblind_nocontext \
    --seed 42

echo ""
echo "======================================================"
echo " Done. Compare checkpoints:"
echo "   checkpoints/dqn_envaware_contextreward/"
echo "   checkpoints/dqn_envblind_nocontext/"
echo ""
echo " View TensorBoard logs:"
echo "   tensorboard --logdir python/logs/"
echo "======================================================"
