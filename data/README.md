# Minecraft HRL Reasoning Path Benchmark

Benchmark dataset and evaluation infrastructure for testing whether LLMs can reason about Minecraft task planning given environmental context (biome, structures, y-level).

Part of the USC CSCI 566 Deep Learning course project on Hierarchical Reinforcement Learning in Minecraft.

## Dataset

**`reasoning_paths.json`** — 105 samples, each containing:

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Unique sample identifier |
| `biome` | string | Minecraft biome (13 biomes covered) |
| `nearby_structures` | list[str] | Structures near the agent (11 types) |
| `y_level` | int | Agent's Y coordinate (depth) |
| `task` | string | Goal to accomplish (8 task types) |
| `reasoning_path` | list[str] | Optimal step sequence (key decision points) |
| `reasoning_text` | string | Natural language explanation of the strategy |
| `context_matters` | bool | Whether biome/structure changes the optimal path |
| `context_explanation` | string | Why context does or doesn't matter |

### Coverage

- **Biomes (13):** plains, forest, desert, mesa, ice_spikes, jungle, swamp, savanna, taiga, mountains, ocean, mushroom_island, dark_forest
- **Structures (11):** none, village, blacksmith, mineshaft, ruined_portal, desert_temple, jungle_temple, igloo, shipwreck, pillager_outpost, stronghold
- **Tasks (8):** obtain_wooden_pickaxe, obtain_stone_pickaxe, obtain_iron_pickaxe, obtain_diamond_pickaxe, obtain_gold, obtain_food, build_shelter, obtain_armor
- **Y-levels:** surface (64+), underground (0-63), deep (-64 to -1)
- **Context split:** 82 samples where context matters, 23 where it doesn't

Targets Minecraft **1.20.1** (1.18+ ore distribution).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Regenerate dataset (optional — reasoning_paths.json is already included)
python generate_dataset.py

# Run benchmark (requires Ollama running locally)
ollama serve  # in another terminal
ollama pull llama3:8b
python benchmark_models.py --models llama3:8b

# Evaluate results
python evaluate_results.py
```

## Scripts

### `generate_dataset.py`
Generates the reasoning path dataset using template-based composition of Minecraft knowledge (biome modifiers, structure shortcuts, y-level strategies).

```bash
python generate_dataset.py                    # generate dataset
python generate_dataset.py --stats-only       # print stats for existing dataset
python generate_dataset.py --output custom.json
```

### `benchmark_models.py`
Runs LLM evaluations via Ollama with two conditions:
- **With context:** biome, structures, y-level provided in prompt
- **Without context:** task only

```bash
python benchmark_models.py --models llama3:8b mistral:7b
python benchmark_models.py --list-models          # show available models
python benchmark_models.py --temperature 0.7      # non-deterministic
python benchmark_models.py --no-resume             # start fresh
```

Results are saved as JSONL for resumability — the script skips already-evaluated (sample, model, condition) tuples.

### `evaluate_results.py`
Computes metrics comparing model outputs to ground truth:

```bash
python evaluate_results.py
python evaluate_results.py --threshold 0.5    # lower fuzzy match threshold
```

## Metrics

| Metric | Description |
|--------|-------------|
| **Step coverage** | Fraction of ground truth steps mentioned by the model (fuzzy matched) |
| **Shortcut detection** | Whether the model leveraged nearby structures when applicable |
| **Efficiency score** | `len(ground_truth) / max(len(predicted), len(ground_truth))` |
| **Context benefit** | Performance delta between with-context and without-context conditions |

Statistical significance tested via paired t-test (coverage, efficiency) and McNemar's test (shortcut detection).

## Output Structure

```
benchmark_results/
├── results.jsonl                    # Raw results (one JSON per line)
├── raw_responses/{model}/{cond}/    # Full model text outputs
├── metrics/
│   ├── per_sample_metrics.csv       # Per-sample computed metrics
│   ├── per_sample_metrics.json      # Same, as JSON for notebooks
│   ├── aggregate_metrics.csv        # Aggregated by model × condition
│   └── context_benefit.csv          # Context benefit with p-values
└── summary.md                       # Human-readable summary
```

## Adding Models

Pull any Ollama model and pass its name:

```bash
ollama pull qwen2:7b
python benchmark_models.py --models llama3:8b qwen2:7b mistral:7b
```

## Dependencies

See `requirements.txt`. Core: `requests`, `scipy`, `numpy`, `tqdm`, `pandas`, `matplotlib`.
