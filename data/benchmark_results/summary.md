# Benchmark Results Summary

## Aggregate Metrics

| Model | Condition | Step Coverage | Shortcut Detection | Efficiency | N |
|-------|-----------|--------------|-------------------|------------|---|
| llama3.2:3b | with_context | 16.6% (±0.21) | 96.0% | 65.2% (±0.25) | 105 |
| llama3.2:3b | without_context | 12.3% (±0.17) | 2.0% | 64.8% (±0.25) | 105 |

## Context Benefit Analysis

| Model | Coverage Δ | Efficiency Δ | Shortcut Δ | p-value (cov) | Significant? |
|-------|-----------|-------------|-----------|--------------|-------------|
| llama3.2:3b | +0.0423 | +0.0041 | +0.9400 | 0.0106 | Yes |
