#!/usr/bin/env python3
"""
Benchmark LLMs on Minecraft Reasoning Path Dataset.

Runs models via Ollama (local) on two conditions:
  - With context: biome, structures, y-level provided
  - Without context: task only

Results saved as JSONL for resumability.

Usage:
    python benchmark_models.py
    python benchmark_models.py --models llama3:8b mistral:7b
    python benchmark_models.py --models llama3:8b --no-resume
    python benchmark_models.py --list-models  # show available Ollama models
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

import requests
from tqdm import tqdm

# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = (
    "You are a Minecraft planning assistant. Given a task (and optionally "
    "environmental context), provide the optimal sequence of steps to accomplish "
    "the task. List each step on a new line, prefixed with a number. Be specific "
    "about Minecraft mechanics. Focus on the key decision points rather than "
    "every sub-step."
)


def build_prompt_with_context(task: str, biome: str, structures: List[str], y_level: int) -> str:
    """Build prompt with full environmental context."""
    task_readable = task.replace("_", " ")
    structures_readable = ", ".join(s.replace("_", " ") for s in structures if s != "none")
    if not structures_readable:
        structures_readable = "none"

    return (
        f"You are playing Minecraft 1.20.1 in survival mode.\n\n"
        f"Your current environment:\n"
        f"- Biome: {biome.replace('_', ' ')}\n"
        f"- Nearby structures: {structures_readable}\n"
        f"- Current Y-level: {y_level}\n\n"
        f"Task: {task_readable}\n\n"
        f"What is the optimal sequence of steps to complete this task? "
        f"Consider how your environment (biome, nearby structures, depth) "
        f"affects your strategy. List each step in order."
    )


def build_prompt_without_context(task: str) -> str:
    """Build prompt with only the task description."""
    task_readable = task.replace("_", " ")
    return (
        f"You are playing Minecraft 1.20.1 in survival mode.\n\n"
        f"Task: {task_readable}\n\n"
        f"What is the optimal sequence of steps to complete this task? "
        f"List each step in order."
    )


# =============================================================================
# RESPONSE PARSING
# =============================================================================

def parse_model_response(response: str) -> List[str]:
    """Extract numbered/bulleted steps from model response.

    Handles formats:
    - "1. Do X\\n2. Do Y"
    - "- Do X\\n- Do Y"
    - "* Do X\\n* Do Y"
    - Numbered with ) like "1) Do X"
    """
    import re

    lines = response.strip().split("\n")
    steps = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Match numbered lists: "1. ...", "1) ...", "1: ..."
        m = re.match(r"^\d+[\.\)\:]\s*(.+)", line)
        if m:
            steps.append(m.group(1).strip())
            continue

        # Match bullet lists: "- ...", "* ...", "• ..."
        m = re.match(r"^[-\*\u2022]\s*(.+)", line)
        if m:
            steps.append(m.group(1).strip())
            continue

    # If no structured steps found, try splitting on sentence boundaries
    if not steps:
        sentences = re.split(r"(?<=[.!])\s+", response.strip())
        steps = [s.strip() for s in sentences if len(s.strip()) > 10]

    return steps


# =============================================================================
# OLLAMA CLIENT
# =============================================================================

class OllamaClient:
    """Client for Ollama local model API."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=30)
            return r.status_code == 200
        except (requests.ConnectionError, requests.ReadTimeout):
            return False

    def list_models(self) -> List[str]:
        """List available local models."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=30)
            r.raise_for_status()
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
        except (requests.ConnectionError, requests.HTTPError, requests.ReadTimeout) as e:
            print(f"Error listing models: {e}")
            return []

    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Tuple[str, float]:
        """Call Ollama generate endpoint.

        Returns (response_text, elapsed_seconds).
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": False,
        }

        start = time.time()
        r = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=300,  # 5 min timeout for slow models
        )
        elapsed = time.time() - start
        r.raise_for_status()

        data = r.json()
        return data.get("response", ""), elapsed


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def load_progress(results_path: str) -> Set[Tuple[str, str, str]]:
    """Load already-completed (sample_id, model, condition) tuples."""
    completed = set()
    if not os.path.exists(results_path):
        return completed

    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
                key = (str(result["sample_id"]), result["model"], result["condition"])
                completed.add(key)
            except (json.JSONDecodeError, KeyError):
                continue

    return completed


def save_result(result: dict, results_path: str):
    """Append a single result to the JSONL file."""
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    with open(results_path, "a") as f:
        f.write(json.dumps(result) + "\n")


def save_raw_response(
    response: str,
    model_name: str,
    condition: str,
    sample_id: int,
    output_dir: str,
):
    """Save raw model response to a text file."""
    # Sanitize model name for directory
    model_dir = model_name.replace("/", "_").replace(":", "_")
    dir_path = os.path.join(output_dir, "raw_responses", model_dir, condition)
    os.makedirs(dir_path, exist_ok=True)

    file_path = os.path.join(dir_path, f"sample_{sample_id:03d}.txt")
    with open(file_path, "w") as f:
        f.write(response)


def run_benchmark(
    dataset_path: str,
    models: List[str],
    output_dir: str,
    ollama_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    skip_existing: bool = True,
):
    """Run the full benchmark."""
    # Load dataset
    with open(dataset_path) as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} samples from {dataset_path}")

    # Setup Ollama client
    client = OllamaClient(ollama_url)
    if not client.is_available():
        print(f"ERROR: Ollama server not available at {ollama_url}")
        print("Start Ollama with: ollama serve")
        return

    available = client.list_models()
    print(f"Available Ollama models: {available}")

    # Check requested models are available
    for model in models:
        if model not in available:
            print(f"WARNING: Model '{model}' not found. Pull it with: ollama pull {model}")

    results_path = os.path.join(output_dir, "results.jsonl")
    completed = load_progress(results_path) if skip_existing else set()
    if completed:
        print(f"Resuming: {len(completed)} evaluations already completed")

    conditions = ["with_context", "without_context"]
    total = len(models) * len(samples) * len(conditions)
    skipped = 0

    with tqdm(total=total, desc="Benchmarking") as pbar:
        for model in models:
            if model not in available:
                pbar.update(len(samples) * len(conditions))
                continue

            for sample in samples:
                for condition in conditions:
                    sample_id = str(sample["id"])
                    key = (sample_id, model, condition)

                    if key in completed:
                        skipped += 1
                        pbar.update(1)
                        continue

                    # Build prompt
                    if condition == "with_context":
                        prompt = build_prompt_with_context(
                            sample["task"],
                            sample["biome"],
                            sample["nearby_structures"],
                            sample["y_level"],
                        )
                    else:
                        prompt = build_prompt_without_context(sample["task"])

                    # Call model
                    try:
                        response_text, elapsed = client.generate(
                            model=model,
                            prompt=prompt,
                            system=SYSTEM_PROMPT,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                    except Exception as e:
                        print(f"\nError on sample {sample_id} with {model}: {e}")
                        pbar.update(1)
                        continue

                    # Parse response
                    parsed_steps = parse_model_response(response_text)

                    # Save result
                    result = {
                        "sample_id": sample["id"],
                        "model": model,
                        "condition": condition,
                        "prompt": prompt,
                        "raw_response": response_text,
                        "parsed_steps": parsed_steps,
                        "response_time_seconds": round(elapsed, 2),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

                    save_result(result, results_path)
                    save_raw_response(
                        response_text, model, condition,
                        sample["id"], output_dir,
                    )

                    pbar.update(1)
                    pbar.set_postfix(model=model, condition=condition[:7])

    print(f"\nBenchmark complete!")
    print(f"  Total evaluations: {total}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLMs on Minecraft reasoning paths"
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Path to reasoning_paths.json (default: same directory)",
    )
    parser.add_argument(
        "--models", nargs="+", default=["llama3:8b"],
        help="Ollama model names to test (default: llama3:8b)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for results (default: benchmark_results/)",
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434",
        help="Ollama server URL",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Model temperature (default: 0.0 for deterministic)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024,
        help="Max tokens in response (default: 1024)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh, don't skip existing evaluations",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List available Ollama models and exit",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = args.dataset or os.path.join(script_dir, "reasoning_paths.json")
    output_dir = args.output_dir or os.path.join(script_dir, "benchmark_results")

    if args.list_models:
        client = OllamaClient(args.ollama_url)
        if not client.is_available():
            print(f"Ollama server not available at {args.ollama_url}")
            return
        models = client.list_models()
        print("Available models:")
        for m in models:
            print(f"  - {m}")
        return

    run_benchmark(
        dataset_path=dataset_path,
        models=args.models,
        output_dir=output_dir,
        ollama_url=args.ollama_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        skip_existing=not args.no_resume,
    )


if __name__ == "__main__":
    main()
