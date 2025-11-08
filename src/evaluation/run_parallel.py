#!/usr/bin/env python3
"""
Parallel execution script for WiFi evaluation
Runs multiple trials simultaneously to speed up experiments
"""

import subprocess
import sys
import time
from pathlib import Path
import argparse

def run_experiment(args_tuple):
    """Run single experiment configuration"""
    model, scenario, trials, prompt_style, seed, idx = args_tuple

    output_file = f"results/parallel_{scenario}_{prompt_style}_seed{seed}_batch{idx}.log"

    cmd = [
        "python", "wifi_eval.py",
        "--model", model,
        "--scenario", scenario,
        "--trials", str(trials),
        "--prompt_style", prompt_style,
        "--seed", str(seed + idx)  # Different seed per batch
    ]

    print(f"[Worker {idx}] Starting: {scenario}/{prompt_style}/seed{seed+idx}")

    with open(output_file, 'w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    print(f"[Worker {idx}] Completed: {scenario}/{prompt_style}/seed{seed+idx}")
    return result.returncode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen/qwen-2.5-72b-instruct")
    parser.add_argument("--scenarios", default="basic_recon,targeted_attack,contextual_chain")
    parser.add_argument("--prompt_styles", default="zero-shot,few-shot-3,cot")
    parser.add_argument("--trials_per_job", type=int, default=5)
    parser.add_argument("--num_jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scenarios = args.scenarios.split(',')
    prompt_styles = args.prompt_styles.split(',')

    print(f"""
╔════════════════════════════════════════════════════════════╗
║          PARALLEL WiFi EVALUATION                          ║
╚════════════════════════════════════════════════════════════╝

Configuration:
  Model: {args.model}
  Scenarios: {len(scenarios)}
  Prompt Styles: {len(prompt_styles)}
  Trials per job: {args.trials_per_job}
  Parallel jobs: {args.num_jobs}

  Total trials: {len(scenarios) * len(prompt_styles) * args.trials_per_job * args.num_jobs}
  Expected speedup: {args.num_jobs}x faster

Starting parallel execution...
""")

    # Create job queue
    jobs = []
    job_idx = 0
    for scenario in scenarios:
        for prompt_style in prompt_styles:
            for batch in range(args.num_jobs):
                jobs.append((
                    args.model,
                    scenario,
                    args.trials_per_job,
                    prompt_style,
                    args.seed,
                    job_idx
                ))
                job_idx += 1

    print(f"Created {len(jobs)} parallel jobs\n")

    # Run in batches
    from multiprocessing import Pool, cpu_count

    num_workers = min(args.num_jobs, cpu_count())
    print(f"Using {num_workers} CPU cores\n")

    start_time = time.time()

    with Pool(num_workers) as pool:
        results = pool.map(run_experiment, jobs)

    elapsed = time.time() - start_time

    print(f"""
╔════════════════════════════════════════════════════════════╗
║                    COMPLETION                              ║
╚════════════════════════════════════════════════════════════╝

Total time: {elapsed/60:.1f} minutes
Failed jobs: {sum(1 for r in results if r != 0)}

Results saved to: results/

Next steps:
  1. Merge results: python merge_results.py
  2. Analyze: python analyze_results.py
""")

if __name__ == "__main__":
    main()
