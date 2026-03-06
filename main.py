#!/usr/bin/env python3
"""
Main runner: Solves Airline Crew Scheduling (Set Partitioning Problem)
using Simulated Annealing, Standard BGA, and Improved BGA.

Runs each algorithm 30 times with different seeds on each problem,
computes mean and standard deviation, and saves results to files.
"""

import os
import time
import math

from utils import (
    find_file, parse_spp, compute_cost, compute_coverage,
    compute_violation, is_feasible, format_result
)
from simulated_annealing import simulated_annealing
from standard_bga import standard_bga
from improved_bga import improved_bga


NUM_RUNS = 30


def run_algorithm_multi(algo_func, algo_name, num_rows, num_cols, costs,
                        col_covers, row_covers, num_runs=NUM_RUNS):
    """Run an algorithm num_runs times with seeds 1..num_runs.
    Returns list of (cost, viol, elapsed) and the best solution."""
    results = []
    best_x, best_cost = None, float('inf')

    for run in range(1, num_runs + 1):
        t0 = time.time()
        x, cost, viol = algo_func(num_rows, num_cols, costs, col_covers,
                                  row_covers, seed=run)
        elapsed = time.time() - t0
        results.append((cost, viol, elapsed))
        print(f"    {algo_name} run {run:2d}/{num_runs}: cost={cost}, "
              f"feasible={viol == 0}, time={elapsed:.2f}s")
        if viol == 0 and cost < best_cost:
            best_cost = cost
            best_x = x[:]

    return results, best_x


def compute_stats(results):
    """Compute statistics from list of (cost, viol, elapsed) tuples."""
    feasible_costs = [c for c, v, _ in results if v == 0]
    all_costs = [c for c, _, _ in results]
    times = [t for _, _, t in results]
    num_feasible = len(feasible_costs)

    stats = {
        'num_runs': len(results),
        'num_feasible': num_feasible,
        'all_costs': all_costs,
        'feasible_costs': feasible_costs,
        'times': times,
    }

    if feasible_costs:
        mean_cost = sum(feasible_costs) / len(feasible_costs)
        std_cost = math.sqrt(sum((c - mean_cost) ** 2 for c in feasible_costs)
                             / len(feasible_costs))
        stats['best_cost'] = min(feasible_costs)
        stats['worst_cost'] = max(feasible_costs)
        stats['mean_cost'] = mean_cost
        stats['std_cost'] = std_cost
    else:
        stats['best_cost'] = min(all_costs)
        stats['worst_cost'] = max(all_costs)
        mean_cost = sum(all_costs) / len(all_costs)
        stats['mean_cost'] = mean_cost
        stats['std_cost'] = math.sqrt(sum((c - mean_cost) ** 2 for c in all_costs)
                                      / len(all_costs))

    stats['mean_time'] = sum(times) / len(times)
    return stats


def write_algorithm_results(filepath, algo_name, problem_results):
    """Write full results for one algorithm across all problems."""
    with open(filepath, 'w') as f:
        f.write("=" * 72 + "\n")
        f.write(f"{algo_name.upper()} RESULTS — {NUM_RUNS} INDEPENDENT RUNS\n")
        f.write("=" * 72 + "\n\n")

        for prob_name, optimal, stats, raw_results in problem_results:
            f.write(f"Problem: {prob_name}  (optimal = {optimal})\n")
            f.write("-" * 72 + "\n")
            f.write(f"  Feasible runs : {stats['num_feasible']} / {stats['num_runs']}\n")
            f.write(f"  Best cost     : {stats['best_cost']}\n")
            f.write(f"  Worst cost    : {stats['worst_cost']}\n")
            f.write(f"  Mean cost     : {stats['mean_cost']:.2f}\n")
            f.write(f"  Std deviation : {stats['std_cost']:.2f}\n")
            if optimal and stats['num_feasible'] > 0:
                gap = (stats['best_cost'] - optimal) / optimal * 100
                mean_gap = (stats['mean_cost'] - optimal) / optimal * 100
                f.write(f"  Best gap      : {gap:.2f}%\n")
                f.write(f"  Mean gap      : {mean_gap:.2f}%\n")
            f.write(f"  Mean time     : {stats['mean_time']:.2f}s\n\n")

            f.write(f"  {'Run':>4s}  {'Cost':>8s}  {'Feasible':>8s}  {'Time (s)':>10s}\n")
            f.write(f"  {'----':>4s}  {'--------':>8s}  {'--------':>8s}  {'----------':>10s}\n")
            for i, (cost, viol, elapsed) in enumerate(raw_results, 1):
                feas = "Yes" if viol == 0 else "No"
                f.write(f"  {i:4d}  {cost:8d}  {feas:>8s}  {elapsed:10.2f}\n")
            f.write("\n" + "=" * 72 + "\n\n")


def main():
    """Main entry point: runs all algorithms 30 times each, saves results."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    problems = ['sppnw41.txt', 'sppnw42.txt', 'sppnw43.txt']
    optimal_values = {'sppnw41.txt': 6942, 'sppnw42.txt': 8405, 'sppnw43.txt': 8853}

    algorithms = [
        ("Simulated Annealing", simulated_annealing, "simulated_annealing_results.txt"),
        ("Standard BGA", standard_bga, "standard_bga_results.txt"),
        ("Improved BGA", improved_bga, "improved_bga_results.txt"),
    ]

    # Collect all stats: all_stats[algo_name][problem] = (stats, raw_results)
    all_stats = {}

    for algo_name, algo_func, result_filename in algorithms:
        print("=" * 72)
        print(f"RUNNING: {algo_name} — {NUM_RUNS} runs per problem")
        print("=" * 72)

        problem_results = []
        all_stats[algo_name] = {}

        for problem in problems:
            filepath = find_file(script_dir, problem)
            num_rows, num_cols, costs, col_covers, row_covers = parse_spp(filepath)
            opt = optimal_values.get(problem, 0)

            print(f"\n  Problem: {problem}  ({num_rows} rows, {num_cols} cols, optimal = {opt})")
            print(f"  {'-' * 60}")

            raw_results, best_x = run_algorithm_multi(
                algo_func, algo_name, num_rows, num_cols, costs,
                col_covers, row_covers, num_runs=NUM_RUNS)

            stats = compute_stats(raw_results)
            problem_results.append((problem, opt, stats, raw_results))
            all_stats[algo_name][problem] = stats

            print(f"\n  Summary for {problem}:")
            print(f"    Best={stats['best_cost']}, Mean={stats['mean_cost']:.2f}, "
                  f"Std={stats['std_cost']:.2f}, Feasible={stats['num_feasible']}/{NUM_RUNS}")

        result_path = os.path.join(results_dir, result_filename)
        write_algorithm_results(result_path, algo_name, problem_results)
        print(f"\n  Results saved to: {result_path}\n")

    # Write combined summary
    summary_path = os.path.join(results_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("=" * 90 + "\n")
        f.write(f"COMBINED RESULTS SUMMARY — {NUM_RUNS} INDEPENDENT RUNS PER ALGORITHM\n")
        f.write("=" * 90 + "\n\n")

        for problem in problems:
            opt = optimal_values.get(problem, 0)
            f.write(f"Problem: {problem}  (optimal = {opt})\n")
            f.write("-" * 90 + "\n")
            f.write(f"  {'Algorithm':<22s}  {'Best':>8s}  {'Worst':>8s}  {'Mean':>10s}  "
                    f"{'Std':>10s}  {'Feasible':>10s}  {'Avg Time':>10s}\n")
            f.write(f"  {'─' * 22}  {'─' * 8}  {'─' * 8}  {'─' * 10}  "
                    f"{'─' * 10}  {'─' * 10}  {'─' * 10}\n")

            for algo_name, _, _ in algorithms:
                s = all_stats[algo_name][problem]
                feas_str = f"{s['num_feasible']}/{s['num_runs']}"
                f.write(f"  {algo_name:<22s}  {s['best_cost']:8d}  {s['worst_cost']:8d}  "
                        f"{s['mean_cost']:10.2f}  {s['std_cost']:10.2f}  "
                        f"{feas_str:>10s}  {s['mean_time']:10.2f}\n")
            f.write("\n")

        f.write("\nNote: Statistics computed over feasible solutions only "
                "(where all rows are covered exactly once).\n")
        f.write("Seeds used: 1 to 30 (one per independent run).\n")

    print(f"\nCombined summary saved to: {summary_path}")
    print("Done!")


if __name__ == '__main__':
    main()
