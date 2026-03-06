"""
Simulated Annealing (SA) for the Set Partitioning Problem.

Multi-restart SA with swap/add/remove neighbourhood.
"""

import random
import math
import time

from utils import (
    compute_cost, compute_coverage, compute_violation,
    greedy_construction, pseudo_random_init, heuristic_improvement
)


def _sa_run(x_init, num_rows, num_cols, costs, col_covers, row_covers,
            col_cover_sets, penalty_weight, alpha, iterations):
    """Single SA run from given initial solution."""
    x = x_init[:]
    coverage = compute_coverage(x, col_covers, num_rows)
    cur_cost = compute_cost(x, costs)
    cur_viol = compute_violation(coverage)
    cur_obj = cur_cost + penalty_weight * cur_viol

    best_x, best_cost, best_viol = x[:], cur_cost, cur_viol
    if cur_viol > 0:
        best_cost = float('inf')

    T = 0.4 * max(cur_cost, 1)
    selected = set(j for j in range(num_cols) if x[j])

    for it in range(iterations):
        mv = random.random()

        if mv < 0.55 and selected:
            j_rem = random.choice(list(selected))
            if random.random() < 0.7:
                cands = set()
                for r in col_covers[j_rem]:
                    for c in row_covers[r]:
                        if c not in selected:
                            cands.add(c)
                if cands:
                    j_add = random.choice(list(cands))
                else:
                    unsel = [j for j in range(num_cols) if j not in selected]
                    if not unsel: continue
                    j_add = random.choice(unsel)
            else:
                unsel = [j for j in range(num_cols) if j not in selected]
                if not unsel: continue
                j_add = random.choice(unsel)

            dc = costs[j_add] - costs[j_rem]
            dv = 0
            for r in col_cover_sets[j_rem] | col_cover_sets[j_add]:
                oc = coverage[r]
                nc = oc - (1 if r in col_cover_sets[j_rem] else 0) \
                       + (1 if r in col_cover_sets[j_add] else 0)
                dv += abs(nc - 1) - abs(oc - 1)

            do = dc + penalty_weight * dv
            if do <= 0 or random.random() < math.exp(-do / max(T, 1e-10)):
                x[j_rem], x[j_add] = 0, 1
                selected.discard(j_rem); selected.add(j_add)
                for r in col_cover_sets[j_rem]: coverage[r] -= 1
                for r in col_cover_sets[j_add]: coverage[r] += 1
                cur_cost += dc; cur_viol += dv; cur_obj += do

        elif mv < 0.75 and selected:
            j = random.choice(list(selected))
            dc = -costs[j]
            dv = sum(abs(coverage[r] - 2) - abs(coverage[r] - 1) for r in col_covers[j])
            do = dc + penalty_weight * dv
            if do <= 0 or random.random() < math.exp(-do / max(T, 1e-10)):
                x[j] = 0; selected.discard(j)
                for r in col_covers[j]: coverage[r] -= 1
                cur_cost += dc; cur_viol += dv; cur_obj += do

        else:
            uncov = [i for i in range(num_rows) if coverage[i] == 0]
            if uncov:
                ri = random.choice(uncov)
                ca = [c for c in row_covers[ri] if c not in selected]
                if not ca: continue
                j = random.choice(ca)
            else:
                j = random.randint(0, num_cols - 1)
                if j in selected: continue
            dc = costs[j]
            dv = sum(abs(coverage[r]) - abs(coverage[r] - 1) for r in col_covers[j])
            do = dc + penalty_weight * dv
            if do <= 0 or random.random() < math.exp(-do / max(T, 1e-10)):
                x[j] = 1; selected.add(j)
                for r in col_covers[j]: coverage[r] += 1
                cur_cost += dc; cur_viol += dv; cur_obj += do

        if cur_viol == 0 and cur_cost < best_cost:
            best_x, best_cost, best_viol = x[:], cur_cost, 0

        T *= alpha

    return best_x, best_cost, best_viol


def simulated_annealing(num_rows, num_cols, costs, col_covers, row_covers,
                        alpha=0.99997, iters_per_run=200000, num_restarts=10,
                        seed=42):
    """Multi-restart SA for SPP with swap/add/remove neighbourhood."""
    random.seed(seed)
    penalty_weight = max(costs) // 2
    col_cover_sets = [set(cc) for cc in col_covers]

    best_x, best_cost, best_viol = None, float('inf'), float('inf')

    for restart in range(num_restarts):
        if restart == 0:
            x0 = greedy_construction(num_rows, num_cols, costs, col_covers, row_covers)
        else:
            x0 = pseudo_random_init(num_rows, num_cols, costs, col_covers, row_covers)
        x0 = heuristic_improvement(x0, num_rows, num_cols, costs, col_covers, row_covers)

        rx, rc, rv = _sa_run(x0, num_rows, num_cols, costs, col_covers, row_covers,
                             col_cover_sets, penalty_weight, alpha, iters_per_run)
        if rv == 0 and rc < best_cost:
            best_x, best_cost, best_viol = rx, rc, rv

    if best_cost == float('inf'):
        x0 = greedy_construction(num_rows, num_cols, costs, col_covers, row_covers)
        x0 = heuristic_improvement(x0, num_rows, num_cols, costs, col_covers, row_covers)
        cov = compute_coverage(x0, col_covers, num_rows)
        best_x, best_cost, best_viol = x0, compute_cost(x0, costs), compute_violation(cov)

    return best_x, best_cost, best_viol
