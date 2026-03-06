#!/usr/bin/env python3
"""
Solving Airline Crew Scheduling Problems (Set Partitioning Problem)
using Simulated Annealing (SA), Standard Binary Genetic Algorithm (BGA),
and Improved BGA.

References:
[1] P.C. Chu and J.E. Beasley, "Constraint handling in genetic algorithms:
    the set partitioning problem", Journal of Heuristics, vol. 4, 1998, 323-357.
[2] T.P. Runarsson and X. Yao, "Stochastic ranking for constrained evolutionary
    optimization", IEEE Trans. Evolutionary Computation, vol. 4, 2000, 284-294.

Problems: sppnw41, sppnw42, sppnw43 from OR-Library.
"""

import random
import math
import time
import os
import glob


# =====================================================================
# DATA PARSER
# =====================================================================

def find_file(directory, basename):
    """Find a data file, handling potential trailing spaces in filename."""
    path = os.path.join(directory, basename)
    if os.path.exists(path):
        return path
    path_space = os.path.join(directory, basename + ' ')
    if os.path.exists(path_space):
        return path_space
    pattern = os.path.join(directory, basename.replace('.txt', '') + '*')
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot find {basename} in {directory}")


def parse_spp(filepath):
    """Parse SPP data file in OR-Library format (rows are 0-indexed internally)."""
    with open(filepath, 'r') as f:
        tokens = f.read().split()
    idx = 0
    num_rows = int(tokens[idx]); idx += 1
    num_cols = int(tokens[idx]); idx += 1
    costs = []
    col_covers = []
    for j in range(num_cols):
        cost = int(tokens[idx]); idx += 1
        num_covered = int(tokens[idx]); idx += 1
        rows = []
        for _ in range(num_covered):
            row = int(tokens[idx]) - 1
            idx += 1
            rows.append(row)
        costs.append(cost)
        col_covers.append(rows)
    row_covers = [[] for _ in range(num_rows)]
    for j in range(num_cols):
        for row in col_covers[j]:
            row_covers[row].append(j)
    return num_rows, num_cols, costs, col_covers, row_covers


# =====================================================================
# COMMON UTILITIES
# =====================================================================

def compute_cost(x, costs):
    return sum(costs[j] for j in range(len(x)) if x[j])

def compute_coverage(x, col_covers, num_rows):
    coverage = [0] * num_rows
    for j in range(len(x)):
        if x[j]:
            for r in col_covers[j]:
                coverage[r] += 1
    return coverage

def compute_violation(coverage):
    return sum(abs(c - 1) for c in coverage)

def is_feasible(coverage):
    return all(c == 1 for c in coverage)


# =====================================================================
# CONSTRUCTION HEURISTICS
# =====================================================================

def greedy_construction(num_rows, num_cols, costs, col_covers, row_covers):
    """Greedy set cover: iteratively add most cost-effective column."""
    x = [0] * num_cols
    uncovered = set(range(num_rows))
    while uncovered:
        best_j, best_ratio = -1, float('inf')
        for j in range(num_cols):
            if x[j]:
                continue
            nc = sum(1 for r in col_covers[j] if r in uncovered)
            if nc > 0:
                ratio = costs[j] / nc
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_j = j
        if best_j < 0:
            break
        x[best_j] = 1
        for r in col_covers[best_j]:
            uncovered.discard(r)
    return x


def basic_repair(x, num_rows, num_cols, costs, col_covers, row_covers):
    """
    Basic repair: drop redundant columns (expensive first),
    then add cheapest column for each uncovered row.
    """
    x = x[:]
    coverage = compute_coverage(x, col_covers, num_rows)
    # Drop redundant
    selected = sorted([j for j in range(num_cols) if x[j]], key=lambda j: -costs[j])
    for j in selected:
        if all(coverage[r] > 1 for r in col_covers[j]):
            x[j] = 0
            for r in col_covers[j]:
                coverage[r] -= 1
    # Cover uncovered
    for i in range(num_rows):
        if coverage[i] == 0:
            cands = [j for j in row_covers[i] if not x[j]]
            if cands:
                best = min(cands, key=lambda j: costs[j])
                x[best] = 1
                for r in col_covers[best]:
                    coverage[r] += 1
    return x


# =====================================================================
# PSEUDO-RANDOM INITIALISATION (Algorithm 2, Chu & Beasley)
# =====================================================================

def pseudo_random_init(num_rows, num_cols, costs, col_covers, row_covers):
    """
    Algorithm 2 (Chu & Beasley, p341): Pseudo-random initialisation.
    Iteratively selects an uncovered row at random and adds a column
    covering it via roulette-wheel selection inversely proportional to cost.
    """
    x = [0] * num_cols
    uncovered = set(range(num_rows))
    while uncovered:
        row_i = random.choice(list(uncovered))
        candidates = [j for j in row_covers[row_i] if not x[j]]
        if not candidates:
            uncovered.discard(row_i)
            continue
        weights = [1.0 / costs[j] for j in candidates]
        total_w = sum(weights)
        r = random.random() * total_w
        cumsum = 0.0
        chosen = candidates[-1]
        for k, j in enumerate(candidates):
            cumsum += weights[k]
            if cumsum >= r:
                chosen = j
                break
        x[chosen] = 1
        for row in col_covers[chosen]:
            uncovered.discard(row)
    return x


# =====================================================================
# HEURISTIC IMPROVEMENT (Algorithm 1, Chu & Beasley)
# =====================================================================

def heuristic_improvement(x, num_rows, num_cols, costs, col_covers, row_covers,
                          fix_overcoverage=True):
    """
    Algorithm 1 (Chu & Beasley, p331): Heuristic improvement operator.
    Phase 1: Remove redundant columns (most expensive first).
    Phase 2: Cover uncovered rows (conflict-aware, most constrained first).
    Phase 3: Remove redundant columns again.
    Phase 4 (optional): Fix over-coverage by trying column replacements.
    """
    x = x[:]
    coverage = compute_coverage(x, col_covers, num_rows)

    def drop_redundant():
        sel = sorted([j for j in range(num_cols) if x[j]], key=lambda j: -costs[j])
        for j in sel:
            if all(coverage[r] > 1 for r in col_covers[j]):
                x[j] = 0
                for r in col_covers[j]:
                    coverage[r] -= 1

    def cover_uncovered():
        uncov = [i for i in range(num_rows) if coverage[i] == 0]
        while uncov:
            uncov.sort(key=lambda i: len([j for j in row_covers[i] if not x[j]]))
            ri = uncov[0]
            cands = [j for j in row_covers[ri] if not x[j]]
            if not cands:
                uncov.pop(0)
                continue
            best_j = min(cands, key=lambda j: (
                sum(1 for rr in col_covers[j] if coverage[rr] >= 1),
                costs[j] / max(1, sum(1 for rr in col_covers[j] if coverage[rr] == 0))
            ))
            x[best_j] = 1
            for r in col_covers[best_j]:
                coverage[r] += 1
            uncov = [i for i in range(num_rows) if coverage[i] == 0]

    # Phases 1-3
    drop_redundant()
    cover_uncovered()
    drop_redundant()

    # Phase 4: fix over-coverage
    if fix_overcoverage:
        for _ in range(num_rows * 3):
            cur_viol = compute_violation(coverage)
            if cur_viol == 0:
                break

            selected = [j for j in range(num_cols) if x[j]]
            best_swap = None
            best_viol = cur_viol
            best_cost = compute_cost(x, costs)

            for j in selected:
                if not any(coverage[r] > 1 for r in col_covers[j]):
                    continue

                tx = x[:]
                tc = coverage[:]
                tx[j] = 0
                for r in col_covers[j]:
                    tc[r] -= 1

                for _ in range(num_rows):
                    ur = [r for r in range(num_rows) if tc[r] == 0]
                    if not ur:
                        break
                    r = ur[0]
                    ca = [c for c in row_covers[r] if not tx[c]]
                    if not ca:
                        break
                    bc = min(ca, key=lambda c: (
                        sum(1 for rr in col_covers[c] if tc[rr] >= 1), costs[c]))
                    tx[bc] = 1
                    for rr in col_covers[bc]:
                        tc[rr] += 1

                ts = sorted([jj for jj in range(num_cols) if tx[jj]],
                            key=lambda jj: -costs[jj])
                for jj in ts:
                    if all(tc[r] > 1 for r in col_covers[jj]):
                        tx[jj] = 0
                        for r in col_covers[jj]:
                            tc[r] -= 1

                nv = compute_violation(tc)
                nc = compute_cost(tx, costs)
                if nv < best_viol or (nv == best_viol and nc < best_cost):
                    best_viol = nv
                    best_cost = nc
                    best_swap = (tx[:], tc[:])

            if best_swap and (best_viol < cur_viol or
                    (best_viol == cur_viol and best_cost < compute_cost(x, costs))):
                x[:] = best_swap[0]
                coverage[:] = best_swap[1]
            else:
                break

    # Phase 5: Cost improvement via column replacement (even if already feasible)
    # Try replacing expensive columns with cheaper alternatives
    for _ in range(num_cols):
        cur_viol = compute_violation(coverage)
        cur_cost_val = compute_cost(x, costs)
        selected = sorted([j for j in range(num_cols) if x[j]],
                          key=lambda j: -costs[j])
        improved_cost = False

        for j in selected:
            unique_rows = [r for r in col_covers[j] if coverage[r] == 1]

            if not unique_rows:
                x[j] = 0
                for r in col_covers[j]:
                    coverage[r] -= 1
                improved_cost = True
                break

            tx = x[:]
            tc = coverage[:]
            tx[j] = 0
            for r in col_covers[j]:
                tc[r] -= 1

            repl_cost = 0
            for _ in range(num_rows):
                ur = [r for r in range(num_rows) if tc[r] == 0]
                if not ur:
                    break
                r = ur[0]
                ca = [c for c in row_covers[r] if not tx[c]]
                if not ca:
                    repl_cost = float('inf')
                    break
                bc = min(ca, key=lambda c: (
                    sum(1 for rr in col_covers[c] if tc[rr] >= 1), costs[c]))
                repl_cost += costs[bc]
                if repl_cost >= costs[j]:
                    break
                tx[bc] = 1
                for rr in col_covers[bc]:
                    tc[rr] += 1

            nv = compute_violation(tc)
            nc = compute_cost(tx, costs)
            if nv <= cur_viol and nc < cur_cost_val:
                # Drop any new redundancies
                ts = sorted([jj for jj in range(num_cols) if tx[jj]],
                            key=lambda jj: -costs[jj])
                for jj in ts:
                    if all(tc[r] > 1 for r in col_covers[jj]):
                        tx[jj] = 0
                        for r in col_covers[jj]:
                            tc[r] -= 1
                nv2 = compute_violation(tc)
                nc2 = compute_cost(tx, costs)
                if nv2 <= cur_viol and nc2 < cur_cost_val:
                    x[:] = tx
                    coverage[:] = tc
                    improved_cost = True
                    break

        if not improved_cost:
            break

    # Phase 6: 2-opt cost improvement (swap pairs of columns)
    for _ in range(num_rows * 2):
        cur_viol = compute_violation(coverage)
        cur_cost_val = compute_cost(x, costs)
        selected = sorted([j for j in range(num_cols) if x[j]],
                          key=lambda j: -costs[j])
        improved_2opt = False

        for i1 in range(len(selected)):
            for i2 in range(i1 + 1, len(selected)):
                j1, j2 = selected[i1], selected[i2]

                tx = x[:]
                tc = coverage[:]
                tx[j1] = 0
                tx[j2] = 0
                for r in col_covers[j1]:
                    tc[r] -= 1
                for r in col_covers[j2]:
                    tc[r] -= 1

                # Greedily cover uncovered rows
                repl_cost = 0
                for _ in range(num_rows):
                    ur = [r for r in range(num_rows) if tc[r] == 0]
                    if not ur:
                        break
                    r = ur[0]
                    ca = [c for c in row_covers[r] if not tx[c]]
                    if not ca:
                        repl_cost = float('inf')
                        break
                    bc = min(ca, key=lambda c: costs[c] / max(
                        1, sum(1 for rr in col_covers[c] if tc[rr] == 0)))
                    repl_cost += costs[bc]
                    tx[bc] = 1
                    for rr in col_covers[bc]:
                        tc[rr] += 1

                if repl_cost >= costs[j1] + costs[j2]:
                    continue

                # Drop redundancies
                ts = sorted([jj for jj in range(num_cols) if tx[jj]],
                            key=lambda jj: -costs[jj])
                for jj in ts:
                    if all(tc[r] > 1 for r in col_covers[jj]):
                        tx[jj] = 0
                        for r in col_covers[jj]:
                            tc[r] -= 1

                nv = compute_violation(tc)
                nc = compute_cost(tx, costs)
                if nv <= cur_viol and nc < cur_cost_val:
                    x[:] = tx
                    coverage[:] = tc
                    improved_2opt = True
                    break
            if improved_2opt:
                break
        if not improved_2opt:
            break

    return x


# =====================================================================
# STOCHASTIC RANKING (Runarsson & Yao, 2000)
# =====================================================================

def stochastic_ranking(pop_size, obj_values, violation_values, pf=0.45):
    """Bubble-sort-based stochastic ranking. Returns indices sorted best-first."""
    indices = list(range(pop_size))
    for i in range(pop_size):
        swapped = False
        for j in range(pop_size - 1):
            a, b = indices[j], indices[j + 1]
            if (violation_values[a] == 0 and violation_values[b] == 0) or random.random() < pf:
                if obj_values[a] > obj_values[b]:
                    indices[j], indices[j + 1] = b, a
                    swapped = True
            else:
                if violation_values[a] > violation_values[b]:
                    indices[j], indices[j + 1] = b, a
                    swapped = True
        if not swapped:
            break
    return indices


# =====================================================================
# GA OPERATORS
# =====================================================================

def tournament_select(population, fitnesses, k):
    indices = random.sample(range(len(population)), k)
    winner = min(indices, key=lambda i: fitnesses[i])
    return population[winner][:]

def single_point_crossover(p1, p2):
    n = len(p1)
    pt = random.randint(1, n - 1)
    return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]

def uniform_crossover(p1, p2):
    n = len(p1)
    c1, c2 = [0] * n, [0] * n
    for j in range(n):
        if random.random() < 0.5:
            c1[j], c2[j] = p1[j], p2[j]
        else:
            c1[j], c2[j] = p2[j], p1[j]
    return c1, c2

def bit_flip_mutation(x, rate):
    for j in range(len(x)):
        if random.random() < rate:
            x[j] = 1 - x[j]
    return x


# =====================================================================
# SIMULATED ANNEALING
# =====================================================================

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


# =====================================================================
# STANDARD BINARY GENETIC ALGORITHM
# =====================================================================

def standard_bga(num_rows, num_cols, costs, col_covers, row_covers,
                 pop_size=100, num_generations=1000,
                 crossover_rate=0.8, tournament_size=3, seed=42):
    """Standard BGA with penalty + basic repair on each offspring."""
    random.seed(seed)
    mutation_rate = 2.0 / num_cols
    big_penalty = 50 * max(costs)

    def evaluate(ind):
        cost = compute_cost(ind, costs)
        cov = compute_coverage(ind, col_covers, num_rows)
        viol = compute_violation(cov)
        return cost + big_penalty * viol, cost, viol

    population = []
    for i in range(pop_size):
        if i < pop_size // 3:
            ind = greedy_construction(num_rows, num_cols, costs, col_covers, row_covers)
            ind = bit_flip_mutation(ind[:], mutation_rate * (1 + 4 * i / max(1, pop_size // 3)))
        else:
            ind = [0] * num_cols
            k = random.randint(max(1, num_rows // 3), num_rows * 2)
            cols = random.sample(range(num_cols), min(k, num_cols))
            for j in cols:
                ind[j] = 1
        ind = basic_repair(ind, num_rows, num_cols, costs, col_covers, row_covers)
        population.append(ind)

    evals = [evaluate(ind) for ind in population]
    fitnesses = [e[0] for e in evals]
    costs_l = [e[1] for e in evals]
    viols = [e[2] for e in evals]

    best_x, best_cost, best_viol = None, float('inf'), float('inf')
    for i in range(pop_size):
        if viols[i] == 0 and costs_l[i] < best_cost:
            best_x, best_cost, best_viol = population[i][:], costs_l[i], 0
    if best_x is None:
        idx = min(range(pop_size), key=lambda i: fitnesses[i])
        best_x, best_cost, best_viol = population[idx][:], costs_l[idx], viols[idx]

    for gen in range(num_generations):
        new_pop = [best_x[:]]
        while len(new_pop) < pop_size:
            p1 = tournament_select(population, fitnesses, tournament_size)
            p2 = tournament_select(population, fitnesses, tournament_size)
            if random.random() < crossover_rate:
                c1, c2 = single_point_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]
            c1 = bit_flip_mutation(c1, mutation_rate)
            c2 = bit_flip_mutation(c2, mutation_rate)
            c1 = basic_repair(c1, num_rows, num_cols, costs, col_covers, row_covers)
            c2 = basic_repair(c2, num_rows, num_cols, costs, col_covers, row_covers)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop[:pop_size]
        evals = [evaluate(ind) for ind in population]
        fitnesses = [e[0] for e in evals]
        costs_l = [e[1] for e in evals]
        viols = [e[2] for e in evals]

        for i in range(pop_size):
            if viols[i] == 0 and costs_l[i] < best_cost:
                best_x, best_cost, best_viol = population[i][:], costs_l[i], 0
            elif best_viol > 0 and viols[i] < best_viol:
                best_x, best_cost, best_viol = population[i][:], costs_l[i], viols[i]

    if best_viol > 0:
        best_x = heuristic_improvement(best_x, num_rows, num_cols, costs,
                                       col_covers, row_covers)
        cov = compute_coverage(best_x, col_covers, num_rows)
        best_cost, best_viol = compute_cost(best_x, costs), compute_violation(cov)

    return best_x, best_cost, best_viol


# =====================================================================
# IMPROVED BGA
# =====================================================================

def improved_bga(num_rows, num_cols, costs, col_covers, row_covers,
                 pop_size=100, num_generations=300,
                 crossover_rate=0.8, tournament_size=3, pf=0.45, seed=42):
    """
    Improved BGA with pseudo-random init, stochastic ranking,
    and heuristic improvement. Uses phases 1-3 for offspring
    and full improvement (with Phase 4) for final refinement.
    """
    random.seed(seed)
    mutation_rate = 5.0 / num_cols

    population = []
    for _ in range(pop_size):
        ind = pseudo_random_init(num_rows, num_cols, costs, col_covers, row_covers)
        ind = heuristic_improvement(ind, num_rows, num_cols, costs,
                                    col_covers, row_covers, fix_overcoverage=True)
        population.append(ind)

    obj_values, violation_values = [], []
    for ind in population:
        obj_values.append(compute_cost(ind, costs))
        cov = compute_coverage(ind, col_covers, num_rows)
        violation_values.append(compute_violation(cov))

    best_x, best_cost, best_viol = None, float('inf'), float('inf')
    for i in range(pop_size):
        if violation_values[i] == 0 and obj_values[i] < best_cost:
            best_x, best_cost, best_viol = population[i][:], obj_values[i], 0

    for gen in range(num_generations):
        ranked = stochastic_ranking(pop_size, obj_values, violation_values, pf)
        rank_of = [0] * pop_size
        for rank, idx in enumerate(ranked):
            rank_of[idx] = rank

        new_pop = []
        if best_x is not None:
            new_pop.append(best_x[:])

        while len(new_pop) < pop_size:
            idxs1 = random.sample(range(pop_size), tournament_size)
            p1_idx = min(idxs1, key=lambda i: rank_of[i])
            idxs2 = random.sample(range(pop_size), tournament_size)
            p2_idx = min(idxs2, key=lambda i: rank_of[i])
            p1, p2 = population[p1_idx][:], population[p2_idx][:]

            if random.random() < crossover_rate:
                c1, c2 = uniform_crossover(p1, p2)
            else:
                c1, c2 = p1, p2

            c1 = bit_flip_mutation(c1, mutation_rate)
            c2 = bit_flip_mutation(c2, mutation_rate)

            # Fast improvement (phases 1-3 only) to preserve diversity
            c1 = heuristic_improvement(c1, num_rows, num_cols, costs,
                                       col_covers, row_covers, fix_overcoverage=False)
            c2 = heuristic_improvement(c2, num_rows, num_cols, costs,
                                       col_covers, row_covers, fix_overcoverage=False)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop[:pop_size]

        obj_values, violation_values = [], []
        for ind in population:
            obj_values.append(compute_cost(ind, costs))
            cov = compute_coverage(ind, col_covers, num_rows)
            violation_values.append(compute_violation(cov))

        for i in range(pop_size):
            if violation_values[i] == 0 and obj_values[i] < best_cost:
                best_x, best_cost, best_viol = population[i][:], obj_values[i], 0

    # Final refinement: apply full improvement to top individuals
    final_ranked = stochastic_ranking(pop_size, obj_values, violation_values, pf)
    for idx in final_ranked[:20]:
        refined = heuristic_improvement(population[idx], num_rows, num_cols, costs,
                                        col_covers, row_covers, fix_overcoverage=True)
        cov = compute_coverage(refined, col_covers, num_rows)
        rc, rv = compute_cost(refined, costs), compute_violation(cov)
        if rv == 0 and rc < best_cost:
            best_x, best_cost, best_viol = refined, rc, rv

    if best_x is None:
        idx = min(range(pop_size), key=lambda i: obj_values[i])
        best_x = population[idx][:]
        best_cost, best_viol = obj_values[idx], violation_values[idx]

    return best_x, best_cost, best_viol


# =====================================================================
# MAIN
# =====================================================================

def format_result(label, x, cost, viol, elapsed, optimal, col_covers, num_rows):
    selected = sorted([j + 1 for j in range(len(x)) if x[j]])
    coverage = compute_coverage(x, col_covers, num_rows)
    feasible = is_feasible(coverage)
    gap = (cost - optimal) / optimal * 100 if optimal and feasible else float('nan')
    lines = []
    lines.append(f"  {label}")
    lines.append(f"    Cost       : {cost}")
    lines.append(f"    Feasible   : {feasible}  (violation = {viol})")
    lines.append(f"    Gap to opt : {gap:.2f}%")
    lines.append(f"    Time       : {elapsed:.2f}s")
    lines.append(f"    # columns  : {len(selected)}")
    lines.append(f"    Columns    : {selected}")
    lines.append(f"    Coverage   : {coverage}")
    lines.append("")
    return "\n".join(lines)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    problems = ['sppnw41.txt', 'sppnw42.txt', 'sppnw43.txt']
    optimal_values = {'sppnw41.txt': 6942, 'sppnw42.txt': 8405, 'sppnw43.txt': 8853}

    # Prepare per-algorithm result files
    sa_file = open(os.path.join(results_dir, "simulated_annealing_results.txt"), "w")
    bga_file = open(os.path.join(results_dir, "standard_bga_results.txt"), "w")
    ibga_file = open(os.path.join(results_dir, "improved_bga_results.txt"), "w")

    sa_file.write("=" * 72 + "\n")
    sa_file.write("SIMULATED ANNEALING RESULTS\n")
    sa_file.write("=" * 72 + "\n\n")

    bga_file.write("=" * 72 + "\n")
    bga_file.write("STANDARD BGA RESULTS\n")
    bga_file.write("=" * 72 + "\n\n")

    ibga_file.write("=" * 72 + "\n")
    ibga_file.write("IMPROVED BGA RESULTS\n")
    ibga_file.write("=" * 72 + "\n\n")

    for problem in problems:
        header = "=" * 72 + "\n"
        filepath = find_file(script_dir, problem)
        num_rows, num_cols, costs, col_covers, row_covers = parse_spp(filepath)
        opt = optimal_values.get(problem, 0)
        prob_header = f"Problem: {problem}  ({num_rows} rows, {num_cols} columns, optimal = {opt})\n"
        print(header + prob_header + header)

        # --- Simulated Annealing ---
        t0 = time.time()
        sa_x, sa_cost, sa_viol = simulated_annealing(
            num_rows, num_cols, costs, col_covers, row_covers)
        t1 = time.time()
        sa_result = format_result("Simulated Annealing", sa_x, sa_cost, sa_viol,
                                   t1 - t0, opt, col_covers, num_rows)
        print(sa_result)
        sa_file.write(f"Problem: {problem}  ({num_rows} rows, {num_cols} cols, optimal = {opt})\n")
        sa_file.write("-" * 72 + "\n")
        sa_file.write(sa_result + "\n")

        # --- Standard BGA ---
        t0 = time.time()
        bga_x, bga_cost, bga_viol = standard_bga(
            num_rows, num_cols, costs, col_covers, row_covers)
        t1 = time.time()
        bga_result = format_result("Standard BGA", bga_x, bga_cost, bga_viol,
                                    t1 - t0, opt, col_covers, num_rows)
        print(bga_result)
        bga_file.write(f"Problem: {problem}  ({num_rows} rows, {num_cols} cols, optimal = {opt})\n")
        bga_file.write("-" * 72 + "\n")
        bga_file.write(bga_result + "\n")

        # --- Improved BGA ---
        t0 = time.time()
        ibga_x, ibga_cost, ibga_viol = improved_bga(
            num_rows, num_cols, costs, col_covers, row_covers)
        t1 = time.time()
        ibga_result = format_result("Improved BGA", ibga_x, ibga_cost, ibga_viol,
                                     t1 - t0, opt, col_covers, num_rows)
        print(ibga_result)
        ibga_file.write(f"Problem: {problem}  ({num_rows} rows, {num_cols} cols, optimal = {opt})\n")
        ibga_file.write("-" * 72 + "\n")
        ibga_file.write(ibga_result + "\n")

    sa_file.close()
    bga_file.close()
    ibga_file.close()
    print(f"Results saved to: {results_dir}/")
    print(f"  - simulated_annealing_results.txt")
    print(f"  - standard_bga_results.txt")
    print(f"  - improved_bga_results.txt")


if __name__ == '__main__':
    main()
