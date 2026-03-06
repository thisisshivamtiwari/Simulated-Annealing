"""
Utility functions for Airline Crew Scheduling (Set Partitioning Problem).

Contains: data parser, common utilities, construction heuristics,
pseudo-random initialisation, heuristic improvement, stochastic ranking,
and GA operators.

References:
[1] P.C. Chu and J.E. Beasley, "Constraint handling in genetic algorithms:
    the set partitioning problem", Journal of Heuristics, vol. 4, 1998, 323-357.
[2] T.P. Runarsson and X. Yao, "Stochastic ranking for constrained evolutionary
    optimization", IEEE Trans. Evolutionary Computation, vol. 4, 2000, 284-294.
"""

import random
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
    Phase 5: Cost improvement via column replacement.
    Phase 6: 2-opt cost improvement (swap pairs of columns).
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
# RESULT FORMATTING
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
