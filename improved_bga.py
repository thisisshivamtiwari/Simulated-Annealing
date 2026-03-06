"""
Improved Binary Genetic Algorithm (BGA) for the Set Partitioning Problem.

Uses pseudo-random initialisation (Algorithm 2), stochastic ranking
(Runarsson & Yao), uniform crossover, and heuristic improvement (Algorithm 1).
"""

import random
import time

from utils import (
    compute_cost, compute_coverage, compute_violation,
    pseudo_random_init, heuristic_improvement, stochastic_ranking,
    uniform_crossover, bit_flip_mutation
)


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
