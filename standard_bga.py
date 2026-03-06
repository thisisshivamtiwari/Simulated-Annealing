"""
Standard Binary Genetic Algorithm (BGA) for the Set Partitioning Problem.

Uses tournament selection, single-point crossover, bit-flip mutation,
basic repair, and penalty-based fitness.
"""

import random
import time

from utils import (
    compute_cost, compute_coverage, compute_violation,
    greedy_construction, basic_repair, heuristic_improvement,
    tournament_select, single_point_crossover, bit_flip_mutation
)


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
