import random
from deap import base
from deap import creator
from deap import tools
import numpy as np


def wta_ga_solver(values, p, weapons=None, population_size=40, crossover_probability=0.5, mutation_probability=0.25,
                  generations_qty=5000, tournament_fraction=5, mutation_fraction=10):
    # TODO: Tune parameters to find a relation between problem size and each parameter
    if weapons is None:
        weapons = [1]*len(p)
    num_weapon_types = len(p)
    num_targets = len(p[0])
    num_weapons = sum(weapons)
    adjusted_p = []
    for i in range(len(weapons)):
        for j in range(weapons[i]):
            adjusted_p.append(p[i])

    def evaluate(individual):
        rem_vals = values.copy()
        for weapon in range(num_weapons):
            rem_vals[individual[weapon]] *= 1 - adjusted_p[weapon][individual[weapon]]
        return sum(rem_vals),

    history = tools.History()
    hall_of_fame = tools.HallOfFame(1)
    if hasattr(creator, "FitnessMin"):  # deap doesn't like it when you recreate Creator methods.
        del creator.FitnessMin
    if hasattr(creator, "Individual"):
        del creator.Individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 0, num_targets-1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=num_weapons)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=num_targets-1, indpb=1//mutation_fraction)
    toolbox.register("select", tools.selRoulette)  # tools.selTournament,tournsize=population_size//tournament_fraction)

    toolbox.decorate("mate", history.decorator)
    toolbox.decorate("mutate", history.decorator)

    population = toolbox.population(n=population_size)
    history.update(population)
    fitnesses = map(toolbox.evaluate, population)

    for individual, fitness in zip(population, fitnesses):
        individual.fitness.values = fitness

    for g in range(generations_qty):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_probability:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutation_probability:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for individual, fitness in zip(invalid_ind, fitnesses):
            individual.fitness.values = fitness
        population[:] = offspring
        hall_of_fame.update(population)
    best = hall_of_fame[0]
    assignment_matrix = np.zeros((num_weapon_types, num_targets))
    for i in range(len(best)):
        assignment_matrix[i][best[i]] = 1
    return best.fitness.values[0], assignment_matrix
