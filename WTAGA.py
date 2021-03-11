import random
from deap import base
from deap import creator
from deap import tools
import numpy as np
# import matplotlib.pyplot as plt
# import networkx


def wta_ga_solver(values, p, weapons=None):
    if weapons is None:
        weapons = [1]*len(p)
    num_weapon_types = len(p)
    num_targets = len(p[0])
    num_weapons = sum(weapons)
    adjusted_p = []
    for i in range(len(weapons)):
        for j in range(weapons[i]):
            adjusted_p.append(p[i])
    WEAPONS = num_weapons
    TARGETS = num_targets
    POPULATION_SIZE = 40
    CXPB, MUTPB, NGEN = 0.5, 0.25, 5000

    def evaluate(individual):
        rem_vals = values.copy()
        for weapon in range(WEAPONS):
            rem_vals[individual[weapon]] *= 1 - adjusted_p[weapon][individual[weapon]]
        return sum(rem_vals),

    history = tools.History()
    hall_of_fame = tools.HallOfFame(1)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 0, TARGETS-1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=WEAPONS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=TARGETS-1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=POPULATION_SIZE//5)

    toolbox.decorate("mate", history.decorator)
    toolbox.decorate("mutate", history.decorator)

    population = toolbox.population(n=POPULATION_SIZE)
    history.update(population)
    fitnesses = map(toolbox.evaluate, population)

    for individual, fitness in zip(population, fitnesses):
        individual.fitness.values = fitness

    for g in range(NGEN):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for individual, fitness in zip(invalid_ind, fitnesses):
            individual.fitness.values = fitness
        population[:] = offspring
        hall_of_fame.update(population)
    best = hall_of_fame[0]
    print(f"hall of fame: {hall_of_fame}: {hall_of_fame[0].fitness}")
    assignment_matrix = np.zeros((num_weapon_types, num_targets))
    for i in range(len(best)):
        assignment_matrix[i][best[i]] = 1
    return assignment_matrix


def main():
    values = [5, 10, 20]
    weapons = [5, 2, 1]
    p = [
        [0.3, 0.2, 0.5],
        [0.1, 0.6, 0.5],
        [0.4, 0.5, 0.4]
    ]

    values = [76, 93, 91, 85, 55, 29, 66, 53]
    weapons = [1, 1, 1, 1, 1, 1, 1, 1]
    p = [
        [0.706943154335022, 0.6360183954238892, 0.8114644885063171, 0.7272420525550842, 0.7375414967536926,
         0.6824339032173157, 0.6231933832168579, 0.6588988304138184],
        [0.8205441236495972, 0.8226460814476013, 0.6846358776092529, 0.6808983087539673, 0.7649776935577393,
         0.6295098662376404, 0.6345000863075256, 0.7816135287284851],
        [0.8206263184547424, 0.6674816012382507, 0.6290864944458008, 0.6989136934280396, 0.7409589886665344,
         0.7029198408126831, 0.6802255511283875, 0.8629313111305237],
        [0.6121272444725037, 0.6316321492195129, 0.6599464416503906, 0.826388955116272, 0.8786106705665588,
         0.7138409614562988, 0.6111229658126831, 0.7680585980415344],
        [0.6966612935066223, 0.8372161984443665, 0.6774555444717407, 0.6943622827529907, 0.7386593818664551,
         0.6733441352844238, 0.8375512957572937, 0.8438227772712708],
        [0.7155847549438477, 0.7222493886947632, 0.6354673504829407, 0.6722898483276367, 0.6532123684883118,
         0.6016350388526917, 0.7587482929229736, 0.8896375298500061],
        [0.8869679570198059, 0.8523036241531372, 0.7461298108100891, 0.776040256023407, 0.7137162685394287,
         0.8593636751174927, 0.7626805305480957, 0.6583917737007141],
        [0.7424099445343018, 0.6186914443969727, 0.6806374192237854, 0.837059736251831, 0.6029490232467651,
         0.8164913058280945, 0.7150862812995911, 0.8988319635391235]
    ]
    print(wta_ga_solver(values, p))


if __name__ == "__main__":
    main()
