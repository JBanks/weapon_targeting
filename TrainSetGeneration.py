#!/usr/bin/env python3

if __package__ is not None and len(__package__) > 0:
    print(f"{__name__} using relative import inside of {__package__}")
    from . import SampleSimulator as Sim
    from . import JFAFeatures as JF
    from . import ProblemGenerators as PG
    from . import JFASolvers as JS
    from . import JFAGA as JG
else:
    import SampleSimulator as Sim
    import JFAFeatures as JF
    import ProblemGenerators as PG
    import JFASolvers as JS
    import JFAGA as JG
import numpy as np
import time
import csv
import os
import argparse
from functools import partial

TOKEN_LENGTH = 5


def log(string):
    print(f"[{time.asctime()}] {string}")


if __name__ == '__main__':
    solvers = [{'name': "Random Choice", 'function': JS.random_solution, 'solve': False},
               {'name': "GA",
                'function': partial(JG.jfa_ga_solver, population_size=240, generations_qty=15000),
                'solve': True},
               {'name': "Greedy", 'function': JS.greedy, 'solve': True},
               {'name': "AStar", 'function': JS.AStar, 'solve': False}]
    # AStar should be the last solver so that its solution get printed
    parser = argparse.ArgumentParser()
    parser.add_argument('--effectors', type=int, help="The number of effectors in each problem", default=3,
                        required=False)
    parser.add_argument('--targets', type=int, help="The number of targets in each problem", default=9, required=False)
    parser.add_argument('--quantity', type=int, help="The number of problems of each size", default=100, required=False)
    parser.add_argument('--offset', type=int, help="Numbering offset for scenarios", default=0, required=False)
    parser.add_argument('--solve', type=bool, help="Whether or not we will solve the problems", default=True,
                        required=False)
    args = parser.parse_args()
    effectors = args.effectors
    targets = args.targets
    num_problems = args.quantity
    numbering_offset = args.offset
    solve_problems = args.solve
    # directory = f"{effectors}x{targets}"
    directory = os.path.join(f"JFA Validation Datasets for DRDC Slides", f"JFA {effectors}x{targets} Validation Set")
    try:
        os.mkdir(directory)
    except Exception as error:
        print(f"Error: {error}")
    csv_content = []
    csv_row = ["filename", "total reward"]
    for solver in solvers:
        if solver['solve']:
            csv_row.append(solver['name'])
    csv_row.append("solution")
    csv_content.append(csv_row)
    for i in range(numbering_offset, num_problems + numbering_offset):
        try:
            identifier = f"validation_{i:05d}_{effectors}x{targets}"
            filename = identifier + ".json"
            filepath = os.path.join(directory, filename)
            if os.path.exists(filepath):
                simProblem = Sim.loadProblem(filepath)
            else:
                simProblem = PG.network_validation(effectors, targets)
                while np.sum(simProblem['Opportunities'][:, :, JF.OpportunityFeatures.SELECTABLE]) < 1:
                    simProblem = PG.network_validation(effectors, targets)
                Sim.saveProblem(simProblem, filepath)

            rewards_available = sum(simProblem['Targets'][:, JF.TaskFeatures.VALUE])
            selectable_opportunities = np.sum(simProblem['Opportunities'][:, :, JF.OpportunityFeatures.SELECTABLE])
            log(f"Scenario {filename[:-5]} with {selectable_opportunities} selectable opportunities")

            if solve_problems:
                csv_row = [filename, rewards_available]
                start_time = time.time()
                solution = []
                for solver in solvers:
                    if solver['solve']:
                        g, solution = solver['function'](simProblem)
                        csv_row.append(g)
                end_time = time.time()
                csv_row.append(solution)

                csv_content.append(csv_row)
                log(f"Solved {i+1}/{num_problems+numbering_offset}: {filename} in: {end_time - start_time:.6f}s")
        except KeyboardInterrupt:
            input("Press Enter to attempt again, or ctrl+c to quit.")
    print()

    if solve_problems:
        csv_filename = os.path.join(directory, f'{time.time()}.csv')
        with open(csv_filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(csv_content)
        log(f"solutions exported to {csv_filename}")
