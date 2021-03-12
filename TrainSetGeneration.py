#!/usr/bin/env python3

import SampleSimulator as Sim
import ProblemGenerators as PG
import JFAFeatures as JF
import JFASolvers as JS
import numpy as np
import time
import csv
import sys
import os
import secrets

TOKEN_LENGTH = 5


def log(string):
    print(f"[{time.asctime()}] {string}")


if __name__ == '__main__':
    effectors = 4
    targets = 8
    quantity = 100
    solve_problems = True
    solvers = [{'name': "Random Choice", 'function': JS.random_solution, 'solve': True},
               {'name': "Greedy", 'function': JS.greedy, 'solve': True},
               {'name': "AStar", 'function': JS.AStar, 'solve': True}]  # AStar should be the last so that its
    # solution get printed
    if len(sys.argv) > 2:
        effectors = int(sys.argv[1])
        targets = int(sys.argv[2])
        if len(sys.argv) > 3:
            quantity = int(sys.argv[3])
            if len(sys.argv) > 4:
                if int(sys.argv[4]) == 0:
                    solve_problems = False
    directory = f"{effectors}x{targets}"
    try:
        os.mkdir(directory)
    except:
        pass
    csv_content = []
    csv_row = ["filename", "total reward"]
    for solver in solvers:
        if solver['solve']:
            csv_row.append(solver['name'])
    csv_row.append("solution")
    csv_content.append(csv_row)
    for i in range(quantity):
        try:
            filename = secrets.token_urlsafe(TOKEN_LENGTH) + ".json"
            simProblem = PG.network_validation(effectors, targets)
            while (np.sum(simProblem['Opportunities'][:,:,JF.OpportunityFeatures.SELECTABLE]) < 1):
                simProblem = PG.network_validation(effectors, targets)
            Sim.saveProblem(simProblem, os.path.join(directory, filename))

            rewards_available = sum(simProblem['Targets'][:, JF.TaskFeatures.VALUE])
            selectable_opportunities = np.sum(simProblem['Opportunities'][:, :, JF.OpportunityFeatures.SELECTABLE])
            log(f"Scenario {filename[:-5]} with {selectable_opportunities} selectable opportunities")

            if solve_problems:
                csv_row = [filename, rewards_available]
                start_time = time.time()
                for solver in solvers:
                    if solver['solve']:
                        g, solution = solver['function'](simProblem)
                        csv_row.append(g)
                end_time = time.time()
                csv_row.append(solution)

                csv_content.append(csv_row)
                log(f"Solved {i+1}/{quantity}: {filename} in: {end_time - start_time:.6f}s")
        except KeyboardInterrupt:
            input("Press Enter to attempt again, or ctrl+c to quit.")
    print()

    if solve_problems:
        csvfilename = os.path.join(directory, f'{time.time()}.csv')
        with open(csvfilename, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(csv_content)
        log(f"solutions exported to {csvfilename}")
