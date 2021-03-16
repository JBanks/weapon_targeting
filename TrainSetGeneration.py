#!/usr/bin/env python3

from . import SampleSimulator as Sim
from . import ProblemGenerators as PG
from . import JFAFeatures as JF
from . import JFASolvers as JS
import numpy as np
import time
import uuid
import base64
import csv
import sys
import re
import os


def log(string):
    print(f"[{time.asctime()}] {string}")


def uuid_url64():
    """Returns a unique, 16 byte, URL safe ID by combining UUID and Base64
    """
    rv = base64.b64encode(uuid.uuid4().bytes).decode('utf-8')
    return re.sub(r'[\=\+\/]', lambda m: {'+': '-', '/': '_', '=': ''}[m.group(0)], rv)

def solver(quantity, directory, start_idx=0, prefix="", suffix="", digits=5, Random=False, Greedy=False, AStar=False, csvfilename=None):

    # ensure that at least one solver was selected in the args
    assert (Random or Greedy or AStar), "At least one solver must be selected. Solvers are Random, Greedy, and AStar"

    # set up the solvers
    solvers = [{'name': "Random Choice", 'function': JS.random_solution, 'solve': Random},
               {'name': "Greedy", 'function': JS.greedy, 'solve': Greedy},
               {'name': "AStar", 'function': JS.AStar, 'solve': AStar}]

    # set up CSV filename if not provided
    if csvfilename is None:
        csvfilename = f'solutions_{prefix}{suffix}'.replace("__", "_")
    # save solutions as CSV file
    csvfilename = os.path.join(directory, f'{csvfilename}.csv')

    # prepare the CSV file
    csv_content = []
    csv_row = ["filename", "total reward"]
    for solver in solvers:
        if solver['solve']:
            csv_row.append(solver['name'])
    csv_row.append("solution")
    with open(csvfilename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(csv_row)



    for i in range(start_idx, start_idx+quantity):
        try:
            # load the appropriate problem from the file
            filename = prefix + str(i).zfill(digits) + suffix + ".json"
            simProblem = PG.loadProblem(os.path.join(directory, filename))

            # determine problem stats
            selectable_opportunities = np.sum(simProblem['Opportunities'][:, :, JF.OpportunityFeatures.SELECTABLE])
            rewards_available = sum(simProblem['Targets'][:, JF.TaskFeatures.VALUE])

            # solve problem using all selected solvers
            csv_row = [filename, rewards_available]
            start_time = time.time()
            for solver in solvers:
                if solver['solve']:
                    g, solution = solver['function'](simProblem)
                    csv_row.append((rewards_available-g))
            end_time = time.time()

            # record the last generated solution (should usually be from A*) in the CSV file
            # this is done at each step to avoid losing data if the solvers hangs on a problem
            csv_row.append(solution)
            with open(csvfilename, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(csv_row)

        # allow keyboard interrupts
        except KeyboardInterrupt:
            input("Press Enter to attempt again, or ctrl+c to quit.")

def generate_dataset(effectors, targets, quantity, solve_problems=False, directory_arg=None, prefix="", suffix="", digits=5):
    """
    Generates a dataset in a given directory with given parameters


    """
    solvers = [{'name': "Random Choice", 'function': JS.random_solution, 'solve': True},
               {'name': "Greedy", 'function': JS.greedy, 'solve': True},
               {'name': "AStar", 'function': JS.AStar, 'solve': True}] #AStar should be the last so that its solution get printed

    #set up the appropriate directory to use
    if directory_arg is not None:
        directory = directory_arg
    else:
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

    for i in range(quantity):
        try:
            filename = prefix + str(i).zfill(digits) + suffix + ".json"
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

if __name__ == '__main__':

    effectors = 4
    targets = 8
    quantity = 100
    solve_problems = True
    directory_arg = None

    if len(sys.argv) > 2:
        effectors = int(sys.argv[1])
        targets = int(sys.argv[2])
        if len(sys.argv) > 3:
            quantity = int(sys.argv[3])
            if len(sys.argv) > 4:
                if int(sys.argv[4]) == 0:
                    solve_problems = False
                    if len(sys.argv) > 5:
                        directory_arg = sys.argv[5]
                        if len(sys.argv) > 6:
                            prefix = sys.argv[6]
                            suffix = sys.argv[7]
                            digits = sys.argv[8]

    generate_dataset(effectors=effectors, targets=targets, quantity=quantity, solve_problem=solve_problems,
                     directory_arg=directory_arg, prefix=prefix, suffix=suffix, digits=digits)