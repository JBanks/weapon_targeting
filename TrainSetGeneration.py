#!/usr/bin/env python3

import SampleSimulator as Sim
import ProblemGenerators as PG
import JFAFeatures as JF
import AStarJFA as AS
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

def find_TSP(problem, max_single=4):
    """
    If there are at least 5 targets where only 1 effector can act on those targets
    and if that effector has enough ammunition to act on all of the targets, we likely have
    a TSP Problem that cannot be solved by the AStar Heuristic.
    """
    solo = [0] * len(problem['Effectors'])
    highest = [0] * len(problem['Effectors'])
    for j in range(len(problem['Targets'])):
        if np.sum(problem['Opportunities'][:,j,JF.OpportunityFeatures.SELECTABLE]) == np.max(problem['Opportunities'][:,j,JF.OpportunityFeatures.SELECTABLE]):
            index = np.argmax(problem['Opportunities'][:,j,JF.OpportunityFeatures.SELECTABLE])
            solo[index] += 1
        index = np.argmax(problem['Opportunities'][:,j,JF.OpportunityFeatures.PSUCCESS])
        highest[index] += 1
    if max(solo) > max_single:
        index = np.argmax(solo)
        if problem['Effectors'][index][JF.EffectorFeatures.AMMORATE] < 1/max_single:
            print(f"Effector {index} is acting alone on {solo[index]} targets")
            return True
    if max(highest) > max_single:
        index = np.argmax(highest)
        if problem['Effectors'][index][JF.EffectorFeatures.AMMORATE] < 1/max_single:
            print(f"Effector {index} is the best effector on {highest[index]} targets")
            return True
    return False 

def uuid_url64():
    """Returns a unique, 16 byte, URL safe ID by combining UUID and Base64
    """
    rv = base64.b64encode(uuid.uuid4().bytes).decode('utf-8')
    return re.sub(r'[\=\+\/]', lambda m: {'+': '-', '/': '_', '=': ''}[m.group(0)], rv)

if __name__ == '__main__':
    effectors = 4
    targets = 8
    quantity = 100
    solve_problems = True
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
    env = Sim.Simulation(Sim.state_to_dict)
    csv_content = []
    for i in range(quantity):
        try:
            start_time = time.time()
            filename = uuid_url64() + ".json"
            simProblem = PG.network_validation(effectors, targets)
            while (np.sum(simProblem['Opportunities'][:,:,JF.OpportunityFeatures.SELECTABLE]) < 1):
                simProblem = PG.network_validation(effectors, targets)
            #while find_TSP(simProblem):
            #    print("Travelling Salesman Problem found.  We cannot validate this with AStar, so we are generating a new problem.")
            #    simProblem = PG.network_validation(effectors, targets)
            Sim.saveProblem(simProblem, os.path.join(directory, filename))

            state = env.reset(simProblem)  # get initial state or load a new problem
            rewards_available = sum(state['Targets'][:, JF.TaskFeatures.VALUE])
            selectable_opportunities = np.sum(state['Opportunities'][:,:,JF.OpportunityFeatures.SELECTABLE])
            log(f"Scenario {filename[:-5]} with {selectable_opportunities} selectable opportunities")
            if solve_problems:
                solution, g, expansions, branchFactor = AS.AStar(state, enviro = env)
                csv_content.append([filename, g, rewards_available, solution])
                end_time = time.time()
                log(f"AStar solved {filename} in: {end_time - start_time:.6f}s")
        except KeyboardInterrupt:
            input("Press Enter to attempt again, or ctrl+c to quit.")
    print()

    if solve_problems:
        csvfilename = os.path.join(directory, f'{time.time()}.csv')
        with open(csvfilename, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(csv_content)
        log(f"solutions exported to {csvfilename}")
