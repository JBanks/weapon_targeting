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

def uuid_url64():
    """Returns a unique, 16 byte, URL safe ID by combining UUID and Base64
    """
    rv = base64.b64encode(uuid.uuid4().bytes).decode('utf-8')
    return re.sub(r'[\=\+\/]', lambda m: {'+': '-', '/': '_', '=': ''}[m.group(0)], rv)

if __name__ == '__main__':
    effectors = 4
    targets = 8
    quantity = 100
    if len(sys.argv) > 2:
        effectors = int(sys.argv[1])
        targets = int(sys.argv[2])
        if len(sys.argv) > 3:
            quantity = int(sys.argv[3])
    directory = f"{effectors}x{targets}"
    try:
        os.mkdir(directory)
    except:
        pass
    env = Sim.Simulation(Sim.state_to_dict)
    csv_content = []
    for i in range(quantity):
        start_time = time.time()
        filename = uuid_url64() + ".json"
        simProblem = PG.network_validation(effectors, targets)
        Sim.saveProblem(simProblem, os.path.join(directory, filename))

        state = env.reset(simProblem)  # get initial state or load a new problem
        rewards_available = sum(state['Targets'][:, JF.TaskFeatures.VALUE])
        selectable_opportunities = np.sum(state['Opportunities'][:,:,JF.OpportunityFeatures.SELECTABLE])
        print(f"Scenario {filename[:-5]} with {selectable_opportunities} selectable opportunities")
        #solution, g, expansions, branchFactor = AS.AStar(state, enviro = env)
        #csv_content.append([filename, g, rewards_available, solution])
        #end_time = time.time()
        #print(f"AStar solved {filename} in: {end_time - start_time:.6f}s")
    #print()

    #with open(os.path.join(directory, f'{time.time()}.csv'), 'wb') as f:
    #    writer = csv.writer(f)
    #    writer.writerows(csv_content)
