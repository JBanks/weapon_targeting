#!/usr/bin/env python3

import heapq
import SampleSimulator as Sim
import ProblemGenerators as PG
import JFAFeatures as JF
import numpy as np
import copy

class Node:
    def __init__(self, g, action, state, reward, terminal = False):
        self.g = g
        self.action = action
        self.state = state
        self.reward = reward
        self.terminal = terminal

    def __eq__(self, other):
        if type(other) == type(self):
            if (self.state['Effectors'] != other.state['Effectors']).any():
                return False
            if (self.state['Targets'] != other.state['Targets']).any():
                return False
            if (self.state['Opportunities'] != other.state['Opportunities']).any():
                return False
            return True
        else:
            if (self.state['Effectors'] != other['Effectors']).any():
                return False
            if (self.state['Targets'] != other['Targets']).any():
                return False
            if (self.state['Opportunities'] != other['Opportunities']).any():
                return False
            return True

    def __lt__(self, other):
        if type(other) == type(self):
            return self.g < other.g
        else:
            return self.g < other

    def __gt__(self, other):
        if type(other) == type(self):
            return self.g > other.g
        else:
            return self.g > other

    def Parent(self, parent):
        self.parent = parent

    def Solution(self):
        solution = ''
        if self.action == None:
            return solution
        if hasattr(self, 'parent'):
            solution += self.parent.Solution() + ','
        return solution + f"{self.action}"

"""
This is an A* implementation to search for a solution to a given JFA problem.
"""
def AStar(state):
    """
    This heuristic is naive and assumes that we can get all of the rewards available to an effector.
    This will extend the search space significantly, but should guarantee the optimal solution.
    The heuristic is optimistic, and assumes that the same effector can more targets than it is capable of.
    """
    def heuristic(state):
        remaining_reward = 0
        for i, target in enumerate(state['Targets']):
            if target[JF.TaskFeatures.SELECTED] == 1:
                continue
            remaining_moves = int(min((1 - target[JF.TaskFeatures.SELECTED]) * 2, sum(state['Opportunities'][:, i, JF.OpportunityFeatures.SELECTABLE])))
            if not remaining_moves:
                continue
            value = target[JF.TaskFeatures.VALUE]
            top = np.argpartition(state['Opportunities'][:, i, JF.OpportunityFeatures.PSUCCESS], remaining_moves)[:remaining_moves]
            for effector in top:
                remaining_reward += value * state['Opportunities'][effector, i, JF.OpportunityFeatures.PSUCCESS]
                value *= (1 - state['Opportunities'][effector, i, JF.OpportunityFeatures.PSUCCESS])
        return remaining_reward

    node = Node(sum(state['Targets'][:,JF.TaskFeatures.VALUE]), None, state, 0)
    expansions = 0
    branchFactor = 0
    frontier = []
    explored = []
    heapq.heappush(frontier, node)
    while frontier:
        node = heapq.heappop(frontier)
        if node in explored:
            continue
        expansions += 1
        if hasattr(node, 'parent'):
            node.g = node.parent.g - node.reward
        if node.terminal == True:
            return node.Solution(), node.g, expansions, branchFactor
        explored.append(node)
        state = node.state
        for effector, target in np.stack(np.where(state['Opportunities'][:, :, JF.OpportunityFeatures.SELECTABLE] == True), axis = 1):
            action = (effector, target)
            new_state, reward, terminal = env.update_state(copy.deepcopy(state), action)
            g = node.g - reward
            h = heuristic(new_state)
            child = Node(g + h, action, new_state, reward, terminal)
            child.Parent(node)
            branchFactor += 1
            print(f"checking action: {action}")
            if child not in explored and child not in frontier:
                print(f"state not found.  g: {g} reward: {reward} prev action {node.action}")
                heapq.heappush(frontier, child)
            elif child in frontier and child.g < frontier[frontier.index(child)].g:
                print(f"state updated")
                heapq.heappush(frontier, child)
    return "failed", None, None, None

if __name__ == '__main__':
    env = Sim.Simulation(Sim.state_to_dict)
    simProblem = PG.Tiny()
    state = env.reset(simProblem) #get initial state or load a new problem
    Sim.printState(Sim.MergeState(env.effectorData, env.taskData, env.opportunityData))
    rewards_available = sum(state['Targets'][:,JF.TaskFeatures.VALUE])
    solution, g, expansions, branchFactor = AStar(state)
    Sim.printState(Sim.MergeState(env.effectorData, env.taskData, env.opportunityData)) 
    print(f"Expansions: {expansions}, branchFactor: {branchFactor}, Reward: {g} / {rewards_available}, Steps: {solution}")
