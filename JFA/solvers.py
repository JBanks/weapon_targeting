#!/usr/bin/env python3

import heapq
if __package__ is not None and len(__package__) > 0:
    print(f"{__name__} using relative import inside of {__package__}")
    from . import simulator as sim
    from . import features as jf
    from . import problem_generators as pg
else:
    import simulator as sim
    import features as jf
    import problem_generators as pg
import numpy as np
import copy
import time
import sys


class Node:
    """
    This node class is used to simplify storage and comparison between different
    states that are reached by the search algorithm.
    """
    def __init__(self, g, action, state, reward, terminal=False):
        self.g = g
        self.action = action
        self.state = state
        self.reward = reward
        self.terminal = terminal
        self.solution = []

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
        self.solution = self.parent.solution().copy()
        self.solution.append(list(self.action))

    def Solution(self):
        return self.solution


def random_solution(problem):
    env = sim.Simulation(sim.state_to_dict)
    state = env.reset(problem)
    node = Node(sum(state['Targets'][:, jf.TaskFeatures.VALUE]), None, state, 0)
    while not node.terminal:
        options = np.asarray(np.where(state['Opportunities'][:, :, jf.OpportunityFeatures.PSUCCESS] > 0)).transpose()
        action_index = np.random.choice(len(options))
        action = tuple(options[action_index])
        state, reward, terminal = env.update_state(action, copy.deepcopy(state))
        child = Node(node.g - reward, action, state, reward, terminal)
        child.Parent(node)
        node = child
    return node.g, node.Solution()


def greedy(problem):
    env = sim.Simulation(sim.state_to_dict)
    state = env.reset(problem)  # get initial state or load a new problem

    node = Node(sum(state['Targets'][:, jf.TaskFeatures.VALUE]), None, state, 0)
    node = greedy_rec(node, env=env)

    return sum(state['Targets'][:, jf.TaskFeatures.VALUE]) - node.g, node.Solution()


def greedy_rec(node, env=None):
    """
    This gives a baseline of choosing the best option first.
    """
    state = node.state

    pSuccesses = state['Opportunities'][:, :, jf.OpportunityFeatures.PSUCCESS]
    values = state['Targets'][:, jf.TaskFeatures.VALUE]
    selectable = state['Opportunities'][:, :, jf.OpportunityFeatures.SELECTABLE]
    action = np.unravel_index(np.argmax(pSuccesses * values * selectable), pSuccesses.shape)

    state, reward, terminal = env.update_state(action, copy.deepcopy(state))
    if terminal:
        child = Node(node.g - reward, action, state, reward, terminal)
        child.Parent(node)
        return child

    once = Node(node.g - reward, action, state, reward, terminal)
    once.Parent(node)
    once = greedy_rec(once, env)

    return once


def astar_heuristic(node):
    """
    This heuristic is naive and assumes that we can get all of the rewards available to an effector.
    This will extend the search space significantly, but should guarantee the optimal solution.
    The heuristic is optimistic, and assumes that the same effector can more targets than it is capable of.
    """
    state = node.state
    remaining_reward = 0
    opportunities = state['Opportunities'][:, :, :].copy()
    for j, target in enumerate(state['Targets']):
        if target[jf.TaskFeatures.SELECTED] == 1:
            continue  # This task has already been selected the maximum number of times.
        for i in range(len(opportunities)):
            if [i, j] in node.solution and node.solution[-1] != [i, j]:
                opportunities[i, j, jf.OpportunityFeatures.PSUCCESS] = 0.00000001
        remaining_moves = int(min((1 - target[jf.TaskFeatures.SELECTED]) * 2,
                                  sum(opportunities[:, j, jf.OpportunityFeatures.SELECTABLE] * 2)))
        if not remaining_moves:
            continue  # The minimum between the number of hits left on a target, and eligible effectors is zero.
        value = target[jf.TaskFeatures.VALUE]
        top = np.argpartition(opportunities[:, j, jf.OpportunityFeatures.PSUCCESS], -1)[
               -1:]  # select the top 'n' effectors.
        for move in range(remaining_moves):
            reward = value * opportunities[top[0], j, jf.OpportunityFeatures.PSUCCESS]
            remaining_reward += reward
            value -= reward
    return remaining_reward  # Return the remaining reward if all moves were possible.


def AStar(problem, heuristic=astar_heuristic, track_progress=False):
    """
    This is an A* implementation to search for a solution to a given JFA problem.
    """
    env = sim.Simulation(sim.state_to_dict)
    state = env.reset(problem)  # get initial state or load a new problem
    node = Node(sum(state['Targets'][:, jf.TaskFeatures.VALUE]), None, state, 0)
    expansions = 0
    branchFactor = 0
    duplicate_states = 0
    frontier = []
    explored = []
    heapq.heappush(frontier, node)
    while frontier:
        node = heapq.heappop(frontier)
        if node in explored:
            continue
        # print(f"pulled g: {node.g}, action: {node.action}")
        expansions += 1
        if track_progress:
            print(f"\rExpansions: {expansions}, Duplicates: {duplicate_states}", end="")
        if hasattr(node, 'parent'):
            if node.terminal is True:
                if node.g == node.parent.g - node.reward:
                    return node.g, node.solution()
                node.g = node.parent.g - node.reward
                heapq.heappush(frontier, node)
                continue
            node.g = node.parent.g - node.reward
            # This may actually not be the optimal.  There may be a more optimal node.
            # If the value is the same, we found it, if the value changes, put it back in the heap.
        explored.append(node)
        state = node.state
        for effector, target in np.stack(
                np.where(state['Opportunities'][:, :, jf.OpportunityFeatures.SELECTABLE] == True), axis=1):
            action = (effector, target)
            new_state, reward, terminal = env.update_state(action, copy.deepcopy(state))
            g = node.g - reward  # The remaining value after the action taken
            child = Node(g, action, new_state, reward, terminal)
            child.Parent(node)
            h = heuristic(child)  # The possible remaining value assuming that all of the best actions can be taken
            child.g = g - h
            if child not in explored and child not in frontier:
                heapq.heappush(frontier, child)
                branchFactor += 1
            else:
                duplicate_states += 1
    return None, "failed"


def ucs_heuristic(state):
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        simProblem = pg.loadProblem(filename)
    else:
        simProblem = pg.network_validation(20, 200)
    print(f"Problem size: {(len(simProblem['Effectors']), len(simProblem['Targets']))}")
    sim.printState(sim.mergeState(simProblem['Effectors'], simProblem['Targets'], simProblem['Opportunities']))
    rewards_available = sum(simProblem['Targets'][:, jf.TaskFeatures.VALUE])

    solvers = [{'name': "AStar", 'function': AStar, 'solve': True},
               {'name': "Greedy", 'function': greedy, 'solve': True},
               {'name': "Random Choice", 'function': random_solution, 'solve': False}]

    for solver in solvers:
        if solver['solve']:
            print(solver['name'])
            start_time = time.time()
            g, solution = solver['function'](simProblem)
            end_time = time.time()
            print(f"{solver['name']} solved in {end_time - start_time}s, reward left: {g} /"
                  f" {rewards_available}, steps: {solution}")
