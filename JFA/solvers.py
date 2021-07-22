#!/usr/bin/env python3

import heapq
if __package__ is not None and len(__package__) > 0:
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
import functools
import sortedcontainers as sc
import zlib


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
        self._solution = []
        self._parent = None

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

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent=None):
        self._parent = parent
        self._solution = self._parent.solution().copy()
        self._solution.append(list(self.action))

    def solution(self):
        return self._solution

    @property
    def candidate_nodes(self):
        return np.stack(np.where(self.state['Opportunities'][:, :, jf.OpportunityFeatures.SELECTABLE] == True), axis=1)

    def cat_string(self):
        return zlib.compress(
            np.concatenate(
                (np.ravel(self.state['Effectors']),
                 np.ravel(self.state['Targets']),
                 np.ravel(self.state['Opportunities']))
            ).tobytes()
        )


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
        child.parent(node)
        node = child
    return node.g, node.solution()


def greedy(problem):
    env = sim.Simulation(sim.state_to_dict)
    state = env.reset(problem)  # get initial state or load a new problem

    node = Node(sum(state['Targets'][:, jf.TaskFeatures.VALUE]), None, state, 0)
    node = greedy_rec(node, env=env)

    return node.g, node.solution()


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
        child.parent = node
        return child

    once = Node(node.g - reward, action, state, reward, terminal)
    once.parent = node
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
    branch_factor = 0
    duplicate_states = 0
    frontier = sc.SortedList()
    explored = sc.SortedDict()
    frontier.add(node)
    while frontier:
        node = frontier.pop(0)
        if node.cat_string() in explored:
            continue
        # print(f"pulled g: {node.g}, action: {node.action}")
        expansions += 1
        if track_progress:
            print(f"\rExpansions: {expansions}, Duplicates: {duplicate_states}", end="")
        if node.parent is not None:
            if node.terminal is True:
                if node.g == node.parent.g - node.reward:
                    return node.g, node.solution()
                node.g = node.parent.g - node.reward
                frontier.add(node)
                continue
            node.g = node.parent.g - node.reward
            # This may actually not be the optimal.  There may be a more optimal node.
            # If the value is the same, we found it, if the value changes, put it back in the heap.
        state = node.state
        explored[node.cat_string()] = node.g
        for effector, target in np.stack(np.where(state['Opportunities'][:, :, jf.OpportunityFeatures.SELECTABLE] == True), axis=1):
            action = (effector, target)
            new_state, reward, terminal = env.update_state(action, copy.deepcopy(state), smart_search=True)
            g = node.g - reward  # The remaining value after the action taken
            child = Node(g, action, new_state, reward, terminal)
            child.parent = node
            h = heuristic(child)  # The possible remaining value assuming that all of the best actions can be taken
            child.g = g - h
            if child.cat_string() not in explored and child not in frontier:
                frontier.add(child)
                if child.action != node.action and node.action:
                    child.state['Opportunities'][node.action[0], node.action[1], jf.OpportunityFeatures.PSUCCESS] = 0
                    child.state['Opportunities'][node.action[0], node.action[1], jf.OpportunityFeatures.SELECTABLE] = 0
                branch_factor += 1
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

    solvers = [{'name': "AStar", 'function': AStar, 'solve': False},
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
