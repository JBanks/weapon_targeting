import numpy as np
import copy
import heapq


class Environment:
    def __init__(self):
        pass

    @staticmethod
    def update_state(action, state):
        weapon, target = action
        reward = state['p'][weapon, target] * state['v'][target]
        new_state = state.copy()
        new_state['v'] *= (1 - state['p'][weapon, target])
        new_state['p'][weapon, :] = 0
        new_state['assignment'][weapon] = target
        terminal = (new_state['assignment'] == -1).any()
        return new_state, reward, terminal


class WTA_Node:
    def __init__(self, g, action, state, reward, terminal=False):
        """

        @param g:
        @param action:
        @param state:
        @param reward:
        @param terminal:
        """
        self.g = g
        self.action = action
        self.state = state
        self.reward = reward
        self.terminal = terminal
        self._solution = []
        self._parent = None

    def __eq__(self, other):
        """

        @param other:
        @return:
        """
        if type(other) == type(self):
            if (self.state['assignment'] != other.state['assignment']).any():
                return False
            return True
        else:
            if (self.state['assignment'] != other['assignment']).any():
                return False
            return True

    def __lt__(self, other):
        """

        @param other:
        @return:
        """
        if type(other) == type(self):
            return self.g < other.g
        else:
            return self.g < other

    def __gt__(self, other):
        """

        @param other:
        @return:
        """
        if type(other) == type(self):
            return self.g > other.g
        else:
            return self.g > other

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        """

        @param parent:
        @return:
        """
        self._parent = parent
        self._solution = self.parent.solution().copy()
        self._solution.append(list(self.action))

    @property
    def solution(self):
        """

        @return:
        """
        return self._solution


def astar_heuristic(node):
    """

    @param node:
    @return:
    """
    state = node.state
    remaining_reward = 0
    opportunities = state['Opportunities'][:, :, :].copy()
    for j, target in enumerate(state['Targets']):
        if target[JF.TaskFeatures.SELECTED] == 1:
            continue  # This task has already been selected the maximum number of times.
        for i in range(len(opportunities)):
            if [i, j] in node.solution and node.solution[-1] != [i, j]:
                opportunities[i, j, JF.OpportunityFeatures.PSUCCESS] = 0.00000001
        remaining_moves = int(min((1 - target[JF.TaskFeatures.SELECTED]) * 2,
                                  sum(opportunities[:, j, JF.OpportunityFeatures.SELECTABLE] * 2)))
        if not remaining_moves:
            continue  # The minimum between the number of hits left on a target, and eligible effectors is zero.
        value = target[JF.TaskFeatures.VALUE]
        top = np.argpartition(opportunities[:, j, JF.OpportunityFeatures.PSUCCESS], -1)[
               -1:]  # select the top 'n' effectors.
        for move in range(remaining_moves):
            reward = value * opportunities[top[0], j, JF.OpportunityFeatures.PSUCCESS]
            remaining_reward += reward
            value -= reward
    return remaining_reward  # Return the remaining reward if all moves were possible.


def wta_mmr_heuristic(node):
    """
    Perform MMR on our BB node.
    @param node:
    @return:
    """
    state = node.state
    remaining_reward = 0
    partial_solution = node.solution
    v = state['v'].copy()
    maximum_p = np.zeros(state['p'].shape) + np.max(state['p'], axis=0)  # Give all weapons the maximum probability
    for assignment in partial_solution:  # Account for all actions up to this point
        v[assignment[1]] *= (1 - state['p'][assignment])  # Vⱼ Πᵢ qᵢⱼ
        maximum_p[assignment[0], :] = 0
    return greedy_heuristic(v, maximum_p)  # Return the remaining reward from MMR


def greedy_heuristic(v=None, p=None, partial_solution=None):
    """
    Uses a greedy heuristic to find the next-best step in the solution.
    @param v: Accepts a different v list for use in other heuristic algorithms
    @param p: Accepts a different p matrix for use in other heuristic algorithms
    @param partial_solution: A partial solution, allowing us to pass in the solution from a node part-way down the
                            path
    @return: either the solution with the remaining value, or just the solution
    """
    if partial_solution is None:
        partial_solution = []
    solution = []
    for assignment in partial_solution:  # Account for all actions up to this point
        solution.append(assignment)
        v[assignment[1]] *= (1 - p[assignment])  # Vⱼ Πᵢ qᵢⱼ
        p[assignment[0], :] = 0
    while (p > 0).any():
        assignment = np.unravel_index(np.argmax(p * v), p.shape)
        solution.append(assignment)
        v[assignment[1]] *= (1 - p[assignment])
        p[assignment[0], :] = 0
    return sum(v)


def AStar(problem, heuristic=astar_heuristic, Node=WTA_Node, track_progress=False):
    """

    @param problem:
    @param heuristic:
    @param track_progress:
    @return:
    """
    env = Environment()
    node = Node(sum(problem['v']), None, problem, 0)
    expansions = 0
    branch_factor = 0
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
                np.where(state['Opportunities'][:, :, JF.OpportunityFeatures.SELECTABLE] == True), axis=1):
            action = (effector, target)
            new_state, reward, terminal = env.update_state(action, copy.deepcopy(state))
            g = node.g - reward  # The remaining value after the action taken
            child = Node(g, action, new_state, reward, terminal)
            child.parent(node)
            h = heuristic(child)  # The possible remaining value assuming that all of the best actions can be taken
            child.g = g - h
            if child not in explored and child not in frontier:
                heapq.heappush(frontier, child)
                branch_factor += 1
            else:
                duplicate_states += 1
    return None, "failed"


def wta_astar_solver(values, p, weapons=None):
    adjusted_p = []
    for i in range(len(weapons)):
        for j in range(weapons[i]):
            adjusted_p.append(p[i])
    state = {'v': values, 'p': adjusted_p, 'assignment': [-1] * len(adjusted_p)}
    AStar(state, heuristic=wta_mmr_heuristic, Node=WTA_Node)
