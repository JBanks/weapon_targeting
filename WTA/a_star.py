import numpy as np
import copy
import sortedcontainers as sc
import zlib


class Environment:
    def __init__(self):
        pass

    @staticmethod
    def update_state(action, state):
        weapon, target = action
        reward = state['p'][weapon, target] * state['v'][target]
        new_state = state.copy()
        new_state['v'][target] *= (1 - state['p'][weapon, target])
        new_state['p'][weapon, :] = 0
        new_state['assignment'][weapon] = target
        terminal = not (new_state['assignment'] == -1).any()
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
        self._solution = state['assignment']
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
        #self._solution = self.parent.solution.copy()
        #self._solution.append(list(self.action))

    @property
    def solution(self):
        """

        @return:
        """
        return self._solution

    @property
    def candidate_nodes(self):
        unassigned = np.where(self.state['assignment'] == -1)[0]
        children = [(i, j) for i in unassigned for j in range(len(self.state['v']))]
        return children

    @property
    def assignment_matrix(self):
        """
        Convert a list of assignments into an assignment matrix. [1, 2, 0] -> [[0, 1, 0],[0, 0, 1],[1, 0, 0]]
        @return: A matrix representing the assignments weapons to targets
        """
        assignment_matrix = np.zeros((len(self._solution), len(self.state['p'][0])))
        for i, assignment in enumerate(self.state['assignment']):
            assignment_matrix[i, assignment] = 1
        return assignment_matrix

    def cat_string(self):
        return zlib.compress(self.state['assignment'])


def wta_mmr_heuristic(node):
    """
    Perform MMR on our BB node.
    @param node:
    @return:
    """
    state = node.state
    v = state['v'].copy()
    maximum_p = (np.zeros(state['p'].shape) + np.max(state['p'], axis=1)).T  # Give all weapons the maximum probability
    return node.g - wta_greedy_heuristic(v, maximum_p)  # Return the remaining reward from MMR


def wta_greedy_heuristic(v=None, p=None, partial_solution=None):
    """
    Uses a greedy heuristic to find the next-best step in the solution.
    @param v: Accepts a different v list for use in other heuristic algorithms
    @param p: Accepts a different p matrix for use in other heuristic algorithms
    @param partial_solution: A partial solution, allowing us to pass in the solution from a node part-way down the
                            path
    @return: either the solution with the remaining value, or just the solution
    """
    while (p > 0).any():
        assignment = np.unravel_index(np.argmax(p * v), p.shape)
        v[assignment[1]] *= (1 - p[assignment])
        p[assignment[0], :] = 0
    return sum(v)


def AStar(problem, heuristic=wta_mmr_heuristic, node_type=WTA_Node, track_progress=False):
    """

    @param problem:
    @param heuristic:
    @param node_type:
    @param track_progress:
    @return:
    """
    Node = node_type
    env = Environment()
    node = Node(sum(problem['v']), None, problem, 0)
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
                    return node.g, node.assignment_matrix
                node.g = node.parent.g - node.reward
                frontier.add(node)
                continue
            node.g = node.parent.g - node.reward
            # This may actually not be the optimal.  There may be a more optimal node.
            # If the value is the same, we found it, if the value changes, put it back in the heap.
        explored[node.cat_string()] = node.g
        state = node.state
        for effector, target in node.candidate_nodes:
            action = (effector, target)
            new_state, reward, terminal = env.update_state(action, copy.deepcopy(state))
            g = node.g - reward  # The remaining value after the action taken
            child = Node(g, action, new_state, reward, terminal)
            child.parent = node
            h = heuristic(child)  # The possible remaining value assuming that all of the best actions can be taken
            child.g = g - h
            if child.cat_string() not in explored and child not in frontier:
                frontier.add(child)
                branch_factor += 1
            else:
                duplicate_states += 1
    return None, "failed"


def wta_astar_solver(values, p, weapons=None):
    adjusted_p = []
    if weapons is None:
        weapons = [1] * len(p)
    for i in range(len(weapons)):
        for j in range(weapons[i]):
            adjusted_p.append(p[i])
    adjusted_p = np.asarray(adjusted_p, dtype=float)
    values = np.asarray(values, dtype=float)
    state = {'v': values, 'p': adjusted_p, 'assignment': np.asarray([-1] * len(adjusted_p))}
    return AStar(state, heuristic=wta_mmr_heuristic, node_type=WTA_Node)
