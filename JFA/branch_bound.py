import copy

import numpy as np
if __package__ is not None and len(__package__) > 0:
    from . import simulator as sim
    from . import features as jf
    from . import problem_generators as pg
else:
    import simulator as sim
    import features as jf
    import problem_generators as pg


class BBSolver:
    class JFANode:
        def __init__(self, solver, action, state, parent=None, terminal=False):
            """
            Initialize a node that represents a step in a solution path
            @param solver: The solver class so that we can reference parent functions without passing large amounts of
                        data
            @param weapon: The weapon that was most recently assigned
            @param target: The target that was most recently assigned
            @param parent: The node for which this node is a candidate solution
            """
            self.solver = solver
            self.action = action
            self.weapon, self.target = action
            self.state = copy.deepcopy(state)
            self.visited_children = []
            self._solution = []
            self._parent = None
            if (self.state['Opportunities'][:, :, jf.OpportunityFeatures.SELECTABLE] != 0).any():
                self.terminal = False
            else:
                self.terminal = True
            if parent is not None:
                self.parent = parent
                self.low_bound = self.solver.jfa_lower_bound(partial_solution=self._solution)

        def __repr__(self):
            return f"JFANode:{self._solution}"

        def visit_child(self, child_pair):
            self.visited_children.append(child_pair)

        def remove_child(self, child_pair):
            self.visited_children.remove(child_pair)

        @property
        def parent(self):
            return self._parent

        @parent.setter
        def parent(self, parent=None):
            parent.visit_child(self.action)
            self._parent = parent
            if parent is not self.solver.root:
                self._solution = self._parent.solution.copy()
            self._solution.append(tuple(self.action))

        @property
        def solution(self):
            return self._solution

        @property
        def candidate_nodes(self):
            """
            Produce a list of candidate nodes which can be sourced from the current node.
            @return: a list of nodes of weapon-target pairs with the current node as their parent
            """
            nodes = np.stack(np.where(self.state['Opportunities'][:, :, jf.OpportunityFeatures.SELECTABLE] == True),
                             axis=1)
            if self._parent is not None:
                nodes = [self.solver.new_node((i, j), self) for i, j in nodes if (i, j) not in self.visited_children and
                         ((i, j) not in self._solution or (i, j) == self.action)]
            else:
                nodes = [self.solver.new_node((i, j), self) for i, j in nodes if (i, j) not in self.visited_children]
            return nodes

    def __init__(self, problem, node_type=JFANode):
        """
        Initialize the search space for the branch and bound algorithm
        @param values: A list of the values of each target
        @param p: A matrix of the probabilities of success for each weapon-target pairing
        """
        self.env = sim.Simulation(sim.state_to_dict)
        self.node = node_type
        self.problem = problem
        self.heuristic = self.greedy_heuristic
        self.root = self.node(self, (-1, -1), copy.deepcopy(problem), None)

    def new_node(self, action, parent=None):
        """
        Create an instance of a node
        @param action: The weapon-target pairing that has been selected.
        @param parent: The parent of the node that is being created
        @return: The instance of the node
        """
        return self.node(self, action, self.env.update_state(action, copy.deepcopy(parent.state))[0], parent)

    def objective_function(self, solution):
        """
        Evaluate the solution of a given node
        @param solution: A list of weapon-target pairings
        @return: The remaining value after application of the solution
        """
        state = copy.deepcopy(self.problem)
        for assignment in solution:
            state, reward, terminal = self.env.update_state(assignment, state)
        return sum(state['Targets'][:, jf.TaskFeatures.VALUE])

    def greedy_heuristic(self, problem=None, partial_solution=None, with_value=False):
        """
        Uses a greedy heuristic to find the next-best step in the solution.
        @param v: Accepts a different v list for use in other heuristic algorithms
        @param p: Accepts a different p matrix for use in other heuristic algorithms
        @param partial_solution: A partial solution, allowing us to pass in the solution from a node part-way down the
                                path
        @param with_value: Chooses to return only the assignments, or to include the value of the solution
        @return: either the solution with the remaining value, or just the solution
        """
        solution = []
        if problem is None:
            state = copy.deepcopy(self.problem)
        else:
            state = copy.deepcopy(problem)  # get initial state or load a new problem
        terminal = False
        if partial_solution:
            for action in partial_solution:
                state, reward, terminal = self.env.update_state(action, state)
        while not terminal:
            pSuccesses = state['Opportunities'][:, :, jf.OpportunityFeatures.PSUCCESS]
            values = state['Targets'][:, jf.TaskFeatures.VALUE]
            selectable = state['Opportunities'][:, :, jf.OpportunityFeatures.SELECTABLE]
            action = np.unravel_index(np.argmax(pSuccesses * values * selectable), pSuccesses.shape)
            state, reward, terminal = self.env.update_state(action, state)
            solution.append(action)
        if with_value:
            return solution, sum(state['Targets'][:, jf.TaskFeatures.VALUE])
        return solution

    def assignment_matrix(self, solution):
        """
        Convert a list of assignments into an assignment matrix. [1, 2, 0] -> [[0, 1, 0],[0, 0, 1],[1, 0, 0]]
        @param solution: A list of assignments
        @return: A matrix representing the assignments weapons to targets
        """
        return [list(pair) for pair in solution]

    def jfa_lower_bound(self, partial_solution):
        """
        This heuristic is naive and assumes that we can get all of the rewards available to an effector.
        This will extend the search space significantly, but should guarantee the optimal solution.
        The heuristic is optimistic, and assumes that the same effector can more targets than it is capable of.
        """
        solution = []
        state = copy.deepcopy(self.problem)
        for action in partial_solution:
            state, reward, terminal = self.env.update_state(action, state)
            solution.append(action)
        remaining_reward = 0
        opportunities = state['Opportunities'][:, :, :].copy()
        for j, target in enumerate(state['Targets']):
            if target[jf.TaskFeatures.SELECTED] == 1:
                continue  # This task has already been selected the maximum number of times.
            for i in range(len(opportunities)):
                if (i, j) in solution and solution[-1] != (i, j):
                    opportunities[i, j, jf.OpportunityFeatures.PSUCCESS] = 0.00000001
            remaining_moves = int(min((1 - target[jf.TaskFeatures.SELECTED]) * 2,
                                      sum(opportunities[:, j, jf.OpportunityFeatures.SELECTABLE] * 2)))
            if not remaining_moves:
                continue  # The minimum between the number of hits left on a target, and eligible effectors is zero.
            value = target[jf.TaskFeatures.VALUE]
            top = np.argpartition(opportunities[:, j, jf.OpportunityFeatures.PSUCCESS], -1)[
                  -1:]  # select the top 'n' effectors.
            for move in range(remaining_moves):  # Use exponent to make this calculate slightly faster?
                reward = value * opportunities[top[0], j, jf.OpportunityFeatures.PSUCCESS]
                remaining_reward += reward
                value -= reward
        return sum(state['Targets'][:, jf.TaskFeatures.VALUE]) - remaining_reward  # Return the remaining reward if all moves were possible.

    def solve(self):
        """
        Iterates through the list of candidate nodes searching for the best solution.
        @return: The value and assignment matrix for the best solution found.
        """
        heuristic_solution, upper_bound = self.heuristic(with_value=True)
        optimum = heuristic_solution
        parent = self.root
        container = [parent]
        for action in heuristic_solution:
            parent = self.new_node(action, parent=parent)
            container.append(parent)
        while container:
            node = container.pop()
            if node.terminal:
                remaining_value = self.objective_function(node.solution)
                if remaining_value < upper_bound:
                    optimum = node.solution
                    upper_bound = remaining_value
            else:
                container.extend([candidate for candidate in node.candidate_nodes if candidate.low_bound < upper_bound])
        return upper_bound, self.assignment_matrix(optimum)


def jfa_branch_bound_solver(problem):
    """
    Executes the branch and bound solving algorithm.
    @param problem:
    @return: A tuple with the remaining value of the targets, and the assignment matrix associated with the value
    """
    solver = BBSolver(problem)
    return solver.solve()
