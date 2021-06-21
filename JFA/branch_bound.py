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
    class Node:
        def __init__(self, solver, action, state, parent=None):
            """
            Initialize a node that represents a step in a solution path
            @param solver: The solver class so that we can reference parent functions without passing large amounts of
                        data
            @param weapon: The weapon that was most recently assigned
            @param target: The target that was most recently assigned
            @param parent: The node for which this node is a candidate solution
            """
            self.solver = solver
            self.weapon, self.target = action
            self.state = state
            self.parent = parent
            self.partial_solution = [action]
            self.children = []
            self.depth = 1
            if parent is not None:
                self.parent.add_child(action)
            while parent is not None and parent is not self.solver.root:
                self.partial_solution.append((parent.weapon, parent.target))
                self.depth += 1
                parent = parent.parent
            self.terminal = self.depth == self.solver.num_weapons
            self.low_bound = self.solver.maximum_marginal(partial_solution=self.partial_solution)

        def add_child(self, child_pair):
            self.children.append(child_pair)

        @property
        def candidate_nodes(self):
            """
            Produce a list of candidate nodes which can be sourced from the current node.
            @return: a list of nodes of weapon-target pairs with the current node as their parent
            """
            assigned = [0] * self.solver.num_weapons
            candidates = []
            for val in self.partial_solution:
                assigned[val[0]] = 1
            # TODO: Take the row for the next-best weapon and use it as the basis for the candidate extension
            # It may be more efficient to sort at problem initialization, but this will guarantee that we don't
            # need to worry about cycles in the graph search.
            for i in range(self.solver.num_weapons):
                if not assigned[i]:
                    candidates.extend([self.solver.new_node((i, j), self) for j in range(self.solver.num_targets) if (i, j) not in self.children])
                    break
            return candidates

    def __init__(self, problem):
        """
        Initialize the search space for the branch and bound algorithm
        @param values: A list of the values of each target
        @param p: A matrix of the probabilities of success for each weapon-target pairing
        """
        self.env = sim.Simulation(sim.state_to_dict, problem)
        self.effectors = problem['Effectors']
        self.targets = problem['Targets']
        self.opportunities = problem['Opportunities']
        self.num_weapons = len(self.effectors)
        self.num_targets = len(self.targets)
        self.heuristic = self.greedy_heuristic
        self.root = self.new_node((-1, -1), None)

    def new_node(self, action, parent=None):
        """
        Create an instance of a node
        @param action: The weapon-target pairing that has been selected.
        @param parent: The parent of the node that is being created
        @return: The instance of the node
        """
        return self.Node(self, action, parent)

    def objective_function(self, solution):
        """
        Evaluate the solution of a given node
        @param solution: A list of weapon-target pairings
        @return: The remaining value after application of the solution
        """
        effectors = self.effectors.copy()
        targets = self.targets.copy()
        opportunities = self.opportunities.copy()
        for assignment in solution:
            v[assignment[1]] *= (1 - p[assignment])
        return sum(v)

    def greedy_heuristic(self, v=None, p=None, partial_solution=None, with_value=False):
        """
        Uses a greedy heuristic to find the next-best step in the solution.
        @param v: Accepts a different v list for use in other heuristic algorithms
        @param p: Accepts a different p matrix for use in other heuristic algorithms
        @param partial_solution: A partial solution, allowing us to pass in the solution from a node part-way down the
                                path
        @param with_value: Chooses to return only the assignments, or to include the value of the solution
        @return: either the solution with the remaining value, or just the solution
        """
        if partial_solution is None:
            partial_solution = []
        if p is None:
            p = self.p.copy()
        if v is None:
            v = self.values.copy()
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
        if with_value:
            return solution, sum(v)
        return solution

    def maximum_marginal(self, partial_solution=None):
        """
        Sets all of the p values to be the maximum of the columns to provide a value that is lower than possible for a
        given path
        @param partial_solution: A list of weapon-target pairings representing the ancestry of the node
        @return: The value of the maximum marginal return solution
        """
        if partial_solution is None:
            partial_solution = []
        v = self.values.copy()
        maximum_p = np.zeros(self.p.shape) + np.max(self.p, axis=0)  # Give all weapons the maximum probability
        for assignment in partial_solution:  # Account for all actions up to this point
            v[assignment[1]] *= (1 - self.p[assignment])  # Vⱼ Πᵢ qᵢⱼ
            maximum_p[assignment[0], :] = 0
        return self.greedy_heuristic(v, maximum_p, with_value=True)[1]

    @staticmethod
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
                if [i, j] in node.solution() and node.solution()[-1] != [i, j]:
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

    def assignment_matrix(self, solution):
        """
        Convert a list of assignments into an assignment matrix. [1, 2, 0] -> [[0, 1, 0],[0, 0, 1],[1, 0, 0]]
        @param solution: A list of assignments
        @return: A matrix representing the assignments weapons to targets
        """
        assignment_matrix = np.zeros((self.num_weapons, self.num_targets))
        for assignment in solution:
            assignment_matrix[assignment[0], assignment[1]] = 1
        return assignment_matrix

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
                remaining_value = self.objective_function(node.partial_solution)
                if remaining_value < upper_bound:
                    optimum = node.partial_solution
                    upper_bound = remaining_value
            else:
                container.extend([candidate for candidate in node.candidate_nodes if candidate.low_bound < upper_bound])
        return upper_bound, self.assignment_matrix(optimum)


def wta_branch_bound_solver(problem):
    """
    Executes the branch and bound solving algorithm.
    @param values: A list of values associated with each target
    @param p: A matrix of the probability of success for each weapon-target pairing
    @param weapons: The number of weapons of a given type
    @return: A tuple with the remaining value of the targets, and the assignment matrix associated with the value
    """
    solver = BBSolver(problem)
    return solver.solve()
