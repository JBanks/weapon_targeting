import numpy as np


class BBSolver:
    class Node:
        def __init__(self, solver, weapon, target, parent=None):
            """
            Initialize a node that represents a step in a solution path
            @param solver: The solver class so that we can reference parent functions without passing large amounts of
                        data
            @param weapon: The weapon that was most recently assigned
            @param target: The target that was most recently assigned
            @param parent: The node for which this node is a candidate solution
            """
            self.solver = solver
            self.weapon = weapon
            self.target = target
            self.parent = parent
            self.children = []
            self.partial_solution = [(weapon, target)]
            self.depth = 1
            if parent is not None:
                self.parent.add_child((weapon, target))
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

    def __init__(self, values, p):
        """
        Initialize the search space for the branch and bound algorithm
        @param values: A list of the values of each target
        @param p: A matrix of the probabilities of success for each weapon-target pairing
        """
        self.values = values
        self.p = p
        self.num_weapons = len(p)
        self.num_targets = len(values)
        self.heuristic = self.greedy_heuristic
        self.root = self.new_node((-1, -1), None)

    def new_node(self, action, parent=None):
        """
        Create an instance of a node
        @param action: The weapon-target pairing that has been selected.
        @param parent: The parent of the node that is being created
        @return: The instance of the node
        """
        return self.Node(self, *action, parent)

    def objective_function(self, solution):
        """
        Evaluate the solution of a given node
        @param solution: A list of weapon-target pairings
        @return: The remaining value after application of the solution
        """
        p = np.asarray(self.p.copy())
        v = np.asarray(self.values.copy())
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


def wta_branch_bound_solver(values, p, weapons=None):
    """
    Executes the branch and bound solving algorithm.
    @param values: A list of values associated with each target
    @param p: A matrix of the probability of success for each weapon-target pairing
    @param weapons: The number of weapons of a given type
    @return: A tuple with the remaining value of the targets, and the assignment matrix associated with the value
    """
    values = list(map(float, values))  # convert all values to floats
    if weapons is not None:  # expand p so that each weapon has its own row in the matrix
        num_weapon_types = len(weapons)
        adjusted_p = []
        for i in range(num_weapon_types):
            for j in range(weapons[i]):
                adjusted_p.append(p[i])
        p = adjusted_p
    p = np.asarray(p)
    solver = BBSolver(values, p)
    return solver.solve()
