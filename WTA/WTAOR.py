from ortools.sat.python import cp_model
import numpy as np
import argparse
import json


def load_problem(filename):
    problem = {}
    with open(filename, 'r') as file:
        from_file = json.load(file)
    for key in from_file.keys():
        problem[key] = np.asarray(from_file[key])
    return problem


def wta_or_solver(values, p, weapons=None):
    if weapons is None:
        weapons = [1]*len(p)
    model = cp_model.CpModel()
    growth_factor = 10000

    values = [value * growth_factor for value in values]

    num_weapon_types = len(weapons)
    num_targets = len(values)
    var = 0
    denominator = model.NewIntVar(growth_factor, growth_factor, f'denominator: {var}')
    var += 1

    q = {}
    for i in range(num_weapon_types):
        for j in range(num_targets):
            for k in range(weapons[i] + 1):
                q_val = int(((1 - p[i][j]) ** k) * growth_factor)
                q[(i, j, k)] = model.NewIntVar(q_val, q_val, f'q[{i,j,k}]: {var}')
                var += 1

    rem_vals = {}
    large_rem_vals = {}
    for j in range(num_targets):
        rem_vals[j] = model.NewIntVar(0, values[j] * 10000, f'target[{j}]: {var}')
        var += 1
        large_rem_vals[j] = model.NewIntVar(0, values[j] * 10000, f'large_target[{j}]: {var}')
        var += 1

    # b[(i,j,k)] is a boolean for if a given weapon qty is associated with a given target
    # i is the weapon type, j is the target, and k is the number of weapons allocated.
    b = {}
    m = {}  # Multiplied values
    cm = {}  # Corrected Multiplication
    for i in range(num_weapon_types):
        for j in range(num_targets):
            cm[(i, j)] = model.NewIntVar(0, values[j] * 10000, f'cm[{i,j}]: {var}')
            var += 1
            for k in range(weapons[i] + 1):
                b[(i, j, k)] = model.NewBoolVar(f'b[{(i,j,k)}]: {var}')
                var += 1
                # Multiply by their boolean values, then take the max equality, then take the product of the return
                # from the maxes.
                m[(i, j, k)] = model.NewIntVar(0, values[j] * 10000, f'm[{i,j,k}]: {var}')
                var += 1
                model.AddMultiplicationEquality(m[(i, j, k)], [q[(i, j, k)], b[(i, j, k)]])

            model.AddMaxEquality(cm[(i, j)], [m[(i, j, k)] for k in range(weapons[i] + 1)])
            model.Add(sum(b[(i, j, k)] for k in range(weapons[i] + 1)) == 1)
        model.Add(sum(b[(i, j, k)] * k for j in range(num_targets) for k in range(weapons[i] + 1)) <= weapons[i])

    running = {}
    c_running = {}
    for j in range(num_targets):
        for i in range(num_weapon_types):
            running[(i, j)] = model.NewIntVar(0, values[j] * 10000, f'running[{i,j}]: {var}')
            var += 1
            c_running[(i, j)] = model.NewIntVar(0, values[j] * 10000, f'c_running[{i,j}]: {var}')
            var += 1
            if i == 0:
                model.Add(c_running[(i, j)] == cm[(i, j)])
            else:
                model.AddMultiplicationEquality(running[(i, j)], [cm[(i, j)], c_running[(i-1, j)]])
                model.AddDivisionEquality(c_running[(i, j)], running[(i, j)], denominator)
        model.AddMultiplicationEquality(large_rem_vals[j], [c_running[(num_weapon_types - 1, j)], values[j]])
        model.AddDivisionEquality(rem_vals[j], large_rem_vals[j], denominator)

    objective_terms = []
    for j in range(num_targets):
        objective_terms.append(rem_vals[j])

    model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        assignment_matrix = np.zeros((num_weapon_types, num_targets))
        # print(f"Found a{'n optimal' if status == cp_model.OPTIMAL else 'feasible'} solution")
        # print(f"Remaining value = {solver.ObjectiveValue()/growth_factor}")
        # for j in range(num_targets):
        #    print(f"Target {j} has {solver.Value(rem_vals[j])/growth_factor} point remaining")
        for j in range(num_targets):
            # print(f"target {j} with value {values[j]/growth_factor} assigned:")
            for i in range(num_weapon_types):
                for k in range(weapons[i] + 1):
                    if k and solver.Value(b[i, j, k]):
                        # print(f"{k} weapons of type {i} with q: {solver.Value(q[(i,j,k)])/growth_factor}")
                        assignment_matrix[i][j] = k
        return solver.ObjectiveValue()/growth_factor, assignment_matrix

    else:
        print("No solution found")
        print("Model:")
        print(model.Proto())
        print(f"Validated: {model.Validate()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, help="The filename of the problem json you would like to load",
                        default='5x5\\5x5-95J0qf8.json')
    args = parser.parse_args()

    problem = load_problem(args.problem)

    result = wta_or_solver(problem['values'], problem['p'])
    print(result)

