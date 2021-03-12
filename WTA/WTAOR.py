#!/usr/bin/env python3

from ortools.sat.python import cp_model
import numpy as np


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


def main():
    """
    values = [5, 10, 20]
    weapons = [5, 2, 1]
    p = [
        [0.3, 0.2, 0.5],
        [0.1, 0.6, 0.5],
        [0.4, 0.5, 0.4]
    ]
    """

    values = [76, 93, 91, 85, 55, 29, 66, 53]
    p = [
        [0.706943154335022, 0.6360183954238892, 0.8114644885063171, 0.7272420525550842, 0.7375414967536926,
         0.6824339032173157, 0.6231933832168579, 0.6588988304138184],
        [0.8205441236495972, 0.8226460814476013, 0.6846358776092529, 0.6808983087539673, 0.7649776935577393,
         0.6295098662376404, 0.6345000863075256, 0.7816135287284851],
        [0.8206263184547424, 0.6674816012382507, 0.6290864944458008, 0.6989136934280396, 0.7409589886665344,
         0.7029198408126831, 0.6802255511283875, 0.8629313111305237],
        [0.6121272444725037, 0.6316321492195129, 0.6599464416503906, 0.826388955116272, 0.8786106705665588,
         0.7138409614562988, 0.6111229658126831, 0.7680585980415344],
        [0.6966612935066223, 0.8372161984443665, 0.6774555444717407, 0.6943622827529907, 0.7386593818664551,
         0.6733441352844238, 0.8375512957572937, 0.8438227772712708],
        [0.7155847549438477, 0.7222493886947632, 0.6354673504829407, 0.6722898483276367, 0.6532123684883118,
         0.6016350388526917, 0.7587482929229736, 0.8896375298500061],
        [0.8869679570198059, 0.8523036241531372, 0.7461298108100891, 0.776040256023407, 0.7137162685394287,
         0.8593636751174927, 0.7626805305480957, 0.6583917737007141],
        [0.7424099445343018, 0.6186914443969727, 0.6806374192237854, 0.837059736251831, 0.6029490232467651,
         0.8164913058280945, 0.7150862812995911, 0.8988319635391235]
    ]
    print(wta_or_solver(values, p))


if __name__ == "__main__":
    main()
