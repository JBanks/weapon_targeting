import numpy as np


def wta_greedy_solver(values, p, weapons=None):
    if weapons is None:
        weapons = [1]*len(p)
    adjusted_p = []
    for i in range(len(weapons)):
        for j in range(weapons[i]):
            adjusted_p.append(p[i])
    p = np.asarray(adjusted_p.copy())
    v = np.asarray(values.copy())
    assignment_matrix = np.zeros(p.shape)
    while (p > 0).any():
        assignment = np.unravel_index(np.argmax(p * v), p.shape)
        assignment_matrix[assignment] = 1
        v[assignment[1]] *= (1 - p[assignment])
        p[assignment[0], :] = 0
    return sum(v), assignment_matrix
