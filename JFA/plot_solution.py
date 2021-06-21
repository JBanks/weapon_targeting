if __package__ is not None and len(__package__) > 0:
    from . import simulator as sim
    from . import features as jf
    from . import problem_generators as pg
    from . import solvers as js
else:
    import simulator as sim
    import features as jf
    import problem_generators as pg
    import solvers as js
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import argparse
import numpy as np
import os


def plot_solution(problem, solution, filename):
    location_effectors = problem['Effectors'][:, :jf.EffectorFeatures.YPOS+1]
    location_targets = problem['Targets'][:, :jf.TaskFeatures.YPOS+1]
    values_targets = problem['Targets'][:, jf.TaskFeatures.VALUE]
    print(location_effectors)
    print(location_targets)
    solution = np.asarray(solution)

    linestyles = []
    colours = ['red', 'green', 'blue', 'purple', 'orange']

    fig = plt.figure()
    ax = plt.subplot()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(os.path.basename(filename))
    ax.scatter(location_effectors[:, 0], location_effectors[:, 1], marker='$A$', s=100)  # , c=colours[:len(location_effectors)])
    ax.scatter(location_targets[:, 0], location_targets[:, 1], marker='$T$', s=[500*values_targets - 200], c='orange')
    for i in range(len(location_effectors)):
        assignments = np.where(solution[:, 0] == i)[0]
        vertices = np.asarray([location_effectors[i]])
        for k in assignments:
            vertices = np.concatenate((vertices, [location_targets[solution[k][1]]]))
        vertices = np.concatenate((vertices, [location_effectors[i]]))
        Path(vertices)
        patch = patches.PathPatch(Path(vertices), facecolor='none', edgecolor=colours[i%len(colours)])
        ax.add_patch(patch)
    # plt.savefig(os.path.join("plots", f"{os.path.basename(filename)[:-4]}.png"))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, help="The number of effectors in each problem", default='',
                        required=True)
    args = parser.parse_args()
    loaded_problem = pg.loadProblem(args.filename)
    reward, solution = js.AStar(loaded_problem)
    plot_solution(loaded_problem, solution, args.filename)
