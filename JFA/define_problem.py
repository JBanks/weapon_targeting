if __package__ is not None and len(__package__) > 0:
    print(f"{__name__} using relative import inside of {__package__}")
    from . import simulator as sim
    from . import features as jf
    from . import problem_generators as pg
else:
    import simulator as sim
    import features as jf
    import problem_generators as pg
import numpy as np
import argparse


def define_problem(problem):
    state = sim.mergeState(problem['Effectors'], problem['Targets'], problem['Opportunities'])
    sim.printState(state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, help="The number of effectors in each problem", default='',
                        required=True)
    args = parser.parse_args()
    loaded_problem = pg.loadProblem(args.filename)
    define_problem(loaded_problem)

