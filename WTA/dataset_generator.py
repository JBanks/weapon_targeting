if __package__ is not None and len(__package__) > 0:
    print(f"{__name__} using relative import inside of {__package__}")
    from .tools import *
else:
    from tools import *

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weapons', type=int, help="The number of effectors in each problem", default=4,
                        required=False)
    parser.add_argument('--targets', type=int, help="The number of targets in each problem", default=4, required=False)
    parser.add_argument('--quantity', type=int, help="The number of problems of each size", default=100, required=False)
    parser.add_argument('--offset', type=int, help="Numbering offset for scenarios", default=0, required=False)
    parser.add_argument('--solve', type=bool, help="Whether or not we will solve the problems", default=True,
                        required=False)
    parser.add_argument('--save', type=bool, help="Save the output to a file", default=False, required=False)
    args = parser.parse_args()
    generate_dataset(weapons=args.weapons, targets=args.targets, quantity=args.quantity, offset=args.offset,
                     problem_set="validation", solve_problems=args.solve, save=args.save,
                     directory=r"old_datasets/WTA/validation_4x4/")


if __name__ == "__main__":
    main()
