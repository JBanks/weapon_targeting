if __package__ is not None and len(__package__) > 0:
    print(f"{__name__} using relative import inside of {__package__}")
    from .WTAtools import *
else:
    from WTAtools import *

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weapons', type=int, help="The number of effectors in each problem", default=15,
                        required=False)
    parser.add_argument('--targets', type=int, help="The number of targets in each problem", default=15, required=False)
    parser.add_argument('--quantity', type=int, help="The number of problems of each size", default=1, required=False)
    parser.add_argument('--offset', type=int, help="Numbering offset for scenarios", default=71, required=False)
    parser.add_argument('--solve', type=bool, help="Whether or not we will solve the problems", default=True,
                        required=False)
    parser.add_argument('--save', type=bool, help="Save the output to a file", default=True, required=False)
    args = parser.parse_args()
    generate_dataset(weapons=args.weapons, targets=args.targets, quantity=args.quantity, offset=args.offset,
                     problem_set="validation", solve_problems=args.solve, save=args.save)


if __name__ == "__main__":
    main()
