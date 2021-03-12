from WTAtools import *
import time
import sys


def main():
    if len(sys.argv) > 1:
        problem = load_problem(sys.argv[1])
    else:
        problem = new_problem()
    solvers = [WTAOR.wta_or_solver, WTAGA.wta_ga_solver]
    values, p = problem['values'], problem['p']
    total_value = sum(values)
    print(f"The total value for the problem is: {total_value}")
    for solver in solvers:
        start_time = time.time()
        value, assignment = solver(values, p)
        end_time = time.time()
        print(f"Solved in {end_time - start_time}s - Value: {value}, assignment: \n{assignment}")


if __name__ == "__main__":
    main()
