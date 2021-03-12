import WTAOR
import WTAGA
import numpy as np
import json
import secrets
import os
import time
import csv

TOKEN_LENGTH = 5


def log(string):
    print(f"[{time.asctime()}] {string}")


def save_problem(problem, filename):
    to_file = {}
    for key in problem.keys():
        to_file[key] = np.asarray(problem[key]).tolist()
    with open(filename, 'w') as file:
        json.dump(to_file, file)


def load_problem(filename):
    problem = {}
    with open(filename, 'r') as file:
        from_file = json.load(file)
    for key in from_file.keys():
        problem[key] = np.asarray(from_file[key])
    return problem


def new_problem(weapons=5, targets=5):
    p = np.random.uniform(0.6, 0.9, (weapons, targets))  # uniform between 0.6 and 0.9
    values = np.random.randint(25, 100, targets, dtype=int).tolist()  # uniform between 25 and 100
    return {"values": values, "p": p}


def safe_filename(directory, size_str, json_prefix):
    pass


def generate_dataset(weapons=5, targets=5, quantity=100, solve_problems=True, csv_filename=time.time(),
                     json_prefix="train"):
    solvers = [{'name': "Genetic Algorithm", 'function': WTAGA.wta_ga_solver, 'solve': True},
               {'name': "OR-Tools", 'function': WTAOR.wta_or_solver, 'solve': True}]

    size_str = f"{weapons}x{targets}"
    try:
        os.mkdir(size_str)
    except:
        pass
    csv_content = []
    csv_row = ["filename", "total reward"]
    for solver in solvers:
        if solver['solve']:
            csv_row.append(solver['name'])
            csv_row.append("solution")
    csv_content.append(csv_row)
    for i in range(quantity):
        try:
            problem = new_problem(weapons, targets)
            filename = json_prefix + "-" + secrets.token_urlsafe(TOKEN_LENGTH) + size_str + ".json"
            save_problem(problem, os.path.join(size_str, filename))

            rewards_available = sum(problem['values'])

            if solve_problems:
                csv_row = [filename, rewards_available]
                start_time = time.time()
                for solver in solvers:
                    if solver['solve']:
                        g, solution = solver['function'](problem['values'], problem['p'])
                        csv_row.append(g)
                        csv_row.append(solution)
                end_time = time.time()
                csv_content.append(csv_row)
                log(f"Solved {i+1}/{quantity}: {filename} in: {end_time - start_time:.6f}s")
        except KeyboardInterrupt:
            input("Press Enter to attempt again, or ctrl+c to quit.")
    print()
    if solve_problems:
        csv_filename = os.path.join(size_str, f'{csv_filename}.csv')
        with open(csv_filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(csv_content)
        log(f"solutions exported to {csv_filename}")
