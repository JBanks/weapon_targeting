import problem_generators as PG
import os


def main(effectors=4, targets=10, numbering_offset=0, num_problems=100):
    directory = f"{effectors}x{targets}"
    try:
        os.mkdir(directory)
    except Exception as error:
        print(f"Error: {error}")

    for i in range(numbering_offset, num_problems + numbering_offset):
        identifier = f"{effectors}x{targets}_{i:04d}"
        new_identifier = f"validation_{i:05d}_{effectors}x{targets}"
        filename = identifier + ".json"
        new_filename = new_identifier + ".json"
        filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, new_filename)
        if not os.path.exists(filepath):
            print(f"file does not exist: {filename}")
            continue
        problem = PG.loadProblem(filepath)
        PG.truncate_effector_data(problem)
        PG.saveProblem(problem, new_filepath)


if __name__ == "__main__":
    main()
