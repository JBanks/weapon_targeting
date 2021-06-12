
if __package__ is not None and len(__package__) > 0:
    print(f"{__name__} using relative import inside of {__package__}")
    from . import JFAGA
else:
    import JFAGA
import argparse
import plotly.graph_objects as go
import numpy as np
import json


def collect_datapoints(problem, populations=(80, 120, 160, 200), generations=100):
    all_rewards = []
    all_times = []
    for population in populations:
        print(f"Calculating population: {population}")
        rewards, times = JFAGA.jfa_ga_explorer(problem, population, generations)
        # rewards = np.asarray([5, 7, 9, 10, 11, 11, 12, 12, 12, 13, 13]) * population / 80  # Make fake numbers
        # times = np.asarray([20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]) * population / 80  # Make fake numbers
        all_rewards.append(rewards)
        all_times.append(times)
    return all_rewards, all_times


def create_surface_plot(datapoints, populations=(80, 120, 160, 200), generations=100):
    axis2 = np.linspace(0, generations, len(datapoints[0][0]))
    greedy = np.ones((len(axis2), len(populations))) * (5.222785 - 0.827704)
    optimum = np.ones((len(axis2), len(populations))) * 4.208206

    rewards, times = datapoints
    rewards = np.asarray(rewards).T
    times = np.asarray(times).T
    fig = go.Figure(data=[go.Surface(z=greedy, x=populations, y=axis2, colorscale=[[0, 'red'], [1, 'red']],
                                     opacity=0.7),
                          # go.Surface(z=optimum, x=populations, y=axis2, colorscale=[[0, 'green'], [1, 'green']],
                          #            opacity=0.5),
                          go.Surface(z=rewards, x=populations, y=axis2, surfacecolor=times, colorscale='viridis',
                                     reversescale=True)
                          ])
    fig.update_layout(scene=dict(
        xaxis_title='Population size',
        yaxis_title='Generations',
        zaxis_title='Score',
        yaxis_type='log'))
    fig.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, help="The filename of the problem to load",
                        default="4x10\\claAGAs4TG6vpaKWslQELg.json")
    parser.add_argument("--results", type=str, help="", default="GS-claAGAs4TG6vpaKWslQELg.json")
    parser.add_argument("--solve", type=bool, help="", default=False)
    args = parser.parse_args()
    # args = parser.parse_args(['--results GS-GVeCUHSIRJGfhz0lHlwcvw.json'])
    if args.solve:
        problem = JFAGA.pg.loadProblem(args.problem)
        datapoints = collect_datapoints(problem, generations=400, populations=[40, 80, 160, 240])
        with open(f"{args.results}", 'w') as file:
            json.dump(datapoints, file)
    else:
        with open(args.results, "r") as file:
            datapoints = json.load(file)

    create_surface_plot(datapoints, generations=400, populations=[40, 80, 160, 240])


if __name__ == "__main__":
    main()
