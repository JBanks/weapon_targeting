#!/usr/bin/env python3

import simulator as sim
import problem_generators as pg
import features as jf
import solvers as js
import numpy as np
import json
import urllib
import copy
from flask import Flask, request, jsonify

app = Flask(__name__)
pythonState = False
env = sim.Simulation(sim.mergeState)
MULTIPLE = True


def get_action_from_solver(state):
	_, solution = js.greedy(state)  # AStar(state)
	return solution[0]


def get_actions_from_solver(state):
	_, solution = js.greedy(state)
	return solution


def compare_results(python_state, unity_state):
	pad = 0  # len(JF.EffectorFeatures) + len(JF.TaskFeatures)
	print(f"U-Energy cost: {unity_state[:, 0, pad + jf.EffectorFeatures.ENERGYLEFT]}")
	print(f"P-Energy cost: {python_state[:, 0, pad + jf.EffectorFeatures.ENERGYLEFT]}")
	print(f"U-Time cost: {unity_state[:, 0, pad + jf.EffectorFeatures.TIMELEFT]}")
	print(f"P-Time cost: {python_state[:, 0, pad + jf.EffectorFeatures.TIMELEFT]}")


@app.route('/', methods=['GET'])
def test_connection():
	return jsonify('The connection to the server was successful')


@app.route('/', methods=['POST'])
def get_action():
	global pythonState
	json_string = urllib.parse.unquote(request.data.decode("UTF-8"))
	problem = json.loads(json_string)
	for key in problem.keys():
		problem[key] = np.asarray(problem[key])
	pg.correct_effector_data(problem)
	state = sim.mergeState(problem['Effectors'], problem['Targets'], problem['Opportunities'])
	if type(pythonState) == np.ndarray:
		compare_results(pythonState, state)
	if MULTIPLE:
		actions = get_actions_from_solver(copy.deepcopy(problem))
		print(f"selected actions: {actions}")
		assets = {'assets': []}
		for action in actions:
			assets['assets'].append([int(action[0]), int(action[1])])
		return jsonify(assets)
	else:
		action = get_action_from_solver(copy.deepcopy(problem))
		print(f"Selected action: {action}")
		pythonState, _, _ = env.update_state((action[0], action[1]), state.copy())
		return jsonify({'assets': [int(action[0]), int(action[1])]})


app.run()
