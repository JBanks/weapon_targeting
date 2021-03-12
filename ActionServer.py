#!/usr/bin/env python3

from . import SampleSimulator as Sim
from . import ProblemGenerators as PG
from . import JFAFeatures as JF
from . import JFASolvers as JS
import numpy as np
import json
import urllib
from flask import Flask, request, jsonify

app = Flask(__name__)
pythonState = False
env = Sim.Simulation(Sim.mergeState)


class AStarAgent:
	def getAction(self, state):
		solution, __ = JS.AStar(state)
		return solution[0]


agent = AStarAgent()  # Sim.JeremyAgent()


def CompareResults(pythonState, unityState):
	print(f"U-Effector energy remaining: {unityState[:, 0, JF.EffectorFeatures.ENERGYLEFT]}")
	print(f"U-Effector time remaining: {unityState[:, 0, JF.EffectorFeatures.TIMELEFT]}")
	print(f"P-Effector energy remaining: {pythonState[:, 0, JF.EffectorFeatures.ENERGYLEFT]}")
	print(f"P-Effector time remaining: {pythonState[:, 0, JF.EffectorFeatures.TIMELEFT]}")


@app.route('/', methods=['GET'])
def test_connection():
	return jsonify('The connection to the server was successful')


@app.route('/', methods=['POST'])
def get_action():
	global pythonState
	# print(f"received: {request.data}")
	json_string = urllib.parse.unquote(request.data.decode("UTF-8"))
	problem = json.loads(json_string)
	for key in problem.keys():
		problem[key] = np.asarray(problem[key])

	state = Sim.mergeState(problem['Effectors'], problem['Targets'], problem['Opportunities'])
	if type(pythonState) == np.ndarray:
		CompareResults(pythonState, state)
	action = agent.getAction(problem)
	print(f"Selected action: {action}")
	pythonState, _, _ = env.update_state((action[0], action[1]), state.copy())
	return jsonify({'assets': [int(action[0]), int(action[1])]})


app.run()
