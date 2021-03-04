#!/usr/bin/env python3

import SampleSimulator as Sim
import ProblemGenerators as PG
import JFAFeatures as JF
import AStarJFA as AS
import numpy as np
import json
import urllib
from flask import Flask, request, jsonify


app = Flask(__name__)

class AStarAgent():
	def getAction(self, state):
		solution, __, __, __ = AS.AStar(state)
		return solution[0]

agent = AStarAgent() #Sim.JeremyAgent()
def CompareResults(pythonState, unityState):
	pass


@app.route('/', methods=['GET'])
def test_connection():
	return jsonify('The connection to the server was successful')


@app.route('/', methods=['POST'])
def get_action():
	#print(f"received: {request.data}")
	json_string = urllib.parse.unquote(request.data.decode("UTF-8"))
	problem = json.loads(json_string)
	for key in problem.keys():
		problem[key] = np.asarray(problem[key])

	state = Sim.mergeState(problem['Effectors'], problem['Targets'], problem['Opportunities'])
	#if pythonState:
	#	CompareResults(pythonState, state)
	action = agent.getAction(problem)
	print(f"Selected action: {action}")
	#pythonState = env.update_state(action, state)
	return jsonify({'assets': [int(action[0]), int(action[1])]})

app.run()