#!/usr/bin/env python3

from enum import IntEnum, auto
import numpy as np
import ProblemGenerators as PG
import JFAFeatures as JF
import random
import json
import math

SPEED_CORRECTION = PG.STANDARDIZED_TIME * PG.MAX_SPEED

def EuclideanDistance(effector, task):
	return math.sqrt((effector[JF.EffectorFeatures.XPOS] - task[JF.TaskFeatures.XPOS])**2 + (effector[JF.EffectorFeatures.YPOS] - task[JF.TaskFeatures.YPOS])**2)


def returnDistance(effector, task):
	EucDistance = Simulation.EuclideanDistance(effector, task)
	travelDistance = max(EucDistance - effector[JF.EffectorFeatures.EFFECTIVEDISTANCE], 0)
	newX = effector[JF.EffectorFeatures.XPOS] + (task[JF.TaskFeatures.XPOS] - effector[JF.EffectorFeatures.XPOS]) * travelDistance / EucDistance
	newY = effector[JF.EffectorFeatures.YPOS] + (task[JF.TaskFeatures.YPOS] - effector[JF.EffectorFeatures.YPOS]) * travelDistance / EucDistance
	returnTrip = math.sqrt((effector[JF.EffectorFeatures.STARTX] - newX)**2 + (effector[JF.EffectorFeatures.STARTY] - newY)**2)
	return travelDistance + returnTrip

def printState(state):
	trueFalseStrings = ["False", "True"]
	print("\n\t (x, y)\t\t\tStatic\tSpeed\tAmmo\tEnergy\tTime\tEffDist\tAmmoRte\tEnergyRate")
	nbEffector = len(state[:,0,0])
	for i in range(0, nbEffector):
		print("Effector:", end="")
		print(f"({state[i,0,JF.EffectorFeatures.XPOS]:.4f}, {state[i,0,JF.EffectorFeatures.YPOS]:.4f})", end="")
		print(f"\t{trueFalseStrings[int(state[i,0,JF.EffectorFeatures.STATIC])]}", end="")
		print(f"\t{state[i,0,JF.EffectorFeatures.SPEED]:4.4f}", end="")
		print(f"\t{state[i,0,JF.EffectorFeatures.AMMOLEFT]:.4f}", end="")
		print(f"\t{state[i,0,JF.EffectorFeatures.ENERGYLEFT]:.4f}", end="")
		print(f"\t{state[i,0,JF.EffectorFeatures.TIMELEFT]:.4f}", end="")
		print(f"\t{state[i,0,JF.EffectorFeatures.EFFECTIVEDISTANCE]:.4f}", end="")
		print(f"\t{state[i,0,JF.EffectorFeatures.AMMORATE]:.4f}", end="")
		print(f"\t{state[i,0,JF.EffectorFeatures.ENERGYRATE]:.4f}")
	nbTask = len(state[0,:,0])
	pad = len(JF.EffectorFeatures)
	print("\n\n\t(x, y)\t\t\tValue\tSelected")
	for i in range(len(state[0,:,0])):
		print(f"Target: ({state[0, i, pad + JF.TaskFeatures.XPOS]:.4f}, {state[0, i, pad + JF.TaskFeatures.YPOS]:.4f})", end="")
		print(f"\t{state[0, i, pad + JF.TaskFeatures.VALUE]:.4f}", end="")
		print(f"\t{state[0, i, pad + JF.TaskFeatures.SELECTED]}")

	pad = len(JF.EffectorFeatures) + len(JF.TaskFeatures)
	print("\n\nAction\t\tPSucc\tEnergy\ttime\tSelectable\tEucDistance\tReturnDist")
	for i in range(nbEffector):
		for j in range(nbTask):
			print(f"({i:2},{j:2}):\t{state[i, j, pad + JF.OpportunityFeatures.PSUCCESS]:.4}\t{state[i, j, pad + JF.OpportunityFeatures.ENERGYCOST]:.4f}\t{state[i, j, pad + JF.OpportunityFeatures.TIMECOST]:.4f}\t{trueFalseStrings[int(state[i, j, pad + JF.OpportunityFeatures.SELECTABLE])]}\t\t{EuclideanDistance(state[i, 0, :], state[0, j, len(JF.EffectorFeatures):]):.6f}\t{returnDistance(state[i, 0, :], state[0, j, len(JF.EffectorFeatures):]):.6f}")


class JeremyAgent():
	def getAction(state):
		"""
		Allows the user to control which action to take next by selecting an agent, and a task.
		"""
		effector, task = None, None
		printState(state)
		while effector == None:
			try:
				effector = int(input("effector: "))
			except ValueError:
				pass
		while task == None:
			try:
				task = int(input("task: "))
			except ValueError:
				pass
		return (effector, task)


	def learn(state, action, reward, new_state, terminal):
		pass


class AlexAgent():
	pass


class Simulation:
	"""
	Simulate a warfare scenario in which a set of effectors complete a set of tasks
	"""
	def __init__(self, FormatState, problem = None, keepstack = False):
		self.keepstack = keepstack
		self.FormatState = FormatState  # In the case of the Neural Network,
										# this function will normalize values and create a 3D tensor
		if problem:
			self.reset(problem)


	def reset(self, problem=None):
		"""
		Set up all values for the initial positions in the scenario and return the state
		"""

		"""
		Reuse the current problem and set all variables back to their original values
		"""
		if problem is None:
			problem = self.InitialProblem
		else:
			self.InitialProblem = problem
		self.reward = 0
		self.scale = problem['Arena'][JF.ArenaFeatures.SCALE]
		self.coastline = problem['Arena'][JF.ArenaFeatures.COASTLINE]
		self.frontline = problem['Arena'][JF.ArenaFeatures.FRONTLINE]
		self.timeHorizon = problem['Arena'][JF.ArenaFeatures.TIMEHORIZON]
		self.problem = problem
		self.nbEffector, self.nbTask = len(problem['Effectors']), len(problem['Targets'])
		self.effectorData = np.ones((self.nbEffector, len(JF.EffectorFeatures)), dtype=float) #Information will all be kept in real values
		self.taskData = np.ones((self.nbTask, len(JF.TaskFeatures)), dtype=float)
		self.opportunityData = np.ones((self.nbEffector, self.nbTask, len(JF.OpportunityFeatures)), dtype=float)
		for i in range(0, self.nbEffector):
			self.effectorData[i] = np.asarray(problem['Effectors'][i])
		for i in range(0, self.nbTask):
			self.taskData[i] = np.asarray(problem['Targets'][i])
		for i in range(0, self.nbEffector):
			for j in range(0, self.nbTask):
				self.opportunityData[i][j] = np.asarray(problem['Opportunities'][i][j])
		self.history = []
		self.schedule = [[] for _ in range(self.nbEffector)]
		self.previousPSuccess = [[] for _ in range(self.nbTask)]
		return self.FormatState(self.effectorData, self.taskData, self.opportunityData)

	def resetState(self, state):
		pass


	def EuclideanDistance(effector, task):
		return math.sqrt((effector[JF.EffectorFeatures.XPOS] - task[JF.TaskFeatures.XPOS])**2 + (effector[JF.EffectorFeatures.YPOS] - task[JF.TaskFeatures.YPOS])**2)


	def returnDistance(effector, task):
		EucDistance = Simulation.EuclideanDistance(effector, task)
		travelDistance = max(EucDistance - effector[JF.EffectorFeatures.EFFECTIVEDISTANCE], 0)
		newX = effector[JF.EffectorFeatures.XPOS] + (task[JF.TaskFeatures.XPOS] - effector[JF.EffectorFeatures.XPOS]) * travelDistance / EucDistance
		newY = effector[JF.EffectorFeatures.YPOS] + (task[JF.TaskFeatures.YPOS] - effector[JF.EffectorFeatures.YPOS]) * travelDistance / EucDistance
		returnTrip = math.sqrt((effector[JF.EffectorFeatures.STARTX] - newX)**2 + (effector[JF.EffectorFeatures.STARTY] - newY)**2)
		return travelDistance + returnTrip


	def getSchedule(self):
		"""
		Return the schedule of events currently chosen.
		"""
		# A schedule should be a list of #effector lists where each element is a target along with some other info (eg timing))

		#[[(2 5min), (1 60min)][3, 2, 3][1]]
		# Effector 1 first hits target 2 (service time 5 min) and then 1 (service time 60 min)
		# Effector 2 first hits target 3 and then 2, and then 3
		# Effector 3 first hits target 1
		return self.schedule


	def update(self, action):
		"""
		Take an action from an agent and apply that action to the effector specified.
		"""
		effectorIndex, taskIndex = action   # Alex passes a 1-hot matrix.  Perform a check and process accordingly
		effector = self.effectorData[effectorIndex, :]
		task = self.taskData[taskIndex, :]
		opportunity = self.opportunityData[effectorIndex, taskIndex, :]
		if opportunity[JF.OpportunityFeatures.SELECTABLE] == False:
			raise Exception(f"This action is not selectable. Effector: {effectorIndex} Task: {taskIndex}")

		#first copy the tensor (not by ref.. make sure to do actual copy)
		if self.keepstack:
			#TODO: make sure this is a copy, not a reference
			self.history.append((self.effectorData, self.taskData, self.opportunityData, self.schedule, self.previousPSuccess))
		#Do not use history, update the schedule directly.
		#Make a separate stack for the schedule so that we don't need to iterate through the whole state history
		self.schedule[effectorIndex].append((taskIndex, opportunity[JF.OpportunityFeatures.TIMECOST]))

		EucDistance = Simulation.EuclideanDistance(effector, task)
		travelDistance = EucDistance - effector[JF.EffectorFeatures.EFFECTIVEDISTANCE]

		if travelDistance > 0: #Calculate the updated position
			effector[JF.EffectorFeatures.XPOS] += (task[JF.TaskFeatures.XPOS] - effector[JF.EffectorFeatures.XPOS]) * travelDistance / EucDistance
			effector[JF.EffectorFeatures.YPOS] += (task[JF.TaskFeatures.YPOS] - effector[JF.EffectorFeatures.YPOS]) * travelDistance / EucDistance
		else:
			pass #We can take action against the target from our current position

		effector[JF.EffectorFeatures.TIMELEFT] -= opportunity[JF.OpportunityFeatures.TIMECOST]
		effector[JF.EffectorFeatures.ENERGYLEFT] -= opportunity[JF.OpportunityFeatures.ENERGYCOST]
		effector[JF.EffectorFeatures.AMMOLEFT] -= effector[JF.EffectorFeatures.AMMORATE]
		#We are dealing with expected plan, not an actual instance of a plan.
		#Damage will be relative to pSuccess rather than sometimes being right and sometimes being wrong
		reward = opportunity[JF.OpportunityFeatures.PSUCCESS] * task[JF.TaskFeatures.VALUE]
		self.taskData[taskIndex][JF.TaskFeatures.VALUE] -= reward

		task[JF.TaskFeatures.SELECTED] += 0.5 #Count the number of engagements so far

		if task[JF.TaskFeatures.SELECTED] >= 1:     #Down the road: opportunities[:,:,selectable] &= task[:,selected] >= 1
			for i in range(0, self.nbEffector):
				self.opportunityData[i][taskIndex][JF.OpportunityFeatures.SELECTABLE] = False
				self.opportunityData[i][taskIndex][JF.OpportunityFeatures.PSUCCESS] = 0

		for i in range(0, self.nbTask):
			EucDistance = Simulation.EuclideanDistance(effector, self.taskData[i])
			#If it wasn't selectable before, could that change?  If not, drop this set of operations whenever something is already unfeasible
			if not effector[JF.EffectorFeatures.STATIC]:
				RTDistance = Simulation.returnDistance(effector, self.taskData[i])
				travelDistance = max(0, EucDistance - effector[JF.EffectorFeatures.EFFECTIVEDISTANCE])
				if (RTDistance > effector[JF.EffectorFeatures.ENERGYLEFT] / (effector[JF.EffectorFeatures.ENERGYRATE]) or
					effector[JF.EffectorFeatures.TIMELEFT] < RTDistance / (effector[JF.EffectorFeatures.SPEED] * PG.STANDARDIZED_TIME)):
					print(f"Return Distance too far: {RTDistance} > {effector[JF.EffectorFeatures.ENERGYLEFT]} / {(effector[JF.EffectorFeatures.ENERGYRATE])} or ")
					print(f"{effector[JF.EffectorFeatures.TIMELEFT]} < {RTDistance} / {(effector[JF.EffectorFeatures.SPEED] * SPEED_CORRECTION)}")
					self.opportunityData[effectorIndex][i][JF.OpportunityFeatures.SELECTABLE] = False
				else:
					self.opportunityData[effectorIndex][i][JF.OpportunityFeatures.TIMECOST] = travelDistance / (effector[JF.EffectorFeatures.SPEED] * SPEED_CORRECTION) #+ effector[JF.EffectorFeatures.DUTYCYCLE]
					self.opportunityData[effectorIndex][i][JF.OpportunityFeatures.ENERGYCOST] = travelDistance * effector[JF.EffectorFeatures.ENERGYRATE] # Energy is related to fuel or essentially range
			else:
				if EucDistance <= effector[JF.EffectorFeatures.EFFECTIVEDISTANCE]:
					self.opportunityData[effectorIndex][i][JF.OpportunityFeatures.TIMECOST] = EucDistance / (effector[JF.EffectorFeatures.SPEED] * SPEED_CORRECTION) #effector[JF.EffectorFeatures.DUTYCYCLE]
					self.opportunityData[effectorIndex][i][JF.OpportunityFeatures.ENERGYCOST] = 0 #Energy is related to fuel or essentially range
				else:
					self.opportunityData[effectorIndex][i][JF.OpportunityFeatures.SELECTABLE] = False

			if self.opportunityData[effectorIndex][i][JF.OpportunityFeatures.TIMECOST] > effector[JF.EffectorFeatures.TIMELEFT]:
				self.opportunityData[effectorIndex][i][JF.OpportunityFeatures.SELECTABLE] = False
			elif self.effectorData[effectorIndex][JF.EffectorFeatures.AMMORATE] > effector[JF.EffectorFeatures.AMMOLEFT]:
				self.opportunityData[effectorIndex][i][JF.OpportunityFeatures.SELECTABLE] = False
			elif self.opportunityData[effectorIndex][i][JF.OpportunityFeatures.ENERGYCOST] > effector[JF.EffectorFeatures.ENERGYLEFT]:
				self.opportunityData[effectorIndex][i][JF.OpportunityFeatures.SELECTABLE] = False

			if self.opportunityData[effectorIndex][i][JF.OpportunityFeatures.SELECTABLE] == False:
				self.opportunityData[effectorIndex][i][JF.OpportunityFeatures.PSUCCESS] = 0

		#self.opportunityData[:,:,JF.OpportunityFeatures.PSUCCESS] &= self.opportunityData[:,:,JF.OpportunityFeatures.SELECTABLE]
		if np.sum(self.opportunityData[:, :, JF.OpportunityFeatures.SELECTABLE]) >= 1:
			terminal = False
		else:
			terminal = True

		return self.FormatState(self.effectorData, self.taskData, self.opportunityData), reward, terminal


	def undo(self):
		"""
		Return to the previous state.  This can help in a depth-first-search style action selection
		"""
		self.effectorData, self.taskData, self.opportunityData, self.schedule, self.previousPSuccess = self.history.pop()


def MergeState(effectorData, taskData, opportunityData):
	"""
	Convert values from real-world to normalized [0,1] and convert representation from vector to 3D tensor

	copy vectors for effector and task, and tensor for opportunity
	remove data that will not be required for the neural network
	normalize vector values according to the scale
	make m copies of each effector vector for an n.m.p tensor
	make n copies of each target vector for an n.m.q tensor
	make a single tensor with each tensor appended in an n.m.(p+q+r)
	"""
	# TODO: change this to use "unsqueeze"
	effectors = np.zeros((len(taskData), len(effectorData), len(JF.EffectorFeatures))) #m.n.p
	tasks = np.zeros((len(effectorData), len(taskData), len(JF.TaskFeatures))) #n.m.q
	effectors += effectorData
	tasks += taskData
	effectors = effectors.transpose([1,0,2]) #Transpose from m.n.p to n.m.p
	return np.concatenate((effectors, tasks, opportunityData), axis=2) #concatenate on the 3rd axis

def UnMergeState(state):
	effectorData = state[:,0,:len(JF.EffectorFeatures)]
	taskData = state[0,:,len(JF.EffectorFeatures):len(JF.EffectorFeatures) + len(JF.TaskFeatures)]
	opportunityData = state[:,:,len(JF.EffectorFeatures) + len(JF.TaskFeatures):]
	return effectorData, taskData, opportunityData

def LoadProblem(filename):
	with open(filename, 'r') as file:
		problem = json.load(file)
	return problem


def SaveProblem(problem, filename):
	with open(filename, 'w') as file:
		json.dump(problem, file)


def main():
	jeremy = 0
	alex = 1
	agents = [JeremyAgent, AlexAgent]
	agentSelection = 0

	problemGenerators = [PG.AllPlanes, PG.InfantryOnly, PG.CombatArms, PG.BoatsBoatsBoats]
	problemGenerator = random.choice(problemGenerators)

	problemGenerator = PG.Tiny

	env = Simulation(MergeState, keepstack=True)
	agent = agents[agentSelection]
	simProblem = problemGenerator()
	state = env.reset(simProblem) #get initial state or load a new problem
	while True:
		terminal = False
		while not terminal:
			action = agent.getAction(state)
			try:
				new_state, reward, terminal =  env.update(action)
			except Exception as e:
				print(f"Error: {e}")
				print(f"Action: {action}")
				continue
			agent.learn(state, action, reward, new_state, terminal)
			state = new_state
			print(f"reward returned: {reward}")
		printState(state)
		print(f"schedule: {env.getSchedule()}")
		state = env.reset()

if __name__ == "__main__":
    main()
