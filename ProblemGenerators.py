import numpy as np
import json
import random
import math
import JFAFeatures as JF

MAX_SPEED = 6000
MIN_RANGE = 10 # Assume that scale will be greater than 10 and less than 1000
MAX_RANGE = 1000
DISPLACEMENT_NORMALIZATION = max(MAX_SPEED, MAX_RANGE)
STANDARDIZED_SCALE = 500
STANDARDIZED_TIME = 12

def loadProblem(filename):
	problem = {}
	with open(filename, 'r') as file:
		fromFile = json.load(file)
	for key in fromFile.keys():
		problem[key] = np.asarray(fromFile[key])
	return problem

def saveProblem(problem, filename):
	toFile = {}
	for key in problem.keys():
		toFile[key] = problem[key].tolist()
	with open(filename, 'w') as file:
		json.dump(toFile, file)

def newEffector(Arena, x_range, y_range, static, speed, range, time, effective_distance, ammo):
	x_low, x_high = x_range
	y_low, y_high = y_range
	TimeFactor = min(time, Arena[JF.ArenaFeatures.TIMEHORIZON])
	effector = np.zeros(len(JF.EffectorFeatures))
	effector[JF.EffectorFeatures.XPOS] = random.uniform(x_low, x_high) / STANDARDIZED_SCALE
	effector[JF.EffectorFeatures.YPOS] = random.uniform(y_low, y_high) / STANDARDIZED_SCALE
	effector[JF.EffectorFeatures.STARTX] = effector[JF.EffectorFeatures.XPOS]
	effector[JF.EffectorFeatures.STARTY] = effector[JF.EffectorFeatures.YPOS]
	effector[JF.EffectorFeatures.STATIC] = static
	effector[JF.EffectorFeatures.SPEED] = speed / (STANDARDIZED_SCALE * MAX_SPEED)
	effector[JF.EffectorFeatures.AMMOLEFT] = 1
	effector[JF.EffectorFeatures.ENERGYLEFT] = range / (STANDARDIZED_SCALE * MAX_RANGE)
	effector[JF.EffectorFeatures.TIMELEFT] = TimeFactor / STANDARDIZED_TIME
	effector[JF.EffectorFeatures.EFFECTIVEDISTANCE] = effective_distance / STANDARDIZED_SCALE
	effector[JF.EffectorFeatures.AMMORATE] = 1 / ammo
	effector[JF.EffectorFeatures.ENERGYRATE] = 1 / MAX_RANGE
	return effector

def newPlane(Arena):
	"""
	Aircraft is limited in munitions (a tank carries almost infinite ammunition).  Aircraft is designed to strike a target and leave.
	It will loiter about 10 minutes away until a target is called, engage, and then return to re-arm.
	Aircraft engage with a high degree of lethality (size of munitions is large compared to target) and they generally operate in pairs.
	May be able to engage every couple of hours.
	"""
	x_range = (0, 0)
	y_range = (0, Arena[JF.ArenaFeatures.FRONTAGE])
	static = False
	speed = 1000
	range = 1000
	time = 2
	effective_distance = 2
	ammo = 2
	return newEffector(Arena, x_range, y_range, static, speed, range, time, effective_distance, ammo)

def newHelicopter(Arena):
	"""
	Extremely lethal with a much quicker cycle time compared to fixed-wing.
	It can engage 2x as many targets, but it is more susceptible to counter-fire (If I can see them, they can see me)
	"""
	pass

def newBoat(Arena):
	"""
	Boats do not engage in gunfire with land targets, they will only use missiles.  Range is considerable, up to 100km for "harpoon"s
	Missiles are very lethal and accurate.  Generally used for small, discrete, high-value targets (command post).  There is a finite number of missiles in a boat.
	We assume Halifax class with 2x 8 cell launchers, and the boat cannot reload at sea.  Duty cycle is extremely short, and will probably expend its stores in 10 minutes.
	Skippy says "whatever is in the launchers is what you have.  You must re-arm at a jetty"
	Skippy also says "The ship must have RADAR lock until the missile hits.  You cannot select a new target until the munitions impact"
	"""
	if Arena[JF.ArenaFeatures.COASTLINE] <= 0:
		raise Exception("No water present in Arena")
	x_range = (0, Arena[JF.ArenaFeatures.COASTLINE])
	y_range = (0, Arena[JF.ArenaFeatures.FRONTAGE])
	static = True
	speed = 4939
	range = 0
	time = 8
	effective_distance = 45
	ammo = 16
	return newEffector(Arena, x_range, y_range, static, speed, range, time, effective_distance, ammo)

def NewArtillery(Arena):
	"""
	This will be a "missile battery", which has 9 launchers and can fire 12 missiles.  Could hit multiple targets at once.
	Doug says "the type of target matters a great deal", TODO: look more into this to ensure we properly match PSuccess
	Range will be 60-80km, and will be placed 30-50km from the front-line.

	This could also be "tube artillery", but that is more about support than destruction.  This will only support direct-fire engagements for front-line units.
	"""
	x_range = (Arena[JF.ArenaFeatures.COASTLINE], Arena[JF.ArenaFeatures.FRONTLINE])
	y_range = (0, Arena[JF.ArenaFeatures.FRONTAGE])
	static = True
	speed = 6000
	range = 0
	time = 8
	effective_distance = 45
	ammo = 20
	return newEffector(Arena, x_range, y_range, static, speed, range, time, effective_distance, ammo)

def NewArmoured(Arena):
	"""
	Tanks have near infinite amount of ammunition.
	"""
	x_range = (Arena[JF.ArenaFeatures.COASTLINE], Arena[JF.ArenaFeatures.FRONTLINE])
	y_range = (0, Arena[JF.ArenaFeatures.FRONTAGE])
	static = False
	speed = 40
	range = 250
	time = 8
	effective_distance = 0.5
	ammo = 13
	return newEffector(Arena, x_range, y_range, static, speed, range, time, effective_distance, ammo)

def newInfantry(Arena):
	"""
	Infantry engaging infantry will generally wipe each other out, so are only able to engage once.
	"""
	x_range = (Arena[JF.ArenaFeatures.COASTLINE], Arena[JF.ArenaFeatures.FRONTLINE])
	y_range = (0, Arena[JF.ArenaFeatures.FRONTAGE])
	static = False
	speed = 5
	range = 20
	time = 8
	effective_distance = 0.75
	ammo = 2
	return newEffector(Arena, x_range, y_range, static, speed, range, time, effective_distance, ammo)

def newTarget(Arena):
	target = np.zeros(len(JF.TaskFeatures))
	target[JF.TaskFeatures.XPOS] = random.uniform(Arena[JF.ArenaFeatures.FRONTLINE], Arena[JF.ArenaFeatures.SCALE]) / STANDARDIZED_SCALE
	target[JF.TaskFeatures.YPOS] = random.uniform(0, Arena[JF.ArenaFeatures.FRONTAGE]) / STANDARDIZED_SCALE
	target[JF.TaskFeatures.VALUE] = random.uniform(0.35, 0.65)
	target[JF.TaskFeatures.SELECTED] = 0
	return target

def EuclideanDistance(effector, task):
	return math.sqrt( (effector[JF.EffectorFeatures.XPOS] - task[JF.TaskFeatures.XPOS])**2 + (effector[JF.EffectorFeatures.YPOS] - task[JF.TaskFeatures.YPOS])**2)

def returnDistance(effector, task):
	EucDistance = EuclideanDistance(effector, task)
	travelDistance = max(EucDistance - effector[JF.EffectorFeatures.EFFECTIVEDISTANCE], 0)
	newX = effector[JF.EffectorFeatures.XPOS] + (task[JF.TaskFeatures.XPOS] - effector[JF.EffectorFeatures.XPOS]) * travelDistance / EucDistance
	newY = effector[JF.EffectorFeatures.YPOS] + (task[JF.TaskFeatures.YPOS] - effector[JF.EffectorFeatures.YPOS]) * travelDistance / EucDistance
	returnTrip = math.sqrt((effector[JF.EffectorFeatures.STARTX] - newX)**2 + (effector[JF.EffectorFeatures.STARTY] - newY)**2)
	return travelDistance + returnTrip


class ProblemGenerator():

	#TODO: Take in a number of types of effectors, and then generate the problem based on these quantities
	def __init__(self):
		self.arena = np.zeros(len(JF.ArenaFeatures))
		self.effectors = []
		self.targets = []
		self.opportunities = []

	def newProblem(self, arena, targets, planes=0, frigates=0, artillery=0, armoured=0, infantry=0):
		self.arena = arena
		self.effectors = []
		for i in range(planes):
			self.effectors.append(newPlane(self.arena))
		for i in range(frigates):
			self.effectors.append(newBoat(self.arena))
		for i in range(artillery):
			self.effectors.append(NewArtillery(self.arena))
		for i in range(armoured):
			self.effectors.append(NewArmoured(self.arena))
		for i in range(infantry):
			self.effectors.append(newInfantry(self.arena))
		self.populateTargets(targets)
		self.populateOpportunities()
		return self.formatProblem()

	def formatProblem(self):
		problem = {}
		problem['Arena'] = self.arena
		problem['Effectors'] = self.effectors
		problem['Targets'] = self.targets
		problem['Opportunities'] = self.opportunities
		return problem

	def saveProblem(filename):
		problem = self.formatProblem()
		with open(filename, 'w') as file:
			json.dump(problem, file)

	def populateTargets(self, qty):
		self.targets = []
		for i in range(qty):
			self.targets.append(newTarget(self.arena))

	def populateOpportunities(self):
		"""
		Engagements have a near 100% lethality.  "If I can see it, I can kill it."
		"""
		self.opportunities = np.zeros((len(self.effectors), len(self.targets), len(JF.OpportunityFeatures)))
		ENERGYRATE_NORMALIZATION = MIN_RANGE / self.arena[JF.ArenaFeatures.SCALE]
		for i in range(0, len(self.effectors)):
			for j in range(0, len(self.targets)):
				self.opportunities[i][j][JF.OpportunityFeatures.SELECTABLE] = True
				EucDistance = EuclideanDistance(self.effectors[i], self.targets[j])
				travelDistance = EucDistance - self.effectors[i][JF.EffectorFeatures.EFFECTIVEDISTANCE]
				if not self.effectors[i][JF.EffectorFeatures.STATIC]:
					RTDistance = returnDistance(self.effectors[i], self.targets[j])
					if travelDistance <= 0:
						self.opportunities[i][j][JF.OpportunityFeatures.TIMECOST] = 0
						self.opportunities[i][j][JF.OpportunityFeatures.ENERGYCOST] = 0
					elif (RTDistance > self.effectors[i][JF.EffectorFeatures.ENERGYLEFT] / (self.effectors[i][JF.EffectorFeatures.ENERGYRATE]) or
						self.effectors[i][JF.EffectorFeatures.TIMELEFT] < RTDistance / (self.effectors[i][JF.EffectorFeatures.SPEED] * STANDARDIZED_TIME * MAX_SPEED)):
						self.opportunities[i][j][JF.OpportunityFeatures.SELECTABLE] = False
						print(f"Effector: {i}, Target: {j}")
						print(f"Dist: {RTDistance > self.effectors[i][JF.EffectorFeatures.ENERGYLEFT] / self.effectors[i][JF.EffectorFeatures.ENERGYRATE]} : {RTDistance} > {self.effectors[i][JF.EffectorFeatures.ENERGYLEFT]} / {self.effectors[i][JF.EffectorFeatures.ENERGYRATE]}")
						print(f"Time: {self.effectors[i][JF.EffectorFeatures.TIMELEFT] < RTDistance / (self.effectors[i][JF.EffectorFeatures.SPEED] * STANDARDIZED_TIME * MAX_SPEED)} : {self.effectors[i][JF.EffectorFeatures.TIMELEFT]} < {RTDistance} / {self.effectors[i][JF.EffectorFeatures.SPEED] * STANDARDIZED_TIME * MAX_SPEED}")
					else:
						self.opportunities[i][j][JF.OpportunityFeatures.TIMECOST] = travelDistance / (self.effectors[i][JF.EffectorFeatures.SPEED] * STANDARDIZED_TIME * MAX_SPEED)
						self.opportunities[i][j][JF.OpportunityFeatures.ENERGYCOST] = travelDistance * self.effectors[i][JF.EffectorFeatures.ENERGYRATE] #Energy is related to fuel or essentially range
				else:
					self.opportunities[i][j][JF.OpportunityFeatures.TIMECOST] = 0
					if EucDistance > self.effectors[i][JF.EffectorFeatures.EFFECTIVEDISTANCE]:
						self.opportunities[i][j][JF.OpportunityFeatures.SELECTABLE] = False

				if self.opportunities[i][j][JF.OpportunityFeatures.SELECTABLE] == True:
					self.opportunities[i][j][JF.OpportunityFeatures.PSUCCESS] = random.uniform(0.4, 0.7)
				else:
					self.opportunities[i][j][JF.OpportunityFeatures.PSUCCESS] = 0

def AllPlanes():
	arena = np.zeros(len(JF.ArenaFeatures))
	arena[JF.ArenaFeatures.SCALE] = 500
	arena[JF.ArenaFeatures.COASTLINE] = 0 * arena[JF.ArenaFeatures.SCALE]
	arena[JF.ArenaFeatures.FRONTLINE] = 0.2 * arena[JF.ArenaFeatures.SCALE]
	arena[JF.ArenaFeatures.TIMEHORIZON] = 4
	planes = random.randint(9,13)
	targets = random.randint(10, 50)
	PG = ProblemGenerator()
	return PG.newProblem(arena, targets, planes=planes)

def BoatsBoatsBoats():
	arena = np.zeros(len(JF.ArenaFeatures))
	arena[JF.ArenaFeatures.SCALE] = 100
	arena[JF.ArenaFeatures.COASTLINE] = 0.3 * arena[JF.ArenaFeatures.SCALE]
	arena[JF.ArenaFeatures.FRONTLINE] = 0.301 * arena[JF.ArenaFeatures.SCALE]
	arena[JF.ArenaFeatures.TIMEHORIZON] = 4
	targets = random.randint(10, 50)
	PG = ProblemGenerator()
	return PG.newProblem(arena, targets, frigates=random.randint(9,13))

def InfantryOnly():
	arena = np.zeros(len(JF.ArenaFeatures))
	arena[JF.ArenaFeatures.SCALE] = 10
	arena[JF.ArenaFeatures.COASTLINE] = 0.01 * arena[JF.ArenaFeatures.SCALE]
	arena[JF.ArenaFeatures.FRONTLINE] = 0.4 * arena[JF.ArenaFeatures.SCALE]
	arena[JF.ArenaFeatures.TIMEHORIZON] = 8
	targets = random.randint(4, 9)
	PG = ProblemGenerator()
	return PG.newProblem(arena, targets, infantry=random.randint(5,15))

def CombatArms():
	arena = np.zeros(len(JF.ArenaFeatures))
	arena[JF.ArenaFeatures.SCALE] = 30
	arena[JF.ArenaFeatures.COASTLINE] = 0.01 * arena[JF.ArenaFeatures.SCALE]
	arena[JF.ArenaFeatures.FRONTLINE] = 0.4 * arena[JF.ArenaFeatures.SCALE]
	arena[JF.ArenaFeatures.TIMEHORIZON] = 4
	rands = []
	total = random.randint(9,13)
	heapq.heappush(rands, random.randint(0,total))
	heapq.heappush(rands, random.randint(0,total))
	artillery = heapq.heappop(rands)
	armoured = heapq.heappop(rands) - artillery
	infantry = total - (artillery + armoured)
	targets = random.randint(10, 30)
	PG = ProblemGenerator()
	return PG.newProblem(arena, targets, artillery=artillery, armoured=armoured, infantry=infantry)

def Tiny():
	arena = np.zeros(len(JF.ArenaFeatures))
	arena[JF.ArenaFeatures.SCALE] = 50
	arena[JF.ArenaFeatures.COASTLINE] = 0.01 * arena[JF.ArenaFeatures.SCALE]
	arena[JF.ArenaFeatures.FRONTLINE] = 0.2 * arena[JF.ArenaFeatures.SCALE]
	arena[JF.ArenaFeatures.FRONTAGE] = 0.4 * arena[JF.ArenaFeatures.SCALE]
	arena[JF.ArenaFeatures.TIMEHORIZON] = 4
	armoured = 1
	artillery = 1
	frigates = 1
	planes = 1
	infantry = 1
	targets = 2
	PG = ProblemGenerator()
	return PG.newProblem(arena, targets, infantry=infantry, armoured=armoured, artillery=artillery)
