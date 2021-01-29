import numpy as np
import json
import random
import math
import JFAFeatures as JF

MAX_SPEED = 6000

SPEED_NORMALIZATION = 1 / MAX_SPEED
MIN_RANGE = 10 # Assume that scale will be greater than 10 and less than 1000
MAX_RANGE = 1000

DISPLACEMENT_NORMALIZATION = max(MAX_SPEED, MAX_RANGE)

#ENERGYRATE_NORMALIZATION = MIN_RANGE / Arena[JF.ArenaFeatures.SCALE]


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


def boundsChecked(array):
	"""
	TODO: This function brings the time into scope, but negatively affects planes' distance.
	"""
	#for i in range(0,array.shape[0]):
		#array[i] = array[i] if array[i] < 1 else 1
	return array


def newPlane(Arena):
	"""
	Aircraft is limited in munitions (a tank carries almost infinite ammunition).  Aircraft is designed to strike a target and leave.
	It will loiter about 10 minutes away until a target is called, engage, and then return to re-arm.
	Aircraft engage with a high degree of lethality (size of munitions is large compared to target) and they generally operate in pairs.
	May be able to engage every couple of hours.
	"""
	ActiveTime = 2
	TimeFactor = min(ActiveTime, Arena[JF.ArenaFeatures.TIMEHORIZON])
	plane = np.zeros(len(JF.EffectorFeatures))
	plane[JF.EffectorFeatures.XPOS] = 0
	plane[JF.EffectorFeatures.YPOS] = random.uniform(0, Arena[JF.ArenaFeatures.FRONTAGE]) / DISPLACEMENT_NORMALIZATION
	plane[JF.EffectorFeatures.STARTX] = plane[JF.EffectorFeatures.XPOS]
	plane[JF.EffectorFeatures.STARTY] = plane[JF.EffectorFeatures.YPOS]
	plane[JF.EffectorFeatures.STATIC] = False
	plane[JF.EffectorFeatures.SPEED] = TimeFactor * 1000 / (Arena[JF.ArenaFeatures.SCALE] * DISPLACEMENT_NORMALIZATION)
	plane[JF.EffectorFeatures.AMMOLEFT] = 1
	plane[JF.EffectorFeatures.ENERGYLEFT] = 1000 / (Arena[JF.ArenaFeatures.SCALE] * DISPLACEMENT_NORMALIZATION)
	plane[JF.EffectorFeatures.TIMELEFT] = min(1, ActiveTime / Arena[JF.ArenaFeatures.TIMEHORIZON])
	plane[JF.EffectorFeatures.EFFECTIVEDISTANCE] = 2 / (Arena[JF.ArenaFeatures.SCALE] * DISPLACEMENT_NORMALIZATION)
	plane[JF.EffectorFeatures.AMMORATE] = 1 / 2
	plane[JF.EffectorFeatures.ENERGYRATE] = 1 # Arena[JF.ArenaFeatures.SCALE] / DISPLACEMENT_NORMALIZATION
	return boundsChecked(plane)


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
	ActiveTime = 8
	TimeFactor = min(ActiveTime, Arena[JF.ArenaFeatures.TIMEHORIZON])
	if Arena[JF.ArenaFeatures.COASTLINE] <= 0:
		raise Exception("No water present in Arena")
	boat = np.zeros(len(JF.EffectorFeatures))
	boat[JF.EffectorFeatures.XPOS] = random.uniform(0, Arena[JF.ArenaFeatures.COASTLINE]) / DISPLACEMENT_NORMALIZATION
	boat[JF.EffectorFeatures.YPOS] = random.uniform(0, Arena[JF.ArenaFeatures.FRONTAGE]) / DISPLACEMENT_NORMALIZATION
	boat[JF.EffectorFeatures.STARTX] = boat[JF.EffectorFeatures.XPOS]
	boat[JF.EffectorFeatures.STARTY] = boat[JF.EffectorFeatures.YPOS]
	boat[JF.EffectorFeatures.STATIC] = True
	boat[JF.EffectorFeatures.SPEED] = TimeFactor * 4939 / (DISPLACEMENT_NORMALIZATION * Arena[JF.ArenaFeatures.SCALE])
	boat[JF.EffectorFeatures.AMMOLEFT] = 1
	boat[JF.EffectorFeatures.ENERGYLEFT] = 0
	boat[JF.EffectorFeatures.TIMELEFT] = 1
	boat[JF.EffectorFeatures.EFFECTIVEDISTANCE] = 45 / (Arena[JF.ArenaFeatures.SCALE] * DISPLACEMENT_NORMALIZATION)
	boat[JF.EffectorFeatures.AMMORATE] = 1 / 16  # Could also pick at random to simulate previous engagements.  random.randint(1,16)
	boat[JF.EffectorFeatures.ENERGYRATE] = 0
	return boundsChecked(boat)


def NewArtillery(Arena):
	"""
	This will be a "missile battery", which has 9 launchers and can fire 12 missiles.  Could hit multiple targets at once.
	Doug says "the type of target matters a great deal", TODO: look more into this to ensure we properly match PSuccess
	Range will be 60-80km, and will be placed 30-50km from the front-line.

	This could also be "tube artillery", but that is more about support than destruction.  This will only support direct-fire engagements for front-line units.
	"""
	ActiveTime = 8
	TimeFactor = min(ActiveTime, Arena[JF.ArenaFeatures.TIMEHORIZON])
	artillery = np.zeros(len(JF.EffectorFeatures))
	artillery[JF.EffectorFeatures.XPOS] = random.uniform(Arena[JF.ArenaFeatures.COASTLINE], Arena[JF.ArenaFeatures.FRONTLINE]) / DISPLACEMENT_NORMALIZATION
	artillery[JF.EffectorFeatures.YPOS] = random.uniform(0, Arena[JF.ArenaFeatures.FRONTAGE]) / DISPLACEMENT_NORMALIZATION
	artillery[JF.EffectorFeatures.STARTX] = artillery[JF.EffectorFeatures.XPOS]
	artillery[JF.EffectorFeatures.STARTY] = artillery[JF.EffectorFeatures.YPOS]
	artillery[JF.EffectorFeatures.STATIC] = True
	artillery[JF.EffectorFeatures.SPEED] = TimeFactor * 6000 / (DISPLACEMENT_NORMALIZATION * Arena[JF.ArenaFeatures.SCALE])
	artillery[JF.EffectorFeatures.AMMOLEFT] = 1
	artillery[JF.EffectorFeatures.ENERGYLEFT] = 0
	artillery[JF.EffectorFeatures.TIMELEFT] = 1
	artillery[JF.EffectorFeatures.EFFECTIVEDISTANCE] = 45 / (Arena[JF.ArenaFeatures.SCALE] * DISPLACEMENT_NORMALIZATION)
	artillery[JF.EffectorFeatures.AMMORATE] = 1 / 20
	artillery[JF.EffectorFeatures.ENERGYRATE] = 0
	return boundsChecked(artillery)


def NewArmoured(Arena):
	"""
	Tanks have near infinite amount of ammunition.
	"""
	ActiveTime = 8
	TimeFactor = min(ActiveTime, Arena[JF.ArenaFeatures.TIMEHORIZON])
	armoured = np.zeros(len(JF.EffectorFeatures))
	armoured[JF.EffectorFeatures.XPOS] = random.uniform(Arena[JF.ArenaFeatures.COASTLINE], Arena[JF.ArenaFeatures.FRONTLINE]) / DISPLACEMENT_NORMALIZATION
	armoured[JF.EffectorFeatures.YPOS] = random.uniform(0, Arena[JF.ArenaFeatures.FRONTAGE]) / DISPLACEMENT_NORMALIZATION
	armoured[JF.EffectorFeatures.STARTX] = armoured[JF.EffectorFeatures.XPOS]
	armoured[JF.EffectorFeatures.STARTY] = armoured[JF.EffectorFeatures.YPOS]
	armoured[JF.EffectorFeatures.STATIC] = False
	armoured[JF.EffectorFeatures.SPEED] = TimeFactor * 40 / (DISPLACEMENT_NORMALIZATION * Arena[JF.ArenaFeatures.SCALE])
	armoured[JF.EffectorFeatures.AMMOLEFT] = 1
	armoured[JF.EffectorFeatures.ENERGYLEFT] = 250 / (Arena[JF.ArenaFeatures.SCALE] * DISPLACEMENT_NORMALIZATION)
	armoured[JF.EffectorFeatures.TIMELEFT] = 1
	armoured[JF.EffectorFeatures.EFFECTIVEDISTANCE] = 0.5 / (Arena[JF.ArenaFeatures.SCALE] * DISPLACEMENT_NORMALIZATION)
	armoured[JF.EffectorFeatures.AMMORATE] = 1 / 13
	armoured[JF.EffectorFeatures.ENERGYRATE] = 1 # Arena[JF.ArenaFeatures.SCALE] / DISPLACEMENT_NORMALIZATION
	return boundsChecked(armoured)


def newInfantry(Arena):
	"""
	Infantry engaging infantry will generally wipe each other out, so are only able to engage once.
	"""
	ActiveTime = 8
	TimeFactor = min(ActiveTime, Arena[JF.ArenaFeatures.TIMEHORIZON])
	infantry = np.zeros(len(JF.EffectorFeatures))
	infantry[JF.EffectorFeatures.XPOS] = random.uniform(Arena[JF.ArenaFeatures.COASTLINE], Arena[JF.ArenaFeatures.FRONTLINE]) / DISPLACEMENT_NORMALIZATION
	infantry[JF.EffectorFeatures.YPOS] = random.uniform(0, Arena[JF.ArenaFeatures.FRONTAGE]) / DISPLACEMENT_NORMALIZATION
	infantry[JF.EffectorFeatures.STARTX] = infantry[JF.EffectorFeatures.XPOS]
	infantry[JF.EffectorFeatures.STARTY] = infantry[JF.EffectorFeatures.YPOS]
	infantry[JF.EffectorFeatures.STATIC] = False
	infantry[JF.EffectorFeatures.SPEED] = TimeFactor * 5 / (DISPLACEMENT_NORMALIZATION * Arena[JF.ArenaFeatures.SCALE])
	infantry[JF.EffectorFeatures.AMMOLEFT] = 1
	infantry[JF.EffectorFeatures.ENERGYLEFT] = 20 / (Arena[JF.ArenaFeatures.SCALE] * DISPLACEMENT_NORMALIZATION)
	infantry[JF.EffectorFeatures.TIMELEFT] = 1
	infantry[JF.EffectorFeatures.EFFECTIVEDISTANCE] = 0.75 / (Arena[JF.ArenaFeatures.SCALE] * DISPLACEMENT_NORMALIZATION)
	infantry[JF.EffectorFeatures.AMMORATE] = 1 / 2
	infantry[JF.EffectorFeatures.ENERGYRATE] = 1 # Arena[JF.ArenaFeatures.SCALE] / DISPLACEMENT_NORMALIZATION
	return boundsChecked(infantry)


def newTarget(Arena):
	target = np.zeros(len(JF.TaskFeatures))
	target[JF.TaskFeatures.XPOS] = random.uniform(Arena[JF.ArenaFeatures.FRONTLINE], 1) / DISPLACEMENT_NORMALIZATION
	target[JF.TaskFeatures.YPOS] = random.uniform(0, Arena[JF.ArenaFeatures.FRONTAGE]) / DISPLACEMENT_NORMALIZATION
	target[JF.TaskFeatures.VALUE] = random.uniform(0.35, 0.65)
	target[JF.TaskFeatures.SELECTED] = 0
	return boundsChecked(target)


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
						self.effectors[i][JF.EffectorFeatures.TIMELEFT] < RTDistance / (self.effectors[i][JF.EffectorFeatures.SPEED])):
						self.opportunities[i][j][JF.OpportunityFeatures.SELECTABLE] = False
						print(f"{RTDistance > self.effectors[i][JF.EffectorFeatures.ENERGYLEFT] / self.effectors[i][JF.EffectorFeatures.ENERGYRATE]} : {RTDistance} > {self.effectors[i][JF.EffectorFeatures.ENERGYLEFT]} / {self.effectors[i][JF.EffectorFeatures.ENERGYRATE]}")
						print(f"{self.effectors[i][JF.EffectorFeatures.TIMELEFT] < RTDistance / (self.effectors[i][JF.EffectorFeatures.SPEED])} : {self.effectors[i][JF.EffectorFeatures.TIMELEFT]} < {RTDistance} / {self.effectors[i][JF.EffectorFeatures.SPEED]}")
						#print(f"Pair ({i},{j}) not selectable RTDist > (ENERGYLEFT / ENERGYRATE): {RTDistance > self.effectors[i][JF.EffectorFeatures.ENERGYLEFT] / (self.effectors[i][JF.EffectorFeatures.ENERGYRATE] / ENERGYRATE_NORMALIZATION)}")
						#print(f" TIMELEFT < RTDist / (Speed * HIGHEST_SPEED): {self.effectors[i][JF.EffectorFeatures.TIMELEFT] < RTDistance / (self.effectors[i][JF.EffectorFeatures.SPEED] / SPEED_NORMALIZATION)}")
						#print(f"TIMELEFT: {self.effectors[i][JF.EffectorFeatures.TIMELEFT]}")
						#print(f"RTDist: {RTDistance}")
						#print(f"Speed: {self.effectors[i][JF.EffectorFeatures.SPEED]}")
						#print(f"Speed Normalize: {SPEED_NORMALIZATION}")
						#print(f"EnergyLeft: {self.effectors[i][JF.EffectorFeatures.ENERGYLEFT]}")
						#print(f"EnergyRate: {(self.effectors[i][JF.EffectorFeatures.ENERGYRATE] / ENERGYRATE_NORMALIZATION)}")
					else:
						self.opportunities[i][j][JF.OpportunityFeatures.TIMECOST] = travelDistance / (self.effectors[i][JF.EffectorFeatures.SPEED]) #+ self.effectors[i][JF.EffectorFeatures.DUTYCYCLE]
						self.opportunities[i][j][JF.OpportunityFeatures.ENERGYCOST] = travelDistance * self.effectors[i][JF.EffectorFeatures.ENERGYRATE] #Energy is related to fuel or essentially range
				else:
					self.opportunities[i][j][JF.OpportunityFeatures.TIMECOST] = 0
					if EucDistance > self.effectors[i][JF.EffectorFeatures.EFFECTIVEDISTANCE]:
						self.opportunities[i][j][JF.OpportunityFeatures.SELECTABLE] = False

				#TODO: Adjust PSuccess based on effector probabilities
				if self.opportunities[i][j][JF.OpportunityFeatures.SELECTABLE] == True:
					self.opportunities[i][j][JF.OpportunityFeatures.PSUCCESS] = random.uniform(0.4, 0.7)
				else:
					self.opportunities[i][j][JF.OpportunityFeatures.PSUCCESS] = 0


def AllPlanes():
	arena = np.zeros(len(JF.ArenaFeatures))
	arena[JF.ArenaFeatures.SCALE] = 500
	arena[JF.ArenaFeatures.COASTLINE] = 0.01
	arena[JF.ArenaFeatures.FRONTLINE] = 0.2
	arena[JF.ArenaFeatures.TIMEHORIZON] = 4
	planes = random.randint(9,13)
	targets = random.randint(10, 50)

	PG = ProblemGenerator()
	return PG.newProblem(arena, targets, planes=planes)


def BoatsBoatsBoats():
	arena = np.zeros(len(JF.ArenaFeatures))
	arena[JF.ArenaFeatures.SCALE] = 100
	arena[JF.ArenaFeatures.COASTLINE] = 0.3
	arena[JF.ArenaFeatures.FRONTLINE] = 0.301
	arena[JF.ArenaFeatures.TIMEHORIZON] = 4
	targets = random.randint(10, 50)

	PG = ProblemGenerator()
	return PG.newProblem(arena, targets, frigates=random.randint(9,13))


def InfantryOnly():
	arena = np.zeros(len(JF.ArenaFeatures))
	arena[JF.ArenaFeatures.SCALE] = 10
	arena[JF.ArenaFeatures.COASTLINE] = 0.01
	arena[JF.ArenaFeatures.FRONTLINE] = 0.4
	arena[JF.ArenaFeatures.TIMEHORIZON] = 8

	targets = random.randint(4, 9)
	PG = ProblemGenerator()
	return PG.newProblem(arena, targets, infantry=random.randint(5,15))

def CombatArms():
	arena = np.zeros(len(JF.ArenaFeatures))
	arena[JF.ArenaFeatures.SCALE] = 30
	arena[JF.ArenaFeatures.COASTLINE] = 0.01
	arena[JF.ArenaFeatures.FRONTLINE] = 0.4
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
	arena[JF.ArenaFeatures.COASTLINE] = 0.01
	arena[JF.ArenaFeatures.FRONTLINE] = 0.2
	arena[JF.ArenaFeatures.FRONTAGE] = 0.4
	arena[JF.ArenaFeatures.TIMEHORIZON] = 4
	armoured = 1
	artillery = 1
	frigates = 1
	planes = 1
	infantry = 1
	targets = 2

	PG = ProblemGenerator()
	return PG.newProblem(arena, targets, infantry=infantry, armoured=armoured, artillery=artillery)
