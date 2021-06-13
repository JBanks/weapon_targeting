import numpy as np
import json
import random
import math

if __package__ is not None and len(__package__) > 0:
    print(f"{__name__} using relative import inside of {__package__}")
    from . import features as jf
else:
    import features as jf

MAX_SPEED = 6000
MIN_RANGE = 10  # Assume that scale will be greater than 10 and less than 1000
MAX_RANGE = 1000
DISPLACEMENT_NORMALIZATION = max(MAX_SPEED, MAX_RANGE)
STANDARDIZED_SCALE = 500
STANDARDIZED_TIME = 12
SPEED_CORRECTION = STANDARDIZED_TIME * MAX_SPEED
ASSET_ENCODING = []
for a_type in jf.AssetTypes:
    encoded = [0] * len(jf.AssetTypes)
    encoded[a_type] = 1
    ASSET_ENCODING.append(encoded)


def correct_effector_data(problem):
    if problem['Effectors'].shape[1] == 12:
        extension = np.zeros((len(problem['Effectors']), len(jf.EffectorFeatures) - problem['Effectors'].shape[1]))
        problem['Effectors'] = np.append(problem['Effectors'], extension, axis=1)


def truncate_effector_data(problem):
    if problem['Effectors'].shape[1] == 18:
        problem['Effectors'] = problem['Effectors'][:, 0:12]


def loadProblem(filename):
    problem = {}
    with open(filename, 'r') as file:
        fromFile = json.load(file)
    for key in fromFile.keys():
        problem[key] = np.asarray(fromFile[key])
    correct_effector_data(problem)
    return problem


def saveProblem(problem, filename):
    toFile = {}
    for key in problem.keys():
        toFile[key] = np.asarray(problem[key]).tolist()
    with open(filename, 'w') as file:
        json.dump(toFile, file)


def newEffector(Arena, x_range, y_range, static, speed, range, time, effective_distance, ammo,
                a_type=jf.AssetTypes.INFANTRY):
    x_low, x_high = x_range
    y_low, y_high = y_range
    TimeFactor = min(time, Arena[jf.ArenaFeatures.TIMEHORIZON])
    effector = np.zeros(len(jf.EffectorFeatures))
    effector[jf.EffectorFeatures.XPOS] = random.uniform(x_low, x_high) / STANDARDIZED_SCALE
    effector[jf.EffectorFeatures.YPOS] = random.uniform(y_low, y_high) / STANDARDIZED_SCALE
    effector[jf.EffectorFeatures.STARTX] = effector[jf.EffectorFeatures.XPOS]
    effector[jf.EffectorFeatures.STARTY] = effector[jf.EffectorFeatures.YPOS]
    effector[jf.EffectorFeatures.STATIC] = static
    effector[jf.EffectorFeatures.SPEED] = speed / (STANDARDIZED_SCALE * MAX_SPEED)
    effector[jf.EffectorFeatures.AMMOLEFT] = 1
    effector[jf.EffectorFeatures.ENERGYLEFT] = range / (STANDARDIZED_SCALE * MAX_RANGE)
    effector[jf.EffectorFeatures.TIMELEFT] = TimeFactor / STANDARDIZED_TIME
    effector[jf.EffectorFeatures.EFFECTIVEDISTANCE] = effective_distance / STANDARDIZED_SCALE
    effector[jf.EffectorFeatures.AMMORATE] = 1 / ammo
    effector[jf.EffectorFeatures.ENERGYRATE] = 1 / MAX_RANGE  # Remove
    effector[jf.EffectorFeatures.TYPE + a_type] = 1
    return effector


def newPlane(Arena):
    """
    Aircraft is limited in munitions (a tank carries almost infinite ammunition).  Aircraft is designed to strike a target and leave.
    It will loiter about 10 minutes away until a target is called, engage, and then return to re-arm.
    Aircraft engage with a high degree of lethality (size of munitions is large compared to target) and they generally operate in pairs.
    May be able to engage every couple of hours.
    """
    x_range = (0, 0)
    y_range = (0, Arena[jf.ArenaFeatures.FRONTAGE])
    static = False
    speed = 1000
    range = 1000
    time = 2
    effective_distance = 2
    ammo = 2
    a_type = jf.AssetTypes.PLANE
    return newEffector(Arena, x_range, y_range, static, speed, range, time, effective_distance, ammo, a_type)


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
    if Arena[jf.ArenaFeatures.COASTLINE] <= 0:
        raise Exception("No water present in Arena")
    x_range = (0, Arena[jf.ArenaFeatures.COASTLINE])
    y_range = (0, Arena[jf.ArenaFeatures.FRONTAGE])
    static = True
    speed = 4939
    range = 0
    time = 8
    effective_distance = 45
    ammo = 16
    a_type = jf.AssetTypes.FRIGATE
    return newEffector(Arena, x_range, y_range, static, speed, range, time, effective_distance, ammo, a_type)


def newArtillery(Arena):
    """
    This will be a "missile battery", which has 9 launchers and can fire 12 missiles.  Could hit multiple targets at once.
    Doug says "the type of target matters a great deal", TODO: look more into this to ensure we properly match PSuccess
    Range will be 60-80km, and will be placed 30-50km from the front-line.

    This could also be "tube artillery", but that is more about support than destruction.  This will only support direct-fire engagements for front-line units.
    """
    x_range = (Arena[jf.ArenaFeatures.COASTLINE], Arena[jf.ArenaFeatures.FRONTLINE])
    y_range = (0, Arena[jf.ArenaFeatures.FRONTAGE])
    static = True
    speed = 6000
    range = 0
    time = 8
    effective_distance = 45
    ammo = 20
    a_type = jf.AssetTypes.ARTILLERY
    return newEffector(Arena, x_range, y_range, static, speed, range, time, effective_distance, ammo, a_type)


def newArmoured(Arena):
    """
    Tanks have near infinite amount of ammunition.
    """
    x_range = (Arena[jf.ArenaFeatures.COASTLINE], Arena[jf.ArenaFeatures.FRONTLINE])
    y_range = (0, Arena[jf.ArenaFeatures.FRONTAGE])
    static = False
    speed = 40
    range = 250
    time = 8
    effective_distance = 0.5
    ammo = 13
    a_type = jf.AssetTypes.ARMOURED
    return newEffector(Arena, x_range, y_range, static, speed, range, time, effective_distance, ammo, a_type)


def newInfantry(Arena):
    """
    Infantry engaging infantry will generally wipe each other out, so are only able to engage once.
    """
    x_range = (Arena[jf.ArenaFeatures.COASTLINE], Arena[jf.ArenaFeatures.FRONTLINE])
    y_range = (0, Arena[jf.ArenaFeatures.FRONTAGE])
    static = False
    speed = 5
    range = 20
    time = 8
    effective_distance = 0.75
    ammo = 2
    a_type = jf.AssetTypes.INFANTRY
    return newEffector(Arena, x_range, y_range, static, speed, range, time, effective_distance, ammo, a_type)


def newTarget(Arena, value=None):
    target = np.zeros(len(jf.TaskFeatures))
    target[jf.TaskFeatures.XPOS] = random.uniform(Arena[jf.ArenaFeatures.FRONTLINE],
                                                  Arena[jf.ArenaFeatures.SCALE]) / STANDARDIZED_SCALE
    target[jf.TaskFeatures.YPOS] = random.uniform(0, Arena[jf.ArenaFeatures.FRONTAGE]) / STANDARDIZED_SCALE
    if value is None:
        target[jf.TaskFeatures.VALUE] = random.uniform(0.35, 0.65)
    else:
        target[jf.TaskFeatures.VALUE] = value
    target[jf.TaskFeatures.SELECTED] = 0
    return target


def euclideanDistance(effector, task):
    return math.sqrt((effector[jf.EffectorFeatures.XPOS] - task[jf.TaskFeatures.XPOS]) ** 2 + (
            effector[jf.EffectorFeatures.YPOS] - task[jf.TaskFeatures.YPOS]) ** 2)


def returnDistance(effector, task):
    EucDistance = euclideanDistance(effector, task)
    travelDistance = max(EucDistance - effector[jf.EffectorFeatures.EFFECTIVEDISTANCE], 0)
    newX = effector[jf.EffectorFeatures.XPOS] + (
            task[jf.TaskFeatures.XPOS] - effector[jf.EffectorFeatures.XPOS]) * travelDistance / EucDistance
    newY = effector[jf.EffectorFeatures.YPOS] + (
            task[jf.TaskFeatures.YPOS] - effector[jf.EffectorFeatures.YPOS]) * travelDistance / EucDistance
    returnTrip = math.sqrt(
        (effector[jf.EffectorFeatures.STARTX] - newX) ** 2 + (effector[jf.EffectorFeatures.STARTY] - newY) ** 2)
    return travelDistance + returnTrip


class ProblemGenerator():

    def __init__(self):
        self.arena = np.zeros(len(jf.ArenaFeatures))
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
            self.effectors.append(newArtillery(self.arena))
        for i in range(armoured):
            self.effectors.append(newArmoured(self.arena))
        for i in range(infantry):
            self.effectors.append(newInfantry(self.arena))
        self.populateTargets(targets)
        self.populateOpportunities()
        return self.formatProblem()

    def formatProblem(self):
        problem = {'Arena': self.arena,
                   'Effectors': np.asarray(self.effectors),
                   'Targets': np.asarray(self.targets),
                   'Opportunities': np.asarray(self.opportunities)}
        return problem

    def populateTargets(self, qty):
        self.targets = []
        for i in range(qty):
            self.targets.append(newTarget(self.arena))

    def populateOpportunities(self):
        """
        Engagements have a near 100% lethality.  "If I can see it, I can kill it." - Mr. Brown
        """
        self.opportunities = np.zeros((len(self.effectors), len(self.targets), len(jf.OpportunityFeatures)))
        ENERGYRATE_NORMALIZATION = MIN_RANGE / self.arena[jf.ArenaFeatures.SCALE]
        for i in range(0, len(self.effectors)):
            for j in range(0, len(self.targets)):
                self.opportunities[i][j][jf.OpportunityFeatures.SELECTABLE] = True
                EucDistance = euclideanDistance(self.effectors[i], self.targets[j])
                travelDistance = EucDistance - self.effectors[i][jf.EffectorFeatures.EFFECTIVEDISTANCE]
                if not self.effectors[i][jf.EffectorFeatures.STATIC]:
                    RTDistance = returnDistance(self.effectors[i], self.targets[j])
                    if travelDistance <= 0:
                        self.opportunities[i][j][jf.OpportunityFeatures.TIMECOST] = 0
                        self.opportunities[i][j][jf.OpportunityFeatures.ENERGYCOST] = 0
                    elif (RTDistance > self.effectors[i][jf.EffectorFeatures.ENERGYLEFT] / (
                            self.effectors[i][jf.EffectorFeatures.ENERGYRATE]) or
                          self.effectors[i][jf.EffectorFeatures.TIMELEFT] < RTDistance / (
                                  self.effectors[i][jf.EffectorFeatures.SPEED] * SPEED_CORRECTION)):
                        self.opportunities[i][j][jf.OpportunityFeatures.SELECTABLE] = False
                    else:
                        self.opportunities[i][j][jf.OpportunityFeatures.TIMECOST] = travelDistance / (
                                self.effectors[i][jf.EffectorFeatures.SPEED] * SPEED_CORRECTION)
                        self.opportunities[i][j][jf.OpportunityFeatures.ENERGYCOST] = travelDistance * \
                                                                                      self.effectors[i][
                                                                                          jf.EffectorFeatures.ENERGYRATE]  # Energy is related to fuel or essentially range
                else:
                    self.opportunities[i][j][jf.OpportunityFeatures.TIMECOST] = 0
                    if EucDistance > self.effectors[i][jf.EffectorFeatures.EFFECTIVEDISTANCE]:
                        self.opportunities[i][j][jf.OpportunityFeatures.SELECTABLE] = False

                if self.opportunities[i][j][jf.OpportunityFeatures.SELECTABLE]:
                    self.opportunities[i][j][jf.OpportunityFeatures.PSUCCESS] = random.uniform(0.35, 0.65)
                else:
                    self.opportunities[i][j][jf.OpportunityFeatures.PSUCCESS] = 0


def network_validation(nb_effectors=7, nb_targets=16):
    arena = np.zeros(len(jf.ArenaFeatures))
    arena[jf.ArenaFeatures.SCALE] = 50
    arena[jf.ArenaFeatures.COASTLINE] = 0.1 * arena[jf.ArenaFeatures.SCALE]
    arena[jf.ArenaFeatures.FRONTLINE] = 0.5 * arena[jf.ArenaFeatures.SCALE]
    arena[jf.ArenaFeatures.FRONTAGE] = 0.4 * arena[jf.ArenaFeatures.SCALE]
    arena[jf.ArenaFeatures.TIMEHORIZON] = 4
    rands = []
    total = nb_effectors
    rands.append(random.randint(0, total))
    rands.append(random.randint(0, total))
    rands.sort()
    artillery = rands[0]
    armoured = rands[1] - artillery
    infantry = total - (artillery + armoured)
    PG = ProblemGenerator()
    return PG.newProblem(arena, targets=nb_targets, artillery=artillery, armoured=armoured, infantry=infantry)


def allPlanes():
    arena = np.zeros(len(jf.ArenaFeatures))
    arena[jf.ArenaFeatures.SCALE] = 500
    arena[jf.ArenaFeatures.COASTLINE] = 0 * arena[jf.ArenaFeatures.SCALE]
    arena[jf.ArenaFeatures.FRONTLINE] = 0.2 * arena[jf.ArenaFeatures.SCALE]
    arena[jf.ArenaFeatures.FRONTAGE] = 0.4 * arena[jf.ArenaFeatures.SCALE]
    arena[jf.ArenaFeatures.TIMEHORIZON] = 4
    planes = random.randint(9, 13)
    targets = random.randint(10, 50)
    PG = ProblemGenerator()
    return PG.newProblem(arena, targets, planes=planes)


def boatsBoatsBoats():
    arena = np.zeros(len(jf.ArenaFeatures))
    arena[jf.ArenaFeatures.SCALE] = 100
    arena[jf.ArenaFeatures.COASTLINE] = 0.3 * arena[jf.ArenaFeatures.SCALE]
    arena[jf.ArenaFeatures.FRONTLINE] = 0.301 * arena[jf.ArenaFeatures.SCALE]
    arena[jf.ArenaFeatures.FRONTAGE] = 0.4 * arena[jf.ArenaFeatures.SCALE]
    arena[jf.ArenaFeatures.TIMEHORIZON] = 4
    targets = random.randint(10, 50)
    PG = ProblemGenerator()
    return PG.newProblem(arena, targets, frigates=random.randint(9, 13))


def infantryOnly():
    arena = np.zeros(len(jf.ArenaFeatures))
    arena[jf.ArenaFeatures.SCALE] = 10
    arena[jf.ArenaFeatures.COASTLINE] = 0.01 * arena[jf.ArenaFeatures.SCALE]
    arena[jf.ArenaFeatures.FRONTLINE] = 0.4 * arena[jf.ArenaFeatures.SCALE]
    arena[jf.ArenaFeatures.FRONTAGE] = 0.4 * arena[jf.ArenaFeatures.SCALE]
    arena[jf.ArenaFeatures.TIMEHORIZON] = 8
    targets = random.randint(4, 9)
    PG = ProblemGenerator()
    return PG.newProblem(arena, targets, infantry=random.randint(5, 15))


def combatArms():
    arena = np.zeros(len(jf.ArenaFeatures))
    arena[jf.ArenaFeatures.SCALE] = 40
    arena[jf.ArenaFeatures.COASTLINE] = 0.01 * arena[jf.ArenaFeatures.SCALE]
    arena[jf.ArenaFeatures.FRONTLINE] = 0.4 * arena[jf.ArenaFeatures.SCALE]
    arena[jf.ArenaFeatures.FRONTAGE] = 0.4 * arena[jf.ArenaFeatures.SCALE]
    arena[jf.ArenaFeatures.TIMEHORIZON] = 4
    rands = []
    total = random.randint(6, 8)
    rands.append(random.randint(0, total))
    rands.append(random.randint(0, total))
    rands.sort()
    artillery = rands[0]
    armoured = rands[1] - artillery
    infantry = total - (artillery + armoured)
    targets = random.randint(20, 30)
    PG = ProblemGenerator()
    return PG.newProblem(arena, targets, artillery=artillery, armoured=armoured, infantry=infantry)


def tiny():
    arena = np.zeros(len(jf.ArenaFeatures))
    arena[jf.ArenaFeatures.SCALE] = 50
    arena[jf.ArenaFeatures.COASTLINE] = 0.01 * arena[jf.ArenaFeatures.SCALE]
    arena[jf.ArenaFeatures.FRONTLINE] = 0.2 * arena[jf.ArenaFeatures.SCALE]
    arena[jf.ArenaFeatures.FRONTAGE] = 0.4 * arena[jf.ArenaFeatures.SCALE]
    arena[jf.ArenaFeatures.TIMEHORIZON] = 4
    armoured = 1
    artillery = 1
    frigates = 1
    planes = 1
    infantry = 1
    targets = 4
    PG = ProblemGenerator()
    return PG.newProblem(arena, targets, infantry=infantry, armoured=armoured, artillery=artillery)


def toy():
    """
    This generates a problem where the best solution is to not choose the items with the highest returns.
    We use a 9, 40, 41 triangle to get distances where the effector must choose between a pair of targets at the same location,
    or a single target at another location.
    """
    arena = np.zeros(len(jf.ArenaFeatures))
    arena[jf.ArenaFeatures.SCALE] = 40
    arena[jf.ArenaFeatures.COASTLINE] = 0.00 * arena[jf.ArenaFeatures.SCALE]
    arena[jf.ArenaFeatures.FRONTLINE] = 0.5 * arena[jf.ArenaFeatures.SCALE]
    arena[jf.ArenaFeatures.FRONTAGE] = 0.5 * arena[jf.ArenaFeatures.SCALE]
    arena[jf.ArenaFeatures.TIMEHORIZON] = 8
    effector_x_range = (0.0, 0.0)
    high_y_range = (9.0, 9.0)
    effectors = []
    targets = []
    effectors.append(
        newEffector(arena, x_range=effector_x_range, y_range=(0, 0), static=False, speed=20, range=85, time=8,
                    effective_distance=0.1, ammo=4))
    effectors.append(
        newEffector(arena, x_range=effector_x_range, y_range=high_y_range, static=False, speed=20, range=85, time=8,
                    effective_distance=0.1, ammo=4))
    target = np.zeros(len(jf.TaskFeatures))
    target[jf.TaskFeatures.XPOS] = 40 / STANDARDIZED_SCALE
    target[jf.TaskFeatures.YPOS] = 9 / STANDARDIZED_SCALE
    target[jf.TaskFeatures.VALUE] = 0.6
    target[jf.TaskFeatures.SELECTED] = 0
    targets.append(target)
    target_x_range = (40, 40)
    target = np.zeros(len(jf.TaskFeatures))
    target[jf.TaskFeatures.XPOS] = 40 / STANDARDIZED_SCALE
    target[jf.TaskFeatures.YPOS] = 0
    target[jf.TaskFeatures.VALUE] = 0.5
    target[jf.TaskFeatures.SELECTED] = 0
    targets.append(target)
    targets.append(target)
    psuccess = np.asarray([
        [0.6, 0.5, 0.5],
        [0.5, 0.3, 0.3]
    ])
    opportunities = np.zeros((len(effectors), len(targets), len(jf.OpportunityFeatures)))
    ENERGYRATE_NORMALIZATION = MIN_RANGE / arena[jf.ArenaFeatures.SCALE]
    for i in range(0, len(effectors)):
        for j in range(0, len(targets)):
            opportunities[i][j][jf.OpportunityFeatures.SELECTABLE] = True
            EucDistance = euclideanDistance(effectors[i], targets[j])
            travelDistance = EucDistance - effectors[i][jf.EffectorFeatures.EFFECTIVEDISTANCE]
            RTDistance = returnDistance(effectors[i], targets[j])
            opportunities[i][j][jf.OpportunityFeatures.TIMECOST] = travelDistance / (
                    effectors[i][jf.EffectorFeatures.SPEED] * SPEED_CORRECTION)
            opportunities[i][j][jf.OpportunityFeatures.ENERGYCOST] = travelDistance * effectors[i][
                jf.EffectorFeatures.ENERGYRATE]  # Energy is related to fuel or essentially range
            opportunities[i][j][jf.OpportunityFeatures.PSUCCESS] = psuccess[i][j]
    problem = {'Arena': arena, 'Effectors': effectors, 'Targets': targets, 'Opportunities': opportunities}
    return problem
