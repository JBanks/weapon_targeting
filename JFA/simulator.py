#!/usr/bin/env python3

import numpy as np

if __package__ is not None and len(__package__) > 0:
    print(f"{__name__} using relative import inside of {__package__}")
    from . import features as jf
    from . import problem_generators as pg
    from .problem_generators import loadProblem, saveProblem, euclideanDistance, returnDistance
else:
    import features as jf
    import problem_generators as pg
    from problem_generators import loadProblem, saveProblem, euclideanDistance, returnDistance
import random
import math
import sys

SPEED_CORRECTION = pg.STANDARDIZED_TIME * pg.MAX_SPEED


def printState(state):
    trueFalseStrings = ["False", "True"]
    print("\n\t (x, y)\t\t\tStatic\tSpeed\tAmmo\tEnergy\tTime\tEffDist\tAmmoRte\tEnergyRate")
    nbEffector = len(state[:, 0, 0])
    for i in range(0, nbEffector):
        print("Effector:", end="")
        print(f"({state[i, 0, jf.EffectorFeatures.XPOS]:.4f}, {state[i, 0, jf.EffectorFeatures.YPOS]:.4f})", end="")
        print(f"\t{trueFalseStrings[int(state[i, 0, jf.EffectorFeatures.STATIC])]}", end="")
        print(f"\t{state[i, 0, jf.EffectorFeatures.SPEED]:4.4f}", end="")
        print(f"\t{state[i, 0, jf.EffectorFeatures.AMMOLEFT]:.4f}", end="")
        print(f"\t{state[i, 0, jf.EffectorFeatures.ENERGYLEFT]:.4f}", end="")
        print(f"\t{state[i, 0, jf.EffectorFeatures.TIMELEFT]:.4f}", end="")
        print(f"\t{state[i, 0, jf.EffectorFeatures.EFFECTIVEDISTANCE]:.4f}", end="")
        print(f"\t{state[i, 0, jf.EffectorFeatures.AMMORATE]:.4f}", end="")
        print(f"\t{state[i, 0, jf.EffectorFeatures.ENERGYRATE]:.4f}")
    nbTask = len(state[0, :, 0])
    pad = len(jf.EffectorFeatures)
    print("\n\n\t(x, y)\t\t\tValue\tSelected")
    for i in range(len(state[0, :, 0])):
        print(f"Target: ({state[0, i, pad + jf.TaskFeatures.XPOS]:.4f}, {state[0, i, pad + jf.TaskFeatures.YPOS]:.4f})",
              end="")
        print(f"\t{state[0, i, pad + jf.TaskFeatures.VALUE]:.4f}", end="")
        print(f"\t{state[0, i, pad + jf.TaskFeatures.SELECTED]}")
    pad = len(jf.EffectorFeatures) + len(jf.TaskFeatures)
    print("\n\nAction\t\tPSucc\tEnergy\ttime\tSelectable\tEucDistance\tReturnDist")
    for i in range(nbEffector):
        for j in range(nbTask):
            if state[i, j, pad + jf.OpportunityFeatures.SELECTABLE]:
                print(
                    f"({i:2},{j:2}):\t{state[i, j, pad + jf.OpportunityFeatures.PSUCCESS]:.4}\t{state[i, j, pad + jf.OpportunityFeatures.ENERGYCOST]:.4f}\t{state[i, j, pad + jf.OpportunityFeatures.TIMECOST]:.4f}\t{trueFalseStrings[int(state[i, j, pad + jf.OpportunityFeatures.SELECTABLE])]}\t\t{euclideanDistance(state[i, 0, :], state[0, j, len(jf.EffectorFeatures):]):.6f}\t{returnDistance(state[i, 0, :], state[0, j, len(jf.EffectorFeatures):]):.6f}")


def print_grid(state):
    trueFalseStrings = ["False", "True"]
    nbEffector = len(state[:, 0, 0])
    nbTask = len(state[0, :, 0])
    pad = len(jf.EffectorFeatures) + len(jf.TaskFeatures)
    print("\t", end="")
    for i in range(nbEffector):
        print(f"{i}\t\t", end="")
    print()
    for j in range(nbTask):
        print(f"{j}\t", end="")
        for i in range(nbEffector):
            if state[i, j, pad + jf.OpportunityFeatures.SELECTABLE]:
                print(f"{state[i, j, pad + jf.OpportunityFeatures.PSUCCESS]:.8f}\t", end="")
            else:
                print("-" * 8, end="")
        print()
        print(f"{state[0, j, len(jf.EffectorFeatures) + jf.TaskFeatures.VALUE]:.3f}", end="")
        print("\t", end="")
        for i in range(nbEffector):
            if state[i, j, pad + jf.OpportunityFeatures.SELECTABLE]:
                print(f"{state[i, j, pad + jf.OpportunityFeatures.ENERGYCOST]:.8f}\t", end="")
            else:
                print("-" * 8, end="")
        print()
        print(f"{state[0, j, len(jf.EffectorFeatures) + jf.TaskFeatures.SELECTED]:.1f}", end="")
        print("\t", end="")
        for i in range(nbEffector):
            if state[i, j, pad + jf.OpportunityFeatures.SELECTABLE]:
                print(f"{state[i, j, pad + jf.OpportunityFeatures.TIMECOST]:.8f}\t", end="")
            else:
                print("-" * 8, end="")
        print("\n")


class JeremyAgent:
    def getAction(self, state):
        """
        Allows the user to control which action to take next by selecting an agent, and a task.
        """
        effector, task = None, None
        printState(state)
        while effector is None:
            try:
                effector = int(input("effector: "))
            except ValueError:
                pass
        while task is None:
            try:
                task = int(input("task: "))
            except ValueError:
                pass
        return effector, task

    def learn(state, action, reward, new_state, terminal):
        pass


class AlexAgent:
    pass


class Simulation:
    """
    Simulate a warfare scenario in which a set of effectors complete a set of tasks
    """

    def __init__(self, formatState, problem=None, keepstack=False):
        self.keepstack = keepstack
        self.formatState = formatState  # In the case of the Neural Network,
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
            if not hasattr(self, 'initialProblem'):
                raise Exception("You must provide an intial problem to solve.")
            problem = self.initialProblem
        else:
            self.initialProblem = problem
        self.reward = 0
        self.scale = problem['Arena'][jf.ArenaFeatures.SCALE]
        self.coastline = problem['Arena'][jf.ArenaFeatures.COASTLINE]
        self.frontline = problem['Arena'][jf.ArenaFeatures.FRONTLINE]
        self.timeHorizon = problem['Arena'][jf.ArenaFeatures.TIMEHORIZON]
        pg.correct_effector_data(problem)
        self.problem = problem
        self.nbEffector, self.nbTask = len(problem['Effectors']), len(problem['Targets'])
        self.effectorData = np.ones((self.nbEffector, len(jf.EffectorFeatures)), dtype=float)
        self.taskData = np.ones((self.nbTask, len(jf.TaskFeatures)), dtype=float)
        self.opportunityData = np.ones((self.nbEffector, self.nbTask, len(jf.OpportunityFeatures)), dtype=float)
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
        return self.formatState(self.effectorData, self.taskData, self.opportunityData)

    def resetState(self, state):
        pass

    def euclidean_distance(self, effector, task):
        return math.sqrt((effector[jf.EffectorFeatures.XPOS] - task[jf.TaskFeatures.XPOS]) ** 2 + (
                effector[jf.EffectorFeatures.YPOS] - task[jf.TaskFeatures.YPOS]) ** 2)

    def return_distance(self, effector, task):
        EucDistance = self.euclidean_distance(effector, task)
        travelDistance = max(EucDistance - effector[jf.EffectorFeatures.EFFECTIVEDISTANCE], 0)
        new_x = effector[jf.EffectorFeatures.XPOS] + (
                task[jf.TaskFeatures.XPOS] - effector[jf.EffectorFeatures.XPOS]) * travelDistance / EucDistance
        new_y = effector[jf.EffectorFeatures.YPOS] + (
                task[jf.TaskFeatures.YPOS] - effector[jf.EffectorFeatures.YPOS]) * travelDistance / EucDistance
        return_trip = math.sqrt(
            (effector[jf.EffectorFeatures.STARTX] - new_x) ** 2 + (effector[jf.EffectorFeatures.STARTY] - new_y) ** 2)
        return travelDistance + return_trip

    def getSchedule(self):
        """
        Return the schedule of events currently chosen.
        """
        # A schedule should be a list of #effector lists where each element is a target along with some other info (eg timing))
        # [[(2 5min), (1 60min)][3, 2, 3][1]]
        # Effector 1 first hits target 2 (service time 5 min) and then 1 (service time 60 min)
        # Effector 2 first hits target 3 and then 2, and then 3
        # Effector 3 first hits target 1
        return self.schedule

    def update(self, action):
        """
        Take an action from an agent and apply that action to the effector specified.
        """
        if type(action) == tuple:
            effectorIndex, taskIndex = action
        else:  # Alex passes a 1-hot matrix.  Process accordingly
            effectorIndex, taskIndex = np.where(action == 1)
            effectorIndex = int(effectorIndex)
            taskIndex = int(taskIndex)
        effector = self.effectorData[effectorIndex, :]
        task = self.taskData[taskIndex, :]
        opportunity = self.opportunityData[effectorIndex, taskIndex, :]
        if opportunity[jf.OpportunityFeatures.SELECTABLE] == False:
            raise IndexError(f"This action is not selectable. Effector: {effectorIndex} Task: {taskIndex}")

        # first copy the tensor (not by ref.. make sure to do actual copy)
        if self.keepstack:
            # TODO: make sure this is a copy, not a reference
            self.history.append(
                (self.effectorData, self.taskData, self.opportunityData, self.schedule, self.previousPSuccess))
        # Do not use history, update the schedule directly.
        # Make a separate stack for the schedule so that we don't need to iterate through the whole state history
        self.schedule[effectorIndex].append((taskIndex, opportunity[jf.OpportunityFeatures.TIMECOST]))

        EucDistance = self.euclidean_distance(effector, task)
        travelDistance = EucDistance - effector[jf.EffectorFeatures.EFFECTIVEDISTANCE]

        if travelDistance > 0:  # Calculate the updated position
            effector[jf.EffectorFeatures.XPOS] += (task[jf.TaskFeatures.XPOS] - effector[
                jf.EffectorFeatures.XPOS]) * travelDistance / EucDistance
            effector[jf.EffectorFeatures.YPOS] += (task[jf.TaskFeatures.YPOS] - effector[
                jf.EffectorFeatures.YPOS]) * travelDistance / EucDistance
        else:
            pass  # We can take action against the target from our current position

        effector[jf.EffectorFeatures.TIMELEFT] -= opportunity[jf.OpportunityFeatures.TIMECOST]
        effector[jf.EffectorFeatures.ENERGYLEFT] -= opportunity[jf.OpportunityFeatures.ENERGYCOST]
        effector[jf.EffectorFeatures.AMMOLEFT] -= effector[jf.EffectorFeatures.AMMORATE]
        # We are dealing with expected plan, not an actual instance of a plan.
        # Damage will be relative to pSuccess rather than sometimes being right and sometimes being wrong
        reward = opportunity[jf.OpportunityFeatures.PSUCCESS] * task[jf.TaskFeatures.VALUE]
        self.taskData[taskIndex][jf.TaskFeatures.VALUE] -= reward

        task[jf.TaskFeatures.SELECTED] += 0.5  # Count the number of engagements so far

        if task[jf.TaskFeatures.SELECTED] >= 1:  # Down the road: opportunities[:,:,selectable] &= task[:,selected] >= 1
            for i in range(0, self.nbEffector):
                self.opportunityData[i][taskIndex][jf.OpportunityFeatures.SELECTABLE] = False
                self.opportunityData[i][taskIndex][jf.OpportunityFeatures.PSUCCESS] = 0

        for i in range(0, self.nbTask):
            EucDistance = self.euclidean_distance(effector, self.taskData[i])
            # If it wasn't selectable before, could that change?  If not, drop this set of operations whenever something is already unfeasible
            if not effector[jf.EffectorFeatures.STATIC]:
                RTDistance = self.return_distance(effector, self.taskData[i])
                travelDistance = max(0, EucDistance - effector[jf.EffectorFeatures.EFFECTIVEDISTANCE])
                if (RTDistance > effector[jf.EffectorFeatures.ENERGYLEFT] / (
                        effector[jf.EffectorFeatures.ENERGYRATE]) or
                        effector[jf.EffectorFeatures.TIMELEFT] < RTDistance / (
                                effector[jf.EffectorFeatures.SPEED] * SPEED_CORRECTION)):
                    # print(f"Effector: {effectorIndex}, Target: {i}")
                    # print(f"Dist: {RTDistance > effector[JF.EffectorFeatures.ENERGYLEFT] / effector[JF.EffectorFeatures.ENERGYRATE]} : {RTDistance} > {effector[JF.EffectorFeatures.ENERGYLEFT]} / {effector[JF.EffectorFeatures.ENERGYRATE]}")
                    # print(f"Time: {effector[JF.EffectorFeatures.TIMELEFT] < RTDistance / (effector[JF.EffectorFeatures.SPEED] * SPEED_CORRECTION)} : {effector[JF.EffectorFeatures.TIMELEFT]} < {RTDistance} / {effector[JF.EffectorFeatures.SPEED] * SPEED_CORRECTION}")
                    self.opportunityData[effectorIndex][i][jf.OpportunityFeatures.SELECTABLE] = False
                else:
                    self.opportunityData[effectorIndex][i][jf.OpportunityFeatures.TIMECOST] = travelDistance / (
                            effector[
                                jf.EffectorFeatures.SPEED] * SPEED_CORRECTION)  # + effector[JF.EffectorFeatures.DUTYCYCLE]
                    self.opportunityData[effectorIndex][i][jf.OpportunityFeatures.ENERGYCOST] = \
                        travelDistance * effector[jf.EffectorFeatures.ENERGYRATE]  # Energy is related to fuel or essentially range
            else:
                if EucDistance <= effector[jf.EffectorFeatures.EFFECTIVEDISTANCE]:
                    self.opportunityData[effectorIndex][i][jf.OpportunityFeatures.TIMECOST] = \
                        EucDistance / (effector[jf.EffectorFeatures.SPEED] * SPEED_CORRECTION)  # effector[JF.EffectorFeatures.DUTYCYCLE]
                    self.opportunityData[effectorIndex][i][
                        jf.OpportunityFeatures.ENERGYCOST] = 0  # Energy is related to fuel or essentially range
                else:
                    self.opportunityData[effectorIndex][i][jf.OpportunityFeatures.SELECTABLE] = False

            if self.opportunityData[effectorIndex][i][jf.OpportunityFeatures.TIMECOST] > effector[
                jf.EffectorFeatures.TIMELEFT]:
                self.opportunityData[effectorIndex][i][jf.OpportunityFeatures.SELECTABLE] = False
            elif self.effectorData[effectorIndex][jf.EffectorFeatures.AMMORATE] > effector[
                jf.EffectorFeatures.AMMOLEFT]:
                self.opportunityData[effectorIndex][i][jf.OpportunityFeatures.SELECTABLE] = False
            elif self.opportunityData[effectorIndex][i][jf.OpportunityFeatures.ENERGYCOST] > effector[
                jf.EffectorFeatures.ENERGYLEFT]:
                self.opportunityData[effectorIndex][i][jf.OpportunityFeatures.SELECTABLE] = False

            if self.opportunityData[effectorIndex][i][jf.OpportunityFeatures.SELECTABLE] == False:
                self.opportunityData[effectorIndex][i][jf.OpportunityFeatures.PSUCCESS] = 0

        # self.opportunityData[:,:,JF.OpportunityFeatures.PSUCCESS] &= self.opportunityData[:,:,JF.OpportunityFeatures.SELECTABLE]
        if np.sum(self.opportunityData[:, :, jf.OpportunityFeatures.SELECTABLE]) >= 1:
            terminal = False
        else:
            terminal = True

        return self.formatState(self.effectorData, self.taskData, self.opportunityData), reward, terminal

    def update_state(self, action, state=None):
        """
        Take an action from an agent and apply that action to the effector specified.
        """
        if state is None:
            effectorData, taskData, opportunityData = self.effectorData, self.taskData, self.opportunityData
            nbEffector = self.nbEffector
            nbTask = self.nbTask
        else:
            pg.correct_effector_data(state)
            effectorData, taskData, opportunityData = unMergeState(state)
            nbEffector = len(effectorData)
            nbTask = len(taskData)

        if type(action) == tuple:
            effectorIndex, taskIndex = action
        else:  # Alex passes a 1-hot matrix.  Process accordingly
            effectorIndex, taskIndex = np.where(action == 1)
            effectorIndex = int(effectorIndex)
            taskIndex = int(taskIndex)

        effector = effectorData[effectorIndex, :]
        task = taskData[taskIndex, :]
        opportunity = opportunityData[effectorIndex, taskIndex, :]
        if not opportunity[jf.OpportunityFeatures.SELECTABLE]:
            raise IndexError(f"This action is not selectable. Effector: {effectorIndex} Task: {taskIndex}")

        EucDistance = self.euclidean_distance(effector, task)
        travel_distance = EucDistance - effector[jf.EffectorFeatures.EFFECTIVEDISTANCE]

        travel_required = travel_distance > 0  # Move the effector to an updated position if required
        effector[jf.EffectorFeatures.XPOS] += travel_required * (task[jf.TaskFeatures.XPOS] - effector[
            jf.EffectorFeatures.XPOS]) * travel_distance / EucDistance
        effector[jf.EffectorFeatures.YPOS] += travel_required * (task[jf.TaskFeatures.YPOS] - effector[
            jf.EffectorFeatures.YPOS]) * travel_distance / EucDistance

        effector[jf.EffectorFeatures.TIMELEFT] -= opportunity[jf.OpportunityFeatures.TIMECOST]
        effector[jf.EffectorFeatures.ENERGYLEFT] -= opportunity[jf.OpportunityFeatures.ENERGYCOST]
        effector[jf.EffectorFeatures.AMMOLEFT] -= effector[jf.EffectorFeatures.AMMORATE]
        # We are dealing with expected plan, not an actual instance of a plan.
        # Damage will be relative to pSuccess rather than sometimes being right and sometimes being wrong
        reward = opportunity[jf.OpportunityFeatures.PSUCCESS] * task[jf.TaskFeatures.VALUE]
        taskData[taskIndex][jf.TaskFeatures.VALUE] -= reward

        task[jf.TaskFeatures.SELECTED] += 0.5  # Count the number of engagements so far

        # The task has been selected the maximum number of times, no other effectors can select this task.
        opportunityData[:, taskIndex, jf.OpportunityFeatures.SELECTABLE] *= (task[jf.TaskFeatures.SELECTED] < 1)
        opportunityData[:, taskIndex, jf.OpportunityFeatures.PSUCCESS] *= \
            opportunityData[:, taskIndex, jf.OpportunityFeatures.SELECTABLE]

        for i in range(0, nbTask):
            if not opportunityData[effectorIndex][i][jf.OpportunityFeatures.SELECTABLE]:
                continue
            EucDistance = self.euclidean_distance(effector, taskData[i])
            mobile = not effector[jf.EffectorFeatures.STATIC]
            if mobile:
                RTDistance = self.return_distance(effector, taskData[i])
                travel_distance = max(0, EucDistance - effector[jf.EffectorFeatures.EFFECTIVEDISTANCE])
                # If the effector is static, then it doesn't move and tasks don't need to be updated here
                # If the effector is mobile, then we need to validate that the moves are still feasible
                maintain_value = (not mobile) or (RTDistance < effector[jf.EffectorFeatures.ENERGYLEFT] /
                                                  effector[jf.EffectorFeatures.ENERGYRATE] and
                                                  effector[jf.EffectorFeatures.TIMELEFT] > RTDistance / (
                                                          effector[jf.EffectorFeatures.SPEED] * SPEED_CORRECTION))
                opportunityData[effectorIndex][i][jf.OpportunityFeatures.SELECTABLE] *= maintain_value
                opportunityData[effectorIndex][i][jf.OpportunityFeatures.PSUCCESS] *= maintain_value

                opportunityData[effectorIndex][i][jf.OpportunityFeatures.TIMECOST] = \
                    (not mobile) * opportunityData[effectorIndex][i][jf.OpportunityFeatures.TIMECOST] + \
                    mobile * travel_distance / (effector[jf.EffectorFeatures.SPEED] * SPEED_CORRECTION)
                opportunityData[effectorIndex][i][jf.OpportunityFeatures.ENERGYCOST] = \
                    (not mobile) * opportunityData[effectorIndex][i][jf.OpportunityFeatures.ENERGYCOST] + \
                    mobile * travel_distance * effector[jf.EffectorFeatures.ENERGYRATE]

            # If there isn't enough time, ammo, or energy, mark the opportunity as infeasible
            feasible = opportunityData[effectorIndex][i][jf.OpportunityFeatures.TIMECOST] <= \
                       effector[jf.EffectorFeatures.TIMELEFT] and \
                       effectorData[effectorIndex][jf.EffectorFeatures.AMMORATE] <= \
                       effector[jf.EffectorFeatures.AMMOLEFT] and \
                       opportunityData[effectorIndex][i][jf.OpportunityFeatures.ENERGYCOST] <= \
                       effector[jf.EffectorFeatures.ENERGYLEFT]
            opportunityData[effectorIndex][i][jf.OpportunityFeatures.SELECTABLE] *= feasible
            opportunityData[effectorIndex][i][jf.OpportunityFeatures.PSUCCESS] *= feasible

        # If no actions are selectable, we are in a terminal state
        terminal = not opportunityData[:, :, jf.OpportunityFeatures.SELECTABLE].any()

        return self.formatState(effectorData, taskData, opportunityData), reward, terminal

    def undo(self):
        """
        Return to the previous state.  This can help in a depth-first-search style action selection
        """
        self.effectorData, self.taskData, self.opportunityData, self.schedule, self.previousPSuccess = self.history.pop()


def mergeState(effectorData, taskData, opportunityData):
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
    effectors = np.zeros((len(taskData), len(effectorData), len(jf.EffectorFeatures)))  # m.n.p
    tasks = np.zeros((len(effectorData), len(taskData), len(jf.TaskFeatures)))  # n.m.q
    effectors += effectorData
    tasks += taskData
    effectors = effectors.transpose([1, 0, 2])  # Transpose from m.n.p to n.m.p
    return np.concatenate((effectors, tasks, opportunityData), axis=2).transpose((2, 0, 1))  # concatenate on the 3rd axis


def state_to_dict(effectorData, taskData, opportunityData):
    state = {}
    state['Effectors'] = effectorData
    state['Targets'] = taskData
    state['Opportunities'] = opportunityData
    return state


def unMergeState(state):
    if type(state) == dict:
        effectorData = state['Effectors']
        taskData = state['Targets']
        opportunityData = state['Opportunities']
    else:
        effectorData = state[:, 0, :len(jf.EffectorFeatures)]
        taskData = state[0, :, len(jf.EffectorFeatures):len(jf.EffectorFeatures) + len(jf.TaskFeatures)]
        opportunityData = state[:, :, len(jf.EffectorFeatures) + len(jf.TaskFeatures):]
    return effectorData, taskData, opportunityData


def main():
    jeremy = 0
    alex = 1
    agents = [JeremyAgent, AlexAgent]
    agentSelection = 0

    env = Simulation(mergeState, keepstack=True)
    if len(sys.argv) > 1:
        simProblem = loadProblem(sys.argv[1])
    else:
        problemGenerators = [pg.allPlanes, pg.infantryOnly, pg.combatArms, pg.boatsBoatsBoats]
        problemGenerator = random.choice(problemGenerators)

        problemGenerator = pg.toy
        simProblem = problemGenerator()

    agent = agents[agentSelection]
    state = env.reset(simProblem)  # get initial state or load a new problem
    total_reward = 0
    while True:
        terminal = False
        while not terminal:
            action = agent.getAction(state)
            try:
                new_state, reward, terminal = env.update(action)
                total_reward += reward
            except Exception as e:
                print(f"Error: {e}")
                print(f"Action: {action}")
                continue
            agent.learn(state, action, reward, new_state, terminal)
            state = new_state
            print(f"reward returned: {reward}")
        printState(state)
        print(f"schedule: {env.getSchedule()}")
        print(f"Reward Returned: {total_reward}")
        state = env.reset()


if __name__ == "__main__":
    main()
