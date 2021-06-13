#!/usr/bin/env python3

import numpy as np

if __package__ is not None and len(__package__) > 0:
    print(f"{__name__} using relative import inside of {__package__}")
    from . import JFAFeatures as JF
    from . import ProblemGenerators as PG
    from .ProblemGenerators import loadProblem, saveProblem, euclideanDistance, returnDistance
else:
    import JFAFeatures as JF
    import ProblemGenerators as PG
    from ProblemGenerators import loadProblem, saveProblem, euclideanDistance, returnDistance
import random
import math
import sys

SPEED_CORRECTION = PG.STANDARDIZED_TIME * PG.MAX_SPEED


def printState(state):
    trueFalseStrings = ["False", "True"]
    print("\n\t (x, y)\t\t\tStatic\tSpeed\tAmmo\tEnergy\tTime\tEffDist\tAmmoRte\tEnergyRate")
    nbEffector = len(state[:, 0, 0])
    for i in range(0, nbEffector):
        print("Effector:", end="")
        print(f"({state[i, 0, JF.EffectorFeatures.XPOS]:.4f}, {state[i, 0, JF.EffectorFeatures.YPOS]:.4f})", end="")
        print(f"\t{trueFalseStrings[int(state[i, 0, JF.EffectorFeatures.STATIC])]}", end="")
        print(f"\t{state[i, 0, JF.EffectorFeatures.SPEED]:4.4f}", end="")
        print(f"\t{state[i, 0, JF.EffectorFeatures.AMMOLEFT]:.4f}", end="")
        print(f"\t{state[i, 0, JF.EffectorFeatures.ENERGYLEFT]:.4f}", end="")
        print(f"\t{state[i, 0, JF.EffectorFeatures.TIMELEFT]:.4f}", end="")
        print(f"\t{state[i, 0, JF.EffectorFeatures.EFFECTIVEDISTANCE]:.4f}", end="")
        print(f"\t{state[i, 0, JF.EffectorFeatures.AMMORATE]:.4f}", end="")
        print(f"\t{state[i, 0, JF.EffectorFeatures.ENERGYRATE]:.4f}")
    nbTask = len(state[0, :, 0])
    pad = len(JF.EffectorFeatures)
    print("\n\n\t(x, y)\t\t\tValue\tSelected")
    for i in range(len(state[0, :, 0])):
        print(f"Target: ({state[0, i, pad + JF.TaskFeatures.XPOS]:.4f}, {state[0, i, pad + JF.TaskFeatures.YPOS]:.4f})",
              end="")
        print(f"\t{state[0, i, pad + JF.TaskFeatures.VALUE]:.4f}", end="")
        print(f"\t{state[0, i, pad + JF.TaskFeatures.SELECTED]}")
    pad = len(JF.EffectorFeatures) + len(JF.TaskFeatures)
    print("\n\nAction\t\tPSucc\tEnergy\ttime\tSelectable\tEucDistance\tReturnDist")
    for i in range(nbEffector):
        for j in range(nbTask):
            if state[i, j, pad + JF.OpportunityFeatures.SELECTABLE]:
                print(
                    f"({i:2},{j:2}):\t{state[i, j, pad + JF.OpportunityFeatures.PSUCCESS]:.4}\t{state[i, j, pad + JF.OpportunityFeatures.ENERGYCOST]:.4f}\t{state[i, j, pad + JF.OpportunityFeatures.TIMECOST]:.4f}\t{trueFalseStrings[int(state[i, j, pad + JF.OpportunityFeatures.SELECTABLE])]}\t\t{euclideanDistance(state[i, 0, :], state[0, j, len(JF.EffectorFeatures):]):.6f}\t{returnDistance(state[i, 0, :], state[0, j, len(JF.EffectorFeatures):]):.6f}")


def print_grid(state):
    trueFalseStrings = ["False", "True"]
    nbEffector = len(state[:, 0, 0])
    nbTask = len(state[0, :, 0])
    pad = len(JF.EffectorFeatures) + len(JF.TaskFeatures)
    print("\t", end="")
    for i in range(nbEffector):
        print(f"{i}\t\t", end="")
    print()
    for j in range(nbTask):
        print(f"{j}\t", end="")
        for i in range(nbEffector):
            if state[i, j, pad + JF.OpportunityFeatures.SELECTABLE]:
                print(f"{state[i, j, pad + JF.OpportunityFeatures.PSUCCESS]:.8f}\t", end="")
            else:
                print("-" * 8, end="")
        print()
        print(f"{state[0, j, len(JF.EffectorFeatures) + JF.TaskFeatures.VALUE]:.3f}", end="")
        print("\t", end="")
        for i in range(nbEffector):
            if state[i, j, pad + JF.OpportunityFeatures.SELECTABLE]:
                print(f"{state[i, j, pad + JF.OpportunityFeatures.ENERGYCOST]:.8f}\t", end="")
            else:
                print("-" * 8, end="")
        print()
        print(f"{state[0, j, len(JF.EffectorFeatures) + JF.TaskFeatures.SELECTED]:.1f}", end="")
        print("\t", end="")
        for i in range(nbEffector):
            if state[i, j, pad + JF.OpportunityFeatures.SELECTABLE]:
                print(f"{state[i, j, pad + JF.OpportunityFeatures.TIMECOST]:.8f}\t", end="")
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
        self.scale = problem['Arena'][JF.ArenaFeatures.SCALE]
        self.coastline = problem['Arena'][JF.ArenaFeatures.COASTLINE]
        self.frontline = problem['Arena'][JF.ArenaFeatures.FRONTLINE]
        self.timeHorizon = problem['Arena'][JF.ArenaFeatures.TIMEHORIZON]
        PG.correct_effector_data(problem)
        self.problem = problem
        self.nbEffector, self.nbTask = len(problem['Effectors']), len(problem['Targets'])
        self.effectorData = np.ones((self.nbEffector, len(JF.EffectorFeatures)), dtype=float)
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
        return self.formatState(self.effectorData, self.taskData, self.opportunityData)

    def resetState(self, state):
        pass

    def euclidean_distance(self, effector, task):
        return math.sqrt((effector[JF.EffectorFeatures.XPOS] - task[JF.TaskFeatures.XPOS]) ** 2 + (
                effector[JF.EffectorFeatures.YPOS] - task[JF.TaskFeatures.YPOS]) ** 2)

    def return_distance(self, effector, task):
        EucDistance = self.euclidean_distance(effector, task)
        travelDistance = max(EucDistance - effector[JF.EffectorFeatures.EFFECTIVEDISTANCE], 0)
        new_x = effector[JF.EffectorFeatures.XPOS] + (
                task[JF.TaskFeatures.XPOS] - effector[JF.EffectorFeatures.XPOS]) * travelDistance / EucDistance
        new_y = effector[JF.EffectorFeatures.YPOS] + (
                task[JF.TaskFeatures.YPOS] - effector[JF.EffectorFeatures.YPOS]) * travelDistance / EucDistance
        return_trip = math.sqrt(
            (effector[JF.EffectorFeatures.STARTX] - new_x) ** 2 + (effector[JF.EffectorFeatures.STARTY] - new_y) ** 2)
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
        if opportunity[JF.OpportunityFeatures.SELECTABLE] == False:
            raise IndexError(f"This action is not selectable. Effector: {effectorIndex} Task: {taskIndex}")

        # first copy the tensor (not by ref.. make sure to do actual copy)
        if self.keepstack:
            # TODO: make sure this is a copy, not a reference
            self.history.append(
                (self.effectorData, self.taskData, self.opportunityData, self.schedule, self.previousPSuccess))
        # Do not use history, update the schedule directly.
        # Make a separate stack for the schedule so that we don't need to iterate through the whole state history
        self.schedule[effectorIndex].append((taskIndex, opportunity[JF.OpportunityFeatures.TIMECOST]))

        EucDistance = self.euclidean_distance(effector, task)
        travelDistance = EucDistance - effector[JF.EffectorFeatures.EFFECTIVEDISTANCE]

        if travelDistance > 0:  # Calculate the updated position
            effector[JF.EffectorFeatures.XPOS] += (task[JF.TaskFeatures.XPOS] - effector[
                JF.EffectorFeatures.XPOS]) * travelDistance / EucDistance
            effector[JF.EffectorFeatures.YPOS] += (task[JF.TaskFeatures.YPOS] - effector[
                JF.EffectorFeatures.YPOS]) * travelDistance / EucDistance
        else:
            pass  # We can take action against the target from our current position

        effector[JF.EffectorFeatures.TIMELEFT] -= opportunity[JF.OpportunityFeatures.TIMECOST]
        effector[JF.EffectorFeatures.ENERGYLEFT] -= opportunity[JF.OpportunityFeatures.ENERGYCOST]
        effector[JF.EffectorFeatures.AMMOLEFT] -= effector[JF.EffectorFeatures.AMMORATE]
        # We are dealing with expected plan, not an actual instance of a plan.
        # Damage will be relative to pSuccess rather than sometimes being right and sometimes being wrong
        reward = opportunity[JF.OpportunityFeatures.PSUCCESS] * task[JF.TaskFeatures.VALUE]
        self.taskData[taskIndex][JF.TaskFeatures.VALUE] -= reward

        task[JF.TaskFeatures.SELECTED] += 0.5  # Count the number of engagements so far

        if task[JF.TaskFeatures.SELECTED] >= 1:  # Down the road: opportunities[:,:,selectable] &= task[:,selected] >= 1
            for i in range(0, self.nbEffector):
                self.opportunityData[i][taskIndex][JF.OpportunityFeatures.SELECTABLE] = False
                self.opportunityData[i][taskIndex][JF.OpportunityFeatures.PSUCCESS] = 0

        for i in range(0, self.nbTask):
            EucDistance = self.euclidean_distance(effector, self.taskData[i])
            # If it wasn't selectable before, could that change?  If not, drop this set of operations whenever something is already unfeasible
            if not effector[JF.EffectorFeatures.STATIC]:
                RTDistance = self.return_distance(effector, self.taskData[i])
                travelDistance = max(0, EucDistance - effector[JF.EffectorFeatures.EFFECTIVEDISTANCE])
                if (RTDistance > effector[JF.EffectorFeatures.ENERGYLEFT] / (
                        effector[JF.EffectorFeatures.ENERGYRATE]) or
                        effector[JF.EffectorFeatures.TIMELEFT] < RTDistance / (
                                effector[JF.EffectorFeatures.SPEED] * SPEED_CORRECTION)):
                    # print(f"Effector: {effectorIndex}, Target: {i}")
                    # print(f"Dist: {RTDistance > effector[JF.EffectorFeatures.ENERGYLEFT] / effector[JF.EffectorFeatures.ENERGYRATE]} : {RTDistance} > {effector[JF.EffectorFeatures.ENERGYLEFT]} / {effector[JF.EffectorFeatures.ENERGYRATE]}")
                    # print(f"Time: {effector[JF.EffectorFeatures.TIMELEFT] < RTDistance / (effector[JF.EffectorFeatures.SPEED] * SPEED_CORRECTION)} : {effector[JF.EffectorFeatures.TIMELEFT]} < {RTDistance} / {effector[JF.EffectorFeatures.SPEED] * SPEED_CORRECTION}")
                    self.opportunityData[effectorIndex][i][JF.OpportunityFeatures.SELECTABLE] = False
                else:
                    self.opportunityData[effectorIndex][i][JF.OpportunityFeatures.TIMECOST] = travelDistance / (
                            effector[
                                JF.EffectorFeatures.SPEED] * SPEED_CORRECTION)  # + effector[JF.EffectorFeatures.DUTYCYCLE]
                    self.opportunityData[effectorIndex][i][JF.OpportunityFeatures.ENERGYCOST] = \
                        travelDistance * effector[JF.EffectorFeatures.ENERGYRATE]  # Energy is related to fuel or essentially range
            else:
                if EucDistance <= effector[JF.EffectorFeatures.EFFECTIVEDISTANCE]:
                    self.opportunityData[effectorIndex][i][JF.OpportunityFeatures.TIMECOST] = \
                        EucDistance / (effector[JF.EffectorFeatures.SPEED] * SPEED_CORRECTION)  # effector[JF.EffectorFeatures.DUTYCYCLE]
                    self.opportunityData[effectorIndex][i][
                        JF.OpportunityFeatures.ENERGYCOST] = 0  # Energy is related to fuel or essentially range
                else:
                    self.opportunityData[effectorIndex][i][JF.OpportunityFeatures.SELECTABLE] = False

            if self.opportunityData[effectorIndex][i][JF.OpportunityFeatures.TIMECOST] > effector[
                JF.EffectorFeatures.TIMELEFT]:
                self.opportunityData[effectorIndex][i][JF.OpportunityFeatures.SELECTABLE] = False
            elif self.effectorData[effectorIndex][JF.EffectorFeatures.AMMORATE] > effector[
                JF.EffectorFeatures.AMMOLEFT]:
                self.opportunityData[effectorIndex][i][JF.OpportunityFeatures.SELECTABLE] = False
            elif self.opportunityData[effectorIndex][i][JF.OpportunityFeatures.ENERGYCOST] > effector[
                JF.EffectorFeatures.ENERGYLEFT]:
                self.opportunityData[effectorIndex][i][JF.OpportunityFeatures.SELECTABLE] = False

            if self.opportunityData[effectorIndex][i][JF.OpportunityFeatures.SELECTABLE] == False:
                self.opportunityData[effectorIndex][i][JF.OpportunityFeatures.PSUCCESS] = 0

        # self.opportunityData[:,:,JF.OpportunityFeatures.PSUCCESS] &= self.opportunityData[:,:,JF.OpportunityFeatures.SELECTABLE]
        if np.sum(self.opportunityData[:, :, JF.OpportunityFeatures.SELECTABLE]) >= 1:
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
            PG.correct_effector_data(state)
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
        if not opportunity[JF.OpportunityFeatures.SELECTABLE]:
            raise IndexError(f"This action is not selectable. Effector: {effectorIndex} Task: {taskIndex}")

        EucDistance = self.euclidean_distance(effector, task)
        travel_distance = EucDistance - effector[JF.EffectorFeatures.EFFECTIVEDISTANCE]

        travel_required = travel_distance > 0  # Move the effector to an updated position if required
        effector[JF.EffectorFeatures.XPOS] += travel_required * (task[JF.TaskFeatures.XPOS] - effector[
            JF.EffectorFeatures.XPOS]) * travel_distance / EucDistance
        effector[JF.EffectorFeatures.YPOS] += travel_required * (task[JF.TaskFeatures.YPOS] - effector[
            JF.EffectorFeatures.YPOS]) * travel_distance / EucDistance

        effector[JF.EffectorFeatures.TIMELEFT] -= opportunity[JF.OpportunityFeatures.TIMECOST]
        effector[JF.EffectorFeatures.ENERGYLEFT] -= opportunity[JF.OpportunityFeatures.ENERGYCOST]
        effector[JF.EffectorFeatures.AMMOLEFT] -= effector[JF.EffectorFeatures.AMMORATE]
        # We are dealing with expected plan, not an actual instance of a plan.
        # Damage will be relative to pSuccess rather than sometimes being right and sometimes being wrong
        reward = opportunity[JF.OpportunityFeatures.PSUCCESS] * task[JF.TaskFeatures.VALUE]
        taskData[taskIndex][JF.TaskFeatures.VALUE] -= reward

        task[JF.TaskFeatures.SELECTED] += 0.5  # Count the number of engagements so far

        # The task has been selected the maximum number of times, no other effectors can select this task.
        opportunityData[:, taskIndex, JF.OpportunityFeatures.SELECTABLE] *= (task[JF.TaskFeatures.SELECTED] < 1)
        opportunityData[:, taskIndex, JF.OpportunityFeatures.PSUCCESS] *= \
            opportunityData[:, taskIndex, JF.OpportunityFeatures.SELECTABLE]

        for i in range(0, nbTask):
            if not opportunityData[effectorIndex][i][JF.OpportunityFeatures.SELECTABLE]:
                continue
            EucDistance = self.euclidean_distance(effector, taskData[i])
            mobile = not effector[JF.EffectorFeatures.STATIC]
            if mobile:
                RTDistance = self.return_distance(effector, taskData[i])
                travel_distance = max(0, EucDistance - effector[JF.EffectorFeatures.EFFECTIVEDISTANCE])
                # If the effector is static, then it doesn't move and tasks don't need to be updated here
                # If the effector is mobile, then we need to validate that the moves are still feasible
                maintain_value = (not mobile) or (RTDistance < effector[JF.EffectorFeatures.ENERGYLEFT] /
                                                  effector[JF.EffectorFeatures.ENERGYRATE] and
                                                  effector[JF.EffectorFeatures.TIMELEFT] > RTDistance / (
                                                    effector[JF.EffectorFeatures.SPEED] * SPEED_CORRECTION))
                opportunityData[effectorIndex][i][JF.OpportunityFeatures.SELECTABLE] *= maintain_value
                opportunityData[effectorIndex][i][JF.OpportunityFeatures.PSUCCESS] *= maintain_value

                opportunityData[effectorIndex][i][JF.OpportunityFeatures.TIMECOST] = \
                    (not mobile) * opportunityData[effectorIndex][i][JF.OpportunityFeatures.TIMECOST] + \
                    mobile * travel_distance / (effector[JF.EffectorFeatures.SPEED] * SPEED_CORRECTION)
                opportunityData[effectorIndex][i][JF.OpportunityFeatures.ENERGYCOST] = \
                    (not mobile) * opportunityData[effectorIndex][i][JF.OpportunityFeatures.ENERGYCOST] + \
                    mobile * travel_distance * effector[JF.EffectorFeatures.ENERGYRATE]

            # If there isn't enough time, ammo, or energy, mark the opportunity as infeasible
            feasible = opportunityData[effectorIndex][i][JF.OpportunityFeatures.TIMECOST] <= \
                effector[JF.EffectorFeatures.TIMELEFT] and \
                effectorData[effectorIndex][JF.EffectorFeatures.AMMORATE] <= \
                effector[JF.EffectorFeatures.AMMOLEFT] and \
                opportunityData[effectorIndex][i][JF.OpportunityFeatures.ENERGYCOST] <= \
                effector[JF.EffectorFeatures.ENERGYLEFT]
            opportunityData[effectorIndex][i][JF.OpportunityFeatures.SELECTABLE] *= feasible
            opportunityData[effectorIndex][i][JF.OpportunityFeatures.PSUCCESS] *= feasible

        # If no actions are selectable, we are in a terminal state
        terminal = not opportunityData[:, :, JF.OpportunityFeatures.SELECTABLE].any()

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
    effectors = np.zeros((len(taskData), len(effectorData), len(JF.EffectorFeatures)))  # m.n.p
    tasks = np.zeros((len(effectorData), len(taskData), len(JF.TaskFeatures)))  # n.m.q
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
        effectorData = state[:, 0, :len(JF.EffectorFeatures)]
        taskData = state[0, :, len(JF.EffectorFeatures):len(JF.EffectorFeatures) + len(JF.TaskFeatures)]
        opportunityData = state[:, :, len(JF.EffectorFeatures) + len(JF.TaskFeatures):]
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
        problemGenerators = [PG.allPlanes, PG.infantryOnly, PG.combatArms, PG.boatsBoatsBoats]
        problemGenerator = random.choice(problemGenerators)

        problemGenerator = PG.toy
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
