#!/usr/bin/env python3

from . import ProblemGenerators as PG
from . import JFAFeatures as JF


def printProblem(problem, arena=False):
	scale = problem['Arena'][JF.ArenaFeatures.SCALE]
	trueFalseStrings = ["False", "True"]

	if arena:
		print(f"Arena Scale: {problem['Arena'][JF.ArenaFeatures.SCALE]} km")
		print(f"Coastline Location: {problem['Arena'][JF.ArenaFeatures.COASTLINE]} km")
		print(f"Frontline Location: {problem['Arena'][JF.ArenaFeatures.FRONTLINE]} km")
		print(f"Time Horizon: {problem['Arena'][JF.ArenaFeatures.TIMEHORIZON]} hours")
	print("\n\t [x, y]\t\t\tStatic\tSpeed\tAmmo\tEnergy\tTime\tEffDist\tAmmoRate\tEnergyRate")
	for effector in problem['Effectors']:
		print("Effector:", end="")
		print(f"[{effector[JF.EffectorFeatures.XPOS]:.4f}, {effector[JF.EffectorFeatures.YPOS]:.4f}]", end="")
		print(f"\t{trueFalseStrings[int(effector[JF.EffectorFeatures.STATIC])]}", end="")
		print(f"\t{effector[JF.EffectorFeatures.SPEED]:4.4f}", end="")
		print(f"\t{effector[JF.EffectorFeatures.AMMOLEFT]:.4f}", end="")
		print(f"\t{effector[JF.EffectorFeatures.ENERGYLEFT]:.4f}", end="")
		print(f"\t{effector[JF.EffectorFeatures.TIMELEFT]:.4f}", end="")
		print(f"\t{effector[JF.EffectorFeatures.EFFECTIVEDISTANCE]:.4f}", end="")
		print(f"\t{effector[JF.EffectorFeatures.AMMORATE]:.4f}", end="")
		print(f"\t\t{effector[JF.EffectorFeatures.ENERGYRATE]:.4f}")

	print("\n\n\t[x, y] (km)\t\tValue\tSelected")
	for target in problem['Targets']:
		print(f"Target: [{target[JF.TaskFeatures.XPOS]:4.4f}, {target[JF.TaskFeatures.YPOS]:4.4f}]\t{target[JF.TaskFeatures.VALUE]:4.4f}\t{target[JF.TaskFeatures.SELECTED]}")

	print("\n\n\t\tPSucc\tEnergy\ttime\tSelectable")
	for i in range(len(problem['Effectors'])):
		for j in range(len(problem['Targets'])):
			opportunity = problem['Opportunities'][i][j]
			if opportunity[JF.OpportunityFeatures.SELECTABLE]:
				print(f"({i:2},{j:2}):\t{opportunity[JF.OpportunityFeatures.PSUCCESS]:.4f}\t{opportunity[JF.OpportunityFeatures.ENERGYCOST]:4.5f}\t{opportunity[JF.OpportunityFeatures.TIMECOST]:4.5f}\t{trueFalseStrings[int(opportunity[JF.OpportunityFeatures.SELECTABLE])]}")


def main():
	problem = PG.network_validation(4,8)
	printProblem(problem, arena=True)
	#PG.saveProblem(problem, "folder/" + "problem.json")


if __name__ == "__main__":
    main()
