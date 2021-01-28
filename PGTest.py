#!/usr/bin/env python3

import ProblemGenerators as PG
import JFAFeatures as JF


def printProblem(problem):
	scale = problem['Arena'][JF.ArenaFeatures.SCALE]
	trueFalseStrings = ["False", "True"]
	print(f"Arena Scale: {problem['Arena'][JF.ArenaFeatures.SCALE]} km")
	print(f"Coastline Location: {problem['Arena'][JF.ArenaFeatures.COASTLINE] * problem['Arena'][JF.ArenaFeatures.SCALE]} km")
	print(f"Frontline Location: {problem['Arena'][JF.ArenaFeatures.FRONTLINE] * problem['Arena'][JF.ArenaFeatures.SCALE]} km")
	print(f"Time Horizon: {problem['Arena'][JF.ArenaFeatures.TIMEHORIZON]} hours")
	print("\n\t [x, y] (km)\t\tStatic\tSpeed(km/h)\tAmmo(%)\tEnergy\tTime(hours)\tEffDist\tAmmoRate\tEnergyRate")
	for effector in problem['Effectors']:
		print("Effector:", end="")
		print(f"[{effector[JF.EffectorFeatures.XPOS] * scale:.6}, {effector[JF.EffectorFeatures.YPOS] * scale:.6}]", end="")
		print(f"\t{trueFalseStrings[int(effector[JF.EffectorFeatures.STATIC])]}", end="")
		print(f"\t{effector[JF.EffectorFeatures.SPEED] * scale:.4}", end="")
		print(f"\t\t{effector[JF.EffectorFeatures.AMMOLEFT]:.4}", end="")
		print(f"\t{effector[JF.EffectorFeatures.ENERGYLEFT]:.4}", end="")
		print(f"\t{effector[JF.EffectorFeatures.TIMELEFT] * problem['Arena'][JF.ArenaFeatures.TIMEHORIZON]}", end="")
		print(f"\t\t{effector[JF.EffectorFeatures.EFFECTIVEDISTANCE] * scale}", end="")
		print(f"\t{1 / effector[JF.EffectorFeatures.AMMORATE]}", end="")
		print(f"\t\t{effector[JF.EffectorFeatures.ENERGYRATE]}")
	
	print("\n\n\t[x, y] (km)\t\t\tValue ([0,1])\tSelected")
	for target in problem['Targets']:
		print(f"Target: [{target[JF.TaskFeatures.XPOS] * scale:6.6}, {target[JF.TaskFeatures.YPOS] * scale:6.6}]\t\t{target[JF.TaskFeatures.VALUE]:6.4}\t\t{target[JF.TaskFeatures.SELECTED]}")
	
	print("\n\n\t\tPSucc\tEnergy\ttime\tSelectable")
	for i in range(len(problem['Effectors'])):
		for j in range(len(problem['Targets'])):
			opportunity = problem['Opportunities'][i][j]
			print(f"({i:2},{j:2}):\t{opportunity[JF.OpportunityFeatures.PSUCCESS]:.4}\t{opportunity[JF.OpportunityFeatures.ENERGYCOST]:5.4}\t{opportunity[JF.OpportunityFeatures.TIMECOST]:.4}\t{trueFalseStrings[int(opportunity[JF.OpportunityFeatures.SELECTABLE])]}")


def main():
	problem = PG.CombatArms()
	printProblem(problem)


if __name__ == "__main__":
    main()