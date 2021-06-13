#!/usr/bin/env python3
if __package__ is not None and len(__package__) > 0:
    print(f"{__name__} using relative import inside of {__package__}")
    from . import features as jf
    from . import problem_generators as pg
else:
    import features as jf
    import problem_generators as pg


def printProblem(problem, arena=False):
	scale = problem['Arena'][jf.ArenaFeatures.SCALE]
	trueFalseStrings = ["False", "True"]

	if arena:
		print(f"Arena Scale: {problem['Arena'][jf.ArenaFeatures.SCALE]} km")
		print(f"Coastline Location: {problem['Arena'][jf.ArenaFeatures.COASTLINE]} km")
		print(f"Frontline Location: {problem['Arena'][jf.ArenaFeatures.FRONTLINE]} km")
		print(f"Time Horizon: {problem['Arena'][jf.ArenaFeatures.TIMEHORIZON]} hours")
	print("\n\t [x, y]\t\t\tStatic\tSpeed\tAmmo\tEnergy\tTime\tEffDist\tAmmoRate\tEnergyRate")
	for effector in problem['Effectors']:
		print("Effector:", end="")
		print(f"[{effector[jf.EffectorFeatures.XPOS]:.4f}, {effector[jf.EffectorFeatures.YPOS]:.4f}]", end="")
		print(f"\t{trueFalseStrings[int(effector[jf.EffectorFeatures.STATIC])]}", end="")
		print(f"\t{effector[jf.EffectorFeatures.SPEED]:4.4f}", end="")
		print(f"\t{effector[jf.EffectorFeatures.AMMOLEFT]:.4f}", end="")
		print(f"\t{effector[jf.EffectorFeatures.ENERGYLEFT]:.4f}", end="")
		print(f"\t{effector[jf.EffectorFeatures.TIMELEFT]:.4f}", end="")
		print(f"\t{effector[jf.EffectorFeatures.EFFECTIVEDISTANCE]:.4f}", end="")
		print(f"\t{effector[jf.EffectorFeatures.AMMORATE]:.4f}", end="")
		print(f"\t\t{effector[jf.EffectorFeatures.ENERGYRATE]:.4f}")

	print("\n\n\t[x, y] (km)\t\tValue\tSelected")
	for target in problem['Targets']:
		print(f"Target: [{target[jf.TaskFeatures.XPOS]:4.4f}, {target[jf.TaskFeatures.YPOS]:4.4f}]\t{target[jf.TaskFeatures.VALUE]:4.4f}\t{target[jf.TaskFeatures.SELECTED]}")

	print("\n\n\t\tPSucc\tEnergy\ttime\tSelectable")
	for i in range(len(problem['Effectors'])):
		for j in range(len(problem['Targets'])):
			opportunity = problem['Opportunities'][i][j]
			if opportunity[jf.OpportunityFeatures.SELECTABLE]:
				print(f"({i:2},{j:2}):\t{opportunity[jf.OpportunityFeatures.PSUCCESS]:.4f}\t{opportunity[jf.OpportunityFeatures.ENERGYCOST]:4.5f}\t{opportunity[jf.OpportunityFeatures.TIMECOST]:4.5f}\t{trueFalseStrings[int(opportunity[jf.OpportunityFeatures.SELECTABLE])]}")


def main():
	problem = pg.network_validation(4, 8)
	printProblem(problem, arena=True)
	#PG.saveProblem(problem, "folder/" + "problem.json")


if __name__ == "__main__":
    main()
