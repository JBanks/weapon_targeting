from enum import IntEnum, auto


class ArenaFeatures(IntEnum):
	COASTLINE = 0
	FRONTLINE = auto()
	FRONTAGE = auto()
	SCALE = auto()
	TIMEHORIZON = auto()


class EffectorFeatures(IntEnum):
	XPOS = 0
	YPOS = auto()
	STARTX = auto()
	STARTY = auto()
	STATIC = auto()
	SPEED = auto()
	AMMOLEFT = auto()
	ENERGYLEFT = auto()
	TIMELEFT = auto()
	# DUTYCYCLE = auto() # Using OpportunityFeatures.TIMECOST for the first iteration
	EFFECTIVEDISTANCE = auto()
	AMMORATE = auto()  # Ammo per engagement
	ENERGYRATE = auto()  # energy per engagement (possibly 0 and only affected by movement from A to B)
	TYPE = auto()
	PLANE = TYPE
	HELICOPTER = auto()
	INFANTRY = auto()
	ARTILLERY = auto()
	ARMOURED = auto()
	FRIGATE = auto()


class TaskFeatures(IntEnum):
	XPOS = 0
	YPOS = auto()
	VALUE = auto()
	SELECTED = auto()


class OpportunityFeatures(IntEnum):
	PSUCCESS = 0
	ENERGYCOST = auto()
	TIMECOST = auto()
	SELECTABLE = auto()


class AssetTypes(IntEnum):
	PLANE = 0
	HELICOPTER = auto()
	INFANTRY = auto()
	ARTILLERY = auto()
	ARMOURED = auto()
	FRIGATE = auto()
