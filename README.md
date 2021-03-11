# JointFireAutomation
This code is related to the Joint Fire Automation problem.  It involves a simulator as well as other helper classes to facilitate the operation and testing

Requirements: Python3, numpy, flask, and deap.

`JFAFeatures` provides a consistent feature set between all files.

`ProblemGenerators` is used to generate a variety of different problems for use in the simulator.  You can change the size of the arena, as well as the quantities of different styles of effectors or the quantity of targets.

`SampleSimulator` provides the core `update` function as well as an interface to interact with a given problem.

`JFASolvers` provides access to various traditional AI solving mechanisms (AStar, Greedy, random walk, etc.)

`TrainSetGeneration` provides a wrapper to generate multiple problems with solutions from various solvers.

`ActionServer` starts up a REST server to receive state information and provide an action, or actions based on the solver selected.
