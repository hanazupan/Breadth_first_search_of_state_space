# Breadth_first_search_of_state_space
Repository for a WS 2021 internship project: Breadth-first search algorithm for exploration of state space.

## Mazes
In the folder Maze, classes and functions for creating N-dimensional mazes and visualizing 2-dimensional mazes are provided. Additionally, several algorithms
for exploration of N-dimensional mazes are provided. Graphs of cell connections and adjacency matrices can be obtained as a result of such exploration. To
obtain visualizations, animations, graphs and/or adjacency matrices of a maze, run the driver script `run_maze.py` as

`python3 run_maze.py`

Using optional flags like:
 - `--size "(20, 30)"` to control the size of the maze (default="(20, 20)")
 - `--name new_maze` to change the names of saved files (default="default")
 - `--path images/` to control where generated files are saved (default="./")
 - `--explorer dijkstra` to change the exploration algorithm (default="bfs")
 - `--animate` to save the animations (default="n")
 - `--visualize` to save the image of the maze (default="y")
 - `--graph` to save the image of the graph

Some examples to try:
 - `python3 run_maze.py --size "(40,40)" --animate y --graph y --name test --path images/`

## Energy surfaces
It is possible to create an energy surface from a maze using a 2D spline
interpolation. It is also possible to use an example two-well potential as
your energy surface or to create it with Lennard-Jones potentials of
randomly positioned atoms. Once such a surface is created, it can be explored using
breadth-first search. The explorer returns an adjacency matrix from
which rates matrix is constructed using SqRA. To explore energy surfaces,
run the driver script `run_energy.py` as

`python3 run_energy.py`

Using optional flags like:
 - `--type atoms` to define the type of energy surface (default="potential")
 - `--size "(20, 30)"` to control the discretization of the surface (default="(20, 20)")
 - `--name new_surface` to change the names of saved files (default="default")
 - `--path images/` to control where generated files are saved (default="./")
 - `--animate y` to create and save all animations (default="n")
 - `--visualize y` to create and save all plots (default="y")
 - `--num_atoms 4` to determine how many atoms to position on the surface (only for type==`atoms`)

Some examples to try:
 - `python3 run_energy.py --type potential --size "(40, 40)" --name test_potential --path images/`
 - `python3 run_energy.py --type maze --size "(15, 20)" --name test_maze --path images/`
 - `python3 run_energy.py --type atoms --size "(15,15)" --num_atoms 4 --name test_atoms --path images/`

## Simulations on surfaces
After an energy surface is created, it is also possible to run a simulation
on it, create a Markov State Model (MSM) and compare the properties of that matrix to SqRA
produced with rates matrix. That is done in the Simulation class. To run a simulation
on any type of energy surface, run the driver script `run_simulation.py` as

`python3 run_simulation.py`

Using optional flags like:
 - `--type atoms` to define the type of energy surface (default="potential")
 - `--size "(20, 30)"` to control the discretization of the surface (default="(20, 20)")
 - `--name new_surface` to change the names of saved files (default="default")
 - `--path images/` to control where generated files are saved (default="./")
 - `--visualize y` to create and save all plots (default="y")
 - `--num_atoms 4` to determine how many atoms to position on the surface (only for type==`atoms`)
 - `--duration 1e7` to set the number of time steps (default=1e6)
 - `--time_step 0.001` to set the size of one time step (default=0.1)
- `-- compare n` to determine whether the Energy plots should also be computed for comparison

Some examples to try:
 - `python3 run_simulation.py --type potential --size "(40, 40)" --name test_potential --path images/ --compare n`
 - `python3 run_simulation.py --type maze --size "(15, 20)" --name test_maze --path images/ --duration 1e7`
 - `ython3 run_simulation.py --type atoms --size "(15,15)" --num_atoms 4 --name test_atoms --path images/ --time_step 0.1`
