# Breadth_first_search_of_state_space
Repository for a WS 2021 internship project: Breadth-first search algorithm for exploration of state space.

## Maze
In the folder Maze, classes and functions for creating N-dimensional mazes and visualizing 2-dimensional mazes are provided. Additionally, several algorithms
for exploration of N-dimensional mazes are provided. Graphs of cell connections and adjacency matrices can be obtained as a result of such exploration. To
obtain visualizations, animations, graphs and/or adjacency matrices of a maze, run the driver script `Maze/main.py` as

`python3 maze.py`

Using optional flags like:
 - `--size "(20, 30)"` to control the size of the maze (default="(20, 20)")
 - `--name new_maze` to change the names of saved files (default="default")
 - `--path Images/` to control where generated files are saved (default="./")
 - `--explorer dijkstra` to change the exploration algorithm (default="bfs")
 - `--animate` to save the animations
 - `--visualize` to save the image of the maze
 - `--graph` to save the image of the graph
 - `--matrix` to save the adjacency matrix
