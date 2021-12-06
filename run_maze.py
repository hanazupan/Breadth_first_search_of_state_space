"""
Try to run, for example: python3 run_maze.py --size "(40,40)" --animate y --graph y

For presentation:
python3 run_maze.py --size "(15,15)" --animate y --explorer all
"""
from maze.create_mazes import Maze
from maze.explore_mazes import BFSExplorer, DijkstraExplorer, DFSExplorer
from plotting.plotting_energies import plot_maze
from ast import literal_eval
from os.path import exists
import argparse

# Warning: saves in images/animations even if animations are not produced
PATH = "images/animations/"

parser = argparse.ArgumentParser()
parser.add_argument('--size', metavar='s', type=str, nargs='?',
                    default="(20, 20)", help='Select the size of maze.')
parser.add_argument('--animate', metavar='a', type=str, nargs='?',
                    default='n', help='Produce animations?')
parser.add_argument('--explorer', metavar='e', type=str, nargs='?',
                    default='bfs', help='Algorithm for exploration of the maze (bfs, dfs, dijkstra, all)?')
parser.add_argument('--graph', metavar='g', type=str, nargs='?',
                    default='n', help='Produce graph image?')
parser.add_argument('--visualize', metavar='v', type=str, nargs='?',
                    default='y', help='Produce maze image?')


def explore(explorer, type_explorer, animate, name, args):
    if animate:
        explorer.explore_and_animate()
        print(f"Animation Maze exploration will be saved in: {PATH}{type_explorer}_{name}.gif")
    else:
        explorer.explore()


def create_and_explore_maze(args):
    # set the name of the file
    name_int = 0
    name = f"maze_only{name_int:03d}"
    while exists(PATH + name + "_maze.pdf"):
        name_int += 1
        name = f"maze_only{name_int:03d}"
    print(f"Maze size: {args.size}")
    if args.animate != "n":
        animate = True
        print(f"Animation Maze creation will be saved in: {PATH}building_{name}.gif")
    else:
        animate = False
    # correctly interpret a tuple for size input
    args.size = literal_eval(args.size)
    maze = Maze(args.size, animate=animate, images_name=name, images_path=PATH)
    # visualization
    if args.visualize != "n":
        plot_maze(name)
        print(f"Visualization of Maze will be saved in: {PATH}{name}_maze.png")
    # exploration and animation
    if args.explorer == "bfs" or args.explorer == "all":
        explorer = BFSExplorer(maze)
        explore(explorer, "bfs", animate, name, args)
    if args.explorer == "dijkstra" or args.explorer == "all":
        explorer = DijkstraExplorer(maze)
        explore(explorer, "dijkstra", animate, name, args)
        explorer.visualize_distances()
        print(f"Visualization of distances will be saved in: {PATH}distances_{name}.png")
    if args.explorer == "dfs" or args.explorer == "all":
        explorer = DFSExplorer(maze)
        explore(explorer, "dfs", animate, name, args)
    # graph
    if args.graph != "n":
        print(f"Visualization of Graph will be saved in: {PATH}{args.explorer}_graph_{name}.png")
        explorer.draw_connections_graph(with_labels=True)
    print("Finished.")


if __name__ == '__main__':
    my_args = parser.parse_args()
    create_and_explore_maze(my_args)
