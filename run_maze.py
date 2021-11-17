"""
Try to run, for example: python3 run_maze.py --size "(40,40)" --animate y --graph y --name test --path images/

For presentation:
python3 run_maze.py --size "(15,15)" --animate y --explorer all --name pres_maze --path presentation/presentation_img/
"""
from maze.create_mazes import Maze
from maze.explore_mazes import BFSExplorer, DijkstraExplorer, DFSExplorer
from ast import literal_eval
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--size', metavar='s', type=str, nargs='?',
                    default="(20, 20)", help='Select the size of maze.')
parser.add_argument('--name', metavar='n', type=str, nargs='?',
                    default='default', help='Provide a name for saved images and animations')
parser.add_argument('--path', metavar='p', type=str, nargs='?',
                    default='./', help='Provide a path where images and animations are saved')
parser.add_argument('--animate', metavar='a', type=str, nargs='?',
                    default='n', help='Produce animations?')
parser.add_argument('--explorer', metavar='e', type=str, nargs='?',
                    default='bfs', help='Algorithm for exploration of the maze (bfs, dfs, dijkstra, all)?')
parser.add_argument('--graph', metavar='g', type=str, nargs='?',
                    default='n', help='Produce graph image?')
parser.add_argument('--visualize', metavar='v', type=str, nargs='?',
                    default='y', help='Produce maze image?')


def explore(explorer, type_explorer, animate, args):
    if animate:
        explorer.explore_and_animate()
        print(f"Animation Maze exploration will be saved in: {args.path}{type_explorer}_{args.name}.gif")
    else:
        explorer.explore()


def create_and_explore_maze(args):
    print(f"Maze size: {args.size}")
    if args.animate != "n":
        animate = True
        print(f"Animation Maze creation will be saved in: {args.path}building_{args.name}.gif")
    else:
        animate = False
    # correctly interpret a tuple for size input
    args.size = literal_eval(args.size)
    maze = Maze(args.size, animate=animate, images_name=args.name, images_path=args.path)
    # visualization
    if args.visualize != "n":
        maze.visualize()
        print(f"Visualization of Maze will be saved in: {args.path}maze_{args.name}.png")
    # exploration and animation
    if args.explorer == "bfs" or args.explorer == "all":
        explorer = BFSExplorer(maze)
        explore(explorer, "bfs", animate, args)
    if args.explorer == "dijkstra" or args.explorer == "all":
        explorer = DijkstraExplorer(maze)
        explore(explorer, "dijkstra", animate, args)
        explorer.visualize_distances()
        print(f"Visualization of distances will be saved in: {args.path}distances_{args.name}.png")
    if args.explorer == "dfs" or args.explorer == "all":
        explorer = DFSExplorer(maze)
        explore(explorer, "dfs", animate, args)
    # graph
    if args.graph != "n":
        print(f"Visualization of Graph will be saved in: {args.path}{args.explorer}_graph_{args.name}.png")
        explorer.draw_connections_graph(with_labels=True)
    print("Finished.")


if __name__ == '__main__':
    my_args = parser.parse_args()
    create_and_explore_maze(my_args)
