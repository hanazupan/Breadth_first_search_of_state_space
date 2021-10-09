"""
Try to run, for exymple: main.py --size (40,40) --animate y --graph y --matrix y --name test --path Images/
"""
from create_mazes import Maze, MazeAnimation
from explore_mazes import BFSExplorer, DijkstraExplorer
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
                    default='bfs', help='Algorithm for exploration of the maze (bfs, dijkstra)?')
parser.add_argument('--graph', metavar='g', type=str, nargs='?',
                    default='n', help='Produce graph image?')
parser.add_argument('--matrix', metavar='m', type=str, nargs='?',
                    default='n', help='Produce adjacency matrix?')
parser.add_argument('--visualize', metavar='v', type=str, nargs='?',
                    default='n', help='Produce maze image?')

# TODO: finish
def create_and_explore_maze(args):
    print(f"Maze size: {args.size}")
    if args.animate == "y" or "yes":
        animate = True
        print(f"Animation Maze creation will be saved in: {args.path}building_{args.name}.gif")
    else:
        animate = False
    # correctly interpret a tuple for size input
    args.size = tuple([int(x) for x in args.size.strip("(").strip(")").split(",")])
    maze = Maze(args.size, animate=animate, images_name=args.name, images_path=args.path)
    # visualization
    if args.visualize == "y" or "yes":
        maze.visualize(show=False)
        print(f"Visualization of Maze will be saved in: {args.path}maze_{args.name}.gif")
    # exploration
    if args.explorer == "bfs":
        explorer = BFSExplorer(maze)
    elif args.explorer == "dijkstra":
        explorer = DijkstraExplorer(maze)
        explorer.visualize_distances(show=False)
        print(f"Visualization of distances will be saved in: {args.path}distances_{args.name}.png")
    else:
        raise ValueError("Not a valid Explorer.")
    if animate:
        explorer.explore_and_animate()
        print(f"Animation Maze exploration will be saved in: {args.path}solving_{args.name}.gif")
    else:
        explorer.explore()
    # graph
    if args.graph == "y" or "yes":
        print(f"Visualization of Graph will be saved in: {args.path}{args.explorer}_graph_{args.name}.gif")
        explorer.draw_connections_graph(show=False, with_labels=True)
    # matrix
    if args.matrix == "y" or "yes":
        print(explorer.get_adjacency_matrix())
    print("Finished.")


if __name__ == '__main__':
    args = parser.parse_args()
    create_and_explore_maze(args)
