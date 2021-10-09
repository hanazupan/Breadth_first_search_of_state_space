from create_mazes import Maze, MazeAnimation
from explore_mazes import BFSExplorer, DijkstraExplorer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('size', metavar='s', type=tuple, nargs='?',
                    default=(20, 20), help='Select the size of maze.')
parser.add_argument('--name', metavar='n', type=str, nargs='?',
                    default='default', help='Provide a name for saved images and images')
parser.add_argument('--animate', metavar='a', type=str, nargs='?',
                    default='n', help='Produce animations?')
parser.add_argument('--explorer', metavar='e', type=str, nargs='?',
                    default='bfs', help='Algorithm for exploration of the maze (bfs, dijkstra)?')
parser.add_argument('--graph', metavar='g', type=str, nargs='?',
                    default='n', help='Produce graph image?')
parser.add_argument('--visualize', metavar='v', type=str, nargs='?',
                    default='bfs', help='Produce maze image?')

# TODO: finish
args = parser.parse_args()
if args.animate == "y":
    animate = True
else:
    animate = False
maze = Maze(args.size, animate=animate, images_name=args.name)
if args.explorer == "bfs":
    explorer = BFSExplorer(maze)
elif args.explorer == "dijkstra":
    explorer = DijkstraExplorer(maze)
if animate:
    explorer.explore()
else:
    explorer.explore_and_animate()
