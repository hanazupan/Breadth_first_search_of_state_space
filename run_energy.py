"""
Try to run, for example:
python3 run_energy.py --type potential --size "(40, 40)" --name test_potential --path images/
python3 run_energy.py --type maze --size "(15, 20)" --name test_maze --path images/
python3 run_energy.py --type atoms --size "(15,15)" --num_atoms 4 --name test_atoms --path images/
"""
from maze.create_mazes import Maze
from maze.create_energies import EnergyFromPotential, EnergyFromMaze, EnergyFromAtoms, Atom
from maze.explore_mazes import BFSExplorer, DFSExplorer
import numpy as np
from ast import literal_eval
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--type', metavar='t', type=str, nargs='?',
                    default="potential", help='Select the type of energy surface (potential, maze, atoms).')
parser.add_argument('--size', metavar='s', type=str, nargs='?',
                    default="(20, 20)", help='Select the size of the energy surface.')
parser.add_argument('--name', metavar='n', type=str, nargs='?',
                    default='energy', help='Provide a name for saved images and animations.')
parser.add_argument('--path', metavar='p', type=str, nargs='?',
                    default='./', help='Provide a path where images and animations will be saved.')
parser.add_argument('--animate', metavar='a', type=str, nargs='?',
                    default='n', help='Produce animations? (y/n)')
parser.add_argument('--visualize', metavar='v', type=str, nargs='?',
                    default='y', help='Produce all plots? (y/n)')
parser.add_argument('--num_atoms', metavar='na', type=str, nargs='?',
                    default='3', help='How many atoms if type==atoms?')


def report_time(start, end):
    duration = end - start
    hours = round(duration // 3600 % 24)
    minutes = round(duration // 60 % 60)
    seconds = round(duration % 60)
    return hours, minutes, seconds


def produce_energies(args):
    print("Setting up the Energy object ...")
    start_time = time.time()
    args.size = literal_eval(args.size)
    if args.type == "potential":
        my_energy = EnergyFromPotential(size=args.size, images_path=args.path, images_name=args.name)
    elif args.type == "maze":
        my_maze = Maze(size=args.size, images_path=args.path, images_name=args.name)
        my_maze.visualize()
        my_energy = EnergyFromMaze(my_maze, images_path=args.path, images_name=args.name)
        my_energy.visualize_underlying_maze()
    elif args.type == "atoms":
        atoms = []
        args.num_atoms = int(args.num_atoms)
        for i in range(args.num_atoms):
            x_coo = -10 + 20*np.random.rand()
            y_coo = -10 + 20*np.random.rand()
            epsilon = np.random.choice([1, 3, 5, 10])
            sigma = np.random.choice([2, 4, 6])
            atom = Atom((x_coo, y_coo), epsilon, sigma)
            atoms.append(atom)
        atoms = tuple(atoms)
        my_energy = EnergyFromAtoms(size=args.size, atoms=atoms, grid_edges=(-12, 12, -12, 12), images_path=args.path,
                                    images_name=args.name)
    else:
        raise ValueError(f"{args.type} is not a valid type of Energy surface! Select from: (potential, maze, atoms).")
    end_setup_time = time.time()
    hours, minutes, seconds = report_time(start_time, end_setup_time)
    print(f" -> time for setup: {hours}h {minutes}min {seconds}s.")
    # visualization
    if args.visualize != "n":
        print("Calculating the rates matrix ...")
        my_energy.get_rates_matix()
        end_matrix_time = time.time()
        hours, minutes, seconds = report_time(end_setup_time, end_matrix_time)
        print(f" -> time for rates matrix: {hours}h {minutes}min {seconds}s.")
        print("Producing images ...")
        my_energy.visualize()
        my_energy.visualize_3d()
        my_energy.visualize_eigenvectors_in_maze(num=8, which="LR")
        my_energy.visualize_eigenvalues()
        end_visualization_time = time.time()
        hours, minutes, seconds = report_time(end_matrix_time, end_visualization_time)
        print(f" -> time for images: {hours}h {minutes}min {seconds}s.")
    # exploration/animation
    if args.animate != "n":
        print("Producing animations ...")
        explorer = BFSExplorer(my_energy)
        explorer.explore_and_animate()
        explorer = DFSExplorer(my_energy)
        explorer.explore_and_animate()
        end_animation_time = time.time()
        if args.visualize != "n":
            hours, minutes, seconds = report_time(end_visualization_time, end_animation_time)
        else:
            hours, minutes, seconds = report_time(end_setup_time, end_animation_time)
        print(f" -> time for animations: {hours}h {minutes}min {seconds}s.")
    my_energy.save_information()
    end_time = time.time()
    hours, minutes, seconds = report_time(start_time, end_time)
    print(f"Finished. Total time: {hours}h {minutes}min {seconds}s.")


if __name__ == '__main__':
    my_args = parser.parse_args()
    produce_energies(my_args)
