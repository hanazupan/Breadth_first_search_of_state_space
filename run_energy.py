"""
Try to run, for example:
python3 run_energy.py --type potential --size "(40, 40)"
python3 run_energy.py --type maze --size "(15, 20)"
python3 run_energy.py --type atoms --size "(15,15)" --num_atoms 4
"""

# internal imports
from plotting.plotting_energies import plot_everything_energy
from maze.create_mazes import Maze
from maze.create_energies import EnergyFromPotential, EnergyFromMaze, EnergyFromAtoms, Atom
from maze.explore_mazes import BFSExplorer, DFSExplorer
# standard library
from ast import literal_eval
import argparse
import time
from os.path import exists
# external imports
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--type', metavar='t', type=str, nargs='?',
                    default="potential", help='Select the type of energy surface (potential, maze, atoms).')
parser.add_argument('--size', metavar='s', type=str, nargs='?',
                    default="(20, 20)", help='Select the size of the energy surface.')
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


def determine_name(args):
    # set the name of the file
    name_int = 0
    if args.type == "potential":
        name = f"potential{name_int:03d}"
        while exists("data/energy_summaries/potentials/" + name + "_summary.txt"):
            name_int += 1
            name = f"potential{name_int:03d}"
    elif args.type == "maze":
        name = f"maze{name_int:03d}"
        while exists("data/energy_summaries/mazes/" + name + "_summary.txt"):
            name_int += 1
            name = f"maze{name_int:03d}"
    elif args.type == "atoms":
        name = f"atoms{name_int:03d}"
        while exists("data/energy_summaries/atoms/" + name + "_summary.txt"):
            name_int += 1
            name = f"atoms{name_int:03d}"
    else:
        raise ValueError(f"{args.type} is not a valid type of Energy surface! Select from: (potential, maze, atoms).")
    return name


def produce_energies(args):
    name = determine_name(args)
    print(f"Given the name: {name}")
    print("Setting up the Energy object ...")
    start_time = time.time()
    args.size = literal_eval(args.size)
    if args.type == "potential":
        my_energy = EnergyFromPotential(size=args.size, images_path="images/potentials/", images_name=name)
    elif args.type == "maze":
        my_maze = Maze(size=args.size, images_path="images/mazes/", images_name=name)
        my_energy = EnergyFromMaze(my_maze, images_path="images/mazes/", images_name=name)
    elif args.type == "atoms":
        atoms = []
        args.num_atoms = int(args.num_atoms)
        for i in range(args.num_atoms):
            x_coo = 10*np.random.rand()
            y_coo = 10*np.random.rand()
            epsilon = np.random.choice([1, 3, 5, 10])
            sigma = np.random.choice([1, 2, 3])
            atom = Atom((x_coo, y_coo), epsilon, sigma)
            atoms.append(atom)
        atoms = tuple(atoms)
        my_energy = EnergyFromAtoms(size=args.size, atoms=atoms, grid_start=(0, 0), grid_end=(10, 10),
                                    images_path="images/atoms/", images_name=name)
    else:
        raise ValueError(f"{args.type} is not a valid type of Energy surface! Select from: (potential, maze, atoms).")
    end_setup_time = time.time()
    hours, minutes, seconds = report_time(start_time, end_setup_time)
    print(f" -> time for setup: {hours}h {minutes}min {seconds}s.")
    # visualization
    if args.visualize != "n":
        print("Calculating the rates matrix ...")
        my_energy.get_rates_matix()
        my_energy.get_eigenval_eigenvec(20, which="LR")
        end_matrix_time = time.time()
        hours, minutes, seconds = report_time(end_setup_time, end_matrix_time)
        print(f" -> time for rates matrix: {hours}h {minutes}min {seconds}s.")
        print("Producing images ...")
        plot_everything_energy(name)
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
    end_time = time.time()
    hours, minutes, seconds = report_time(start_time, end_time)
    print(f"Finished. Total time: {hours}h {minutes}min {seconds}s.")


if __name__ == '__main__':
    my_args = parser.parse_args()
    produce_energies(my_args)
