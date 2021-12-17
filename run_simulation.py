"""
Try to run, for example:
python3 run_simulation.py --type potential --size "(40, 40)" --compare n
python3 run_simulation.py --type maze --size "(15, 20)" --duration 1e7
python3 run_simulation.py --type atoms --size "(15,15)" --num_atoms 4 --time_step 0.1
"""

# internal imports
from maze.create_mazes import Maze
from maze.create_energies import EnergyFromPotential, EnergyFromMaze, EnergyFromAtoms, Atom
from run_energy import determine_name
from simulation.create_simulation import Simulation
from simulation.create_msm import MSM
from plotting.plotting_simulations import plot_everything_simulation
from plotting.plotting_energies import plot_everything_energy
from constants import *
# standard library
from ast import literal_eval
import argparse
import time
import random
# external imports
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--type', metavar='t', type=str, nargs='?',
                    default="potential", help='Select the type of energy surface (potential, maze, atoms).')
parser.add_argument('--size', metavar='s', type=str, nargs='?',
                    default="(20, 20)", help='Select the size of the energy surface.')
parser.add_argument('--visualize', metavar='v', type=str, nargs='?',
                    default='y', help='Produce all plots? (y/n)')
parser.add_argument('--num_atoms', metavar='na', type=str, nargs='?',
                    default='3', help='How many atoms if type==atoms?')
parser.add_argument('--duration', metavar='d', type=str, nargs='?',
                    default='1e6', help='How many time steps in the simulation?')
parser.add_argument('--time_step', metavar='ts', type=str, nargs='?',
                    default='0.01', help='What should the time step be?')
parser.add_argument('--compare', metavar='c', type=str, nargs='?',
                    default='y', help='Compare to SqRA?')
parser.add_argument('--seed', type=str, default='n', help="Set a seed for random processes?")


def report_time(start, end):
    duration = end - start
    hours = round(duration // 3600 % 24)
    minutes = round(duration // 60 % 60)
    seconds = round(duration % 60)
    return hours, minutes, seconds


def produce_energies(args):
    # set a seed if given by the user
    if my_args.seed != "n":
        random.seed(int(my_args.seed))
        np.random.seed(int(my_args.seed))
    # generate the name and create Energy object
    name = determine_name(args)
    print(f"Given the name: {name}")
    print("Setting up the Energy object ...")
    start_time = time.time()
    args.size = literal_eval(args.size)
    if args.type == "potential":
        my_energy = EnergyFromPotential(size=args.size, images_path=PATH_IMG_POTENTIALS, images_name=name, friction=10,
                                        T=400)
    elif args.type == "maze":
        my_maze = Maze(size=args.size, images_path=PATH_IMG_MAZES, images_name=name, edge_is_wall=True, no_branching=True)
        my_energy = EnergyFromMaze(my_maze, images_path=PATH_IMG_MAZES, images_name=name, factor_grid=3, friction=5,
                                   grid_start=(0, 0), grid_end=(10, 10))
    elif args.type == "atoms":
        atoms = []
        args.num_atoms = int(args.num_atoms)
        for i in range(args.num_atoms):
            x_coo = 0 + 10*np.random.rand()
            y_coo = 0 + 10*np.random.rand()
            epsilon = np.random.choice([1, 3, 5, 10])
            sigma = np.random.choice([1, 2, 3])
            atom = Atom((x_coo, y_coo), epsilon, sigma)
            atoms.append(atom)
        atoms = tuple(atoms)
        my_energy = EnergyFromAtoms(size=args.size, atoms=atoms, images_path=PATH_IMG_ATOMS, grid_start=(0, 0),
                                    grid_end=(10, 10), images_name=name, friction=1, m=1, T=400)
    else:
        raise ValueError(f"{args.type} is not a valid type of Energy surface! Select from: (potential, maze, atoms).")
    end_setup_time = time.time()
    hours, minutes, seconds = report_time(start_time, end_setup_time)
    print(f" -> time for setup: {hours}h {minutes}min {seconds}s.")
    # visualization
    if args.visualize != "n" and args.compare != "n":
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
    end_time = time.time()
    hours, minutes, seconds = report_time(start_time, end_time)
    print(f"-------- Total Energy time: {hours}h {minutes}min {seconds}s. --------")
    return my_energy


def produce_simulation(args, energy):
    print("Setting up the Simulation object ...")
    start_time = time.time()
    my_simulation = Simulation(energy, images_path=energy.images_path, images_name=energy.images_name)
    args.duration = int(float(args.duration))
    args.time_step = float(args.time_step)
    end_setup_time = time.time()
    hours, minutes, seconds = report_time(start_time, end_setup_time)
    print(f" -> time for setup: {hours}h {minutes}min {seconds}s.")
    print("Simulating the trajectory ...")
    my_simulation.integrate(N=args.duration, dt=args.time_step, save_trajectory=False)
    end_simulation_time = time.time()
    hours, minutes, seconds = report_time(start_time, end_simulation_time)
    print(f" -> time for simulation: {hours}h {minutes}min {seconds}s.")
    # visualization
    if args.visualize != "n":
        print("Calculating the MSM ...")
        msm = MSM(energy.images_name, images_path=energy.images_path)
        msm.get_transitions_matrix(noncorr=True)
        msm.get_eigenval_eigenvec(num_eigv=20, which="LR")
        end_matrix_time = time.time()
        hours, minutes, seconds = report_time(end_simulation_time, end_matrix_time)
        print(f" -> time for MSM: {hours}h {minutes}min {seconds}s.")
        print("Producing images ...")
        plot_everything_simulation(energy.images_name, traj=False)
        end_visualization_time = time.time()
        hours, minutes, seconds = report_time(end_matrix_time, end_visualization_time)
        print(f" -> time for images: {hours}h {minutes}min {seconds}s.")
    end_time = time.time()
    hours, minutes, seconds = report_time(start_time, end_time)
    # write the seed to file
    path = my_simulation.path_to_summary()
    with open(path + f"{my_simulation.images_name}_summary.txt", "a+", encoding='utf-8') as f:
        f.write(f"seed = {args.seed}")
    print(f"-------- Total Simulation time: {hours}h {minutes}min {seconds}s. --------")


if __name__ == '__main__':
    my_args = parser.parse_args()
    energy_object = produce_energies(my_args)
    produce_simulation(my_args, energy_object)
