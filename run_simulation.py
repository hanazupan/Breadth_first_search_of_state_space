"""
Try to run, for example:
python3 run_simulation.py --type potential --size "(40, 40)" --name test_potential --path images/ --compare n
python3 run_simulation.py --type maze --size "(15, 20)" --name test_maze --path images/ --duration 1e7
python3 run_simulation.py --type atoms --size "(15,15)" --num_atoms 4 --name test_atoms --path images/ --time_step 0.1
"""
from maze.create_mazes import Maze
from maze.create_energies import EnergyFromPotential, EnergyFromMaze, EnergyFromAtoms, Atom
from simulation.create_simulation import Simulation
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
        my_energy = EnergyFromPotential(size=args.size, images_path=args.path, images_name=args.name, friction=10)
    elif args.type == "maze":
        my_maze = Maze(size=args.size, images_path=args.path, images_name=args.name, edge_is_wall=True)
        my_energy = EnergyFromMaze(my_maze, images_path=args.path, images_name=args.name, factor_grid=1)
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
    if args.visualize != "n" and args.compare != "n":
        print("Calculating the rates matrix ...")
        my_energy.get_rates_matix()
        end_matrix_time = time.time()
        hours, minutes, seconds = report_time(end_setup_time, end_matrix_time)
        print(f" -> time for rates matrix: {hours}h {minutes}min {seconds}s.")
        print("Producing images ...")
        my_energy.visualize()
        my_energy.visualize_3d()
        my_energy.visualize_eigenvectors_in_maze(num=6, which="LR")
        my_energy.visualize_eigenvalues()
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
    my_simulation = Simulation(energy, images_path=args.path, images_name=args.name)
    args.duration = int(float(args.duration))
    args.time_step = float(args.time_step)
    end_setup_time = time.time()
    hours, minutes, seconds = report_time(start_time, end_setup_time)
    print(f" -> time for setup: {hours}h {minutes}min {seconds}s.")
    print("Simulating the trajectory ...")
    my_simulation.integrate(N=args.duration, dt=args.time_step)
    end_simulation_time = time.time()
    hours, minutes, seconds = report_time(start_time, end_simulation_time)
    print(f" -> time for simulation: {hours}h {minutes}min {seconds}s.")
    my_simulation.save_information()
    # visualization
    if args.visualize != "n":
        print("Calculating the MSM ...")
        my_simulation.get_transitions_matrix()
        end_matrix_time = time.time()
        hours, minutes, seconds = report_time(end_simulation_time, end_matrix_time)
        print(f" -> time for MSM: {hours}h {minutes}min {seconds}s.")
        print("Producing images ...")
        my_simulation.visualize_hist_2D()
        my_simulation.visualize_population_per_energy()
        my_simulation.visualize_eigenvec(6, which="LR")
        if args.compare != "n":
            e_eigval, e_eigvec = energy.get_eigenval_eigenvec(6, which="LR")
            my_simulation.visualize_its(num_eigv=6, which="LR", rates_eigenvalues=e_eigval)
        else:
            my_simulation.visualize_its(num_eigv=6, which="LR")
        my_simulation.visualize_eigenvalues()
        end_visualization_time = time.time()
        hours, minutes, seconds = report_time(end_matrix_time, end_visualization_time)
        print(f" -> time for images: {hours}h {minutes}min {seconds}s.")
    end_time = time.time()
    hours, minutes, seconds = report_time(start_time, end_time)
    print(f"-------- Total Simulation time: {hours}h {minutes}min {seconds}s. --------")


if __name__ == '__main__':
    my_args = parser.parse_args()
    energy_object = produce_energies(my_args)
    produce_simulation(my_args, energy_object)
