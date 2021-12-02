from maze.create_energies import Energy, EnergyFromPotential, EnergyFromMaze, Atom, EnergyFromAtoms  # need all
from maze.create_mazes import Maze  # need this import
from simulation.create_simulation import Simulation
import matplotlib.pyplot as plt
from constants import DATA_PATH, IMG_PATH
import pandas as pd
import seaborn as sns
import numpy as np
import time
from tqdm import tqdm

sns.set_style("ticks")
sns.set_context("talk")

img_compare_path = IMG_PATH + "compare_runtimes/"


def produce_important_images(my_energy: Energy = None, my_simulation: Simulation = None):
    """Just produce all the typical visualizations for your Energy and/or Simulation object."""
    if my_energy:
        my_energy.visualize()
        my_energy.visualize_3d()
        my_energy.visualize_eigenvectors_in_maze(num=6, which="LR")
        my_energy.visualize_eigenvalues()
    if my_simulation:
        my_simulation.visualize_hist_2D()
        my_simulation.visualize_population_per_energy()
        my_simulation.visualize_eigenvec(6, which="LR")
        if my_energy:
            e_eigval, e_eigvec = my_energy.get_eigenval_eigenvec(6, which="LR")
            my_simulation.visualize_its(num_eigv=6, which="LR", rates_eigenvalues=e_eigval)

def compare_grids_double_well():
    """
    Compare the time needed to get the rates matrix + first 4 eigenvectors/values vs the time needed to
    get the transitions matrix + first 4 eigenvectors/values at tau=0.1 (N=1e7, dt=0.001).

    Returns:

    """
    mass = 1
    friction = 10
    step = 0.001
    duration = int(1e6)
    tau_array = np.array([100])
    side_sizes = [5, 7]
    #side_sizes = [5, 7, 10, 12, 15, 20, 25, 30]
    grids = [(a, a) for a in side_sizes]
    collected_data = pd.DataFrame(columns=["Num. of cells", r"$Time_{SqRA}$ [s]", r"$Time_{MSM}$ [s]"])
    for i, grid in enumerate(grids):
        name = f"compare_grids_double_well_{i}"
        start_sqra_time = time.time()
        my_energy = EnergyFromPotential(size=grid, images_path=img_compare_path, images_name=name, friction=friction, m=mass)
        my_energy.get_rates_matix()
        my_energy.get_eigenval_eigenvec(4)
        end_sqra_time = time.time()
        start_msm_time = time.time()
        my_simulation = Simulation(my_energy, dt=step, N=duration)
        my_simulation.integrate()
        my_simulation.get_transitions_matrix(tau_array=tau_array)
        my_simulation.get_eigenval_eigenvec(4)
        end_msm_time = time.time()
        # add to dataframe
        collected_data.append({"Num. of cells": grid[0]*grid[1], r"$Time_{SqRA}$ [s]": end_sqra_time-start_sqra_time,
                               r"$Time_{MSM}$ [s]": end_msm_time - start_msm_time}, ignore_index=True)
        # pictures not timed, just to check
        produce_important_images(my_energy, my_simulation)
        my_simulation.save_information()
    name_file = DATA_PATH+"compare_grids_double_well.csv"
    collected_data.to_csv(path_or_buf=name_file)


def plot_time_comp(file_path):
    data = pd.read_csv(file_path)
    sns.lineplot(x="Num. of cells", y={r"$Time_{SqRA}$ [s]", r"$Time_{MSM}$ [s]"}, data=data)
    plt.savefig(img_compare_path+"time_comparison.pdf")


def scan_cutoffs(e_type="potential"):
    """
    Compare how long it takes to peform SqRA with full matrix or just the part within cutoff (BFS,
    could also try comparing with DFS). Scan for a few different cutoffs to see how properties like eigenvalues
    converge towards full SqRA.

    Returns:

    """
    num_eigv = 6
    friction = 10
    if e_type.startswith("potential"):
        e_cutoffs = np.linspace(0.5, 20.5, num=101)
        size = (30, 30)
        my_energy = EnergyFromPotential(size=size, images_path=img_compare_path, friction=friction)
    elif e_type.startswith("maze"):
        e_cutoffs = np.linspace(5, 18, num=51)
        size = (10, 10)
        my_maze = Maze(size=size, images_path=img_compare_path, edge_is_wall=True, no_branching=True)
        my_energy = EnergyFromMaze(my_maze, friction=friction, images_path=img_compare_path)
    else:
        e_cutoffs = np.linspace(-1, 10, num=21)
        additional = np.array([20, 23, 25, 28, 30, 35, 40, 45, 50])
        e_cutoffs = np.concatenate((e_cutoffs, additional))
        size = (30, 30)
        atoms = []
        num_atoms = 4
        for i in range(num_atoms):
            x_coo = -10 + 20*np.random.rand()
            y_coo = -10 + 20*np.random.rand()
            epsilon = np.random.choice([1, 3, 5, 10])
            sigma = np.random.choice([2, 4, 6])
            atom = Atom((x_coo, y_coo), epsilon, sigma)
            atoms.append(atom)
        atoms = tuple(atoms)
        my_energy = EnergyFromAtoms(size=size, atoms=atoms, grid_edges=(-12, 12, -12, 12), images_path=img_compare_path)
    all_eigenvalues = [f"Eigenvalue {i}" for i in range(num_eigv)]
    data = pd.DataFrame(columns=["Num. of cells", "Cutoff", "% explored cells", "Time [s]"]+all_eigenvalues)
    for i, cutoff in enumerate(tqdm(e_cutoffs)):
        name = f"scan_cutoff_{e_type}_{i}"
        start_sqra_time = time.time()
        my_energy.images_name = name
        my_energy.explorer = None
        my_energy.rates_matrix = None
        my_energy.energy_cutoff = cutoff
        my_energy.get_rates_matix()
        eigenval, eigenvec = my_energy.get_eigenval_eigenvec(num_eigv, which="LR")
        end_sqra_time = time.time()
        dict_values = {f"Eigenvalue {i+1}": eigv for i, eigv in enumerate(eigenval)}
        percentage_explored = len(my_energy.explorer.get_sorted_accessible_cells())/(my_energy.size[0]*my_energy.size[1])*100
        dict_values.update({"Num. of cells": my_energy.size[0]*my_energy.size[1], "Cutoff": my_energy.energy_cutoff,
                            "% explored cells": percentage_explored, "Time [s]": end_sqra_time - start_sqra_time})
        data = data.append(dict_values, ignore_index=True)
        #visualize every 10-th energy surface/its eigenvectors
        if i % 10 == 0:
            my_energy.visualize_eigenvectors_in_maze(num_eigv, which="LR")
            my_energy.save_information()
    name_file = DATA_PATH+f"scan_cutoff_{e_type}.csv"
    data.to_csv(path_or_buf=name_file)