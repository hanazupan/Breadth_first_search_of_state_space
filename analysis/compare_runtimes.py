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


def produce_important_images(my_energy=None, my_simulation=None):
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
    plt.savefig(img_compare_path+"time_comparison")


def run_double_well(my_energy, my_explorer, cutoff):
    num_eigv = 4
    start_sqra_time = time.time()
    my_energy.explorer = None
    my_energy.rates_matrix = None
    my_energy.energy_cutoff = cutoff
    my_energy.get_rates_matix(explorer=my_explorer)
    eigenval, eigenvec = my_energy.get_eigenval_eigenvec(num_eigv, which="LR")
    end_sqra_time = time.time()
    return end_sqra_time - start_sqra_time, eigenval


def compare_time_double_well(e_type="potential"):
    friction = 10
    data = pd.DataFrame()
    # you could also do a scan for diff sizes
    if e_type.startswith("potential"):
        cutoffs = np.linspace(3, 100, num=30)
        additional = np.array([150, 200, 250, 300, 350, 400])
        cutoffs = np.concatenate((cutoffs, additional))
        num_cells = 50
        size = (num_cells, num_cells)
        my_energy = EnergyFromPotential(size=size, images_path=img_compare_path, friction=friction,
                                        grid_start=-2.5, grid_end=2.5)
    elif e_type.startswith("maze"):
        cutoffs = np.linspace(5, 13, num=21)
        size = (40, 40)
        my_maze = Maze(size=size, images_path=img_compare_path, edge_is_wall=True, no_branching=False)
        my_energy = EnergyFromMaze(my_maze, friction=friction, images_path=img_compare_path, factor_grid=1,
                                   grid_start=-1, grid_end=1)
    else:
        cutoffs = np.linspace(-2, 10, num=51)
        additional = np.array([20, 23, 25, 28, 30, 35, 40, 45, 50])
        cutoffs = np.concatenate((cutoffs, additional))
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
    for j, co in enumerate(tqdm(cutoffs)):
        my_energy.images_name = f"cutoff_{int(co)}_{e_type}"
        for i in range(1):
            # option 3 - no cutoff, no explorer
            full_time, eigenval_ss = run_double_well(my_energy, "none", co)
            # option 1 - cutoff 5 and bfs explorer
            time_bfs, eigenval_bfs = run_double_well(my_energy, "bfs", co)
            explored_bfs = len(my_energy.explorer.get_sorted_accessible_cells()) / (my_energy.size[0] * my_energy.size[1]) * 100
            # option 2 - cutoff 5 and dfs explorer
            time_dfs, eigenval_dfs = run_double_well(my_energy, "dfs", co)
            explored_dfs = len(my_energy.explorer.get_sorted_accessible_cells()) / (my_energy.size[0] * my_energy.size[1]) * 100
            dict_values = {f"Eigenvalue {i + 1}": eigv for i, eigv in enumerate(eigenval_bfs)}
            dict_values.update({f"Eigenvalue DFS {i + 1}": eigv for i, eigv in enumerate(eigenval_dfs)})
            dict_values.update({f"Eigenvalue Full SS {i + 1}": eigv for i, eigv in enumerate(eigenval_ss)})
            dict_values.update({"Num. of cells": size[0]*size[1],
                                "Cutoff": co,
                           "% explored": explored_bfs,
                           "BFS time [s]": time_bfs,
                           "DFS time [s]": time_dfs,
                           "Full state space time [s]": full_time})
            data = data.append(dict_values, ignore_index=True)
        # plot the first and the last one
        if j % 5 == 0:
            my_energy.visualize_eigenvectors_in_maze(4, which="LR")
    name_file = DATA_PATH+f"compare_time_double_well_{e_type}.csv"
    data.to_csv(path_or_buf=name_file)


def plot_compare_time_double_well(file_path, name):
    data = pd.read_csv(file_path)
    data = data.loc[data["% explored"] > 10]
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(x="% explored", y="BFS time [s]", data=data, label="BFS",
                 ax=ax, ci="sd")
    sns.lineplot(x="% explored", y="DFS time [s]", data=data, label="DFS",
                 ax=ax, ci="sd")
    sns.lineplot(x="% explored", y="Full state space time [s]", data=data, label="Full SS",
                 ax=ax, ci="sd")
    plt.legend(loc="lower right", framealpha=0.8)
    ax.set(yscale="log")
    ax.set_ylabel("Time [s]")
    ax.set_xlabel("% explored")
    #ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(img_compare_path + f"plot_compare_time_double_well_{name}")


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


def plot_scan_cutoff(file_path, e_type):
    data = pd.read_csv(file_path)
    data = data.loc[data["% explored"] > 20]
    fig, ax1 = plt.subplots(1, 1)
    all_eigenvalues = [f"Eigenvalue {i+1}" for i in range(4)]
    for i, eigenvalue in enumerate(all_eigenvalues):
        sns.lineplot(x="% explored", y=eigenvalue, data=data, label=f"Eigv {i+1}", ax=ax1, ci="sd", legend=False)
    ax1.set_ylabel("Eigenvalues")
    #plt.legend(loc="lower right", framealpha=0.8)
    #ax2 = ax1.twinx()
    #sns.lineplot(x="Cutoff", y="% explored", data=data, label="% explored", color="black", ax=ax2, legend=False)
    plt.tight_layout()
    plt.savefig(img_compare_path+f"scan_cutoff_{e_type}")


if __name__ == '__main__':
    names = ["maze19"]
    for name in names:
        #scan_cutoffs(e_type=name)
        #compare_time_double_well(e_type=name)
        plot_compare_time_double_well(DATA_PATH + f"compare_time_double_well_{name}.csv", name)
        plot_scan_cutoff(DATA_PATH+f"compare_time_double_well_{name}.csv", name)
