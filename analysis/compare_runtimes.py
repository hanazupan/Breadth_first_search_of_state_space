from maze.create_energies import Energy, EnergyFromPotential, EnergyFromMaze, Atom, EnergyFromAtoms  # need all
from maze.create_mazes import Maze  # need this import
from plotting.plotting_energies import *
from plotting.read_files import read_everything_energies
from constants import *
import matplotlib.pyplot as plt
from constants import DATA_PATH, IMG_PATH
import pandas as pd
import seaborn as sns
import numpy as np
import time
from tqdm import tqdm

sns.set_style("ticks")
sns.set_context("talk")


def run_energy(my_energy: Energy, my_explorer: str, cutoff: float) -> tuple:
    """
    Conduct the creation of rates matrix with a specific explorer (or full state space) and a specific cutoff.

    Args:
        my_energy: Energy object we are working with
        my_explorer: string that is either "bfs", "dfs" or anything else - will be interpreted as full state space
        cutoff: float, how large should the cutoff be

    Returns:
        (time, eigenvalues)
    """
    num_eigv = 10
    start_sqra_time = time.time()
    my_energy.explorer = None
    my_energy.rates_matrix = None
    my_energy.energy_cutoff = cutoff
    my_energy.get_rates_matix(explorer=my_explorer)
    eigenval, eigenvec = my_energy.get_eigenval_eigenvec(num_eigv, which="LR")
    end_sqra_time = time.time()
    return end_sqra_time - start_sqra_time, eigenval


def time_comparison_explorers(e_type: str = "potential"):
    """
    Compare how long it takes to construct rate matix and find its first 6 eigenvectors/values when using full
    state space vs using BFS or DFS to limit the state space that should be explored.

    Saves data to: DATA_PATH + f"time_comparison_explorers_{e_type}.csv"

    Args:
        e_type: which type of energy surface to use ("potential", "maze", "atoms")
    """
    friction = 10
    data = pd.DataFrame()
    # prepare an energy surface of the specified type
    if e_type.startswith("potential"):
        cutoffs = np.linspace(3, 100, num=30)
        additional = np.array([150, 200, 250, 300, 350, 400])
        cutoffs = np.concatenate((cutoffs, additional))
        my_energy = EnergyFromPotential(size=(50, 50), images_path=PATH_IMG_ANALYSIS, friction=friction,
                                        grid_start=(-2.5, -2.5), grid_end=(2.5, 2.5))
    elif e_type.startswith("maze"):
        cutoffs = np.linspace(5, 9, num=11)
        additional = np.linspace(10.5, 15, num=11)
        cutoffs = np.concatenate((cutoffs, additional))
        my_maze = Maze(size=(25, 25), images_path=PATH_IMG_ANALYSIS, edge_is_wall=True, no_branching=True)
        plot_maze(my_maze.images_name)
        my_energy = EnergyFromMaze(my_maze, friction=friction, images_path=PATH_IMG_ANALYSIS, factor_grid=1,
                                   grid_start=(-1, -1), grid_end=(1, 1))
        properties, e, x, y, rm, eigenvec, eigenval, underlying_maze = read_everything_energies(my_energy.images_name)
        plot_energy(properties, e, x, y)
    else:
        cutoffs = np.linspace(-2, 19, num=21)
        #additional = np.array([20, 23, 25, 28, 30, 35, 40, 45, 50])
        #cutoffs = np.concatenate((cutoffs, additional))
        atoms = []
        num_atoms = 8  #TODO: increase
        for i in range(num_atoms):
            x_coo = 0 + 10*np.random.rand()
            y_coo = 0 + 10*np.random.rand()
            epsilon = np.random.choice([1, 3, 5])
            sigma = np.random.choice([1, 2, 3])
            atom = Atom((x_coo, y_coo), epsilon, sigma)
            atoms.append(atom)
        atoms = tuple(atoms)
        my_energy = EnergyFromAtoms(size=(25, 25), atoms=atoms, grid_start=(0, 0), grid_end=(10, 10),
                                    images_path=PATH_IMG_ANALYSIS)
        properties, e, x, y, rm, eigenvec, eigenval, underlying_maze = read_everything_energies(my_energy.images_name)
        plot_energy(properties, e, x, y)
    # loop over cutoff
    for j, co in enumerate(tqdm(cutoffs)):
        my_energy.images_name = f"cutoff_{int(co)}_{e_type}"
        for i in range(3):
            num_cells = my_energy.size[0] * my_energy.size[1]
            # option 3 - no cutoff, no explorer
            full_time, eigenval_ss = run_energy(my_energy, "none", co)
            # option 1 - cutoff 5 and bfs explorer
            time_bfs, eigenval_bfs = run_energy(my_energy, "bfs", co)
            explored_bfs = len(my_energy.explorer.get_sorted_accessible_cells()) / num_cells * 100
            # option 2 - cutoff 5 and dfs explorer
            time_dfs, eigenval_dfs = run_energy(my_energy, "dfs", co)
            dict_values = {f"Eigenvalue {i + 1}": eigv for i, eigv in enumerate(eigenval_bfs)}
            dict_values.update({f"Eigenvalue DFS {i + 1}": eigv for i, eigv in enumerate(eigenval_dfs)})
            dict_values.update({f"Eigenvalue Full SS {i + 1}": eigv for i, eigv in enumerate(eigenval_ss)})
            dict_values.update({"Num. of cells": num_cells,
                                "Cutoff": co,
                           "% explored": explored_bfs,
                           "BFS time [s]": time_bfs,
                           "DFS time [s]": time_dfs,
                           "Full state space time [s]": full_time})
            data = data.append(dict_values, ignore_index=True)
        # plot the first and the last one
        if j % 3 == 0:
            my_energy.visualize_eigenvectors_in_maze(4, which="LR")
        name_file = DATA_PATH+f"time_comparison_explorers_{e_type}.csv"
        data.to_csv(path_or_buf=name_file)
    run_energy(my_energy, "none", 1)
    my_energy.images_name += "full_ss"
    my_energy.visualize_eigenvectors_in_maze(4, which="LR")


def plot_time_comparison_explorers(file_path, name):
    data = pd.read_csv(file_path)
    data = data.loc[data["Cutoff"] > 7]
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(x="Cutoff", y="BFS time [s]", data=data, label="BFS",
                 ax=ax, ci="sd")
    sns.lineplot(x="Cutoff", y="DFS time [s]", data=data, label="DFS",
                 ax=ax, ci="sd")
    #sns.lineplot(x="% explored", y="Full state space time [s]", data=data, label="Full SS",
    #             ax=ax, ci="sd")
    plt.hlines(data["Full state space time [s]"].mean(), data["Cutoff"].min(), data["Cutoff"].max(),
               label="Full SS", color="green")
    plus_sd = data["Full state space time [s]"].mean() + data["Full state space time [s]"].std()
    minus_sd = data["Full state space time [s]"].mean() - data["Full state space time [s]"].std()
    plt.fill_between(x=[data["Cutoff"].min(), data["Cutoff"].max()],
                     y1=[minus_sd, minus_sd], y2=[plus_sd, plus_sd], color="green", alpha=0.3, linewidth=0)
    plt.legend(loc="lower right", framealpha=0.8)
    #ax.set(yscale="log")
    ax.set_ylabel("Time [s]")
    ax.set_xlabel("Cutoff")
    #ax.invert_xaxis()
    #ax2 = ax.twinx()
    #sns.lineplot(x="Cutoff", y="% explored", data=data, label="% explored", color="black", ax=ax2, legend=False)
    #plt.tight_layout()
    plt.tight_layout()
    plt.savefig(PATH_IMG_ANALYSIS + f"plot_time_comparison_explorers_{name}.pdf")


def plot_scan_cutoff(file_path, e_type):
    data = pd.read_csv(file_path)
    data = data.loc[data["Cutoff"] > 7]
    fig, ax1 = plt.subplots(1, 1)
    all_eigenvalues = [f"Eigenvalue {i+1}" for i in range(6)]
    all_ss_eigenvalues = [f"Eigenvalue Full SS {i+1}" for i in range(6)]
    for i, eigenvalue in enumerate(all_eigenvalues):
        sns.lineplot(x="Cutoff", y=eigenvalue, data=data, label=f"Eigv {i+1}", ax=ax1, ci="sd", legend=False)
    for i, ss_eigenvalue in enumerate(all_ss_eigenvalues):
        plt.hlines(data[ss_eigenvalue].mean(), data["Cutoff"].min(), data["Cutoff"].max(), linestyles="dotted",
                   color="black")
    ax1.set_ylabel("Eigenvalues")
    #plt.legend(loc="lower right", framealpha=0.8)
    #ax2 = ax1.twinx()
    #sns.lineplot(x="Cutoff", y="% explored", data=data, label="% explored", color="black", ax=ax2, legend=False)
    plt.tight_layout()
    plt.savefig(PATH_IMG_ANALYSIS + f"scan_cutoff_{e_type}.pdf")


if __name__ == '__main__':
    names = ["maze25"]
    for name in names:
        #time_comparison_explorers(e_type=name)
        plot_time_comparison_explorers(DATA_PATH + f"time_comparison_explorers_{name}.csv", name)
        plot_scan_cutoff(DATA_PATH+f"time_comparison_explorers_{name}.csv", name)
