from maze.create_energies import Energy, EnergyFromPotential, EnergyFromMaze, Atom, EnergyFromAtoms  # need all
from maze.create_mazes import Maze  # need this import
from simulation.create_simulation import Simulation
import matplotlib.pyplot as plt
from constants import DATA_PATH, IMG_PATH
import pandas as pd
import seaborn as sns
import numpy as np
import time
import gc

sns.set_style("ticks")


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
        my_energy = EnergyFromPotential(size=grid, images_path=IMG_PATH, images_name=name, friction=friction, m=mass)
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
    plt.savefig(IMG_PATH+"time_comparison")


def compare_len_simulation_double_well():
    pass


def compare_corr_noncorr_window():
    pass


def scan_cutoffs(e_type="potential"):
    """
    Compare how long it takes to peform SqRA with full matrix or just the part within cutoff (BFS,
    could also try comparing with DFS). Scan for a few different cutoffs to see how properties like eigenvalues
    converge towards full SqRA.

    Returns:

    """
    if type == "potential":
        e_cutoffs = np.linspace(0.5, 5.5, num=100)
    else:
        e_cutoffs = np.linspace(0, 15, num=50)
        size = (10, 10)
        my_maze = Maze(size=size, images_path=IMG_PATH)
    friction = 10
    all_eigenvalues = [f"Eigenvalue {i}" for i in range(6)]
    data = pd.DataFrame(columns=["Num. of cells", "Cutoff", "% explored cells", "Time [s]"]+all_eigenvalues)
    for i, cutoff in enumerate(e_cutoffs):
        name = f"scan_cutoff_{e_type}_{i}"
        start_sqra_time = time.time()
        if type == "potential":
            size = (30, 30)
            my_energy = EnergyFromPotential(size=size, images_path=IMG_PATH, images_name=name, friction=friction)
        else:
            my_energy = EnergyFromMaze(my_maze, friction=friction, images_path=IMG_PATH, images_name=name)
        my_energy.energy_cutoff = cutoff
        my_energy.get_rates_matix()
        eigenval, eigenvec = my_energy.get_eigenval_eigenvec(6, which="LR")
        end_sqra_time = time.time()
        dict_values = {f"Eigenvalue {i}": eigv for i, eigv in enumerate(eigenval)}
        dict_values.update({"Num. of cells": size[0]*size[1], "Cutoff": my_energy.energy_cutoff,
                            "% explored cells": len(set(my_energy.explorer.get_sorted_accessible_cells()))/(size[0]*size[1])*100,
                            "Time [s]": end_sqra_time - start_sqra_time})
        data = data.append(dict_values, ignore_index=True)
        if i%10 == 0:
            my_energy.visualize_eigenvectors_in_maze(6, which="LR")
        del my_energy
        gc.collect()
    name_file = DATA_PATH+f"scan_cutoff_{e_type}.csv"
    data.to_csv(path_or_buf=name_file)


def plot_scan_cutoff(file_path, e_type):
    data = pd.read_csv(file_path)
    fig, ax1 = plt.subplots(1, 1)
    all_eigenvalues = [f"Eigenvalue {i}" for i in range(6)]
    for i, eigenvalue in enumerate(all_eigenvalues):
        sns.lineplot(x="Cutoff", y=eigenvalue, data=data, label=f"Eigenvalue {i}", ax=ax1)
    ax1.set_ylabel("Eigenvalues")
    ax2 = ax1.twinx()
    sns.lineplot(x="Cutoff", y="% explored cells", data=data, label="% explored", color="black", ax=ax2)
    plt.savefig(IMG_PATH+f"scan_cutoff_{e_type}")


if __name__ == '__main__':
    #compare_grids_double_well()
    #plot_time_comp(DATA_PATH+"compare_grids_double_well.csv")
    scan_cutoffs(e_type="maze")
    plot_scan_cutoff(DATA_PATH+"scan_cutoff_maze.csv", "maze")
