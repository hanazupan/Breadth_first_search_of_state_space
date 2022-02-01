"""
This file is not documented. Use with caution.
"""

# internal imports
from maze.create_energies import EnergyFromPotential, EnergyFromMaze, EnergyFromAtoms, Atom, Energy
from maze.create_mazes import Maze
from simulation.create_simulation import Simulation
from simulation.create_msm import MSM
from plotting.plotting_energies import *
from plotting.plotting_simulations import *
from constants import *
import matplotlib.pyplot as plt
# standard library
import random
import time
# external imports
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import interpolate
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
    num_eigv = 20
    start_sqra_time = time.time()
    # clears any residual information in case the object already exists
    my_energy.explorer = None
    my_energy.rates_matrix = None
    my_energy.energy_cutoff = cutoff
    my_energy.get_rates_matix(explorer=my_explorer)
    eigenval, eigenvec = my_energy.get_eigenval_eigenvec(num_eigv, which="LR")
    my_energy.save_information()
    # need to save what is usually saved in __init__
    np.save(PATH_ENERGY_SURFACES + f"surface_{my_energy.images_name}", my_energy.energies)
    np.savez(PATH_ENERGY_GRIDS + f"grid_x_y_{my_energy.images_name}", x=my_energy.grid_x, y=my_energy.grid_y)
    if my_energy.images_name.startswith("maze"):
        np.save(PATH_ENERGY_SURFACES + f"underlying_maze_{my_energy.images_name}.npy", my_energy.underlying_maze)
    end_sqra_time = time.time()
    return end_sqra_time - start_sqra_time, eigenval


def grid_scan_potential(name: str = "potential", num_eigenv: int = 20, plot=False, dt=0.01):
    friction = 50
    sides_scan = [6, 8, 9, 10, 12, 15, 17, 20, 22, 25, 30, 35, 40, 45, 50, 55]
    sizes_scan = [(s, s) for s in sides_scan]
    data = pd.DataFrame()
    for size_i, size in enumerate(sizes_scan):
        start_time = time.time()
        full_name = f"{name}_{size_i}"
        my_energy = EnergyFromPotential(size=size, images_path=PATH_IMG_ANALYSIS, friction=friction,
                                        grid_start=(-2, -2), grid_end=(2, 2), images_name=full_name)
        # option 3 - cutoff 5 and dfs explorer
        time_dfs, eigenval_dfs = run_energy(my_energy, "dfs", 15)
        # option 2 - no cutoff, no explorer
        full_time, eigenval_ss = run_energy(my_energy, "none", 15)
        # option 1 - cutoff 5 and bfs explorer
        time_bfs, eigenval_bfs = run_energy(my_energy, "bfs", 15)
        if plot and size[0] in [6, 8, 15, 20, 30, 40, 55]:
            plot_everything_energy(full_name)
        dict_values = {f"Eigenvalue {i + 1}": eigv.real for i, eigv in enumerate(eigenval_bfs)}
        dict_values.update({"Grid side": size[0],
                            "Time SqRA [s]": full_time,
                            "Time BFS-SqRA [s]": time_bfs,
                            "Time DFS-SqRA [s]": time_dfs})
        data = data.append(dict_values, ignore_index=True)
        # last item - also do a simulation
        if size == (55, 55):
            my_simulation = Simulation(my_energy, images_path=my_energy.images_path, images_name=my_energy.images_name)
            my_simulation.integrate(N=int(5e7), dt=dt, save_trajectory=False)
            msm = MSM(my_energy.images_name, images_path=my_energy.images_path, change_tau=[5, 10, 20, 50, 100, 150, 200])
            msm.get_transitions_matrix(noncorr=True)
            msm.get_eigenval_eigenvec(num_eigv=num_eigenv, which="LR")
            plot_everything_simulation(my_simulation.images_name)
    name_file = PATH_DATA_ANALYSIS + f"grid_scan_{name}.csv"
    data.to_csv(path_or_buf=name_file)


def grid_scan_maze(name: str = "maze", num_eigenv: int = 20, plot=False, dt=0.01):
    orig_size = (12, 12)
    my_maze = Maze(size=orig_size, no_branching=True, edge_is_wall=True)
    factor_scan = np.linspace(1, 2, num=30)
    my_energy = EnergyFromMaze(my_maze, images_name=name, images_path=PATH_IMG_ANALYSIS,
                               grid_start=(0, 0), grid_end=(10, 10), factor_grid=1, cutoff=80, T=600)
    data = pd.DataFrame()
    for factor_i, factor in enumerate(tqdm(factor_scan)):
        full_name = f"{name}_{factor_i}"

        my_energy.images_name = full_name
        my_energy.grid_x, my_energy.grid_y = my_energy._prepare_grid(factor=1.085)
        my_energy.energies = interpolate.bisplev(my_energy.grid_x[:, 0], my_energy.grid_y[0, :], my_energy.spline)
        my_energy.size = my_energy.energies.shape
        my_energy._prepare_geometry()
        my_energy.save_information()

        # option 3 - cutoff 5 and dfs explorer
        time_dfs, eigenval_dfs = run_energy(my_energy, "dfs", 15)
        # option 2 - no cutoff, no explorer
        full_time, eigenval_ss = run_energy(my_energy, "none", 15)
        # option 1 - cutoff 5 and bfs explorer
        time_bfs, eigenval_bfs = run_energy(my_energy, "bfs", 15)
        if plot and factor_i in [0, 5, 9, 15, 19]:
            np.save(PATH_ENERGY_SURFACES + f"surface_{my_energy.images_name}", my_energy.energies)
            np.save(PATH_ENERGY_SURFACES + f"underlying_maze_{my_energy.images_name}", my_energy.underlying_maze)
            np.savez(PATH_ENERGY_GRIDS + f"grid_x_y_{my_energy.images_name}", x=my_energy.grid_x, y=my_energy.grid_y)
        if plot and factor_i in [0, 5, 9, 15, 19]:
            plot_everything_energy(full_name, num_eigenvec=0, num_eigenval=5)
        dict_values = {f"Eigenvalue {i + 1}": eigv.real for i, eigv in enumerate(eigenval_ss)}
        dict_values.update({"Grid side": my_energy.size[0],
                            "Time SqRA [s]": full_time,
                            "Time BFS-SqRA [s]": time_bfs,
                            "Time DFS-SqRA [s]": time_dfs})
        data = data.append(dict_values, ignore_index=True)
        # last item - also do a simulation
        if factor == factor_scan[15]:
            my_simulation = Simulation(my_energy, images_path=my_energy.images_path, images_name=my_energy.images_name)
            my_simulation.integrate(N=int(3e7), dt=dt, save_trajectory=False)
            my_simulation.save_information()
            msm = MSM(my_energy.images_name, images_path=my_energy.images_path, change_tau=[10, 20, 50, 100, 150, 200, 500, 1000, 2000, 3000, 5000])
            msm.get_transitions_matrix(noncorr=True)
            msm.get_eigenval_eigenvec(num_eigv=num_eigenv, which="LR")
            plot_everything_simulation(my_simulation.images_name)
        name_file = PATH_DATA_ANALYSIS + f"grid_scan_{name}.csv"
        data.to_csv(path_or_buf=name_file)


def grid_scan_atoms(name: str = "atoms", num_eigenv: int = 20, plot=False, dt=0.01):
    friction = 5
    sides_scan = [10, 12, 15, 17, 20, 22, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 100]
    sizes_scan = [(s, s) for s in sides_scan]

    data = pd.DataFrame()
    for size_i, size in enumerate(tqdm(sizes_scan)):
        full_name = f"{name}_{size_i}"
        epsilon = 3
        sigma = 5
        atom_1 = Atom((30.3, 20.5), epsilon, sigma + 1)
        atom_2 = Atom((14.3, 9.3), epsilon, sigma - 2)
        atom_3 = Atom((9.3, 27.3), epsilon / 5, sigma + 2)
        atom_4 = Atom((22.3, 10.3), epsilon / 10, sigma)
        atom_5 = Atom((22.3, 30.3), epsilon, sigma - 2.5)
        my_energy = EnergyFromAtoms(size, (atom_1, atom_2, atom_3, atom_4, atom_5), grid_start=(5, 5),
                                    grid_end=(35, 35), T=500,
                                    images_name=full_name, friction=friction, images_path=PATH_IMG_ANALYSIS)

        # option 3 - cutoff 5 and dfs explorer
        time_dfs, eigenval_dfs = run_energy(my_energy, "dfs", 10)
        # option 2 - no cutoff, no explorer
        full_time, eigenval_ss = run_energy(my_energy, "none", 10)
        # option 1 - cutoff 5 and bfs explorer
        time_bfs, eigenval_bfs = run_energy(my_energy, "bfs", 10)

        if plot and size_i in [0, 3, 5, 9, 12, 15, 19]:
            np.save(PATH_ENERGY_SURFACES + f"surface_{my_energy.images_name}", my_energy.energies)
            np.savez(PATH_ENERGY_GRIDS + f"grid_x_y_{my_energy.images_name}", x=my_energy.grid_x, y=my_energy.grid_y)

        if plot and size_i in [0, 3, 5, 9, 12, 15, 19]:
            plot_everything_energy(full_name, num_eigenvec=0, num_eigenval=5)
        dict_values = {f"Eigenvalue {i + 1}": eigv.real for i, eigv in enumerate(eigenval_bfs)}
        dict_values.update({"Grid side": my_energy.size[0],
                            "Time SqRA [s]": full_time,
                            "Time BFS-SqRA [s]": time_bfs,
                            "Time DFS-SqRA [s]": time_dfs})
        data = data.append(dict_values, ignore_index=True)
        # last item - also do a simulation
        if size == sizes_scan[11]:
            my_simulation = Simulation(my_energy, images_path=my_energy.images_path, images_name=my_energy.images_name)
            my_simulation.integrate(N=int(6e7), dt=dt, save_trajectory=False)
            my_simulation.save_information()
            msm = MSM(my_energy.images_name, images_path=my_energy.images_path, change_tau=[10, 20, 50, 100, 150, 200, 300, 500, 1000, 2000])
            msm.get_transitions_matrix(noncorr=True)
            msm.get_eigenval_eigenvec(num_eigv=num_eigenv, which="LR")
            plot_everything_simulation(my_simulation.images_name)
        name_file = PATH_DATA_ANALYSIS + f"grid_scan_{name}.csv"
        data.to_csv(path_or_buf=name_file)


def plot_grid_scan(name: str = "potential", num_eigenv: int = 20, dt=0.01):
    file_path = PATH_DATA_ANALYSIS + f"grid_scan_{name}.csv"
    data = pd.read_csv(file_path)
    # WARNING! If using simulation data, make sure you know what you are using!
    #sim_eigenval_vec = np.load(PATH_MSM_EIGEN + f"eigv_{5}_{name}_11.npz")
    #sim_eigenval_vec = np.load(PATH_MSM_EIGEN + f"eigv_{5}_{name}_15.npz")
    #sim_eigenval = sim_eigenval_vec["eigenval"]
    fig, ax = plt.subplots(1, 2, figsize=DIM_LANDSCAPE)
    sns.lineplot(x="Grid side", y="Time SqRA [s]", data=data, ax=ax[0], color="black", label="full SqRA")
    sns.lineplot(x="Grid side", y="Time BFS-SqRA [s]", data=data, ax=ax[0], color="black", linestyle='--',
                 label="BFS-SqRA")
    #sns.lineplot(x="Grid side", y="Time DFS-SqRA [s]", data=data, ax=ax[0], color="black", linestyle='.',
    # label="DFS-SqRA")
    for i in range(1, 8):
        data[f"ITS {i + 1}"] = -1 / data[f"Eigenvalue {i + 1}"]
        sns.lineplot(x="Grid side", y=f"ITS {i + 1}", data=data, ax=ax[1])
    #    repeated_sim_eigv = np.array([-(200 * dt) / np.log(np.abs(sim_eigenval[i])) for _ in range(data["Grid side"].size)])
    #    ax[1].plot(data["Grid side"], repeated_sim_eigv, color="black", ls="dotted")
    ax[1].set_ylabel("ITS")
    ax[0].set_yscale('log')
    ax[0].set_xlabel("Cells per side")
    ax[1].set_xlabel("Cells per side")
    ax[1].set_yscale('log')
    ax[0].set_ylabel('Time [s]')
    ax[0].legend(prop={'size': 14})
    plt.tight_layout()
    plt.savefig(PATH_IMG_ANALYSIS + f"scan_grid_{name}.pdf")


if __name__ == '__main__':
    num_eig = 20
    random.seed(1)
    np.random.seed(1)
    dt_step = 0.01
    #grid_scan_maze(name="maze10", num_eigenv=num_eig, plot=True, dt=dt_step)
    #grid_scan_atoms("atoms03", num_eigenv=num_eig, plot=True, dt=dt_step)
    #grid_scan_potential(name="potential", num_eigenv=num_eig, dt=dt_step, plot=True)
    plot_grid_scan(name="atoms03", num_eigenv=num_eig, dt=dt_step)
    plot_grid_scan(name="potential", num_eigenv=num_eig, dt=dt_step)
    plot_grid_scan(name="maze10", num_eigenv=num_eig, dt=dt_step)
