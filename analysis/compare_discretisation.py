# internal imports
import random

from maze.create_energies import Energy, EnergyFromPotential, EnergyFromMaze
from maze.create_mazes import Maze
from simulation.create_simulation import Simulation
from simulation.create_msm import MSM
from plotting.plotting_energies import *
from plotting.plotting_simulations import *
from constants import *
import matplotlib.pyplot as plt
# standard library
import time
# external imports
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import interpolate
from tqdm import tqdm

sns.set_style("ticks")
sns.set_context("paper")


def grid_scan_potential(name: str = "potential", num_eigenv: int = 20, plot=False, dt=0.01):
    friction = 10
    mass = 1
    sides_scan = [6, 7, 8, 9, 10, 12, 15, 17, 20, 22, 25, 30, 35, 40, 45, 50, 55]
    sizes_scan = [(s, s) for s in sides_scan]
    data = pd.DataFrame()
    for size_i, size in enumerate(sizes_scan):
        start_time = time.time()
        full_name = f"{name}_{size_i}"
        my_energy = EnergyFromPotential(size=size, images_path=PATH_IMG_ANALYSIS, friction=friction,
                                        grid_start=(-1.5, -1.5), grid_end=(1.5, 1.5), images_name=full_name)
        my_energy.get_rates_matix()
        eigenval, eigenvec = my_energy.get_eigenval_eigenvec(num_eigenv, which="LR")
        end_time = time.time()
        if plot:
            plot_everything_energy(full_name)
        dict_values = {f"Eigenvalue {i + 1}": eigv.real for i, eigv in enumerate(eigenval)}
        dict_values.update({"Grid side": size[0],
                            "Time [s]": end_time - start_time})
        data = data.append(dict_values, ignore_index=True)
        # last item - also do a simulation
        if size == sizes_scan[-1]:
            my_simulation = Simulation(my_energy, images_path=my_energy.images_path, images_name=my_energy.images_name)
            my_simulation.integrate(N=int(1e7), dt=dt, save_trajectory=False)
            msm = MSM(my_energy.images_name, images_path=my_energy.images_path, change_tau=[100])
            msm.get_transitions_matrix(noncorr=True)
            msm.get_eigenval_eigenvec(num_eigv=num_eigenv, which="LR")
    name_file = PATH_DATA_ANALYSIS + f"grid_scan_{name}.csv"
    data.to_csv(path_or_buf=name_file)


def grid_scan_maze(name: str = "maze", num_eigenv: int = 20, plot=False, dt=0.01):
    orig_size = (12, 12)
    my_maze = Maze(size=orig_size)
    factor_scan = np.linspace(1, 2, num=20)
    #factor_scan = [1, 1.1, 1.3, 1.5, 1.8, 2, 2.6, 3]
    my_energy = EnergyFromMaze(my_maze, images_name=name, images_path=PATH_IMG_ANALYSIS,
                               grid_start=(0, 0), grid_end=(10, 10), factor_grid=1)
    data = pd.DataFrame()
    for factor_i, factor in enumerate(tqdm(factor_scan)):
        start_time = time.time()
        full_name = f"{name}_{factor_i}"
        my_energy.images_name = full_name
        my_energy.grid_x, my_energy.grid_y = my_energy._prepare_grid(factor=1.1)
        my_energy.energies = interpolate.bisplev(my_energy.grid_x[:, 0], my_energy.grid_y[0, :], my_energy.spline)
        my_energy.size = my_energy.energies.shape
        my_energy._prepare_geometry()
        my_energy.explorer = None
        if plot and factor_i in [0, 5, 10, 15, 19]:
            np.save(PATH_ENERGY_SURFACES + f"surface_{my_energy.images_name}", my_energy.energies)
            np.save(PATH_ENERGY_SURFACES + f"underlying_maze_{my_energy.images_name}", my_energy.underlying_maze)
            np.savez(PATH_ENERGY_GRIDS + f"grid_x_y_{my_energy.images_name}", x=my_energy.grid_x, y=my_energy.grid_y)
        my_energy.get_rates_matix()
        eigenval, eigenvec = my_energy.get_eigenval_eigenvec(num_eigenv, which="LR")
        end_time = time.time()
        if plot and factor_i in [0, 5, 10, 15, 19]:
            plot_everything_energy(full_name, num_eigenvec=0, num_eigenval=5)
        dict_values = {f"Eigenvalue {i + 1}": eigv.real for i, eigv in enumerate(eigenval)}
        dict_values.update({"Grid side": my_energy.size[0],
                            "Time [s]": end_time - start_time})
        data = data.append(dict_values, ignore_index=True)
        # last item - also do a simulation
        if factor_i == 9:
            my_simulation = Simulation(my_energy, images_path=my_energy.images_path, images_name=my_energy.images_name)
            my_simulation.integrate(N=int(5e7), dt=dt, save_trajectory=False)
            msm = MSM(my_energy.images_name, images_path=my_energy.images_path, change_tau=[10, 50, 100, 200, 500])
            msm.get_transitions_matrix(noncorr=True)
            msm.get_eigenval_eigenvec(num_eigv=num_eigenv, which="LR")
            plot_everything_simulation(my_simulation.images_name)
    name_file = PATH_DATA_ANALYSIS + f"grid_scan_{name}.csv"
    data.to_csv(path_or_buf=name_file)


def plot_grid_scan(name: str = "potential", num_eigenv: int = 20, dt=0.01):
    file_path = PATH_DATA_ANALYSIS + f"grid_scan_{name}.csv"
    data = pd.read_csv(file_path)
    sim_eigenval_vec = np.load(PATH_MSM_EIGEN + f"eigv_{2}_{name}_9.npz")
    sim_eigenval = sim_eigenval_vec["eigenval"]
    fig, ax = plt.subplots(1, 2, figsize=DIM_LANDSCAPE)
    sns.lineplot(x="Grid side", y="Time [s]", data=data, ax=ax[0], color="black")
    for i in range(1, num_eigenv):
        data[f"ITS {i + 1}"] = -1 / data[f"Eigenvalue {i + 1}"]
        sns.lineplot(x="Grid side", y=f"ITS {i + 1}", data=data, ax=ax[1])
        repeated_sim_eigv = np.array([-(100 * dt) / np.log(np.abs(sim_eigenval[i])) for _ in range(data["Grid side"].size)])
        ax[1].plot(data["Grid side"], repeated_sim_eigv, color="black", ls="dotted")
    ax[1].set_ylabel("ITS")
    ax[1].set_yscale('log')
    plt.tight_layout()
    plt.savefig(PATH_IMG_ANALYSIS + f"scan_grid_{name}.pdf")


if __name__ == '__main__':
    num_eig = 10
    random.seed(12)
    np.random.seed(12)
    dt_step = 0.01
    # TODO: repeat for mazes/atoms
    grid_scan_maze(num_eigenv=num_eig, plot=True, dt=dt_step)
    #grid_scan_potential(num_eigenv=num_eig, dt=dt_step)
    plot_grid_scan(name="maze", num_eigenv=num_eig, dt=dt_step)
