# internal imports
from maze.create_energies import Energy, EnergyFromPotential
from simulation.create_simulation import Simulation
from simulation.create_msm import MSM
from plotting.plotting_energies import *
from plotting.read_files import read_everything_energies
from constants import *
import matplotlib.pyplot as plt
# standard library
import time
# external imports
import pandas as pd
import seaborn as sns
import numpy as np
from tqdm import tqdm

sns.set_style("ticks")
sns.set_context("paper")


def grid_scan(name: str = "potential", num_eigenv: int = 20):
    friction = 10
    mass = 1
    sides_scan = [6, 7, 8, 9, 10, 12, 15, 17, 20, 22, 25, 30, 35, 40, 45, 50, 55]
    sizes_scan = [(s, s) for s in sides_scan]
    data = pd.DataFrame()
    for size in sizes_scan:
        start_time = time.time()
        my_energy = EnergyFromPotential(size=size, images_path=PATH_IMG_ANALYSIS, friction=friction,
                                        grid_start=(-1.5, -1.5), grid_end=(1.5, 1.5), images_name=name)
        # save info in the dataframe
        #properties, energies, grid_x, grid_y, rates_matrix, eigenvec, eigenval, *extra = read_everything_energies(name)
        my_energy.get_rates_matix()
        eigenval, eigenvec = my_energy.get_eigenval_eigenvec(num_eigenv, which="LR")
        end_time = time.time()
        dict_values = {f"Eigenvalue {i + 1}": eigv.real for i, eigv in enumerate(eigenval)}
        dict_values.update({"Grid side": size[0],
                            "Time [s]": end_time - start_time})
        data = data.append(dict_values, ignore_index=True)
        # last item - also do a simulation
        if size == sizes_scan[-1]:
            my_simulation = Simulation(my_energy, images_path=my_energy.images_path, images_name=my_energy.images_name)
            my_simulation.integrate(N=int(1e7), dt=0.005, save_trajectory=False)
            msm = MSM(my_energy.images_name, images_path=my_energy.images_path, change_tau=[100])
            msm.get_transitions_matrix(noncorr=True)
            msm.get_eigenval_eigenvec(num_eigv=num_eigenv, which="LR")
    name_file = PATH_DATA_ANALYSIS + f"grid_scan_{name}.csv"
    data.to_csv(path_or_buf=name_file)


def plot_grid_scan(name: str = "potential", num_eigenv: int = 20):
    file_path = PATH_DATA_ANALYSIS + f"grid_scan_{name}.csv"
    data = pd.read_csv(file_path)
    sim_eigenval_vec = np.load(PATH_MSM_EIGEN + f"eigv_{0}_{name}.npz")
    sim_eigenval = sim_eigenval_vec["eigenval"]
    fig, ax = plt.subplots(1, 2, figsize=DIM_LANDSCAPE)
    sns.lineplot(x="Grid side", y="Time [s]", data=data, ax=ax[0], color="black")
    for i in range(1, num_eigenv):
        data[f"ITS {i + 1}"] = -1 / data[f"Eigenvalue {i + 1}"]
        sns.lineplot(x="Grid side", y=f"ITS {i + 1}", data=data, ax=ax[1])
        repeated_sim_eigv = np.array([-(100 * 0.005) / np.log(np.abs(sim_eigenval[i])) for _ in range(data["Grid side"].size)])
        ax[1].plot(data["Grid side"], repeated_sim_eigv, color="black", ls="dotted")
    ax[1].set_ylabel("ITS")
    plt.tight_layout()
    plt.savefig(PATH_IMG_ANALYSIS + f"scan_grid_{name}.pdf")


if __name__ == '__main__':
    num_eig = 10
    # TODO: save images of surface
    # TODO: repeat for mazes/atoms
    #grid_scan(num_eigenv=num_eig)
    plot_grid_scan(num_eigenv=num_eig)