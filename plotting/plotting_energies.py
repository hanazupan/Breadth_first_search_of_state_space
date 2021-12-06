from plotting.read_files import read_summary_file
from constants import DIM_SQUARE
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, cm
from scipy.sparse import load_npz
import seaborn as sns
import pandas as pd


path_data_mazes = "data/mazes/"
path_data_summary = "data/mazes_summaries/"
path_energy_summary = "data/energy_summaries/"
path_energy_surfaces = "data/energy_surfaces/"
path_energy_rates = "data/sqra_rates_matrices/"
path_energy_eigen = "data/sqra_eigenvectors_eigenvalues/"


def plot_maze(file_id):
    """
    Visualize the maze with black squares (walls) and white squares (halls).
    """
    energies = np.load(path_data_mazes + "maze_" + file_id + ".npy")
    dict_properties = read_summary_file(file_id)
    with plt.style.context(['Stylesheets/maze_style.mplstyle', 'Stylesheets/not_animation.mplstyle']):
        ax = plt.imshow(energies, cmap="Greys")
        ax.figure.savefig(dict_properties["images path"] + f"{dict_properties['images name']}_maze.pdf")
        plt.close()


def plot_with_cutoff(file_id):
    energies = np.load(path_energy_surfaces + "surface_" + file_id + ".npy")
    grids = np.load(f"data/energy_grids/grid_x_y_{file_id}.npz")
    grid_x = grids["x"]
    grid_y = grids["y"]
    dict_properties = read_summary_file(file_id, summary_type="energy")
    with plt.style.context(['Stylesheets/not_animation.mplstyle']):
        fig, ax = plt.subplots(1, 1)
        df = pd.DataFrame(data=energies, index=grid_x[:, 0], columns=grid_y[0, :])
        if "atom_positions" in dict_properties.keys():
            sns.heatmap(df, cmap="RdBu_r", norm=colors.TwoSlopeNorm(vcenter=0, vmax=dict_properties["energy cutoff"]),
                    fmt='.2f', square=True, ax=ax, yticklabels=[], xticklabels=[])
        else:
            sns.heatmap(df, cmap="RdBu",
                    fmt='.2f', square=True, ax=ax, yticklabels=[], xticklabels=[])
        # if you want labels: yticklabels=[f"{ind:.2f}" for ind in df.index]
        # xticklabels=[f"{col:.2f}" for col in df.columns]
        if "atom_positions" in dict_properties.keys():
            for atom in dict_properties["atom_positions"]:
                range_x_grid = dict_properties["grid_end"][0] - dict_properties["grid_start"][0]
                range_y_grid = dict_properties["grid_end"][1] - dict_properties["grid_start"][1]
                ax.scatter((atom[1] - dict_properties["grid_start"][1]) * dict_properties["size"][1] / range_y_grid,
                           (atom[0] - dict_properties["grid_start"][0]) * dict_properties["size"][0] / range_x_grid,
                           marker="o", c="white")
        ax.figure.savefig(dict_properties["images path"] + f"{dict_properties['images name']}_energy.pdf")
        plt.close()


def plot_energy_3d(file_id):
    """
    Visualizes the array self.energies in 3D.

    Raises:
        ValueError: if there are no self.energies
    """
    energies = np.load(path_energy_surfaces + "surface_" + file_id + ".npy")
    grids = np.load(f"data/energy_grids/grid_x_y_{file_id}.npz")
    grid_x = grids["x"]
    grid_y = grids["y"]
    dict_properties = read_summary_file(file_id, summary_type="energy")
    with plt.style.context('Stylesheets/not_animation.mplstyle'):
        ax = plt.axes(projection='3d')
        if "atom_positions" in dict_properties.keys():
            ax.plot_surface(grid_x, grid_y, energies, rstride=1, cstride=1, cmap='RdBu_r',
                            norm=colors.SymLogNorm(linthresh=1e-13, vmax=np.max(energies), vmin=-np.max(energies)),
                            edgecolor='none')
        else:
            ax.plot_surface(grid_x, grid_y, energies, rstride=1, cstride=1,
                            cmap='RdBu', edgecolor='none')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.figure.savefig(dict_properties["images path"] + f"{dict_properties['images name']}_3D_energy.pdf")
        plt.close()


def plot_underlying_maze(file_id):
    """
    Visualization of the maze (with eventually added noise) from which the Energy object was created.

    Raises:
        Value error: if there is no self.underlying_maze (if self.from_maze has not been used).
    """
    underlying_maze = np.load("data/energy_surfaces/" + f"underlaying_maze_{file_id}.npy")
    dict_properties = read_summary_file(file_id, summary_type="energy")
    with plt.style.context(['Stylesheets/maze_style.mplstyle', 'Stylesheets/not_animation.mplstyle']):
        fig, ax = plt.subplots(1, 1)
        sns.heatmap(underlying_maze, cmap='RdBu_r', norm=colors.TwoSlopeNorm(vcenter=0))
        ax.figure.savefig(dict_properties["images path"] + f"{dict_properties['images name']}_underlying_maze.pdf")
        plt.close()


def plot_rates_matrix(file_id):
    """
    Visualizes the array self.rates_matrix.
    """
    rates_matrix = load_npz(path_energy_rates + f"rates_{file_id}.npz")
    dict_properties = read_summary_file(file_id, summary_type="energy")
    with plt.style.context(['Stylesheets/maze_style.mplstyle', 'Stylesheets/not_animation.mplstyle']):
        norm = colors.TwoSlopeNorm(vcenter=0)
        fig, ax = plt.subplots(1, 1)
        sns.heatmap(rates_matrix.toarray(), cmap="RdBu_r", norm=norm)
        ax.set_title("Rates matrix")
        fig.savefig(dict_properties["images path"] + f"{dict_properties['images name']}_rates_matrix.pdf")
        plt.close()


if __name__ == '__main__':
    file = "maze011"
    #plot_maze(file)
    plot_with_cutoff(file)
    plot_energy_3d(file)
    plot_underlying_maze(file)
    plot_rates_matrix(file)
