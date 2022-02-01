# internal imports
from plotting.read_files import read_summary_file, read_everything_energies
from constants import DIM_LANDSCAPE, PATH_DATA_MAZES
# external imports
from matplotlib import colors, cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

sns.set_style("ticks")
sns.set_context("talk")


def plot_maze(file_id: str):
    """
    Visualize the maze with black squares (walls) and white squares (halls).

    Args:
        file_id: the img_name of the saved data
    """
    energies = np.load(PATH_DATA_MAZES + "maze_" + file_id + ".npy")
    dict_properties = read_summary_file(file_id)
    with plt.style.context(['Stylesheets/maze_style.mplstyle', 'Stylesheets/not_animation.mplstyle']):
        ax = plt.imshow(energies, cmap="Greys")
        ax.figure.savefig(dict_properties["images path"] + f"{dict_properties['images name']}_maze.pdf")
        plt.close()


def plot_energy(properties: dict, energies: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray):
    """
    Plot the energy surface from data.

    Args:
        properties: dictionary of properties read from the summary file
        energies: array of saved energies
        grid_x: array of saved grid x
        grid_y: array of saved grid y
    """
    with plt.style.context(['Stylesheets/not_animation.mplstyle']):
        fig, ax = plt.subplots(1, 1)
        df = pd.DataFrame(data=energies, index=grid_x[:, 0], columns=grid_y[0, :])
        if "atom_positions" in properties.keys():
            sns.heatmap(df, cmap="RdBu_r", norm=colors.TwoSlopeNorm(vcenter=0, vmax=properties["energy cutoff"]),
                        fmt='.2f', square=True, ax=ax, yticklabels=[], xticklabels=[])
        else:
            sns.heatmap(df, cmap="RdBu", fmt='.2f', square=True, ax=ax, yticklabels=[], xticklabels=[])
        if "atom_positions" in properties.keys():
            for atom in properties["atom_positions"]:
                range_x_grid = properties["grid_end"][0] - properties["grid_start"][0]
                range_y_grid = properties["grid_end"][1] - properties["grid_start"][1]
                ax.scatter((atom[1] - properties["grid_start"][1]) * properties["size"][1] / range_y_grid,
                           (atom[0] - properties["grid_start"][0]) * properties["size"][0] / range_x_grid,
                           marker="o", c="white")
        ax.figure.savefig(properties["images path"] + f"{properties['images name']}_energy.pdf")
        plt.close()


def plot_energy_3d(properties: dict, energies: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray):
    """
    Visualizes the array of energies in 3D.

    Args:
        properties: dictionary of properties read from the summary file
        energies: array of saved energies
        grid_x: array of saved grid x
        grid_y: array of saved grid y
    """
    with plt.style.context('Stylesheets/not_animation.mplstyle'):
        ax = plt.axes(projection='3d')
        if "atom_positions" in properties.keys():
            ax.plot_surface(grid_x, grid_y, energies, rstride=1, cstride=1, cmap='RdBu_r',
                            norm=colors.SymLogNorm(linthresh=1e-13, vmax=np.max(energies), vmin=-np.max(energies)),
                            edgecolor='none')
        else:
            ax.plot_surface(grid_x, grid_y, energies, rstride=1, cstride=1,
                            cmap='RdBu', edgecolor='none')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.figure.savefig(properties["images path"] + f"{properties['images name']}_3D_energy.pdf")
        plt.close()


def plot_underlying_maze(properties: dict, maze: np.ndarray):
    """
    Visualization of the maze (with eventually added noise) from which the Energy object was created.

    Args:
        properties: dictionary of properties read from the summary file
        maze: array of the underlying maze read from the file
    """
    with plt.style.context(['Stylesheets/maze_style.mplstyle', 'Stylesheets/not_animation.mplstyle']):
        fig, ax = plt.subplots(1, 1)
        sns.heatmap(maze, cmap='RdBu_r', norm=colors.TwoSlopeNorm(vcenter=0))
        ax.figure.savefig(properties["images path"] + f"{properties['images name']}_underlying_maze.pdf")
        plt.close()


def plot_rates_matrix(properties: dict, rates):
    """
    Visualizes the array rates_matrix.

    Args:
        properties: dictionary of properties read from the summary file
        rates: array of the rates matrix read from the file
    """
    with plt.style.context(['Stylesheets/maze_style.mplstyle', 'Stylesheets/not_animation.mplstyle']):
        norm = colors.TwoSlopeNorm(vcenter=0)
        fig, ax = plt.subplots(1, 1)
        sns.heatmap(rates.toarray(), cmap="RdBu_r", norm=norm)
        ax.set_title("Rates matrix")
        fig.savefig(properties["images path"] + f"{properties['images name']}_rates_matrix.pdf")
        plt.close()


def plot_eigenvectors(properties: dict, eigenvectors: np.ndarray, num: int = 3):
    """
    Visualize the energy surface and the first num (default=3) eigenvectors as a 2D image in a maze.
    Black squares mean the cells are not accessible.

    Args:
        properties: dictionary of properties read from the summary file
        eigenvectors: array of the eigenvectors read from the file
        num: int, how many eigenvectors of rates matrix to show
    """
    eigenvectors = eigenvectors[:, :num]
    with plt.style.context(['Stylesheets/not_animation.mplstyle', 'Stylesheets/maze_style.mplstyle']):
        full_width = DIM_LANDSCAPE[0]
        fig, ax = plt.subplots(1, num, sharey="row", figsize=(full_width, full_width/num))
        cmap = cm.get_cmap("RdBu").copy()
        try:
            accesible = properties["accessible cells"]
        except AttributeError or KeyError:
            accesible = [(i, j) for i in range(properties["size"][0]) for j in range(properties["size"][1])]
        len_acc = len(accesible)
        assert eigenvectors.shape[0] == len_acc, "The length of the eigenvector should equal the num of accesible cells"
        vmax = np.max(eigenvectors[:, :num+1])
        vmin = np.min(eigenvectors[:, :num+1])
        if "factor" in properties.keys():
            size = int(properties["size"][0])*properties["factor"], int(properties["size"][1])*properties["factor"]
        else:
            size = properties["size"]
        for i in range(num):
            array = np.full(size, vmax+1)
            for index, cell in enumerate(accesible):
                if eigenvectors[index, 0] > 0:
                    array[cell] = eigenvectors[index, i]
                else:
                    array[cell] = - eigenvectors[index, i]
            ax[i].imshow(array, cmap=cmap, norm=colors.TwoSlopeNorm(vmax=vmax, vcenter=0, vmin=vmin))
            #ax[i].set_title(f"Eigenvector {i+1}", fontsize=7, fontweight="bold")
        plt.savefig(properties["images path"] + f"{properties['images name']}_eigenvectors_sqra.pdf")
        plt.close()


def plot_eigenvalues(properties: dict, eigenvalues: np.ndarray, num: int = None, calc_type: str = "sqra"):
    """
    Visualize the eigenvalues of rate matrix.

    Args:
        properties: dictionary of properties read from the summary file
        eigenvalues: array of the eigenvalues read from the file
        num: int, how many eigenvalues of rates matrix to show - if None, all available
        calc_type: 'sqra' or 'msm', for correct labeling
    """
    if num:
        eigenvalues = eigenvalues[:num]
    with plt.style.context(['Stylesheets/not_animation.mplstyle']):
        fig, ax = plt.subplots(1, 1, figsize=DIM_LANDSCAPE)
        xs = np.linspace(0, 1, num=len(eigenvalues))
        plt.scatter(xs, eigenvalues, s=5, c="black")
        for i, eigenw in enumerate(eigenvalues):
            plt.vlines(xs[i], eigenw, 0, linewidth=0.5)
        plt.hlines(0, 0, 1)
        ax.set_ylabel(f"Eigenvalues ({calc_type.upper()})")
        ax.axes.get_xaxis().set_visible(False)
        plt.savefig(properties["images path"] + f"{properties['images name']}_eigenvalues_{calc_type}.pdf")
        plt.close()


def plot_everything_energy(file_id: str, num_eigenvec: int = 6, num_eigenval: int = None, plot_rates: bool = False):
    """
    Save all the energy plots connected with some file_id.

    Args:
        file_id: the img_name of the saved data
        num_eigenvec: int, how many eigenvectors of rates matrix to show
        num_eigenval: int, how many eigenvalues of rates matrix to show - if None, all available
        plot_rates: whether to plot rates matrix (impacts performance)

    Returns:

    """
    properties = read_everything_energies(file_id)
    if file_id.startswith("maze"):
        dict_properties, energies, grid_x, grid_y, rates_matrix, eigenvec, eigenval, underlying_maze = properties
        plot_underlying_maze(dict_properties, underlying_maze)
    else:
        dict_properties, energies, grid_x, grid_y, rates_matrix, eigenvec, eigenval = properties
    if plot_rates:
        plot_rates_matrix(dict_properties, rates_matrix)
    plot_energy(dict_properties, energies, grid_x, grid_y)
    plot_energy_3d(dict_properties, energies, grid_x, grid_y)
    if int(num_eigenvec) > 0:
        plot_eigenvectors(dict_properties, eigenvec, num=num_eigenvec)
    if num_eigenval == None or int(num_eigenval) > 0:
        plot_eigenvalues(dict_properties, eigenval, num=num_eigenval)


if __name__ == '__main__':
    file = "maze017"
    plot_everything_energy(file)

