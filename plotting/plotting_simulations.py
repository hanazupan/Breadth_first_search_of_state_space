# internal imports
from constants import *
from plotting.plotting_energies import plot_eigenvalues
from plotting.read_files import read_everything_simulations
# external imports
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import seaborn as sns

sns.set_style("ticks")
sns.set_context("paper")


def plot_eigenvec(properties, file_id, num_eigv: int = 6):
    """
    Visualize first num_eiv of the transitions matrix for all tau-s in the self.tau_array.

    Args:
        properties: dictionary of properties read from the summary file
        file_id: identifier of the file
        num_eigv: number of eigenvectors to visualize
    """
    taus_to_plot = 5
    full_width = DIM_LANDSCAPE[0]
    fig, ax = plt.subplots(len(properties["tau_array"][:taus_to_plot]), num_eigv, sharey="row",
                           figsize=(full_width, full_width / num_eigv * len(properties["tau_array"][:taus_to_plot])))
    # TODO: plot better taus
    cmap = cm.get_cmap("RdBu").copy()
    cmap.set_over("black")
    cmap.set_under("black")
    for i, tau in enumerate(properties["tau_array"][:taus_to_plot]):
        # load eigenvectors
        eig = np.load(PATH_MSM_EIGEN + f"eigv_{i}_{file_id}.npz")
        eigenvec = eig["eigenvec"][:, :num_eigv]
        vmin = np.min(eigenvec)
        vmax = np.max(eigenvec)
        for j in range(num_eigv):
            array = np.full(properties["size"], np.max(eigenvec[:, j]) + 1)
            index = 0
            must_flip = np.sum(eigenvec[:, 0])
            for m in range(properties["size"][0]):
                for n in range(properties["size"][1]):
                    if must_flip > 0 or (j + 1) % 2 == 0:
                        array[m, n] = eigenvec[index, j]
                    else:
                        array[m, n] = - eigenvec[index, j]
                    index += 1
            ax[i][j].imshow(array.real, cmap=cmap, norm=colors.TwoSlopeNorm(vmax=vmax, vcenter=0, vmin=vmin))
            ax[0][j].set_title(f"Eigenvector {j + 1}", fontsize=7, fontweight="bold")
            ax[i][0].set_ylabel(f"tau = {tau}", fontsize=7, fontweight="bold")
            ax[i][j].axes.get_xaxis().set_visible(False)
            ax[i][j].set_yticks([])
            ax[i][j].set_yticklabels([])
    fig.savefig(properties["images path"] + f"{properties['images name']}_eigenvectors_msm.pdf",
                bbox_inches='tight', dpi=1200)
    plt.close()


def plot_its(properties, file_id, num_eigv: int = 6, rates_eigenvalues=None):
    """
    Plot iterative timescales.

    Args:
        properties: dictionary of properties read from the summary file
        file_id: identifier of the file
        num_eigv: how many eigenvalues/timescales to plot
        rates_eigenvalues: if not None, provide a list of SqRA eigenvalues so that it is possible to
                           plot ITS of SqRA as dashed lines for comparison
    """
    tau_array = np.array(properties["tau_array"])
    tau_eigenvals = np.zeros((num_eigv, properties["len_tau"]))
    for tau_i in range(properties["len_tau"]):
        eig = np.load(PATH_MSM_EIGEN + f"eigv_{tau_i}_{file_id}.npz")
        eigenval = eig["eigenval"][:num_eigv]
        tau_eigenvals[:, tau_i] = eigenval
    fig, ax = plt.subplots(1, 1, figsize=DIM_SQUARE)
    for j in range(1, num_eigv):
        to_plot_abs = np.array(-tau_array * properties["dt"] / np.log(np.abs(tau_eigenvals[j, :])))
        sns.lineplot(x=tau_array * properties["dt"], y=to_plot_abs,
                     palette=sns.color_palette("hls", len(tau_array)), ax=ax, legend=False)
    if np.any(rates_eigenvalues):
        for j in range(1, len(rates_eigenvalues[:num_eigv])):
            absolute_its = np.array([- 1 / rates_eigenvalues[j] for _ in tau_array])
            ax.plot(tau_array * properties["dt"], absolute_its, color="black", ls="--")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=tau_array[-1] * properties["dt"])
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"ITS")
    tau_array_with_zero = (tau_array * properties["dt"]).tolist()
    tau_array_with_zero.append(0)
    tau_array_with_zero.sort()
    ax.fill_between(tau_array_with_zero, tau_array_with_zero, color="grey", alpha=0.5)
    fig.savefig(properties["images path"] + f"{properties['images name']}_implied_timescales.pdf",
                bbox_inches='tight', dpi=1200)
    plt.close()


def plot_hist_2D(properties: dict, histogram: np.ndarray):
    """
    Plot the histogram of the simulation. Should correspond to the 2D Boltzmann distribution of the energy
    surface.

    Args:
        properties: dictionary of properties read from the summary file
        histogram: array of the 2D histogram read from the file
    """
    with plt.style.context(['Stylesheets/not_animation.mplstyle', 'Stylesheets/maze_style.mplstyle']):
        fig, ax = plt.subplots(1, 1)
        cmap = cm.get_cmap("RdBu").copy()
        sns.heatmap(histogram, cmap=cmap, fmt='.2f', square=True, ax=ax, yticklabels=[], xticklabels=[])
        ax.figure.savefig(properties["images path"] + f"{properties['images name']}_hist_2D.pdf")
        plt.close()


def plot_population_per_energy(properties: dict, energy: np.ndarray, histogram: np.ndarray):
    """
    Check whether the number of visits of the cell exponentially decrease with the energy of that cell.

    Args:
        properties: dictionary of properties read from the summary file
        energy: the energy surface array read from the file
        histogram: array of the 2D histogram read from the file
    """
    x_len, y_len = histogram.shape
    energies = []
    population = []
    for x in range(x_len):
        for y in range(y_len):
            cell = (x, y)
            energies.append(energy[cell])
            population.append(histogram[cell])
    with plt.style.context(['Stylesheets/not_animation.mplstyle']):
        fig, ax = plt.subplots(1, 1, figsize=DIM_SQUARE)
        energies = np.array(energies)
        E_population = np.histogram(energies, bins=25)
        print(E_population)
        E_pop = np.zeros(energies.shape)
        for i, e in enumerate(energies):
            for j, ep in enumerate(E_population[1][:]):
                if e <= E_population[1][j]:
                    E_pop[i] = E_population[0][j - 1]
                    break
        population = np.array(population)
        population[E_pop == 0] = 0
        E_pop[E_pop == 0] = 1
        weights = population/E_pop
        # weights must be the shape of energies
        # kde=True, kde_kws=dict(gridsize=3000, cut=0)
        sns.histplot(x=energies, bins=25, weights=weights, ax=ax, element="step",
                     color="black", fill=False)
        ax.set_xlabel("Cell energy")
        ax.set_ylabel("Relative cell population")
        plt.savefig(properties["images path"] + f"{properties['images name']}_population_per_energy.pdf")
        plt.close()


def plot_trajectory(properties: dict, traj_x: np.ndarray, traj_y: np.ndarray):
    """
    Plot the points visited by the trajectory.

    Args:
        properties: dictionary of properties read from the summary file
        traj_x: array of the trajectory x coordinates read from the file
        traj_y: array of the trajectory y coordinates read from the file
    """
    with plt.style.context(['Stylesheets/not_animation.mplstyle']):
        plt.subplots(1, 1, figsize=properties["size"])
        plt.scatter(traj_y, traj_x, marker="o", c="black", s=1)
        plt.gca().invert_yaxis()
        plt.savefig(properties["images path"] + f"{properties['images name']}_trajectory.pdf")
        plt.close()


def plot_everything_simulation(file_id: str, traj: bool = False):
    """
    Produce all relevant plots of a simulation.

    Args:
        file_id: the identifier of the file
        traj: bool, should the x-y trajectories also be plotted
    """
    if traj:
        dict_properties, energies, histogram, sqra_eigenval, traj_x, traj_y = read_everything_simulations(file_id,
                                                                                                          traj_x_y=traj)
        plot_trajectory(dict_properties, traj_x, traj_y)
    else:
        dict_properties, energies, histogram, sqra_eigenval = read_everything_simulations(file_id, traj_x_y=traj)
    last_eigenval = np.load(PATH_MSM_EIGEN + f"eigv_{dict_properties['len_tau'] - 1}_{file_id}.npz")["eigenval"]
    plot_eigenvalues(dict_properties, last_eigenval, calc_type="msm")
    plot_hist_2D(dict_properties, histogram)
    plot_population_per_energy(dict_properties, energies, histogram)
    plot_eigenvec(dict_properties, file_id)
    plot_its(dict_properties, file_id, rates_eigenvalues=sqra_eigenval)


if __name__ == '__main__':
    file = "maze017"
    plot_everything_simulation(file)
