from constants import *
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import seaborn as sns

sns.set_style("ticks")
sns.set_context("talk")

def visualize_eigenvalues(self):
    """
    Visualize the eigenvalues of rate matrix.
    """
    if not np.any(self.transition_matrices):
        self.get_transitions_matrix()
    num = self.transition_matrices.shape[1] - 2
    eigenval, eigenvec = self.get_eigenval_eigenvec(num_eigv=num, which="LR")
    with plt.style.context(['Stylesheets/not_animation.mplstyle']):
        fig, ax = plt.subplots(1, 1, figsize=DIM_LANDSCAPE)
        xs = np.linspace(0, 1, num=num)
        plt.scatter(xs, eigenval[-1], s=5, c="black")
        for i, eigenw in enumerate(eigenval[-1]):
            plt.vlines(xs[i], eigenw, 0, linewidth=0.5)
        plt.hlines(0, 0, 1)
        ax.set_ylabel("Eigenvalues (MSM)")
        ax.axes.get_xaxis().set_visible(False)
        plt.savefig(self.images_path + f"{self.images_name}_eigenvalues_msm.pdf")
        plt.close()

def visualize_eigenvec(self, num_eigv: int = 6, **kwargs):
    """
    Visualize first num_eiv of the transitions matrix for all tau-s in the self.tau_array
    Args:
        num_eigv: number of eigenvectors to visualize
        **kwargs: named arguments to forward to eigs()
    """
    taus_to_plot = 5
    tau_eigenvals, tau_eigenvec = self.get_eigenval_eigenvec(num_eigv=num_eigv, **kwargs)
    full_width = DIM_LANDSCAPE[0]
    fig, ax = plt.subplots(len(self.tau_array[:taus_to_plot]), num_eigv, sharey="row",
                           figsize=(full_width, full_widt h /num_eig v *len(self.tau_array[:taus_to_plot])))
    # TODO: plot better taus
    cmap = cm.get_cmap("RdBu").copy()
    cmap.set_over("black")
    cmap.set_under("black")
    # show max first taus_to_plot (5) taus
    vmin = np.min(tau_eigenvec[:taus_to_plot])
    vmax = np.max(tau_eigenvec[:taus_to_plot])
    for i, tau in enumerate(self.tau_array[:taus_to_plot]):
        for j in range(num_eigv):
            array = np.full(self.histogram.shape, np.max(tau_eigenvec[i, :, j]) + 1)
            index = 0
            must_flip = np.sum(tau_eigenvec[i, :, 0])
            for m in range(self.histogram.shape[0]):
                for n in range(self.histogram.shape[1]):
                    # if self.energy.is_accessible((m, n)):
                    if must_flip > 0 or ( j +1) % 2 == 0:
                        array[m, n] = tau_eigenvec[i, index, j]
                    else:
                        array[m, n] = - tau_eigenvec[i, index, j]
                    index += 1
            ax[i][j].imshow(array, cmap=cmap, norm=colors.TwoSlopeNorm(vmax=vmax, vcenter=0, vmin=vmin))
            ax[0][j].set_title(f"Eigenvector {j + 1}", fontsize=7, fontweight="bold")
            ax[i][0].set_ylabel(f"tau = {tau}", fontsize=7, fontweight="bold")
            ax[i][j].axes.get_xaxis().set_visible(False)
            ax[i][j].set_yticks([])
            ax[i][j].set_yticklabels([])
    fig.savefig(self.images_path + f"{self.images_name}_eigenvectors_msm.pdf", bbox_inches='tight', dpi=1200)
    plt.close()

def visualize_its(self, num_eigv: int = 6, rates_eigenvalues=None, **kwargs):
    """
    Plot iterative timescales.

    Args:
        num_eigv: how many eigenvalues/timescales to plot

        rates_eigenvalues: if not None, provide a list of SqRA eigenvalues so that it is possible to
                           plot ITS of SqRA as dashed lines for comparison
    """
    if len(self.eigenvals[0]) >= num_eigv:
        tau_eigenvals, tau_eigenvec = self.eigenvals, self.eigenvec
    else:
        tau_eigenvals, tau_eigenvec = self.get_eigenval_eigenvec(num_eigv=num_eigv, **kwargs)
    tau_eigenvals = tau_eigenvals.T
    fig, ax = plt.subplots(1, 1, figsize=DIM_SQUARE)
    colors_circle = ["blue", "red", "green", "orange", "black", "yellow", "purple", "pink"]
    for j in range(1, num_eigv):
        to_plot_abs = np.array(-self.tau_array * self.dt / np.log(np.abs(tau_eigenvals[j, :])))
        sns.lineplot(x=self.tau_array * self.dt, y=to_plot_abs,
                     palette=sns.color_palette("hls", len(self.tau_array)), ax=ax, legend=False)
        # ax.plot(self.tau_array * self.dt, to_plot_abs, color=colors_circle[j])
    if np.any(rates_eigenvalues):
        for j in range(1, len(rates_eigenvalues)):
            absolute_its = np.array([- 1 /rates_eigenvalues[j] for _ in self.tau_array])
            ax.plot(self.tau_array * self.dt, absolute_its, color="black", ls="--")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=self.tau_array[-1] * self.dt)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"ITS")
    tau_array_with_zero = (self.tau_array * self.dt).tolist()
    tau_array_with_zero.append(0)
    tau_array_with_zero.sort()
    ax.fill_between(tau_array_with_zero, tau_array_with_zero, color="grey", alpha=0.5)
    fig.savefig(self.images_path + f"{self.images_name}_implied_timescales.pdf", bbox_inches='tight', dpi=1200)
    plt.close()

def visualize_hist_2D(self):
    """
    Plot the histogram of the simulation. Should correspond to the 2D Boltzmann distribution of the energy
    surface.
    """
    with plt.style.context(['Stylesheets/not_animation.mplstyle', 'Stylesheets/maze_style.mplstyle']):
        fig, ax = plt.subplots(1, 1)
        cmap = cm.get_cmap("RdBu").copy()
        # im = plt.imshow(self.histogram, cmap=cmap)
        sns.heatmap(self.histogram, cmap=cmap, fmt='.2f',
                    square=True, ax=ax, yticklabels=[], xticklabels=[])
        # fig.colorbar(im, ax=ax)
        ax.figure.savefig(self.images_path + f"{self.images_name}_hist_2D.pdf")
        plt.close()

def visualize_population_per_energy(self):
    """
    Check whether the number of visits of the cell exponentially decrease with the energy of that cell.
    """
    x_len, y_len = self.energy.size
    energies = []
    population = []
    for x in range(x_len):
        for y in range(y_len):
            cell = (x, y)
            energies.append(self.energy.get_energy(cell))
            population.append(self.histogram[cell])
    with plt.style.context(['Stylesheets/not_animation.mplstyle']):
        fig, ax = plt.subplots(1, 1, figsize=DIM_SQUARE)
        energies = np.array(energies)
        E_population = np.histogram(energies, bins=25)
        E_pop = np.zeros(energies.shape)
        for i, e in enumerate(energies):
            for j, ep in enumerate(E_population[1][1:]):
                if E_population[1][ j -1] < e <= E_population[1][j]:
                    E_pop[i] = E_population[0][ j -1]
        population = np.array(population)
        population[E_pop == 0] = 0
        E_pop[E_pop == 0] = 1
        sns.histplot(x=energies, bins=25, weights=populatio n /E_pop, ax=ax, element="step",
                     color="black", fill=False, kde=False)
        # plt.hist(energies, bins=25, weights=population/E_pop, histtype='step')
        ax.set_xlabel("Cell energy")
        ax.set_ylabel("Relative cell population")
        plt.savefig(self.images_path + f"{self.images_name}_population_per_energy.pdf")
        plt.close()

def visualize_trajectory(self):
    """
    Plot the points visited by the trajectory.
    """
    with plt.style.context(['Stylesheets/not_animation.mplstyle']):
        plt.subplots(1, 1, figsize=self.energy.size)
        plt.scatter(self.traj_y, self.traj_x, marker="o", c="black", s=1)
        plt.gca().invert_yaxis()
        plt.savefig(self.images_path + f"{self.images_name}_trajectory.pdf")
        plt.close()