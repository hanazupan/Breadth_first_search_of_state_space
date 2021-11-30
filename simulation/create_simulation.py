"""
In this file, molecular dynamics simulation on an Energy surface is performed, a MSM constructed and evaluated.
Finally, characteristic ITS are plotted. All these calculations are performed in Simulation class.
"""

from maze.create_energies import Energy, EnergyFromPotential, EnergyFromMaze, Atom, EnergyFromAtoms  # need all
from maze.create_mazes import Maze  # need this import
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm, colors
from scipy.sparse.linalg import eigs
import seaborn as sns
import time
from datetime import datetime
from constants import DIM_LANDSCAPE, DIM_SQUARE, DIM_PORTRAIT, DATA_PATH, IMG_PATH

sns.set_style("ticks")
sns.set_context("talk")


class Simulation:

    def __init__(self, energy: Energy, dt: float = 0.01, N: int = int(1e7),
                 images_path: str = IMG_PATH, images_name: str = "simulation"):
        """
        Class Simulation intended to run simulations on the Energy object using Euler–Maruyama integration
        as well as performing Markov State Modelling (MSM) and calculate eigenvectors and eigenvalues of so
        determined transitions matrix.

        Args:
            energy: Energy object containing the E surface and a way to calculate its derivatives
            dt: time step for integration
            N: total number of time steps
            images_path: where should images be saved
            images_name: how should images be identified
        """
        self.energy = energy
        self.images_name = images_name
        self.images_path = images_path
        # TD/particle properties inherited from Energy
        self.m = self.energy.m
        self.friction = self.energy.friction
        self.temperature = self.energy.temperature
        self.dt = dt
        self.N = N
        self.D = self.energy.D
        # TODO: tau array should probably be calculated (what are appropriate values?) and not pre-determined
        if type(energy) == EnergyFromPotential:
            self.tau_array = np.array([5, 7, 10, 20, 30, 50, 70, 100, 150, 250, 500, 700, 1000])
        elif type(energy) == EnergyFromMaze:
            #
            self.tau_array = np.array([5, 7, 10, 20, 30, 10, 50, 100, 500, 700, 1000, 1500, 2000, 2500, 3000])
        else:
            self.tau_array = np.array([10, 20, 50, 70, 100, 250, 500, 700, 1000, 1500, 2000, 2500, 3000])
        # prepare empty objects
        self.histogram = np.zeros(self.energy.size)
        self.outside_hist = 0
        self.traj_x = []
        self.traj_y = []
        self.traj_cell = []
        self.eigenvals = []
        self.eigenvec = []
        self.transition_matrices = None
        # everything to do with the grid
        self.step_x = self.energy.grid_x[1, 0] - self.energy.grid_x[0, 0]
        self.step_y = self.energy.grid_y[0, 1] - self.energy.grid_y[0, 0]
        # xmin, xmax, ymin, ymax - NOT middle of cells, but actual min/max
        xmin = self.energy.grid_x[0, 0] - self.step_x / 2
        xmax = self.energy.grid_x[-1, -1] + self.step_x / 2
        ymin = self.energy.grid_y[0, 0] - self.step_y / 2
        ymax = self.energy.grid_y[-1, -1] + self.step_y / 2
        self.grid_edges = (xmin, xmax, ymin, ymax)

    def integrate(self, dt: float = None, N: int = None, save_trajectory: bool = False):
        """
        Implement the Euler–Maruyama method for integration of the trajectory on self.energy surface.

        Args:
            dt: float, time step
            N: int, number of time steps
            save_trajectory: bool, should the trajectory be saved
                             (not needed for MSM analysis, only for plotting trajectories)
        """
        # the time step and number of steps can be redefined for every simulation
        if dt:
            self.dt = dt
            if type(self.energy) == EnergyFromMaze and self.dt > 0.01:
                print("Warning! Chosen dt possibly too large for mazes!")
        if N:
            self.N = N
        self.histogram = np.zeros(self.energy.size)
        # start at a random position somewhere on the grid
        x_n = self.step_x*self.energy.size[0]*np.random.random() + self.grid_edges[0]
        y_n = self.step_y*self.energy.size[1]*np.random.random() + self.grid_edges[2]
        # figure out in which cell the trajectory starts and increase the count of the histogram
        cell = self._point_to_cell((x_n, y_n))
        self.histogram[cell] += 1
        self.traj_cell.append(cell)
        for _ in tqdm(range(self.N)):
            # integrate the trajectory one step and increase the histogram count
            x_n, y_n = self._euler_maruyama(x_n, y_n)
            if self.energy.pbc:
                x_n, y_n = self._point_within_bound((x_n, y_n))
                cell = self._point_to_cell((x_n, y_n))
            elif not (self.grid_edges[0] <= x_n <= self.grid_edges[1]) or not\
                    (self.grid_edges[2] <= y_n <= self.grid_edges[3]):
                cell = (-1, -1)
            else:
                cell = self._point_to_cell((x_n, y_n))
            # if cell fits into the histogram
            if np.all([0 <= cell[k] < self.histogram.shape[k] for k in range(len(self.histogram.shape))]):
                self.histogram[cell] += 1
                self.traj_cell.append(cell)
                # if applicable, save trajectory
                if save_trajectory:
                    self.traj_x.append(x_n)
                    self.traj_y.append(y_n)
            else:
                # if not using periodic boundaries, points can land outside the histogram
                self.outside_hist += 1
                assert not self.energy.pbc
        # normalizing the histogram
        self.histogram = self.histogram / np.sum(self.histogram)
        self.traj_cell = np.array(self.traj_cell)

    def _euler_maruyama(self, x_n: float, y_n: float) -> tuple:
        """
        Complete a step of trajectory integration using an Euler-Maruyama integrator. If PBC, the (x_n, y_n) point
        can be assumed to be in the primary grid.

        Args:
            x_n: float, x-coordinate of the current trajectory point
            y_n: float, y-coordinate of the current trajectory point

        Returns: tuple (x_n, y_n), the integrated new trajectory points (not necessarily between -1 and 1)

        """
        if self.energy.pbc:
            assert self.grid_edges[0] <= x_n <= self.grid_edges[1], "Pbc used but x not in the original grid."
            assert self.grid_edges[2] <= y_n <= self.grid_edges[3], "Pbc used but y not in the original grid."
        # update x
        dV_dx = self.energy.get_x_derivative((x_n, y_n))
        eta_x = np.random.normal(loc=0.0, scale=np.sqrt(self.dt))
        x_deterministic = dV_dx * self.dt / self.m / self.friction
        x_random = np.sqrt(2 * self.D) * eta_x
        x_n = x_n - x_deterministic + x_random
        # update y
        dV_dy = self.energy.get_y_derivative((x_n, y_n))
        eta_y = np.random.normal(loc=0.0, scale=np.sqrt(self.dt))
        y_deterministic = dV_dy * self.dt / self.m / self.friction
        y_random = np.sqrt(2 * self.D) * eta_y
        y_n = y_n - y_deterministic + y_random
        return x_n, y_n

    def _point_within_bound(self, point: tuple) -> tuple:
        """
        Apply periodic boundary conditions so that the argument point is transformed into an equivalent point
        within the original grid.

        Args:
            point: tuple (x_n, y_n) a 2D point potentially outside self.grid_edges

        Returns:
            tuple (x_n, y_n), a 2D point corrected to an equivalent position within self.grid_edges
        """
        x_n, y_n = point
        range_x_grid = self.grid_edges[1] - self.grid_edges[0]
        range_y_grid = self.grid_edges[3] - self.grid_edges[2]
        # move to zero and stretch/shrink to the number of cells width
        x_cell = (x_n - self.grid_edges[0]) * self.energy.size[0] / range_x_grid
        y_cell = (y_n - self.grid_edges[2]) * self.energy.size[1] / range_y_grid
        # go to the "original" simulation box
        x_cell = x_cell % self.energy.size[0]
        y_cell = y_cell % self.energy.size[1]
        # transform back to grid
        x_n = x_cell * range_x_grid / self.energy.size[0] + self.grid_edges[0]
        y_n = y_cell * range_y_grid / self.energy.size[1] + self.grid_edges[2]
        return x_n, y_n

    def _point_to_cell(self, point: tuple) -> tuple:
        """
        Given a trajectory point (x_n, y_n), determine to which cell of the histogram this trajectory point belongs to.

        Args:
            point: tuple (x_n, y_n) a 2D point always inside self.grid_edges (if pbc)

        Returns:
            tuple (row, column) in which cell of the histogram this point lands
        """
        x_n, y_n = point
        range_x_grid = self.grid_edges[1] - self.grid_edges[0]
        range_y_grid = self.grid_edges[3] - self.grid_edges[2]
        # move to zero and strech/shrink to the number of cells width
        x_cell = (x_n - self.grid_edges[0]) * self.energy.size[0] / range_x_grid
        y_cell = (y_n - self.grid_edges[2]) * self.energy.size[1] / range_y_grid
        # always round down - points between 0 and 1 belong all in cell 0, between 1 and 2 all in cell 1 and so on
        return int(x_cell), int(y_cell)

    def _cell_to_index(self, cell: tuple) -> int:
        """
        Given a cell of the histogram, get the sequential (1D) index of that cell, for example for the index in
        transitions matrix.

        Args:
            cell: tuple (row, column), a cell of the histogram

        Returns:
            int: the unique 1D index of that cell
        """
        index = cell[0]
        for i in range(1, len(cell)):
            index = index * self.histogram.shape[i] + cell[i]
        return index

    def get_transitions_matrix(self, tau_array: np.ndarray = None, noncorr: bool = False) -> np.ndarray:
        """
        Obtain a set of transition matrices for different tau-s specified in tau_array.

        Args:
            tau_array: 1D array of tau values for which the transition matrices should be constructed
            noncorr: bool, should only every tau-th frame be used for MSM construction
                     (if False, use sliding window - much more expensive but throws away less data)
        Returns:
            an array of transition matrices
        """
        if tau_array:
            self.tau_array = tau_array

        def window(seq, len_window):
            # in this case always move the window by 1 and use all points in simulations to count transitions
            return [seq[k: k + len_window:len_window-1] for k in range(0, (len(seq)+1)-len_window)]

        def noncorr_window(seq, len_window):
            # in this case, only use every tau-th element for MSM. Faster but loses a lot of data
            cut_seq = seq[0:-1:len_window]
            return [[a, b] for a, b in zip(cut_seq[0:-2], cut_seq[1:])]

        # now we are creating transitions matrix only for accessible cells - already ordered
        acc_cells = [(i, j) for i in range(self.histogram.shape[0]) for j in range(self.histogram.shape[1])
                     if self.energy.is_accessible((i, j))]
        cells = [(i, j) for i in range(self.histogram.shape[0]) for j in range(self.histogram.shape[1])]
        #all_cells = len(acc_cells)
        all_cells = len(cells)
        self.transition_matrices = np.zeros(shape=(len(self.tau_array), all_cells, all_cells))
        for tau_i, tau in enumerate(tqdm(self.tau_array)):
            count_per_cell = {(i, j, m, n): 0 for i, j in cells for m, n in cells}
            if not noncorr:
                window_cell = window(self.traj_cell, int(tau))
            else:
                window_cell = noncorr_window(self.traj_cell, int(tau))
            for cell_slice in window_cell:
                start_cell = cell_slice[0]
                end_cell = cell_slice[1]
                count_per_cell[(start_cell[0], start_cell[1], end_cell[0], end_cell[1])] += 1
            for key, value in count_per_cell.items():
                a, b, c, d = key
                start_cell = (a, b)
                end_cell = (c, d)
                #if self.energy.is_accessible(tuple(start_cell)) and self.energy.is_accessible(tuple(end_cell)) and \
                #        value != 0:
                #i = acc_cells.index(start_cell)
                #j = acc_cells.index(end_cell)
                i = cells.index(start_cell)
                j = cells.index(end_cell)
                self.transition_matrices[tau_i][i, j] += value
                # enforce detailed balance
                self.transition_matrices[tau_i][j, i] += value
        # divide each row of each matrix by the sum of that row
        sums = self.transition_matrices.sum(axis=-1, keepdims=True)
        sums[sums == 0] = 1
        self.transition_matrices = self.transition_matrices / sums
        return self.transition_matrices

    def get_eigenval_eigenvec(self, num_eigv: int = 6, **kwargs) -> tuple:
        """
        Obtain eigenvectors and eigenvalues of the transition matrices.

        Args:
            num_eigv: how many eigenvalues/vectors pairs
            **kwargs: named arguments to forward to eigs()
        Returns:
            (eigenval, eigenvec) a tuple of eigenvalues and eigenvectors, first num_eigv given for all tau-s
        """
        tau_eigenvals = np.zeros((len(self.tau_array), num_eigv))
        tau_eigenvec = np.zeros((len(self.tau_array), self.transition_matrices[0].shape[0], num_eigv))
        for i, tau in enumerate(self.tau_array):
            tm = self.transition_matrices[i].T
            eigenval, eigenvec = eigs(tm, num_eigv, **kwargs)
            if eigenvec.imag.max() == 0 and eigenval.imag.max() == 0:
                eigenvec = eigenvec.real
                eigenval = eigenval.real
            # sort eigenvectors according to their eigenvalues
            idx = eigenval.argsort()[::-1]
            eigenval = eigenval[idx]
            tau_eigenvals[i] = eigenval.real
            eigenvec = eigenvec[:, idx]
            tau_eigenvec[i] = eigenvec
        self.eigenvals = tau_eigenvals
        self.eigenvec = tau_eigenvec
        return tau_eigenvals, tau_eigenvec

    ############################################################################
    # ------------------------   VISUALIZATION  --------------------------------
    ############################################################################

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
            plt.savefig(self.images_path + f"{self.images_name}_eigenvalues_msm.png")
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
                               figsize=(full_width, full_width/num_eigv*len(self.tau_array[:taus_to_plot])))
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
                        #if self.energy.is_accessible((m, n)):
                        if must_flip > 0 or (j+1) % 2 == 0:
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
        fig.savefig(self.images_path + f"{self.images_name}_eigenvectors_msm.png", bbox_inches='tight', dpi=1200)
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
            #ax.plot(self.tau_array * self.dt, to_plot_abs, color=colors_circle[j])
        if np.any(rates_eigenvalues):
            for j in range(1, len(rates_eigenvalues)):
                absolute_its = np.array([-1/rates_eigenvalues[j] for _ in self.tau_array])
                ax.plot(self.tau_array * self.dt, absolute_its, color="black", ls="--")
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=self.tau_array[-1] * self.dt)
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel(r"ITS")
        tau_array_with_zero = (self.tau_array * self.dt).tolist()
        tau_array_with_zero.append(0)
        tau_array_with_zero.sort()
        ax.fill_between(tau_array_with_zero, tau_array_with_zero, color="grey", alpha=0.5)
        fig.savefig(self.images_path + f"{self.images_name}_implied_timescales.png", bbox_inches='tight', dpi=1200)
        plt.close()

    def visualize_hist_2D(self):
        """
        Plot the histogram of the simulation. Should correspond to the 2D Boltzmann distribution of the energy
        surface.
        """
        with plt.style.context(['Stylesheets/not_animation.mplstyle', 'Stylesheets/maze_style.mplstyle']):
            fig, ax = plt.subplots(1, 1)
            cmap = cm.get_cmap("RdBu").copy()
            #im = plt.imshow(self.histogram, cmap=cmap)
            sns.heatmap(self.histogram, cmap=cmap, fmt='.2f',
                        square=True, ax=ax, yticklabels=[], xticklabels=[])
            #fig.colorbar(im, ax=ax)
            ax.figure.savefig(self.images_path + f"{self.images_name}_hist_2D.png")
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
                    if E_population[1][j-1] < e <= E_population[1][j]:
                        E_pop[i] = E_population[0][j-1]
            population = np.array(population)
            population[E_pop == 0] = 0
            E_pop[E_pop == 0] = 1
            sns.histplot(x=energies, bins=25, weights=population/E_pop, ax=ax, element="step",
                         color="black", fill=False, kde=False)
            #plt.hist(energies, bins=25, weights=population/E_pop, histtype='step')
            ax.set_xlabel("Cell energy")
            ax.set_ylabel("Relative cell population")
            plt.savefig(self.images_path + f"{self.images_name}_population_per_energy.png")
            plt.close()

    def visualize_trajectory(self):
        """
        Plot the points visited by the trajectory.
        """
        with plt.style.context(['Stylesheets/not_animation.mplstyle']):
            plt.subplots(1, 1, figsize=self.energy.size)
            plt.scatter(self.traj_y, self.traj_x, marker="o", c="black", s=1)
            plt.gca().invert_yaxis()
            plt.savefig(self.images_path + f"{self.images_name}_trajectory.png")
            plt.close()

    def save_information(self):
        with open(DATA_PATH + f"{self.images_name}_summary.txt", "w") as f:
            describe_types = {EnergyFromMaze: "maze", EnergyFromPotential: "double_well", EnergyFromAtoms: "atoms"}
            f.write(f"# Simulation performed with the script simulation.create_simulation.py.\n")
            f.write(f"# Time of execution: {datetime.now()}\n")
            f.write(f"# --------- PARAMETERS ----------\n")
            f.write(f"energy type = {describe_types[type(self.energy)]}\n")
            f.write(f"energy cutoff = {self.energy.energy_cutoff}\n")
            f.write(f"size = {self.energy.size}\n")
            f.write(f"grid_edges = {self.grid_edges}\n")
            f.write(f"step_x = {self.step_x}\n")
            f.write(f"step_y = {self.step_y}\n")
            f.write(f"images path = {self.images_path}\n")
            f.write(f"images name = {self.images_name}\n")
            f.write(f"mass = {self.m}\n")
            f.write(f"friction = {self.friction}\n")
            f.write(f"temperature = {self.temperature}\n")
            f.write(f"D = {self.D}\n")
            f.write(f"N = {self.N}\n")
            f.write(f"dt = {self.dt}\n")
            f.write(f"tau_array = {self.tau_array}\n")


if __name__ == '__main__':
    start_time = time.time()
    # ------------------- MAZE ------------------
    my_maze = Maze((8, 8), images_name="test20", no_branching=True, edge_is_wall=True)
    my_energy = EnergyFromMaze(my_maze, images_name=my_maze.images_name,
                               factor_grid=1, m=1, friction=30)
    my_maze.visualize()
    my_energy.visualize_underlying_maze()
    # ------------------- POTENTIAL ------------------
    # my_energy = EnergyFromPotential((50, 50), images_name="potentials", m=1,
    #                                 friction=20)
    # ------------------- ATOMS ------------------
    # epsilon = 3
    # sigma = 5
    # atom_1 = Atom((3.3, 20.5), epsilon, sigma)
    # atom_2 = Atom((14.3, 9.3), epsilon, sigma-2)
    # atom_3 = Atom((9.3, 35.3), epsilon/5, sigma)
    # my_energy = EnergyFromAtoms((40, 40), (atom_1, atom_2, atom_3), grid_edges=(0, 40, 0, 40),
    #                             images_name="atoms_big", friction=10)
    # ------------------- GENERAL FUNCTIONS ------------------
    # my_energy.visualize_boltzmann()
    my_energy.visualize()
    my_energy.visualize_eigenvectors_in_maze(num=6, which="LR")
    #my_energy.visualize_eigenvalues()
    #my_energy.visualize_rates_matrix()
    e_eigval, e_eigvec = my_energy.get_eigenval_eigenvec(6, which="LR")
    my_simulation = Simulation(my_energy, images_name=my_energy.images_name)
    print(f"Performing a simulation named {my_simulation.images_name}.")
    to_save_trajectory = False
    my_simulation.integrate(N=int(1e6), dt=0.01, save_trajectory=to_save_trajectory)
    my_simulation.visualize_hist_2D()
    my_simulation.visualize_population_per_energy()
    if to_save_trajectory:
        my_simulation.visualize_trajectory()
    my_simulation.save_information()
    my_simulation.get_transitions_matrix(noncorr=False)
    #my_simulation.visualize_transition_matrices()
    my_simulation.visualize_eigenvec(6, which="LR")
    my_simulation.visualize_its(num_eigv=6, which="LR", rates_eigenvalues=e_eigval)
    #my_simulation.visualize_eigenvalues()
    # ----------   TIMING -----------------
    end_time = time.time()
    duration = end_time - start_time
    hours = round(duration // 3600 % 24)
    minutes = round(duration // 60 % 60)
    seconds = round(duration % 60)
    print(f"Total time: {hours}h {minutes}min {seconds}s")

