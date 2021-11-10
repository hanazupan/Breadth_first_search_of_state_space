"""
In this file, molecular dynamics simulation on an Energy surface is performed, a MSM constructed and evaluated.
Finally, characteristic ITS are plotted. All these calculations are performed in Simulation class.
"""

from maze.create_energies import EnergyFromPotential, EnergyFromMaze, Atom, EnergyFromAtoms  # need all imports
from maze.create_mazes import Maze  # need this import
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm, colors
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
import time
import pandas as pd
import seaborn as sns

# DEFINING BOLTZMANN CONSTANT
kB = 0.008314463  # kJ/mol/K

DIM_LANDSCAPE = (7.25, 4.45)
DIM_PORTRAIT = (3.45, 4.45)
DIM_SQUARE = (4.45, 4.45)


class Simulation:

    def __init__(self, energy, dt: float = 0.01, N: int = int(1e7),
                 images_path: str = "./", images_name: str = "simulation"):
        self.energy = energy
        self.images_name = images_name
        self.images_path = images_path
        # TD/particle properties inherited from Energy
        self.m = self.energy.m
        self.friction = self.energy.friction
        self.T = self.energy.T
        self.dt = dt
        self.N = N
        self.D = kB*self.T/self.m/self.friction
        # TODO: tau array should probably be calculated (what are appropriate values?) and not pre-determined
        self.tau_array = np.array([5, 10, 50, 100, 150, 200, 250, 350, 500, 700, 1000])
        # prepare empty objects
        self.histogram = np.zeros(self.energy.size)
        self.outside_hist = 0
        self.traj_x = None
        self.traj_y = None
        self.traj_cell = None
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

    def integrate(self, dt: float = None, N: int = None, save_trajectory: bool = False,
                  restart_after: int = -1):
        """
        Implement the Euler–Maruyama method for integration of the trajectory on self.energy surface. It is
        possible to simulate several shorter trajectories using the option restart_after.

        Args:
            dt: float, time step
            N: int, number of time steps
            save_trajectory: bool, should the trajectory be saved (True needed for MSM analysis)
            restart_after: int, after how many time steps the trajectory should restart (-1 if never)
        """
        # the time step and number of steps can be redefined for every simulation
        if dt:
            self.dt = dt
        if N:
            self.N = N
        self.histogram = np.zeros(self.energy.size)
        # start at a random position somewhere on the grid
        x_n = self.step_x*self.energy.size[0]*np.random.random() + self.grid_edges[0]
        y_n = self.step_y*self.energy.size[1]*np.random.random() + self.grid_edges[2]
        # figure out in which cell the trajectory starts and increase the count of the histogram
        cell = self._point_to_cell((x_n, y_n))
        self.histogram[cell] += 1
        self.traj_cell = [cell]
        # not necessary for MSM, but for trajectory visualization the x and y positions can also be saved
        if save_trajectory:
            self.traj_x = np.zeros(N)
            self.traj_y = np.zeros(N)
        for n in tqdm(range(self.N)):
            # every restart_after-th step start the trajectory in a new random place
            if restart_after != -1 and n % restart_after == 0:
                x_n = self.step_x * self.energy.size[0] * np.random.random() + self.grid_edges[0]
                y_n = self.step_y * self.energy.size[1] * np.random.random() + self.grid_edges[2]
            # integrate the trajectory one step and increase the histogram count
            x_n, y_n = self._euler_maruyama(x_n, y_n)
            if self.energy.pbc:
                x_n, y_n = self._point_within_bound((x_n, y_n))
            cell = self._point_to_cell((x_n, y_n))
            # if cell fits into the histogram
            if np.all([0 <= cell[k] < self.histogram.shape[k] for k in range(len(self.histogram.shape))]):
                self.histogram[cell] += 1
                self.traj_cell.append(cell)
                # if applicable, save trajectory
                if save_trajectory:
                    self.traj_x[n] = x_n
                    self.traj_y[n] = y_n
            else:
                # if not using periodic boundaries, points can land outside the histogram
                self.outside_hist += 1
                assert not self.energy.pbc
        # normalizing the histogram
        self.histogram = self.histogram / np.sum(self.histogram)
        self.traj_cell = np.array(self.traj_cell)

    def _euler_maruyama(self, x_n: float, y_n: float) -> tuple:
        """
        Complete a step of trajectory integration using an Euler-Maruyama integrator.
        Args:
            x_n: float, x-coordinate of the current trajectory point
            y_n: float, y-coordinate of the current trajectory point

        Returns: tuple (x_n, y_n), the integrated new trajectory points (not necessarily between -1 and 1)

        """
        dV_dx = self.energy.get_x_derivative((x_n, y_n))
        dV_dy = self.energy.get_y_derivative((x_n, y_n))
        eta_x = np.random.normal(loc=0.0, scale=np.sqrt(self.dt))
        x_n = x_n - dV_dx * self.dt / self.m / self.friction + np.sqrt(2 * self.D) * eta_x
        eta_y = np.random.normal(loc=0.0, scale=np.sqrt(self.dt))
        y_n = y_n - dV_dy * self.dt / self.m / self.friction + np.sqrt(2 * self.D) * eta_y
        return x_n, y_n

    def _point_within_bound(self, point: tuple) -> tuple:
        """
        Apply periodic boundary conditions so that the argument point is transformed into an equivalent point
        within the original grid (-1 to 1).

        Args:
            point: tuple (x_n, y_n) a 2D point potentially outside (-1, 1)

        Returns:
            tuple (x_n, y_n), a 2D point corrected to an equivalent position within (-1, 1)
        """
        x_n, y_n = point
        range_x_grid = self.grid_edges[1] - self.grid_edges[0]
        range_y_grid = self.grid_edges[3] - self.grid_edges[2]
        x_cell = (x_n - self.grid_edges[0]) * self.energy.size[0] / range_x_grid
        y_cell = (y_n - self.grid_edges[2]) * self.energy.size[1] / range_y_grid
        x_cell = x_cell % self.energy.size[0]
        y_cell = y_cell % self.energy.size[1]
        x_n = x_cell * range_x_grid / self.energy.size[0] + self.grid_edges[0]
        y_n = y_cell * range_y_grid / self.energy.size[1] + self.grid_edges[2]
        return x_n, y_n

    def _point_to_cell(self, point: tuple) -> tuple:
        """
        Given a trajectory point (x_n, y_n), determine to which cell of the histogram this trajectory point belongs to.

        Args:
            point: tuple (x_n, y_n) a 2D point potentially outside (-1, 1)

        Returns:
            tuple (row, column) in which cell of the histogram this point lands
        """
        x_n, y_n = point
        if self.energy.pbc:
            x_n, y_n = self._point_within_bound(point)
        range_x_grid = self.grid_edges[1] - self.grid_edges[0]
        range_y_grid = self.grid_edges[3] - self.grid_edges[2]
        # conversion from grid to cell -------------------THIS WORKS
        x_cell = (x_n - self.grid_edges[0])*self.energy.size[0]/range_x_grid
        y_cell = (y_n - self.grid_edges[2])*self.energy.size[1]/range_y_grid
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

    def get_transitions_matrix(self, tau_array: np.ndarray = None, noncorr=False) -> np.ndarray:
        """
        Obtain a set of transition matrices for different tau-s specified in tau_array.

        Args:
            tau_array: 1D array of tau values for which the transition matrices should be constructed

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

        self.acc_cells = [(i, j) for i in range(self.histogram.shape[0]) for j in range(self.histogram.shape[1])
                          if self.energy.is_accessible((i, j))]
        all_cells = len(self.acc_cells)
        self.transition_matrices = np.zeros(shape=(len(self.tau_array), all_cells, all_cells))
        for tau_i, tau in enumerate(self.tau_array):
            if not noncorr:
                window_cell = window(self.traj_cell, int(tau))
            else:
                window_cell = noncorr_window(self.traj_cell, int(tau))
            for cell_slice in window_cell:
                start_cell = cell_slice[0]
                end_cell = cell_slice[1]
                if self.energy.is_accessible(tuple(start_cell)) and self.energy.is_accessible(tuple(end_cell)):
                    i = self.acc_cells.index(tuple(start_cell))
                    j = self.acc_cells.index(tuple(end_cell))
                    try:
                        self.transition_matrices[tau_i, i, j] += 1
                        # enforce detailed balance
                        self.transition_matrices[tau_i, j, i] += 1
                    except IndexError:
                        if self.energy.pbc:
                            #print(i, j, self.transition_matrices.shape)
                            raise IndexError("If PBC used, all points on a trajectory should fit in the histogram!")
        # with plt.style.context(['Stylesheets/not_animation.mplstyle']):
        #     fig, ax = plt.subplots(1, len(self.transition_matrices), sharey="row")
        # for i, tm in enumerate(self.transition_matrices):
        #     vmax = np.max(self.transition_matrices[i, 43, :])
        #     vmin = np.min(self.transition_matrices[i, 43, :])
        #     ax[i].imshow(self.transition_matrices[i, 43, :].reshape((self.energy.energies.shape[0], self.energy.energies.shape[1])),
        #                  cmap="RdBu_r", vmin=vmin, vmax=vmax)
        #     ax[i].axes.get_xaxis().set_visible(False)
        #     ax[i].axes.get_yaxis().set_visible(False)
        #     ax[i].set_title(f"tau = {self.tau_array[i]}", fontsize=7)
        # plt.savefig(self.images_path + f"row_{self.images_name}.png", bbox_inches='tight', dpi=1200)
        # plt.close()
        # divide each row of each matrix by the sum of that row
        sums = self.transition_matrices.sum(axis=-1, keepdims=True)
        sums[sums == 0] = 1
        self.transition_matrices = self.transition_matrices / sums
        return self.transition_matrices

    def get_eigenval_eigenvec(self, num_eigv: int = 6, **kwargs) -> tuple:
        """
        Obtain eigenvectors and eigenvalues of the transition matrices.
        Returns:
            (eigenval, eigenvec) a tuple of eigenvalues and eigenvectors, first num_eigv given for all tau-s
        """
        tau_eigenvals = np.zeros((len(self.tau_array), num_eigv))
        tau_eigenvec = np.zeros((len(self.tau_array), len(self.transition_matrices[0]), num_eigv))
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
        return tau_eigenvals, tau_eigenvec

    ############################################################################
    # ------------------------   VISUALIZATION  --------------------------------
    ############################################################################

    def visualize_eigenvalues(self):
        """
        Visualize the eigenvalues of rate matrix.

        Args:
            show: bool, whether to display the image
        """
        if not np.any(self.transition_matrices):
            self.get_transitions_matrix()
        num = self.transition_matrices.shape[1] - 2
        eigenval, eigenvec = self.get_eigenval_eigenvec(num_eigv=num, which="LR")
        with plt.style.context(['Stylesheets/not_animation.mplstyle']):
            plt.subplots(1, 1, figsize=DIM_LANDSCAPE)
            xs = np.linspace(0, 1, num=num)
            plt.scatter(xs, eigenval[-1], s=5, c="black")
            for i, eigenw in enumerate(eigenval[-1]):
                plt.vlines(xs[i], eigenw, 0, linewidth=0.5)
            plt.hlines(0, 0, 1)
            plt.title("Eigenvalues")
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.savefig(self.images_path + f"eigenvalues_msm_{self.images_name}.png")
            plt.close()

    def visualize_eigenvec(self, num_eigv: int = 6, **kwargs):
        """
        Visualize first num_eiv of the transitions matrix for all tau-s in the self.tau_array
        Args:
            num_eigv: number of eigenvectors to visualize
        """
        tau_eigenvals, tau_eigenvec = self.get_eigenval_eigenvec(num_eigv=num_eigv, **kwargs)
        full_width = DIM_LANDSCAPE[0]
        fig, ax = plt.subplots(len(self.tau_array), num_eigv, sharey="row",
                               figsize=(full_width, full_width/num_eigv*len(self.tau_array)))
        vmin = np.min(tau_eigenvec)
        vmax = np.max(tau_eigenvec)
        cmap = cm.get_cmap("RdBu").copy()
        cmap.set_over("black")
        cmap.set_under("black")
        for i, tau in enumerate(self.tau_array):
            for j in range(num_eigv):
                array = np.full(self.histogram.shape, np.max(tau_eigenvec[i, :, j]) + 1)
                index = 0
                for m in range(self.histogram.shape[0]):
                    for n in range(self.histogram.shape[1]):
                        if self.energy.is_accessible((m, n)):
                            array[m, n] = tau_eigenvec[i, index, j]
                            index += 1
                ax[i][j].imshow(array, cmap=cmap, vmax=vmax, vmin=vmin)
                ax[0][j].set_title(f"Eigenvector {j + 1}", fontsize=7)
                ax[i][0].set_ylabel(f"tau = {tau}", fontsize=7)
                ax[i][j].axes.get_xaxis().set_visible(False)
                ax[i][j].set_yticklabels([])
        fig.savefig(self.images_path + f"eigenvectors_{self.images_name}.png", bbox_inches='tight', dpi=1200)
        plt.close()

    def visualize_its(self, num_eigv: int = 6, rates_eigenvalues=None, **kwargs):
        """
        Plot iterative timescales.

        Args:
            num_eigv: how many eigenvalues/timescales to plot
        """
        tau_eigenvals, tau_eigenvec = self.get_eigenval_eigenvec(num_eigv=num_eigv, **kwargs)
        tau_eigenvals = tau_eigenvals.T
        fig2, ax2 = plt.subplots(1, 1)
        colors = ["blue", "red", "green", "orange", "black", "yellow", "purple", "pink"]
        for j in range(1, num_eigv):
            print("S eigv ", tau_eigenvals[j, :])
            to_plot = -self.tau_array * self.dt / np.log(np.abs(tau_eigenvals[j, :]))
            ax2.plot(self.tau_array * self.dt, to_plot, label=f"its {j}", color=colors[j])
        if np.any(rates_eigenvalues):
           for j in range(1, len(rates_eigenvalues)):
               ax2.plot(self.tau_array * self.dt, [-1/rates_eigenvalues[j] for _ in self.tau_array], color="black", ls="--")
        ax2.legend()
        ax2.fill_between(self.tau_array * self.dt, self.tau_array*self.dt, color="grey", alpha=0.5)
        fig2.savefig(self.images_path + f"implied_timescales_{self.images_name}.png", bbox_inches='tight', dpi=1200)
        plt.close()

    def visualize_transition_matrices(self):
        """
        Plot the transition matrices corresponding to different taus.
        """
        if not np.any(self.transition_matrices):
            self.get_transitions_matrix()
        with plt.style.context(['Stylesheets/not_animation.mplstyle']):
            fig, ax = plt.subplots(1, len(self.transition_matrices), sharey="row")
        for i, tm in enumerate(self.transition_matrices):
            vmax = np.max(tm)
            vmin = np.min(tm)
            ax[i].imshow(tm, cmap="RdBu_r", vmin=vmin, vmax=vmax)
            ax[i].axes.get_xaxis().set_visible(False)
            ax[i].axes.get_yaxis().set_visible(False)
            ax[i].set_title(f"tau = {self.tau_array[i]}", fontsize=7)
        plt.savefig(self.images_path + f"trans_mat_{self.images_name}.png", bbox_inches='tight', dpi=1200)
        plt.close()

    def visualize_hist_2D(self):
        """
        Plot the histogram of the simulation. Should correspond to the 2D Boltzmann distribution of the energy
        surface.
        """
        with plt.style.context(['Stylesheets/not_animation.mplstyle', 'Stylesheets/maze_style.mplstyle']):
            fig, ax = plt.subplots(1, 1)
            cmap = cm.get_cmap("RdBu").copy()
            im = plt.imshow(self.histogram, cmap=cmap)
            fig.colorbar(im, ax=ax)
            ax.figure.savefig(self.images_path + f"hist_2D_{self.images_name}.png")
            plt.close()

    def visualize_sim_Boltzmann(self):
        """
        Visualize a 1D histogram with cells sorted as they were explored with BFS. Only for
        comparison with a Boltzmann distribution obtained from rates matrix.
        """
        list_of_cells = self.energy.explorer.get_sorted_accessible_cells()
        boltzmanns = np.array([self.histogram[cell] for cell in list_of_cells])
        with plt.style.context(['Stylesheets/not_animation.mplstyle']):
            fig, ax = plt.subplots(1, 1)
            ax.plot(boltzmanns)
            ax.set_xlabel("Accessible cell index")
            ax.set_ylabel("Relative cell visitation frequency")
            ax.set_title("Simulated Boltzmann distribution")
            fig.savefig(self.images_path + f"sim_boltzmann_{self.images_name}.png")
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
            energies = np.array(energies)
            E_population = np.histogram(energies, bins=25)
            E_pop = np.zeros(energies.shape)
            for i, e in enumerate(energies):
                for j, ep in enumerate(E_population[1][1:]):
                    if E_population[1][j-1] < e <= E_population[1][j]:
                        E_pop[i] = E_population[0][j-1]
            plt.hist(energies, bins=25, weights=np.array(population)/E_pop, histtype='step')
            plt.savefig(self.images_path + f"population_per_energy_{self.images_name}.png")
            plt.close()

    def visualize_trajectory(self):
        """
        Plot the points visited by the trajectory.
        """
        with plt.style.context(['Stylesheets/not_animation.mplstyle']):
            plt.subplots(1, 1, figsize=self.energy.size)
            plt.scatter(self.traj_y, self.traj_x, marker="o", c="black", s=1)
            plt.gca().invert_yaxis()
            plt.savefig(self.images_path + f"traj_{self.images_name}.png")
            plt.close()


if __name__ == '__main__':
    start_time = time.time()
    img_path = "images/"
    # ------------------- MAZE ------------------
    # my_maze = Maze((6, 8), images_path=img_path, images_name="mazes", no_branching=False, edge_is_wall=False)
    # my_energy = EnergyFromMaze(my_maze, images_path=img_path, images_name=my_maze.images_name, m=1, friction=20, T=1600)
    # my_maze.visualize()
    # my_energy.visualize_underlying_maze()
    # ------------------- POTENTIAL ------------------
    # my_energy = EnergyFromPotential((20, 30), images_path=img_path, images_name="potentials", m=1,
    #                                 friction=20, T=200)
    # ------------------- ATOMS ------------------
    epsilon = 3.18*1.6022e-22
    sigma = 5.928
    atom_1 = Atom((3.3, 20.5), epsilon, sigma)
    atom_2 = Atom((14.3, 9.3), epsilon, sigma-2)
    atom_3 = Atom((5.3, 45.3), epsilon/5, sigma)
    my_energy = EnergyFromAtoms((12, 13), (atom_1, atom_2, atom_3), grid_edges=(2, 20, 2, 50),
                                images_name="atoms", images_path=img_path, friction=10, T=1600)
    arr_x = np.zeros(my_energy.size)
    arr_y = np.zeros(my_energy.size)
    for i in range(my_energy.size[0]):
        for j in range(my_energy.size[1]):
            arr_x[i, j] = my_energy.get_x_derivative((my_energy.grid_x[i, j], my_energy.grid_y[i, j]))
            arr_y[i, j] = my_energy.get_y_derivative((my_energy.grid_x[i, j], my_energy.grid_y[i, j]))
    vec_lens = np.sqrt(arr_x**2 + arr_y**2)
    df = pd.DataFrame(data=my_energy.energies, index=my_energy.grid_x[:, 0], columns=my_energy.grid_y[0, :])
    cmap = cm.get_cmap("RdBu_r").copy()
    fig, im = plt.subplots(1, 1)
    sns.heatmap(df, cmap=cmap, norm=colors.TwoSlopeNorm(vcenter=0, vmax=my_energy.energy_cutoff), fmt='.2f',
                     yticklabels=[f"{ind:.2f}" for ind in df.index],
                     xticklabels=[f"{col:.2f}" for col in df.columns], ax=im)
    im.quiver(arr_y/vec_lens, arr_x/vec_lens, pivot='mid')
    plt.savefig("derivatives.png")
    plt.close()
    # ------------------- GENERAL FUNCTIONS ------------------
    my_energy.visualize_boltzmann()
    my_energy.visualize()
    my_energy.visualize_eigenvectors_in_maze(num=6, which="SR", sigma=0)
    my_energy.visualize_eigenvalues()
    my_energy.visualize_rates_matrix()
    e_eigval, e_eigvec = my_energy.get_eigenval_eigenvec(8, which="SR", sigma=0)
    print("ITS energies ", -1/e_eigval)
    print("E eigv ", e_eigval)
    my_simulation = Simulation(my_energy, images_path=img_path, images_name=my_energy.images_name)
    to_save_trajectory = True
    my_simulation.integrate(N=int(1e6), dt=0.01, save_trajectory=to_save_trajectory)
    my_simulation.visualize_hist_2D()
    my_simulation.visualize_sim_Boltzmann()
    my_simulation.visualize_population_per_energy()
    if to_save_trajectory:
        my_simulation.visualize_trajectory()
    my_simulation.get_transitions_matrix(noncorr=True)
    s_eigval, s_eigvec = my_simulation.get_eigenval_eigenvec(8, which="LR")
    my_simulation.visualize_transition_matrices()
    my_simulation.visualize_eigenvec(8, which="LR")
    my_simulation.visualize_its(num_eigv=8, which="LR", rates_eigenvalues=e_eigval)
    my_simulation.visualize_eigenvalues()
    print("outsiders: ", my_simulation.outside_hist)
    # ----------   TIMING -----------------
    end_time = time.time()
    duration = end_time - start_time
    hours = round(duration // 3600 % 24)
    minutes = round(duration // 60 % 60)
    seconds = round(duration % 60)
    print(f"Total time: {hours}h {minutes}min {seconds}s")


