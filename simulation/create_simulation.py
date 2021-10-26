from maze.create_energies import Energy
from maze.create_mazes import Maze
from scipy.interpolate import bisplev
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm, colors
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
import itertools as it

# DEFINING BOLTZMANN CONSTANT
kB = 0.008314463  # kJ/mol/K

DIM_LANDSCAPE = (7.25, 4.45)
DIM_PORTRAIT = (3.45, 4.45)
DIM_SQUARE = (4.45, 4.45)


class Simulation:

    def __init__(self, energy, m: float = 1, friction: float = 1, T: float = 293, dt: float = 0.01, N: int = int(1e7),
                 images_path: str = "./", images_name: str = "simulation"):
        self.energy = energy
        self.spline = self.energy.get_spline()
        self.images_name = images_name
        self.images_path = images_path
        self.m = m
        self.friction = friction
        self.T = T
        self.dt = dt
        self.N = N
        self.D = kB*self.T/self.m/self.friction
        # prepare empty objects
        self.histogram = np.zeros(self.energy.size)
        self.tau_array = np.array([5, 7, 10, 12, 15, 20, 25, 40, 70, 100, 150, 200])
        self.traj_x = None
        self.traj_y = None
        self.transition_matrices = None
        self.step_x = 2/self.histogram.shape[0]
        self.step_y = 2/self.histogram.shape[1]

    def integrate(self, dt: float = None, N: int = None, save_trajectory: bool = False,
                  restart_after: int = -1):
        """
        Implement the Euler–Maruyama method for integration of the trajectory on self.energy surface.

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
        x_n = 2*np.random.random() - 1               # random between (-1, 1)
        y_n = 2*np.random.random() - 1               # random between (-1, 1)
        # figure out in which cell the trajectory starts and increase the count
        cell = self._point_to_cell((x_n, y_n))
        self.histogram[cell] += 1
        if save_trajectory:
            self.traj_x = np.zeros(N)
            self.traj_y = np.zeros(N)
        for n in tqdm(range(self.N)):
            # every restart_after-th step start the trajectory in a new random place
            if restart_after != -1 and n % restart_after == 0:
                x_n = 2 * np.random.random() - 1                # random between (-1, 1)
                y_n = 2 * np.random.random() - 1                # random between (-1, 1)
            # integrate the trajectory one step and increase the histogram count
            x_n, y_n = self._euler_maruyama(x_n, y_n)
            x_n, y_n = self._point_within_bound((x_n, y_n))
            cell = self._point_to_cell((x_n, y_n))
            self.histogram[cell] += 1
            # if applicable, save trajectory
            if save_trajectory:
                self.traj_x[n] = x_n
                self.traj_y[n] = y_n
        # normalizing the histogram
        self.histogram = self.histogram / self.N

    def _euler_maruyama(self, x_n: float, y_n: float) -> tuple:
        """
        Complete a step of trajectory integration using an Euler-Maruyama integrator.
        Args:
            x_n: float, x-coordinate of the current trajectory point
            y_n: float, y-coordinate of the current trajectory point

        Returns: tuple (x_n, y_n), the integrated new trajectory points (not necessarily between -1 and 1)

        """
        if self.spline:
            dV_dx = bisplev(x_n, y_n, self.spline, dx=1)
            dV_dy = bisplev(x_n, y_n, self.spline, dy=1)
        else:
            dV_dx = self.energy.dV_dx(x_n)
            dV_dy = self.energy.dV_dy(y_n)
        eta_x = np.random.normal()
        x_n = x_n - dV_dx * self.dt / self.m / self.friction + np.sqrt(2 * self.D) * eta_x * np.sqrt(self.dt)
        eta_y = np.random.normal()
        y_n = y_n - dV_dy * self.dt / self.m / self.friction + np.sqrt(2 * self.D) * eta_y * np.sqrt(self.dt)
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
        dist_x_n = (abs(x_n) + 1) // 2
        x_n = x_n - np.sign(x_n) * 2 * dist_x_n
        dist_y_n = (abs(y_n) + 1) // 2
        y_n = y_n - np.sign(y_n) * 2 * dist_y_n
        return x_n, y_n

    def _point_to_cell(self, point: tuple) -> tuple:
        """
        Given a trajectory point (x_n, y_n), determine to which cell of the histogram this trajectory point belongs to.

        Args:
            point: tuple (x_n, y_n) a 2D point potentially outside (-1, 1)

        Returns:
            tuple (row, column) in which cell of the histogram this point lands
        """
        # changing to an equivalent point within the original energy surface (-1, 1)
        point = self._point_within_bound(point)
        x_n, y_n = point
        # determine the cell of the histogram
        cell = int((x_n + 1) // self.step_x), int((y_n + 1) // self.step_y)
        return cell

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

    def get_transitions_matrix(self, tau_array: np.ndarray = None) -> np.ndarray:
        """
        Obtain a set of transition matrices for different tau-s specified in tau_array.

        Args:
            tau_array: 1D array of tau values for which the transition matrices should be constructed

        Returns:
            an array of transition matrices
        """
        if tau_array:
            self.tau_array = tau_array
        if not np.any(self.traj_x):
            raise ValueError("No trajectories found! Check if using the setting save_trajectory=False.")

        # def window(seq, n):
        #     """Sliding window width tau from seq.  From old itertools recipes."""
        #     it = iter(seq)
        #     result = tuple(islice(it, n))
        #     if len(result) == n:
        #         yield result
        #     for elem in it:
        #         result = result[1:] + (elem,)
        #         yield result

        def window(seq, len_window):
            return [seq[k: k + len_window:len_window-1] for k in range(0, (len(seq)+1)-len_window)]

        all_cells = len(self.histogram.flatten())
        self.transition_matrices = np.zeros(shape=(len(self.tau_array), all_cells, all_cells))
        for tau_i, tau in enumerate(self.tau_array):
            window_x = window(self.traj_x, int(tau))
            window_y = window(self.traj_y, int(tau))
            for w_x, w_y in zip(window_x, window_y):
                start_cell = self._point_to_cell((w_x[0], w_y[0]))
                end_cell = self._point_to_cell((w_x[1], w_y[1]))
                i = self._cell_to_index(start_cell)
                j = self._cell_to_index(end_cell)
                self.transition_matrices[tau_i, i, j] += 1
                # enforce detailed balance
                self.transition_matrices[tau_i, j, i] += 1
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

    def visualize_eigenvec(self, num_eigv: int = 6):
        """
        Visualize first num_eiv of the transitions matrix for all tau-s in the self.tau_array
        Args:
            num_eigv: number of eigenvectors to visualize
        """
        tau_eigenvals, tau_eigenvec = self.get_eigenval_eigenvec(num_eigv=num_eigv)
        full_width = DIM_LANDSCAPE[0]
        fig, ax = plt.subplots(len(self.tau_array), num_eigv, sharey="row",
                               figsize=(full_width, full_width/num_eigv*len(self.tau_array)))
        vmin = np.min(tau_eigenvec)
        vmax = np.max(tau_eigenvec)
        for i, tau in enumerate(self.tau_array):
            for j in range(num_eigv):
                to_plot = tau_eigenvec[i, :, j]
                to_plot = to_plot.reshape(self.energy.size)
                ax[i][j].imshow(to_plot, cmap="RdBu_r", vmin=vmin, vmax=vmax)
                ax[0][j].set_title(f"Eigenvector {j + 1}", fontsize=7)
                ax[i][0].set_ylabel(f"tau = {tau}", fontsize=7)
                ax[i][j].axes.get_xaxis().set_visible(False)
                ax[i][j].set_yticklabels([])
                #ax[i][j].axes.get_yaxis().set_visible(False)
        fig.savefig(self.images_path + f"eigenvectors_{self.images_name}.png", bbox_inches='tight', dpi=1200)
        plt.close()

    def visualize_its(self, num_eigv: int = 6, rates_eigenvalues=None):
        """
        Plot iterative timescales.

        Args:
            num_eigv: how many eigenvalues/timescales to plot
        """
        tau_eigenvals, tau_eigenvec = self.get_eigenval_eigenvec(num_eigv=num_eigv, which="LM")
        tau_eigenvals = tau_eigenvals.T
        fig2, ax2 = plt.subplots(1, 1)
        colors = ["blue", "red", "green", "orange", "black", "yellow", "purple", "pink"]
        for j in range(1, num_eigv):
            tau_filter = self.tau_array
            to_plot = -self.tau_array / np.log(np.abs(tau_eigenvals[j, :]))
            ax2.plot(tau_filter, to_plot, label=f"its {j}", color=colors[j])
        #if np.any(rates_eigenvalues):
        #    for j in range(1, len(rates_eigenvalues)):
        #        ax2.plot(self.tau_array, [-1/rates_eigenvalues[j] for _ in self.tau_array], color="black", ls="--")
        ax2.legend()
        fig2.savefig(self.images_path + f"implied_timescales_{self.images_name}.png", bbox_inches='tight', dpi=1200)
        plt.close()

    def visualize_transition_matrices(self):
        """
        Plot the transition matrices corresponding to different taus.
        """
        if not np.any(self.transition_matrices):
            self.get_transitions_matrix()
        with plt.style.context(['stylesheets/not_animation.mplstyle']):
            fig, ax = plt.subplots(1, len(self.transition_matrices), sharey="row")
        vmax = np.max(self.transition_matrices)
        vmin = np.min(self.transition_matrices)
        for i, tm in enumerate(self.transition_matrices):
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
        with plt.style.context(['stylesheets/not_animation.mplstyle', 'stylesheets/maze_style.mplstyle']):
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
        with plt.style.context(['stylesheets/not_animation.mplstyle']):
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
        with plt.style.context(['stylesheets/not_animation.mplstyle']):
            plt.hist(energies, weights=population, histtype='step')
            plt.savefig(self.images_path + f"population_per_energy_{self.images_name}.png")
            plt.close()

    def visualize_trajectory(self):
        """
        Plot the points visited by the trajectory.
        """
        with plt.style.context(['stylesheets/not_animation.mplstyle']):
            plt.subplots(1, 1, figsize=(8, 10))
            plt.scatter(self.traj_y, self.traj_x, marker="o", c="black", s=1)
            plt.gca().invert_yaxis()
            plt.savefig(self.images_path + f"traj_{self.images_name}.png")
            plt.close()


if __name__ == '__main__':
    img_path = "simulation/Images/"
    my_energy = Energy(images_path=img_path, images_name="energy")
    my_maze = Maze((7, 9), images_path=img_path, no_branching=True, edge_is_wall=False, animate=False)
    #my_energy.from_potential(size=(30, 30))
    my_energy.from_maze(my_maze, add_noise=True)
    my_energy.visualize()
    my_energy.visualize_boltzmann()
    my_energy.visualize_eigenvectors(num=6)
    my_energy.visualize_eigenvectors_in_maze(num=6)
    e_eigval, e_eigvec = my_energy.get_eigenval_eigenvec(8, which="SM")
    my_simulation = Simulation(my_energy, images_path=img_path, m=1, friction=20)
    #TODO: do a test for different dt
    my_simulation.integrate(N=int(1e7), dt=0.001, save_trajectory=True)
    my_simulation.visualize_hist_2D()
    my_simulation.visualize_sim_Boltzmann()
    my_simulation.visualize_population_per_energy()
    my_simulation.visualize_trajectory()
    my_simulation.get_transitions_matrix()
    s_eigval, s_eigvec = my_simulation.get_eigenval_eigenvec(8)
    my_simulation.visualize_transition_matrices()
    my_simulation.visualize_eigenvec(8)
    my_simulation.visualize_its(num_eigv=8, rates_eigenvalues=e_eigval)


