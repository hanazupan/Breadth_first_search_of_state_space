from Maze.create_energies import Energy
from Maze.create_mazes import Maze
from scipy.interpolate import bisplev
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix

# DEFINING BOLTZMANN CONSTANT
kB = 0.008314463  # kJ/mol/K


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
        self.traj_x = None
        self.traj_y = None
        self.step_x = 2/self.histogram.shape[0]
        self.step_y = 2/self.histogram.shape[1]

    def integrate(self, dt: float = None, N: int = None, save_trajectory = False):
        """
        Implement the Euler–Maruyama method for integration of the trajectory.

        Args:
            dt: float, time step
            N: int, number of time steps
        """
        if dt:
            self.dt = dt
        if N:
            self.N = N
        self.histogram = np.zeros(self.energy.size)
        x_n = 2*np.random.random() - 1    # random between (-1, 1)
        y_n = 2*np.random.random() - 1    # random between (-1, 1)
        cell = self._point_to_cell((x_n, y_n))
        self.histogram[cell] += 1
        if save_trajectory:
            self.traj_x = np.zeros(N)
            self.traj_y = np.zeros(N)
        for n in tqdm(range(self.N)):
            if n % 1000 == 0:
                x_n = 2 * np.random.random() - 1  # random between (-1, 1)
                y_n = 2 * np.random.random() - 1  # random between (-1, 1)
            x_n, y_n = self._euler_maruyama(x_n, y_n)
            # histogram
            x_n, y_n = self._point_within_bound((x_n, y_n))
            cell = self._point_to_cell((x_n, y_n))
            self.histogram[cell] += 1
            # trajectory
            if save_trajectory:
                self.traj_x[n] = x_n
                self.traj_y[n] = y_n
        # normalizing the histogram
        self.histogram = self.histogram / self.N

    def _euler_maruyama(self, x_n, y_n) -> tuple:
        dV_dx = bisplev(x_n, y_n, self.spline, dx=1)
        dV_dy = bisplev(x_n, y_n, self.spline, dy=1)
        eta_x = np.random.normal()
        x_n = x_n - dV_dx * self.dt / self.m / self.friction + np.sqrt(2 * self.D) * eta_x * np.sqrt(self.dt)
        eta_y = np.random.normal()
        y_n = y_n - dV_dy * self.dt / self.m / self.friction + np.sqrt(2 * self.D) * eta_y * np.sqrt(self.dt)
        return x_n, y_n

    def _point_within_bound(self, point):
        x_n, y_n = point
        # calculate the corrected x, y between (-1, 1) with help of periodic boundaries
        dist_x_n = (abs(x_n) + 1) // 2
        x_n = x_n - np.sign(x_n) * 2 * dist_x_n
        dist_y_n = (abs(y_n) + 1) // 2
        y_n = y_n - np.sign(y_n) * 2 * dist_y_n
        return x_n, y_n

    def _point_to_cell(self, point):
        x_n, y_n = point
        # determine the cell of the histogram
        cell = int((x_n + 1) // self.step_x), int((y_n + 1) // self.step_y)
        return cell

    def _cell_to_index(self, cell):
        index = cell[0]
        for i in range(1, len(cell)):
            index = index * self.histogram.shape[i] + cell[i]
        return index

    def get_transitions_matrix(self, tau_array = np.array([10, 20, 50, 100, 200, 500])):
        if not np.any(self.traj_x):
            raise ValueError("No trajectories found! Check if using the setting save_trajectory=False.")
        all_cells = len(self.histogram.flatten())
        transition_matrices = np.zeros(shape=(len(tau_array), all_cells, all_cells))
        for tau_i, tau in enumerate(tau_array):
            filtered_traj_x = self.traj_x[::tau]
            filtered_traj_y = self.traj_x[::tau]
            previous_cell = self._point_to_cell((self.traj_x[0], self.traj_y[0]))
            for x,y in zip(filtered_traj_x[1:], filtered_traj_y[1:]):
                cell = self._point_to_cell((x, y))
                i = self._cell_to_index(previous_cell)
                j = self._cell_to_index(cell)
                transition_matrices[tau_i, i, j] += 1
        transition_matrices = csr_matrix(transition_matrices / transition_matrices.sum(axis=-1, keepdims=True))
        eigenval, eigenvec = eigs(transition_matrices[3], 6, which='LR')
        if eigenvec.imag.max() == 0 and eigenval.imag.max() == 0:
            eigenvec = eigenvec.real
            eigenval = eigenval.real
        # sort eigenvectors according to their eigenvalues
        idx = eigenval.argsort()[::-1]
        eigenval = eigenval[idx]
        eigenvec = eigenvec[:, idx]
        fig, ax = plt.subplots(1, 6, sharey="row")
        xs = np.linspace(-0.5, 0.5, num=len(eigenvec))
        for i in range(6):
            # plot eigenvectors corresponding to the largest (most negative) eigenvalues
            ax[i].plot(xs, eigenvec[:, i])
            ax[i].set_title(f"Eigenvector {i + 1}", fontsize=7)
            ax[i].axes.get_xaxis().set_visible(False)
        plt.savefig(self.images_path + f"eigenvectors_{self.images_name}.png", bbox_inches='tight', dpi=1200)
        plt.close()

    def visualize_hist_2D(self):
        fig, ax = plt.subplots(1, 1)
        cmap = cm.get_cmap("RdBu").copy()
        im = plt.imshow(self.histogram, cmap=cmap)
        fig.colorbar(im, ax=ax)
        ax.figure.savefig(self.images_path + f"hist_2D_{self.images_name}.png")
        plt.close()

    def visualize_sim_Boltzmann(self):
        list_of_cells = self.energy.explorer.get_sorted_accessible_cells()
        boltzmanns = np.array([self.histogram[cell] for cell in list_of_cells])
        with plt.style.context(['Stylesheets/not_animation.mplstyle']):
            fig, ax = plt.subplots(1, 1)
            ax.plot(boltzmanns)
            ax.set_xlabel("Accessible cell index")
            ax.set_ylabel("Relative cell population")
            ax.set_title("Boltzmann distribution")
            fig.savefig(self.images_path + f"sim_boltzmann_{self.images_name}.png")
            plt.close()

    def visualize_population_per_energy(self):
        x_len, y_len = self.energy.size
        energies = []
        population = []
        for x in range(x_len):
            for y in range(y_len):
                cell = (x, y)
                energies.append(self.energy.get_energy(cell))
                population.append(self.histogram[cell])
        plt.scatter(energies, population, marker="o", color="black")
        plt.savefig(self.images_path + f"population_per_energy_{self.images_name}.png")
        plt.close()

    def visualize_trajectory(self):
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        plt.scatter(self.traj_y, self.traj_x, marker="o", c="black")
        plt.gca().invert_yaxis()
        plt.savefig(self.images_path + f"traj_{self.images_name}.png")
        plt.close()


if __name__ == '__main__':
    img_path = "Simulation/Images/"
    my_energy = Energy(images_path=img_path, images_name="energy")
    my_maze = Maze((6, 7), images_path=img_path, no_branching=True, edge_is_wall=False, animate=False)
    my_energy.from_maze(my_maze, add_noise=True)
    my_energy.visualize()
    my_energy.visualize_boltzmann()
    my_simulation = Simulation(my_energy, images_path=img_path)
    my_simulation.integrate(N=int(1e7), dt=0.001, save_trajectory=True)
    my_simulation.visualize_hist_2D()
    my_simulation.visualize_sim_Boltzmann()
    my_simulation.visualize_population_per_energy()
    my_simulation.visualize_trajectory()
    my_simulation.get_transitions_matrix()


