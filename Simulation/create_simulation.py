import sys
sys.path.append("..")
from Maze.create_energies import Energy
from Maze.create_mazes import Maze
from scipy.interpolate import bisplev
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm
import math

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
        self.step_x = 2/self.histogram.shape[0]
        self.step_y = 2/self.histogram.shape[1]

    def gradient(self, x_n, y_n) -> float:
        """
        Obtain the first derivatives of the spline interpolation in the point (x_n, y_n).
        Args:
            x_n:
            y_n:

        Returns:

        """
        return bisplev(x_n, y_n, self.spline, dx=1, dy=1)

    def integrate(self, dt: float = None, N: int = None):
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
        self._add_to_hist((x_n, y_n))
        for n in tqdm(range(self.N)):
            # if n % 1000 == 0:
            #     # start a new trajectory every 1000 steps
            #     x_n = 2 * np.random.random() - 1  # random between (-1, 1)
            #     y_n = 2 * np.random.random() - 1  # random between (-1, 1)
            x_n, y_n = self._euler_maruyama(x_n, y_n)
            # calculate the correct x, y with periodic boundaries
            dist_x_n = (abs(x_n) + 1) // 2
            x_n = x_n - np.sign(x_n)*2*dist_x_n
            dist_y_n = (abs(y_n) + 1) // 2
            y_n = y_n - np.sign(y_n) * 2 * dist_y_n
            # histogram
            self._add_to_hist((x_n, y_n))

    def _euler_maruyama(self, x_n, y_n) -> tuple:
        dV = self.gradient(x_n, y_n)
        eta_x = np.random.normal()
        x_n = x_n - dV * self.dt / self.m / self.friction + np.sqrt(2 * self.D) * eta_x * np.sqrt(self.dt)
        eta_y = np.random.normal()
        y_n = y_n - dV * self.dt / self.m / self.friction + np.sqrt(2 * self.D) * eta_y * np.sqrt(self.dt)
        return x_n, y_n

    def _add_to_hist(self, point):
        x_n, y_n = point
        cell = int((x_n + 1)//self.step_x), int((y_n + 1)//self.step_y)
        self.histogram[cell] += 1

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
        with plt.style.context(['../Stylesheets/not_animation.mplstyle']):
            fig, ax = plt.subplots(1, 1)
            ax.plot(boltzmanns)
            ax.set_xlabel("Accessible cell index")
            ax.set_ylabel("Relative cell population")
            ax.set_title("Boltzmann distribution")
            fig.savefig(self.images_path + f"sim_boltzmann_{self.images_name}.png")
            plt.close()


if __name__ == '__main__':
    img_path = "Images/"
    my_energy = Energy(images_path=img_path, images_name="sim")
    my_maze = Maze((12, 15), images_path=img_path, no_branching=True, edge_is_wall=True, animate=False)
    my_energy.from_maze(my_maze, add_noise=True)
    my_energy.visualize()
    my_energy.visualize_boltzmann()
    my_simulation = Simulation(my_energy, images_path=img_path, m=10)
    my_simulation.integrate(N=int(1e8), dt=0.005)
    my_simulation.visualize_hist_2D()
    my_simulation.visualize_sim_Boltzmann()


