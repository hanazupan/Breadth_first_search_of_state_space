"""
In this file, molecular dynamics simulation on an Energy surface is performed.
"""

# internal imports
from maze.create_energies import Energy, EnergyFromPotential, EnergyFromMaze, Atom, EnergyFromAtoms
from maze.create_mazes import Maze
from constants import *
# standard library
import time
from datetime import datetime
# external imports
import numpy as np
from tqdm import tqdm


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
        # prepare empty objects
        self.histogram = np.zeros(self.energy.size)
        self.outside_hist = 0
        self.traj_x = []
        self.traj_y = []
        self.traj_cell = []
        # everything to do with the grid
        self.step_x = self.energy.grid_x[1, 0] - self.energy.grid_x[0, 0]
        self.step_y = self.energy.grid_y[0, 1] - self.energy.grid_y[0, 0]
        # xmin, xmax, ymin, ymax - NOT middle of cells, but actual min/max
        xmin = self.energy.grid_x[0, 0] - self.step_x / 2
        xmax = self.energy.grid_x[-1, -1] + self.step_x / 2
        ymin = self.energy.grid_y[0, 0] - self.step_y / 2
        ymax = self.energy.grid_y[-1, -1] + self.step_y / 2
        self.grid_edges = (xmin, xmax, ymin, ymax)
        self.save_information()

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
        np.save(PATH_HISTOGRAMS + f"histogram_{self.images_name}", self.histogram)
        self.traj_cell = np.array(self.traj_cell)
        np.save(PATH_TRAJECTORIES + f"cell_trajectory_{self.images_name}", self.traj_cell)
        if save_trajectory:
            np.savez(PATH_TRAJECTORIES + f"trajectory_x_y_{self.images_name}",
                     x=np.array(self.traj_x), y=np.array(self.traj_y))
        self.save_information()

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

    def path_to_summary(self):
        data_path = DATA_PATH + "simulation_summaries/"
        if type(self.energy) == EnergyFromPotential:
            data_path += "potentials/"
        elif type(self.energy) == EnergyFromMaze:
            data_path += "mazes/"
        elif type(self.energy) == EnergyFromAtoms:
            data_path += "atoms/"
        return data_path

    def save_information(self):
        with open(self.path_to_summary() + f"{self.images_name}_summary.txt", "w") as f:
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


if __name__ == '__main__':
    start_time = time.time()
    # ------------------- MAZE ------------------
    my_maze = Maze((8, 8), images_name="test20", no_branching=True, edge_is_wall=True)
    my_energy = EnergyFromMaze(my_maze, images_name=my_maze.images_name,
                               factor_grid=1, m=1, friction=30)
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
    e_eigval, e_eigvec = my_energy.get_eigenval_eigenvec(6, which="LR")
    my_simulation = Simulation(my_energy, images_name=my_energy.images_name)
    print(f"Performing a simulation named {my_simulation.images_name}.")
    to_save_trajectory = False
    my_simulation.integrate(N=int(1e6), dt=0.01, save_trajectory=to_save_trajectory)
    # ----------   TIMING -----------------
    end_time = time.time()
    duration = end_time - start_time
    hours = round(duration // 3600 % 24)
    minutes = round(duration // 60 % 60)
    seconds = round(duration % 60)
    print(f"Total time: {hours}h {minutes}min {seconds}s")

