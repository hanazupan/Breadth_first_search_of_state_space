"""
In this file, Energy surfaces are created - from Mazes, double well potential or atoms positioned at various points.
This is implemented in subclasses EnergyFromMaze, EnergyFromPotential and EnergyFromAtoms.
Square root approximation is implemented using the rates matrices of those surfaces.
"""

from abc import abstractmethod
from .create_mazes import Maze, AbstractEnergy
from .explore_mazes import BFSExplorer, DFSExplorer, DijkstraExplorer
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib import colors, cm
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
from scipy.interpolate import bisplev
import seaborn as sns
import pandas as pd
from datetime import datetime
from constants import DIM_LANDSCAPE, DIM_SQUARE, DIM_PORTRAIT
from mpl_toolkits import mplot3d  # a necessary import

# DEFINING BOLTZMANN CONSTANT
kB = 0.008314463  # kJ/mol/K


class Energy(AbstractEnergy):

    def __init__(self, images_path: str = "./", images_name: str = "energy", m: float = 1, friction: float = 10,
                 temperature: float = 293):
        """
        An Energy object has an array in which energies at the midpoint of cells are saved. It also has general
        thermodynamic/atomic properties (mass, friction, temperature) and geometric properties (area between cells,
        volume of cells, distances between cell centers).

        Args:
            images_path: where to save all resulting images
            images_name: an identifier of saved images
            m: mass of a particle
            friction: friction coefficient (for now assumed constant)
            T: temperature
        """
        # energy cutoff is generally 5, at least for mazes
        cutoff = 5
        # for now assume uniform, square cells so that geom. parameters = 1
        super().__init__(None, cutoff, None, None, images_path, images_name)
        self.h = 1
        self.S = 1
        self.V = 1
        # and let´s assume room temperature
        self.temperature = temperature  # 293K <==> 20°C
        self.m = m
        self.friction = friction
        # diffusion coefficient
        self.D = kB * self.temperature / self.m / self.friction
        # in preparation
        self.grid_start = 0
        self.grid_end = 5
        self.grid_full_len = self.grid_end - self.grid_start
        self.rates_matrix = None
        self.grid_x = None
        self.grid_y = None
        self.explorer = None

    def _prepare_grid(self, factor=1) -> tuple:
        cell_step_x = self.grid_full_len / (self.size[0] * factor)
        cell_step_y = self.grid_full_len / (self.size[1] * factor)
        start_x = self.grid_start + cell_step_x / 2
        end_x = self.grid_end - cell_step_x / 2
        start_y = self.grid_start + cell_step_y / 2
        end_y = self.grid_end - cell_step_y / 2
        size_x, size_y = complex(self.size[0]), complex(self.size[1])
        return np.mgrid[start_x:end_x:factor*size_x, start_y:end_y:factor*size_y]

    @abstractmethod
    def get_x_derivative(self, point: tuple) -> float:
        """Obtain the derivative of the energy surface dV/dx at point (x, y)."""

    @abstractmethod
    def get_y_derivative(self, point: tuple) -> float:
        """Obtain the derivative of the energy surface dV/dy at point (x, y)."""

    ############################################################################
    # ------------------------   RATES MATRIX  ---------------------------------
    ############################################################################

    def _calculate_rates_matrix_ij(self, cell_i: tuple, cell_j: tuple) -> float:
        """
        Implements the formula for square root approximation:
        Q_ij = D*S/(h*V) * sqrt(pi_j/pi_i)

        Args:
            cell_i: tuple, coordinates of the i-th cell
            cell_j: tuple, coordinates of the j-th cell

        Returns:
            float, the ij-th element of the rates matrix Q
        """
        energy_i = self.get_energy(cell_i)
        energy_j = self.get_energy(cell_j)
        return min(self.D * self.S / self.h / self.V * np.sqrt(np.exp(-(energy_j - energy_i)/(kB*self.temperature))),
                   self.D * self.S / self.h / self.V * np.sqrt(np.exp(10*self.energy_cutoff/(kB*self.temperature))))

    def _calculate_rates_matrix(self):
        """
        Explores the self.energies matrix using breadth-first search, t.i. starting in a random accessible
        (energy < energy_cutoff) cell and then exploring accessible neighbours of the cell. This creates an adjacency
        matrix. For every 1 in adj_matrix, the SqRA formula is applied to calculate the rate of adjacent cells and this
        value is saved in the ij-position of the rates_matrix. The diagonal elements of rates_matrix are determined
        so that the rowsum of rates_matrix = 0.
        """
        self.explorer = BFSExplorer(self)
        adj_matrix = self.explorer.get_adjacency_matrix()
        self.adj_matrix = adj_matrix
        self.rates_matrix = np.zeros(adj_matrix.shape)
        # get the adjacent elements
        rows, cols = adj_matrix.nonzero()
        for r, c in zip(rows, cols):
            # important! Index in adj cell is not the same as index in self.energies because non-accessible
            # cells are skipped! Will not work if you use node_to_cell!
            # TODO: create an Energy method that gets a cell from index of accessible and vice versa
            cell_i = self.explorer.get_cell_from_adj(r)
            cell_j = self.explorer.get_cell_from_adj(c)
            # TODO: should here be += or =?
            self.rates_matrix[r, c] += self._calculate_rates_matrix_ij(cell_i, cell_j)
        # get the i == j elements
        for i, row in enumerate(self.rates_matrix):
            self.rates_matrix[i, i] = - np.sum(row)
        self.rates_matrix = csr_matrix(self.rates_matrix)

    def get_rates_matix(self) -> np.ndarray:
        """
        Get (and create if not yet created) the rate matrix of the energy surface.

        Returns:
            np.ndarray, rates matrix Q

        Raises:
            ValueError: if there are no self.energies
        """
        if not self.explorer:
            self._calculate_rates_matrix()
        return self.rates_matrix

    ############################################################################
    # --------------------------   GETTERS  -----------------------------------
    ############################################################################

    def get_boltzmann(self) -> np.ndarray:
        """
        Obtain a Boltzmann distribution for all accessible cells (ordered as nodes in the graph, meaning in the
        order they were discovered by the explorer.

        Returns:
            an array of floats, each containing Boltzmann distribution at the energy of one accessible cell
        """
        if not self.explorer:
            self._calculate_rates_matrix()
        list_of_cells = self.explorer.get_sorted_accessible_cells()
        boltzmanns = np.array([np.exp(-self.get_energy(cell) / (kB * self.temperature)) for cell in list_of_cells])
        return boltzmanns

    def get_eigenval_eigenvec(self, num: int = 10, **kwargs) -> tuple:
        """
        Obtain num (default 10) eigenvalues and eigenvectors of the rates matrix.

        Args:
            num: int, how many eigenvalues and eigenvectors to calculate (up to N-1)

        Returns:
            (eigenvals, eigenvecs) a tuple of eigenvalues and eigenvectors that are both numpy arrays

        Raises:
            ValueError if no self.energies
        """
        if not self.explorer:
            self._calculate_rates_matrix()
        # left eigenvectors and eigenvalues
        eigenval, eigenvec = eigs(self.rates_matrix.transpose(copy=True), num, **kwargs)
        if eigenvec.imag.max() == 0 and eigenval.imag.max() == 0:
            eigenvec = eigenvec.real
            eigenval = eigenval.real
        # sort eigenvectors according to their eigenvalues
        idx = eigenval.argsort()[::-1]
        eigenval = eigenval[idx]
        eigenvec = eigenvec[:, idx]
        return eigenval, eigenvec

    ############################################################################
    # -----------------------   VISUALIZATION  ---------------------------------
    ############################################################################

    def _add_colorbar(self, fig, ax, im):
        """
        Add a colorbar to the image.

        Args:
            ax: Axis
            fig: Figure
            im: a colorful plot, eg imshow
        """
        im_ratio = self.size[0] / self.size[1]
        fig.colorbar(im, ax=ax, fraction=0.05 * im_ratio, pad=0.04)

    def visualize(self, **kwargs):
        """
        Visualizes the array self.energies.

        Raises:
            ValueError: if there are no self.energies
        """
        with plt.style.context(['Stylesheets/maze_style.mplstyle', 'Stylesheets/not_animation.mplstyle']):
            fig, ax = plt.subplots(1, 1)
            cmap = cm.get_cmap("RdBu").copy()
            im = plt.imshow(self.energies, cmap=cmap, **kwargs)
            self._add_colorbar(fig, ax, im)
            ax.figure.savefig(self.images_path + f"{self.images_name}_energy.png")
            plt.close()
            return ax

    def visualize_3d(self, **kwargs):
        """
        Visualizes the array self.energies in 3D.

        Raises:
            ValueError: if there are no self.energies
        """
        with plt.style.context('Stylesheets/not_animation.mplstyle'):
            ax = plt.axes(projection='3d')
            ax.plot_surface(self.grid_x, self.grid_y, self.energies, rstride=1, cstride=1,
                            cmap='RdBu_r', edgecolor='none', **kwargs)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.figure.savefig(self.images_path + f"{self.images_name}_3D_energy.png")
            plt.close()
            return ax

    def visualize_eigenvectors_in_maze(self, num: int = 3, **kwargs):
        """
        Visualize the energy surface and the first num (default=3) eigenvectors as a 2D image in a maze.
        Black squares mean the cells are not accessible.

        Args:
            num: int, how many eigenvectors of rates matrix to show
            **kwargs: named arguments that can be passed to self.get_eigenval_eigenvec()
        """
        eigenval, eigenvec = self.get_eigenval_eigenvec(num=num, **kwargs)
        with plt.style.context(['Stylesheets/not_animation.mplstyle', 'Stylesheets/maze_style.mplstyle']):
            full_width = DIM_LANDSCAPE[0]
            fig, ax = plt.subplots(1, num, sharey="row", figsize=(full_width, full_width/num))
            cmap = cm.get_cmap("RdBu").copy()
            # cmap.set_over("black")
            # cmap.set_under("black")
            # ax[0].imshow(self.energies, cmap=cmap, norm=colors.TwoSlopeNorm(vcenter=0, vmax=self.energy_cutoff))
            # ax[0].set_title("Energy surface", fontsize=7, fontweight="bold")
            accesible = self.explorer.get_sorted_accessible_cells()
            len_acc = len(accesible)
            assert eigenvec.shape[0] == len_acc, "The length of the eigenvector should equal the num of accesible cells"
            vmax = np.max(eigenvec[:, :num+1])
            vmin = np.min(eigenvec[:, :num+1])
            for i in range(num):
                array = np.full(self.size, vmax+1)
                for index, cell in enumerate(accesible):
                    if eigenvec[index, 0] > 0:
                        array[cell] = eigenvec[index, i]
                    else:
                        array[cell] = - eigenvec[index, i]
                ax[i].imshow(array, cmap=cmap, norm=colors.TwoSlopeNorm(vmax=vmax, vcenter=0, vmin=vmin))
                ax[i].set_title(f"Eigenvector {i+1}", fontsize=7, fontweight="bold")
            plt.savefig(self.images_path + f"{self.images_name}_eigenvectors_sqra.png")
            plt.close()

    def visualize_eigenvalues(self):
        """
        Visualize the eigenvalues of rate matrix.
        """
        if not self.explorer:
            self._calculate_rates_matrix()
        num = self.rates_matrix.shape[0] - 2
        eigenval, eigenvec = self.get_eigenval_eigenvec(num=num, which="LR")
        with plt.style.context(['Stylesheets/not_animation.mplstyle']):
            fig, ax = plt.subplots(1, 1, figsize=DIM_LANDSCAPE)
            xs = np.linspace(0, 1, num=num)
            plt.scatter(xs, eigenval, s=5, c="black")
            for i, eigenw in enumerate(eigenval):
                plt.vlines(xs[i], eigenw, 0, linewidth=0.5)
            plt.hlines(0, 0, 1)
            ax.set_ylabel("Eigenvalues (SqRA)")
            ax.axes.get_xaxis().set_visible(False)
            plt.savefig(self.images_path + f"{self.images_name}_eigenvalues_sqra.png")
            plt.close()

    def visualize_rates_matrix(self):
        """
        Visualizes the array self.rates_matrix.

        Raises:
            ValueError: if there are no self.energies
        """
        if not self.explorer:
            self._calculate_rates_matrix()
        with plt.style.context(['Stylesheets/maze_style.mplstyle', 'Stylesheets/not_animation.mplstyle']):
            norm = colors.TwoSlopeNorm(vcenter=0)
            fig, ax = plt.subplots(1, 1, figsize=DIM_SQUARE)
            im = plt.imshow(self.rates_matrix.toarray(), cmap="RdBu_r", norm=norm)
            self._add_colorbar(fig, ax, im)
            ax.set_title("Rates matrix")
            fig.savefig(self.images_path + f"{self.images_name}_rates_matrix.png")
            plt.close()

    def save_information(self):
        with open(self.images_path + f"{self.images_name}_summary.txt", "w") as f:
            describe_types = {EnergyFromMaze: "maze", EnergyFromPotential: "double_well", EnergyFromAtoms: "atoms",
                              Energy: "not determined"}
            f.write(f"# Simulation performed with the script simulation.create_simulation.py.\n")
            f.write(f"# Time of execution: {datetime.now()}\n")
            f.write(f"# --------- PARAMETERS ----------\n")
            f.write(f"energy type = {describe_types[type(self)]}\n")
            f.write(f"energy cutoff = {self.energy_cutoff}\n")
            f.write(f"size = {self.size}\n")
            f.write(f"grid_start = {self.grid_start}\n")
            f.write(f"grid_end = {self.grid_end}\n")
            f.write(f"images path = {self.images_path}\n")
            f.write(f"images name = {self.images_name}\n")
            f.write(f"mass = {self.m}\n")
            f.write(f"friction = {self.friction}\n")
            f.write(f"temperature = {self.temperature}\n")
            f.write(f"D = {self.D}\n")


class EnergyFromMaze(Energy):

    def __init__(self, maze: Maze, add_noise: bool = True, factor_grid: int = 2, images_path: str = "./",
                 images_name: str = "energy", m: float = 1, friction: float = 10, T: float = 293):
        """
        Creating a energy surface from a 2D maze object.
        Grid x is the same for the first row, changes row for row.
        Grid y changes column for column.

        Args:
            maze: a maze object that should be changed into an energy surface.
            add_noise: boolean, if False, the maze is not changed, if True, some of 0s in the maze -> -1 or -2
            factor_grid: int, how many times more points for interpolation than in original maze
                         (note: factor_grid > 2 produces very localized min/max)
        """
        super().__init__(images_path, images_name, m, friction, T)
        # interpolation only available for 2D mazes
        if len(maze.size) != 2:
            raise ValueError("Maze does not have the right dimensionality.")
        self.size = maze.size
        # sparse grid
        x_edges, y_edges = self._prepare_grid()
        # dense grid
        self.grid_x, self.grid_y = self._prepare_grid(factor=factor_grid)
        z = maze.energies.copy()
        # change some random zeroes into -1 and -2
        if add_noise:
            for _ in range(int(0.05 * np.prod(maze.size))):
                cell = maze.find_random_accessible()
                z[cell] = -1
            for _ in range(int(0.04 * np.prod(maze.size))):
                cell = maze.find_random_accessible()
                z[cell] = -2
        z = z * 10
        self.underlying_maze = z
        m = max(maze.size)
        tck = interpolate.bisplrep(x_edges, y_edges, z, nxest=factor_grid * m, nyest=factor_grid * m, task=-1,
                                   tx=self.grid_x[:, 0], ty=self.grid_y[0, :])
        self.grid_x, self.grid_y = self._prepare_grid(factor=5)
        self.energies = interpolate.bisplev(self.grid_x[:, 0], self.grid_y[0, :], tck)
        self.size = self.energies.shape
        self.h = self.grid_full_len / self.size[0]
        self.S = self.grid_full_len / self.size[1]
        self.V = self.grid_full_len / self.size[0] * self.grid_full_len / self.size[1]
        self.spline = tck
        self.energy_cutoff = 8
        self.deltas = np.ones(len(self.size), dtype=int)

    def get_x_derivative(self, point: tuple) -> float:
        return bisplev(point[0], point[1], self.spline, dx=1)  # do not change, this is correct

    def get_y_derivative(self, point: tuple) -> float:
        return bisplev(point[0], point[1], self.spline, dy=1)   # do not change, this is correct

    def visualize_underlying_maze(self):
        """
        Visualization of the maze (with eventually added noise) from which the Energy object was created.

        Raises:
            Value error: if there is no self.underlying_maze (if self.from_maze has not been used).
        """
        with plt.style.context(['Stylesheets/maze_style.mplstyle', 'Stylesheets/not_animation.mplstyle']):
            lims = dict(cmap='RdBu_r', norm=colors.TwoSlopeNorm(vcenter=0))
            fig, ax = plt.subplots(1, 1)
            im = plt.imshow(self.underlying_maze, **lims)
            self._add_colorbar(fig, ax, im)
            ax.figure.savefig(self.images_path + f"{self.images_name}_underlying_maze.png")
            plt.close()


class EnergyFromPotential(Energy):

    def __init__(self, size: tuple = (12, 16), images_path: str = "./",
                 images_name: str = "energy", m: float = 1, friction: float = 10, T: float = 293):
        """
        Initiate an energy surface with a 2D potential well.
        Grid x is the same for the first row, changes row for row.
        Grid y changes column for column.
        """
        super().__init__(images_path, images_name, m, friction, T)
        self.size = size
        # making sure that the grid is set up in the middle of the cell
        self.grid_start = -1
        self.grid_end = 1
        self.grid_full_len = self.grid_end - self.grid_start
        self.grid_x, self.grid_y = self._prepare_grid()
        self.energies = self.square_well(self.grid_x, self.grid_y)
        self.energy_cutoff = 10
        self.deltas = np.ones(len(self.size), dtype=int)
        self.pbc = False
        self.h = self.grid_full_len / (self.size[0])
        self.S = self.grid_full_len / (self.size[1])
        self.V = self.grid_full_len / (self.size[0]) * self.grid_full_len / (self.size[1])

    def square_well(self, x, y, a=5, b=10):
        return a * (x ** 2 - 0.3) ** 2 + b * (y ** 2 - 0.5) ** 2

    def get_x_derivative(self, point: tuple) -> float:
        return 4 * 5 * point[0] * (point[0] ** 2 - 0.3)

    def get_y_derivative(self, point: tuple) -> float:
        return 4 * 10 * point[1] * (point[1] ** 2 - 0.5)


class Atom:

    def __init__(self, position: tuple, epsilon: float, sigma: float):
        """
        An instance of an atom with LJ potential. For now no real atoms, parameters must be explicitly given.

        Args:
            position: (x, y) coordinates of the atom on the grid in Angstrom
            epsilon: parameter of the LJ potential, the depth of the potential well
            sigma: parameter of the LJ potential, the distance at which the particle-particle potential energy is zero
        """
        self.epsilon = epsilon
        self.sigma = sigma
        self.position = position

    def _find_r(self, point: tuple, pbc_atom: tuple) -> float:
        """
        Find the distance between a point on grid and the, possibly mirrored, position of the atom (Euclidean distance).

        Args:
            point: coordinates of a point in grid space

        Returns:
            distance between the point and the atom
        """
        x, y = point
        x0, y0 = pbc_atom
        r = np.sqrt((x - x0)**2 + (y - y0)**2)
        return r

    def get_potential(self, point: tuple, grid_edges: tuple) -> float:
        """
        Find the LJ potential that the atom causes at a certain point.

        Args:
            point: coordinates of a point in grid space
            grid_edges: (xmin, xmax, ymin, ymax) sizes of the grid - for mirroring

        Returns:
            energy contribution of this atom at point
        """
        atom_mirrored = self.get_closest_mirror(point, grid_edges)
        r = self._find_r(point, atom_mirrored)
        return 4*self.epsilon*((self.sigma/r)**12 - (self.sigma/r)**6)

    def get_dV_dx(self, point: tuple, grid_edges: tuple) -> float:
        """
        Find the x derivative of the LJ potential that the atom causes at a certain point.

        Args:
            point: coordinates of a point in grid space
            grid_edges: (xmin, xmax, ymin, ymax) sizes of the grid - for mirroring

        Returns:
            dV/dx where V is the potential energy contribution of this atom at point
        """
        atom_mirrored = self.get_closest_mirror(point, grid_edges)
        x_a, y_a = atom_mirrored
        r = self._find_r(point, atom_mirrored)
        x, y = point
        return 4*self.epsilon*(-12*(self.sigma/r)**12/r + 6*(self.sigma/r)**6/r)*(x-x_a)/r

    def get_dV_dy(self, point: tuple, grid_edges: tuple) -> float:
        """
        Find the y derivative of the LJ potential that the atom causes at a certain point.

        Args:
            point: coordinates of a point in grid space
            grid_edges: (xmin, xmax, ymin, ymax) sizes of the grid - for mirroring

        Returns:
            dV/dy where V is the potential energy contribution of this atom at point
        """
        atom_mirrored = self.get_closest_mirror(point, grid_edges)
        x_a, y_a = atom_mirrored
        r = self._find_r(point, atom_mirrored)
        x, y = point
        return 4*self.epsilon*(-12*(self.sigma/r)**12/r + 6*(self.sigma/r)**6/r)*(y-y_a)/r

    def get_closest_mirror(self, point: tuple, grid_edges: tuple) -> tuple:
        """
        Instead of atom position, get an equivalent atom in one of neighbouring mirror images that has smaller x and y
        distance to the atom.

        Args:
            point: tuple, position at which we try to calculate something
            grid_edges: (xmin, xmax, ymin, ymax) sizes of the grid - for mirroring

        Returns:
            a point in the same or one of the mirroring simulation boxes
        """
        dx = self.position[0] - point[0]
        dy = self.position[1] - point[1]
        range_x = grid_edges[1] - grid_edges[0]
        range_y = grid_edges[3] - grid_edges[2]
        pos_x = self.position[0]
        pos_y = self.position[1]
        if dx > range_x * 0.5:
            pos_x = pos_x - range_x
        if dx <= -range_x * 0.5:
            pos_x = pos_x + range_x
        if dy > range_y * 0.5:
            pos_y = pos_y - range_y
        if dy <= -range_y * 0.5:
            pos_y = pos_y + range_y
        assert abs(pos_x - point[0]) <= abs(dx)
        assert abs(pos_y - point[1]) <= abs(dy)
        return pos_x, pos_y


class EnergyFromAtoms(Energy):

    def __init__(self, size: tuple, atoms: tuple, images_path: str = "./", grid_edges: tuple or None = None,
                 images_name: str = "energy", m: float = 1, friction: float = 10, T: float = 293):
        """
        Initiate an energy surface with LJ potentials induced by atoms placed on the surface.
        Atoms must be placed on positions between 0 and size.
        Grid x is the same for the first row, changes row for row.
        Grid y changes column for column.
        """
        super().__init__(images_path, images_name, m, friction, T)
        self.atoms = atoms
        self.epsilon = np.max([atom.epsilon for atom in atoms])
        # plotting is problematic if including only one atom and not prescribing where the grid starts and ends
        if len(self.atoms) < 2:
            if not grid_edges:
                raise AttributeError("Add at least two atoms or use grid_edges!")
        self.size = size
        # cutoff is 4* max epsilon
        self.energy_cutoff = np.max([4*atom.epsilon for atom in atoms])
        # create the grid, determine grid edges automatically if not given
        x_positions = [atom.position[0] for atom in self.atoms]
        y_positions = [atom.position[1] for atom in self.atoms]
        if grid_edges:
            xmin, xmax, ymin, ymax = grid_edges
        else:
            # if grid edges not given, create a grid a bit larger than what covers all atoms
            xmin = 0.9*np.min(x_positions)
            xmax = 1.1*np.max(x_positions)
            ymin = 0.9*np.min(y_positions)
            ymax = 1.1*np.max(y_positions)
        size_x, size_y = complex(self.size[0]), complex(self.size[1])
        self.grid_x, self.grid_y = np.mgrid[xmin:xmax:size_x, ymin:ymax:size_y]
        self.step_x = self.grid_x[1, 0] - self.grid_x[0, 0]
        self.step_y = self.grid_y[0, 1] - self.grid_y[0, 0]
        # xmin, xmax, ymin, ymax - NOT middle of cells, but actual min/max
        xmin = self.grid_x[0, 0] - self.step_x / 2
        xmax = self.grid_x[-1, -1] + self.step_x / 2
        ymin = self.grid_y[0, 0] - self.step_y / 2
        ymax = self.grid_y[-1, -1] + self.step_y / 2
        self.grid_edges = (xmin, xmax, ymin, ymax)
        # calculate energies by looping over contributions of all atoms and adding them up
        self.energies = np.zeros(self.size)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                # be sure to use the grid, not the cell index!
                point_x = self.grid_x[i, j]
                point_y = self.grid_y[i, j]
                self.energies[i, j] = self.get_full_potential((point_x, point_y))
        self.energies[self.energies > 4*self.epsilon] = 4*self.epsilon
        self.deltas = np.ones(len(self.size), dtype=int)

    def get_full_potential(self, point: tuple) -> float:
        full_potential = 0
        for atom in self.atoms:
            full_potential += atom.get_potential(point, self.grid_edges)
        return full_potential

    def get_x_derivative(self, point: tuple) -> float:
        """
        Obtain the x derivative of energy surface at a certain point by summing over contributions of all atoms
        while respecting the mirror image convention

        Args:
            point: (x, y) - needs to be coordinates in actual space (not cells!)

        Returns:
            value of dV/dx at point
        """
        total_derivative = 0
        for atom in self.atoms:
            total_derivative += atom.get_dV_dx(point, self.grid_edges)
        return total_derivative

    def get_y_derivative(self, point: tuple) -> float:
        """
        Obtain the y derivative of energy surface at a certain point by summing over contributions of all atoms
        while respecting the mirror image convention

        Args:
            point: (x, y) - needs to be coordinates in actual space (not cells!)

        Returns:
            value of dV/dy at point
        """
        total_derivative = 0
        for atom in self.atoms:
            total_derivative += atom.get_dV_dy(point, self.grid_edges)
        return total_derivative

    def visualize(self, **kwargs):
        """
        Visualizes the array self.energies. Use same color for all values above cutoff.

        Raises:
            ValueError: if there are no self.energies
        """
        with plt.style.context(['Stylesheets/not_animation.mplstyle']):
            fig, ax = plt.subplots(1, 1)
            df = pd.DataFrame(data=self.energies, index=self.grid_x[:, 0], columns=self.grid_y[0, :])
            sns.heatmap(df, cmap="RdBu_r", norm=colors.TwoSlopeNorm(vcenter=0, vmax=self.energy_cutoff), fmt='.2f',
                        square=True, ax=ax, yticklabels=[], xticklabels=[])
            # if you want labels: yticklabels=[f"{ind:.2f}" for ind in df.index]
            # xticklabels=[f"{col:.2f}" for col in df.columns]
            for atom in self.atoms:
                range_x_grid = self.grid_edges[1] - self.grid_edges[0]
                range_y_grid = self.grid_edges[3] - self.grid_edges[2]
                ax.scatter((atom.position[1]-self.grid_edges[2])*self.size[1]/range_y_grid,
                           (atom.position[0]-self.grid_edges[0])*self.size[0]/range_x_grid, marker="o", c="white")
            ax.figure.savefig(self.images_path + f"{self.images_name}_energy_with_cutoff.png")
            plt.close()
            return ax
    
    def visualize_3d(self, **kwargs):
        super(EnergyFromAtoms, self).visualize_3d(norm=colors.SymLogNorm(linthresh=1e-13, vmax=np.max(self.energies),
                                                                         vmin=-np.max(self.energies)))


if __name__ == '__main__':
    img_path = "images/"
    # ------------------- ATOMS -----------------------
    # my_epsilon = 3
    # my_sigma = 5
    # atom_1 = Atom((3.3, 20.5), my_epsilon, my_sigma)
    # atom_2 = Atom((14.3, 9.3), my_epsilon, my_sigma-2)
    # atom_3 = Atom((5.3, 45.3), my_epsilon/5, my_sigma)
    # my_energy = EnergyFromAtoms((50, 60), (atom_1, atom_2, atom_3), grid_edges=(-8, 20, 5, 50),
    #                             images_name="atoms", images_path=img_path)
    # ------------------- MAZES -----------------------
    my_maze = Maze((9, 9), images_path=img_path, images_name="testing", no_branching=True, edge_is_wall=True)
    my_energy = EnergyFromMaze(my_maze, images_path=img_path, images_name="testing", friction=10)
    print(my_energy.grid_x)
    my_maze.visualize()
    my_energy.visualize_underlying_maze()
    # ------------------- POTENTIAL -----------------------
    # my_energy = EnergyFromPotential((30, 20), images_path=img_path, images_name="potential", friction=10)
    # ------------------- EXPLORERS -----------------------
    # me = BFSExplorer(my_energy)
    # me.explore_and_animate()
    # me = DFSExplorer(my_energy)
    # me.explore_and_animate()
    # ------------------- GENERAL FUNCTIONS -----------------------
    my_energy.visualize()
    my_energy.visualize_3d()
    my_energy.visualize_rates_matrix()
    my_energy.visualize_eigenvectors_in_maze(num=6, which="LR")
    my_energy.visualize_eigenvalues()
    my_energy.save_information()

