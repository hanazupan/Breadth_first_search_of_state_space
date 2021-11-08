"""
In this file, Energy surfaces are created - from Mazes, double well potential or atoms positioned at various points.
This is implemented in subclasses EnergyFromMaze, EnergyFromPotential and EnergyFromAtoms.
Square root approximation is implemented using the rates matrices of those surfaces.
"""

from abc import abstractmethod
from .create_mazes import Maze, AbstractEnergy
from .explore_mazes import BFSExplorer
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib import colors, cm
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
from scipy.interpolate import bisplev
import math
from mpl_toolkits import mplot3d  # a necessary import

# DEFINING BOLTZMANN CONSTANT
kB = 0.008314463  # kJ/mol/K

DIM_LANDSCAPE = (7.25, 4.45)
DIM_PORTRAIT = (3.45, 4.45)
DIM_SQUARE = (4.45, 4.45)


class Energy(AbstractEnergy):

    def __init__(self, images_path: str = "./", images_name: str = "energy", m: float = 1, friction: float = 10,
                 T: float = 293):
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
        self.T = T  # 293K <==> 20°C
        self.m = m
        self.friction = friction
        # diffusion coefficient
        self.D = kB * self.T / self.m / self.friction
        # in preparation
        self.rates_matrix = None
        self.grid_x = None
        self.grid_y = None
        self.explorer = None

    def _prepare_grid(self, factor=1) -> tuple:
        cell_step_x = 2 / (self.size[0] * factor)
        cell_step_y = 2 / (self.size[1] * factor)
        start_x = -1 + cell_step_x / 2
        end_x = 1 - cell_step_x / 2
        start_y = -1 + cell_step_y / 2
        end_y = 1 - cell_step_y / 2
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
        return self.D * self.S / self.h / self.V * np.sqrt(np.exp(-(energy_j - energy_i)/(kB*self.T)))

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
        self.rates_matrix = np.zeros(adj_matrix.shape)
        # get the adjacent elements
        rows, cols = adj_matrix.nonzero()
        for r, c in zip(rows, cols):
            cell_i = self.node_to_cell(r)
            cell_j = self.node_to_cell(c)
            self.rates_matrix[r, c] = self._calculate_rates_matrix_ij(cell_i, cell_j)
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

        Raises:
            ValueError if no self.energies
        """
        if not self.explorer:
            self._calculate_rates_matrix()
        list_of_cells = self.explorer.get_sorted_accessible_cells()
        boltzmanns = np.array([np.exp(-self.get_energy(cell) / (kB * self.T)) for cell in list_of_cells])
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
        eigenval, eigenvec = eigs(self.rates_matrix.T, num, **kwargs)
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
            im = plt.imshow(self.energies, cmap='RdBu_r', **kwargs)
            self._add_colorbar(fig, ax, im)
            ax.figure.savefig(self.images_path + f"energy_{self.images_name}.png")
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
            ax.figure.savefig(self.images_path + f"3D_energy_{self.images_name}.png")
            plt.close()
            return ax

    def visualize_eigenvectors(self, num: int = 3, **kwargs):
        """
        Visualize the eigenvectors of rate matrix.

        Args:
            num: int, how many eigenvectors to display
        """
        eigenval, eigenvec = self.get_eigenval_eigenvec(num=num, **kwargs)
        # sort the eigenvectors according to value of eigenvalues
        with plt.style.context('Stylesheets/not_animation.mplstyle'):
            full_width = DIM_LANDSCAPE[0]
            fig, ax = plt.subplots(1, num, sharey="row", figsize=(full_width, full_width/num))
            xs = np.linspace(-0.5, 0.5, num=len(eigenvec))
            for i in range(num):
                # plot eigenvectors corresponding to the largest (most negative) eigenvalues
                ax[i].plot(xs, eigenvec[:, i])
                ax[i].set_title(f"Eigenvector {i+1}", fontsize=7)
                ax[i].axes.get_xaxis().set_visible(False)
            plt.savefig(self.images_path + f"eigenvectors_{self.images_name}.png", bbox_inches='tight', dpi=1200)
            plt.close()

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
            fig, ax = plt.subplots(1, num + 1, sharey="row", figsize=(full_width, full_width/(num+1)))
            cmap = cm.get_cmap("RdBu").copy()
            cmap.set_over("black")
            cmap.set_under("black")
            ax[0].imshow(self.energies, cmap=cmap, vmax=self.energy_cutoff)
            ax[0].set_title("Energy surface", fontsize=7)
            len_acc = len([1 for j in range(self.size[0]) for k in range(self.size[1]) if self.is_accessible((j, k))])
            assert eigenvec.shape[0] == len_acc, "The length of the eigenvector should equal the num of accesible cells"
            for i in range(1, num+1):
                array = np.full(self.size, np.max(eigenvec[:, i-1])+1)
                index = 0
                for j in range(self.size[0]):
                    for k in range(self.size[1]):
                        if self.is_accessible((j, k)):
                            array[j, k] = eigenvec[index, i-1]
                            index += 1
                ax[i].imshow(array, cmap=cmap, vmax=np.max(eigenvec[:, :num+1]), vmin=np.min(eigenvec[:, :num+1]))
                ax[i].set_title(f"Eigenvector {i}", fontsize=7)
            plt.savefig(self.images_path + f"eigenvectors_in_maze_{self.images_name}.png")
            plt.close()

    def visualize_eigenvalues(self):
        """
        Visualize the eigenvalues of rate matrix.
        """
        if not self.explorer:
            self._calculate_rates_matrix()
        num = self.rates_matrix.shape[0] - 2
        eigenval, eigenvec = self.get_eigenval_eigenvec(num=num)
        with plt.style.context(['Stylesheets/not_animation.mplstyle']):
            plt.subplots(1, 1, figsize=DIM_LANDSCAPE)
            xs = np.linspace(0, 1, num=num)
            plt.scatter(xs, eigenval, s=5, c="black")
            for i, eigenw in enumerate(eigenval):
                plt.vlines(xs[i], eigenw, 0, linewidth=0.5)
            plt.hlines(0, 0, 1)
            plt.title("Eigenvalues")
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.savefig(self.images_path + f"eigenvalues_{self.images_name}.png")
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
            fig.savefig(self.images_path + f"rates_matrix_{self.images_name}.png")
            plt.close()

    def visualize_boltzmann(self):
        """
        Visualizes both the energies and the Boltzmann distribution on that energy surface.
        """
        boltzmanns = self.get_boltzmann()
        with plt.style.context(['Stylesheets/not_animation.mplstyle']):
            fig, ax = plt.subplots(1, 1, figsize=DIM_LANDSCAPE)
            ax.plot(boltzmanns)
            ax.set_xlabel("Accessible cell index")
            ax.set_ylabel("Relative cell population")
            ax.set_title("Boltzmann distribution")
            fig.savefig(self.images_path + f"boltzmann_{self.images_name}.png")
            plt.close()


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
        z = maze.energies
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
        self.energies = interpolate.bisplev(self.grid_x[:, 0], self.grid_y[0, :], tck)
        self.size = self.energies.shape
        self.spline = tck
        self.energy_cutoff = 5
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
            ax.figure.savefig(self.images_path + f"underlying_maze_{self.images_name}.png")
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
        self.grid_x, self.grid_y = self._prepare_grid()
        self.energies = self.square_well(self.grid_x, self.grid_y)
        self.energy_cutoff = 10
        self.deltas = np.ones(len(self.size), dtype=int)
        self.pbc = False
        self.h = 2 / (self.size[0])
        self.S = 2 / (self.size[1])
        self.V = 2 / (self.size[0]) * 2 / (self.size[1])

    def square_well(self, x, y, a=5, b=10):
        return a * (x ** 2 - 0.3) ** 2 + b * (y ** 2 - 0.5) ** 2

    def get_x_derivative(self, point: tuple) -> float:
        return 4 * 5 * point[0] * (point[0] ** 2 - 0.3)

    def get_y_derivative(self, point: tuple) -> float:
        return 4 * 10 * point[1] * (point[1] ** 2 - 0.5)


class Atom:

    def __init__(self, position: tuple, epsilon: float, sigma: float):
        self.epsilon = epsilon
        self.sigma = sigma
        self.position = position

    def _find_r(self, point: tuple) -> float:
        x, y = point
        x0, y0 = self.position
        r = np.sqrt((x - x0)**2 + (y - y0)**2)
        return r

    def get_potential(self, point: tuple) -> float:
        r = self._find_r(point)
        return 4*self.epsilon*((self.sigma/r)**12 - (self.sigma/r)**6)

    def get_dV_dx(self, point: tuple) -> float:
        r = self._find_r(point)
        return 4*self.epsilon*(-12*self.sigma**12*r**(-13) + 6*self.sigma**6*r**(-7))*r**(-0.5)*point[0]

    def get_dV_dy(self, point: tuple) -> float:
        r = self._find_r(point)
        return 4*self.epsilon*(-12*self.sigma**12*r**(-13) + 6*self.sigma**6*r**(-7))*r**(-0.5)*point[1]

    def plot_atom(self):
        x0, y0 = self.position
        xs = np.linspace(x0-1, x0+1, num=100)
        ys = np.linspace(y0-1, y0+1, num=100)
        array = np.zeros((100, 100))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                array[i, j] = self.get_potential((x, y))
        plt.imshow(array, norm=colors.LogNorm())
        plt.colorbar()
        plt.savefig(f"single_atom.png")


class EnergyFromAtoms(Energy):

    def __init__(self, size: tuple, atoms: tuple, images_path: str = "./",
                 images_name: str = "energy", m: float = 1, friction: float = 10, T: float = 293):
        """
        Initiate an energy surface with LJ potentials induced by atoms placed on the surface.
        Atoms must be placed on positions between 0 and size.
        Grid x is the same for the first row, changes row for row.
        Grid y changes column for column.
        """
        super().__init__(images_path, images_name, m, friction, T)
        self.atoms = atoms
        self.size = size
        self.energies = np.zeros(self.size)
        for atom in atoms:
            # check whether atoms within size
            for i in range(len(size)):
                if atom.position[i] > size[i] or atom.position[i] < 0:
                    raise IndexError(f"Atoms must positioned within the energy surface of the size {self.size}!")
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    self.energies[i, j] += atom.get_potential((i, j))
        self.energy_cutoff = 10
        self.deltas = np.ones(len(self.size), dtype=int)
        self.grid_x, self.grid_y = np.mgrid[0:self.size[0]:1, 0:self.size[1]:1]

    def get_x_derivative(self, point: tuple) -> float:
        total_derivative = 0
        for atom in self.atoms:
            total_derivative += atom.get_dV_dx(point)
        return total_derivative

    def get_y_derivative(self, point: tuple) -> float:
        total_derivative = 0
        for atom in self.atoms:
            total_derivative += atom.get_dV_dy(point)
        return total_derivative
    
    def visualize(self):
        super(EnergyFromAtoms, self).visualize(norm=colors.LogNorm())
    
    def visualize_3d(self, **kwargs):
        super(EnergyFromAtoms, self).visualize_3d(norm=colors.LogNorm())


if __name__ == '__main__':
    img_path = "images/"
    atom_1 = Atom((3.1, 10.5), 3.18*1.6022e-22, 3)
    atom_2 = Atom((5.3, 5.3), 3.18*1.6022e-22, 3)
    my_energy = EnergyFromAtoms((15, 15), (atom_2, atom_1), images_name="testing", images_path=img_path)
    #my_maze = Maze((30, 20), images_path=img_path, images_name="testing", no_branching=True, edge_is_wall=False)
    #my_energy = EnergyFromMaze(my_maze, images_path=img_path, images_name="testing", friction=10)
    #my_maze.visualize()
    #my_energy.visualize_underlying_maze()
    #my_energy = EnergyFromPotential((30, 20), images_path=img_path, images_name="testing", friction=10)
    my_energy.visualize_boltzmann()
    my_energy.visualize()
    my_energy.visualize_3d()
    my_energy.visualize_rates_matrix()
    my_energy.visualize_eigenvectors(num=6, which="LR", sigma=0)
    my_energy.visualize_eigenvectors_in_maze(num=6, which="LR", sigma=0)
    my_energy.visualize_eigenvalues()


