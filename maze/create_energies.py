"""
In this file, Energy surfaces are created - from Mazes, double well potential or atoms positioned at various points.
This is implemented in subclasses EnergyFromMaze, EnergyFromPotential and EnergyFromAtoms.
Square root approximation is implemented using the rates matrices of those surfaces.
"""

# internal imports
from .create_mazes import Maze, AbstractEnergy
from .explore_mazes import BFSExplorer, DFSExplorer
from constants import *
# standard library
from abc import abstractmethod
from datetime import datetime
# external imports
import numpy as np
from scipy import interpolate
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix, diags, save_npz
from scipy.interpolate import bisplev


class Energy(AbstractEnergy):

    def __init__(self, images_path: str = "./", images_name: str = "energy", m: float = 1, friction: float = 10,
                 temperature: float = 293, grid_start: tuple = (0, 0), grid_end: tuple = (5, 5), cutoff: float = 5,
                 size: tuple = (20, 20)):
        """
        An Energy object has an array in which energies at the midpoint of cells are saved. It also has general
        thermodynamic/atomic properties (mass, friction, temperature) and geometric properties (area between cells,
        volume of cells, distances between cell centers).

        Args:
            images_path: where to save all resulting images
            images_name: an identifier of saved images
            m: mass of a particle
            friction: friction coefficient (for now assumed constant)
            temperature: temperature
        """
        # initiates AbstractEnergy that is also common with Maze objects
        super().__init__(energies=None, energy_cutoff=cutoff, size=size, images_path=images_path,
                         images_name=images_name)
        # and let´s assume room temperature
        self.temperature = temperature  # 293K <==> 20°C
        self.m = m
        self.friction = friction
        # diffusion coefficient
        self.D = kB * self.temperature / self.m / self.friction
        # empty objects
        self.rates_matrix = None
        self.explorer = None
        # prepare the grid
        self.grid_start = grid_start
        self.grid_end = grid_end
        self.grid_full_len = tuple(np.array(self.grid_end) - np.array(self.grid_start))
        self.grid_x, self.grid_y = self._prepare_grid()
        np.savez(PATH_ENERGY_GRIDS + f"grid_x_y_{self.images_name}", x=self.grid_x, y=self.grid_y)
        # prepare geometry
        self._prepare_geometry()
        self.save_information()

    def _prepare_geometry(self):
        # distances between centers of cells: (horizontal, vertical)
        self.hs = (self.grid_full_len[0] / self.size[0], self.grid_full_len[1] / self.size[1])
        # surface areas between cells: (horizontal, vertical)
        self.Ss = (self.grid_full_len[1] / self.size[1], self.grid_full_len[0] / self.size[0])
        # volumes of cells
        self.V = np.prod(np.array(self.hs))  # volume of cells

    def _prepare_grid(self, factor=1) -> tuple:
        cell_step_x = self.grid_full_len[0] / (self.size[0] * factor)
        cell_step_y = self.grid_full_len[1] / (self.size[1] * factor)
        start_x = self.grid_start[0] + cell_step_x / 2
        end_x = self.grid_end[0] - cell_step_x / 2
        start_y = self.grid_start[1] + cell_step_y / 2
        end_y = self.grid_end[1] - cell_step_y / 2
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
        # 0 if vertical neighbours, 1 if horizontal neighbours
        if self.are_neighbours(cell_i, cell_j, axis=0):
            n_dim = 0
        elif self.are_neighbours(cell_i, cell_j, axis=1):
            n_dim = 1
        else:
            raise TypeError(f"Trying to calculate Q_ij for non-neighbours {cell_i} and {cell_j}!")
        Q_ij = self.D * self.Ss[n_dim] / self.hs[n_dim] / self.V * np.sqrt(np.exp(-(energy_j - energy_i)/(
                kB*self.temperature)))
        limit = self.D * self.Ss[n_dim] / self.hs[n_dim] / self.V * np.sqrt(np.exp(10*self.energy_cutoff/(
                kB*self.temperature)))
        return min(Q_ij, limit)

    def _calculate_rates_matrix(self, selected_explorer: str):
        """
        Explores the self.energies matrix using breadth-first search, t.i. starting in a random accessible
        (energy < energy_cutoff) cell and then exploring accessible neighbours of the cell. This creates an adjacency
        matrix. For every 1 in adj_matrix, the SqRA formula is applied to calculate the rate of adjacent cells and this
        value is saved in the ij-position of the rates_matrix. The diagonal elements of rates_matrix are determined
        so that the row sum of rates_matrix = 0.

        Args:
            selected_explorer: if 'bfs' od 'dfs', use that explorer, else use the full state space
        """
        if selected_explorer == "bfs":
            self.explorer = BFSExplorer(self)
            adj_matrix = self.explorer.get_adjacency_matrix()
        elif selected_explorer == "dfs":
            self.explorer = DFSExplorer(self)
            adj_matrix = self.explorer.get_adjacency_matrix()
        else:
            self.explorer = "other"
            self.energy_cutoff = np.max(self.energies) + 1
            full_len = self.size[0]*self.size[1]
            ones = [1]*full_len
            mixed = [1]*(self.size[1] - 1)
            mixed.append(0)
            mixed = mixed * (self.size[0] + 1)
            inverse_mixed = [1]
            inverse_mixed.extend([0]*(self.size[1] - 1))
            inverse_mixed = inverse_mixed * (self.size[0] + 1)
            # if PBC, the off-seted diagonals of neighbours continue on another diagonal
            if self.pbc:
                adj_matrix = diags((mixed, ones, mixed, ones, inverse_mixed, inverse_mixed, ones, ones),
                                   offsets=(1, self.size[1], -1, -self.size[1],
                                            self.size[1] - 1, -self.size[0] + 1,
                                            full_len - self.size[0], - full_len + self.size[1]),
                                   shape=(full_len, full_len))
            else:
                adj_matrix = diags((mixed, ones, mixed, ones),
                                   offsets=(1, self.size[1], -1, -self.size[1]),
                                   shape=(full_len, full_len))
        self.adj_matrix = adj_matrix
        self.rates_matrix = np.zeros(adj_matrix.shape)
        # get the adjacent elements
        rows, cols = adj_matrix.nonzero()
        for r, c in zip(rows, cols):
            # important! Index in adj cell is not the same as index in self.energies because non-accessible
            # cells are skipped! Will not work if you use node_to_cell!
            # TODO: create an Energy method that gets a cell from index of accessible and vice versa
            if selected_explorer == "bfs" or selected_explorer == "dfs":
                cell_i = self.explorer.get_cell_from_adj(r)
                cell_j = self.explorer.get_cell_from_adj(c)
            else:
                cell_i = self.node_to_cell(r)
                cell_j = self.node_to_cell(c)

            self.rates_matrix[r, c] += self._calculate_rates_matrix_ij(cell_i, cell_j)
        # get the i == j elements
        for i, row in enumerate(self.rates_matrix):
            self.rates_matrix[i, i] = - np.sum(row)
        # save the rates matrix and connected info
        self.rates_matrix = csr_matrix(self.rates_matrix)
        with open(self.path_to_summary() + f"{self.images_name}_summary.txt", "a+", encoding='utf-8') as f:
            f.write(f"explorer type = {selected_explorer}\n")
            if selected_explorer == "bfs" or selected_explorer == "dfs":
                f.write(f"accessible cells = {self.explorer.get_sorted_accessible_cells()}\n")
        save_npz(PATH_ENERGY_RATES + f"rates_{self.images_name}", self.rates_matrix)

    ############################################################################
    # --------------------------   GETTERS  -----------------------------------
    ############################################################################

    def get_rates_matix(self, explorer: str = "bfs") -> np.ndarray:
        """
        Get (and create if not yet created) the rate matrix of the energy surface.

        Returns:
            np.ndarray, rates matrix Q

        Raises:
            ValueError: if there are no self.energies
        """
        if not self.explorer:
            self._calculate_rates_matrix(explorer)
        return self.rates_matrix

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
            self._calculate_rates_matrix("bfs")
        # left eigenvectors and eigenvalues
        eigenval, eigenvec = eigs(self.rates_matrix.transpose(copy=True), num, **kwargs)
        if eigenvec.imag.max() == 0 and eigenval.imag.max() == 0:
            eigenvec = eigenvec.real
            eigenval = eigenval.real
        # sort eigenvectors according to their eigenvalues
        idx = eigenval.argsort()[::-1]
        eigenval = eigenval[idx]
        eigenvec = eigenvec[:, idx]
        np.savez(PATH_ENERGY_EIGEN + f"eigv_{self.images_name}", eigenval=eigenval, eigenvec=eigenvec)
        return eigenval, eigenvec

    ############################################################################
    # --------------------------   SUMMARY  -----------------------------------
    ############################################################################

    def path_to_summary(self):
        data_path = PATH_ENERGY_SUMMARY
        if type(self) == EnergyFromPotential:
            data_path += "potentials/"
        elif type(self) == EnergyFromMaze:
            data_path += "mazes/"
        elif type(self) == EnergyFromAtoms:
            data_path += "atoms/"
        return data_path

    def save_information(self):
        path = self.path_to_summary()
        with open(path + f"{self.images_name}_summary.txt", "a+", encoding='utf-8') as f:
            describe_types = {EnergyFromMaze: "maze", EnergyFromPotential: "double_well", EnergyFromAtoms: "atoms",
                              Energy: "not determined"}
            f.write(f"# Simulation performed with the script simulation.create_energies.py.\n")
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
            f.write(f"hs = {self.hs}\n")
            f.write(f"Ss = {self.Ss}\n")
            f.write(f"V = {self.V}\n")


class EnergyFromMaze(Energy):

    def __init__(self, maze: Maze, add_noise: bool = True, factor_grid: int = 2, images_path: str = PATH_IMG_MAZES,
                 images_name: str = "energy", m: float = 1, friction: float = 10, T: float = 293,
                 grid_start: tuple = (0, 0), grid_end: tuple = (5, 5), cutoff: float = 60):
        """
        Creating a energy surface from a 2D maze object.
        Grid x is the same for the first row, changes row for row.
        Grid y changes column for column.

        Args:
            maze: a maze object that should be changed into an energy surface.
            add_noise: boolean, if False, the maze is not changed, if True, some of 0s in the maze -> -1 or -2
            factor_grid: int, how many times more points for discretization than in original maze
        """
        super().__init__(images_path=images_path, images_name=images_name, m=m, friction=friction, temperature=T,
                         grid_start=grid_start, grid_end=grid_end, cutoff=cutoff, size=maze.size)
        # interpolation only available for 2D mazes
        if len(maze.size) != 2:
            raise ValueError("Maze does not have the right dimensionality.")
        z = maze.energies.copy()
        # change some random zeroes into -1 and -2
        z = z * 100   # TODO: test increasing the energy of walls
        if add_noise:
            for _ in range(int(0.05 * np.prod(maze.size))):
                cell = maze.find_random_accessible()
                z[cell] = -5
            for _ in range(int(0.04 * np.prod(maze.size))):
                cell = maze.find_random_accessible()
                z[cell] = -10
        self.underlying_maze = z
        np.save(PATH_ENERGY_SURFACES + f"underlying_maze_{self.images_name}", self.underlying_maze)
        m = max(maze.size)
        tck = interpolate.bisplrep(self.grid_x, self.grid_y, z, nxest=factor_grid * m, nyest=factor_grid * m, task=-1,
                                   tx=self.grid_x[:, 0], ty=self.grid_y[0, :])
        # WARNING! We change the size, so need to update geometry and the grid
        self.grid_x, self.grid_y = self._prepare_grid(factor=factor_grid)
        np.savez(PATH_ENERGY_GRIDS + f"grid_x_y_{self.images_name}", x=self.grid_x, y=self.grid_y)
        self.energies = interpolate.bisplev(self.grid_x[:, 0], self.grid_y[0, :], tck)
        self.size = self.energies.shape
        self._prepare_geometry()
        self.spline = tck
        np.save(PATH_ENERGY_SURFACES + f"surface_{self.images_name}", self.energies)
        with open(self.path_to_summary() + f"{self.images_name}_summary.txt", "a+", encoding='utf-8') as f:
            f.write(f"factor = {factor_grid}\n")
            arr1, arr2, arr3, num1, num2 = tuple(self.spline)
            f.write(f"spline = {tuple(arr1), tuple(arr2), tuple(arr3), num1, num2}\n")

    def get_x_derivative(self, point: tuple) -> float:
        return bisplev(point[0], point[1], self.spline, dx=1)  # do not change, this is correct

    def get_y_derivative(self, point: tuple) -> float:
        return bisplev(point[0], point[1], self.spline, dy=1)   # do not change, this is correct


class EnergyFromPotential(Energy):

    def __init__(self, size: tuple = (12, 16), images_path: str = PATH_IMG_POTENTIALS,
                 images_name: str = "energy", m: float = 1, friction: float = 10, T: float = 293,
                 grid_start: tuple = (-1.4, -1.4), grid_end: tuple = (1.4, 1.4), cutoff: float = 10):
        """
        Initiate an energy surface with a 2D potential well.
        Grid x is the same for the first row, changes row for row.
        Grid y changes column for column.
        """
        super().__init__(images_path=images_path, images_name=images_name, m=m, friction=friction, temperature=T,
                         grid_start=grid_start, grid_end=grid_end, cutoff=cutoff, size=size)
        self.energies = self.square_well(self.grid_x, self.grid_y)
        self.pbc = False
        np.save(PATH_ENERGY_SURFACES + f"surface_{self.images_name}", self.energies)

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

    def __init__(self, size: tuple, atoms: tuple, images_path: str = PATH_IMG_ATOMS,
                 images_name: str = "energy", m: float = 1, friction: float = 10, T: float = 293,
                 grid_start: tuple = (0, 0), grid_end: tuple = (12, 12)):
        """
        Initiate an energy surface with LJ potentials induced by atoms placed on the surface.
        Atoms must be placed on positions between 0 and size.
        Grid x is the same for the first row, changes row for row.
        Grid y changes column for column.
        """

        self.atoms = atoms
        self.epsilon = np.max([atom.epsilon for atom in atoms])
        # plotting is problematic if including only one atom and not prescribing where the grid starts and ends
        if len(self.atoms) < 2:
            raise AttributeError("Add at least two atoms!")
        # cutoff is 4* max epsilon
        energy_cutoff = np.max([4*atom.epsilon for atom in atoms])
        self.grid_edges = (grid_start[0], grid_end[0], grid_start[1], grid_end[1])
        super().__init__(images_path=images_path, images_name=images_name, m=m, friction=friction, temperature=T,
                         grid_start=grid_start, grid_end=grid_end, cutoff=energy_cutoff, size=size)
        # calculate energies by looping over contributions of all atoms and adding them up
        self.energies = np.zeros(self.size)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                # be sure to use the grid, not the cell index!
                point_x = self.grid_x[i, j]
                point_y = self.grid_y[i, j]
                self.energies[i, j] = self.get_full_potential((point_x, point_y))
        self.energies[self.energies > 4*self.epsilon] = 4*self.epsilon
        with open(self.path_to_summary() + f"{self.images_name}_summary.txt", "a+", encoding='utf-8') as f:
            f.write(f"atom_positions = {[tuple(atom.position) for atom in self.atoms]}\n")
            f.write(f"epsilons = {[atom.epsilon for atom in atoms]}\n")
            f.write(f"sigmas = {[atom.sigma for atom in atoms]}\n")
        np.save(PATH_ENERGY_SURFACES + f"surface_{self.images_name}", self.energies)

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


if __name__ == '__main__':
    img_path = "images/"
    # ------------------- ATOMS -----------------------
    # my_epsilon = 3
    # my_sigma = 5
    # atom_1 = Atom((3.3, 20.5), my_epsilon, my_sigma)
    # atom_2 = Atom((14.3, 9.3), my_epsilon, my_sigma-2)
    # atom_3 = Atom((5.3, 45.3), my_epsilon/5, my_sigma)
    # my_energy = EnergyFromAtoms((50, 60), (atom_1, atom_2, atom_3), grid_start=(-8, 5), grid_end=(20, 50),
    #                             images_name="atoms", images_path=img_path)
    # ------------------- MAZES -----------------------
    my_maze = Maze((9, 9), images_path=img_path, images_name="testing", no_branching=True, edge_is_wall=True)
    my_energy = EnergyFromMaze(my_maze, images_path=img_path, images_name="testing", friction=10)
    # ------------------- POTENTIAL -----------------------
    # my_energy = EnergyFromPotential((30, 20), images_path=img_path, images_name="potential", friction=10)
    # ------------------- EXPLORERS -----------------------
    # me = BFSExplorer(my_energy)
    # me.explore_and_animate()
    # me = DFSExplorer(my_energy)
    # me.explore_and_animate()
