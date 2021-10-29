from .create_mazes import Maze, AbstractEnergy
from .explore_mazes import BFSExplorer, DFSExplorer
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib import colors, cm
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
from mpl_toolkits import mplot3d  # a necessary import

# DEFINING BOLTZMANN CONSTANT
kB = 0.008314463  # kJ/mol/K

DIM_LANDSCAPE = (7.25, 4.45)
DIM_PORTRAIT = (3.45, 4.45)
DIM_SQUARE = (4.45, 4.45)


class Energy(AbstractEnergy):

    def __init__(self, images_path: str = "./", images_name: str = "energy", m: float = 1, friction: float = 10):
        # for now assume uniform, square cells so that geom. parameters = 1 and also diffusion coeff is 1
        super().__init__(None, 5, None, None, images_path, images_name)
        self.h = 1
        self.S = 1
        self.V = 1
        # and let´s assume room temperature (does that make sense?)
        self.T = 293  # 293K <==> 20°C
        self.m = m
        self.friction = friction
        self.D = kB * self.T / self.m / self.friction
        # energy cutoff - let's see if it should be changed
        # will only have a value if energy created from maze
        self.underlying_maze = None
        self.rates_matrix = None
        # in preparation
        self.grid_x = None
        self.grid_y = None
        self.explorer = None
        self.spline = None

    def from_potential(self, size=(12, 16)):
        """
        For testing - initiate an energy surface with a 2D potential well.
        """
        self.size = size[::-1]
        cell_step_x = 2 / self.size[0]
        cell_step_y = 2 / self.size[1]
        start_x = -1 + cell_step_x/2
        end_x = 1 - cell_step_x/2
        start_y = -1 + cell_step_y / 2
        end_y = 1 - cell_step_y / 2
        size_x, size_y = complex(self.size[0]), complex(self.size[1])
        self.grid_x, self.grid_y = np.mgrid[start_x:end_x:size_x, start_y:end_y:size_y]

        def square_well(x, y, a=5, b=10):
            return a*(x**2 - 0.3)**2 + b*(y**2 - 0.5)**2

        self.dV_dx = lambda x: 4*5*x*(x**2 - 0.3)
        self.dV_dy = lambda y: 4*10*y*(y**2 - 0.5)
        self.energies = square_well(self.grid_y, self.grid_x)
        self.energy_cutoff = 10
        self.deltas = np.ones(len(self.size), dtype=int)
        self.pbc = False
        self.h = cell_step_x
        self.S = cell_step_y
        self.V = cell_step_x*cell_step_y

    def from_maze(self, maze: Maze, add_noise: bool = True, factor_grid: int = 2):
        """
        A function for creating a energy surface from a 2D maze object.

        Args:
            maze: a maze object that should be changed into an energy surface.
            add_noise: boolean, if False, the maze is not changed, if True, some of 0s in the maze -> -1 or -2
            factor_grid: int, how many times more points for interpolation than in original maze
                         (note: factor_grid > 2 produces very localized min/max)
        """
        # interpolation only available for 2D mazes
        if len(maze.size) != 2:
            raise ValueError("maze does not have the right dimensionality.")
        # sparse grid
        size_x, size_y = complex(maze.size[0]), complex(maze.size[1])
        x_edges, y_edges = np.mgrid[-1:1:size_x, -1:1:size_y]
        # could also do that to work with any dimensionality in case the interpolation scheme also changes
        # dense_size = tuple(np.linspace(-1, 1, num=factor_grid*ms) for ms in maze.size)
        # grid_dense = np.meshgrid(*dense_size)
        # dense grid
        self.grid_x, self.grid_y = np.mgrid[-1:1:factor_grid*size_x, -1:1:factor_grid*size_y]
        z = maze.energies * 10
        # change some random zeroes into -1 and -2
        if add_noise:
            for _ in range(int(0.05*np.prod(maze.size))):
                cell = maze.find_random_accessible()
                z[cell] = -1
            for _ in range(int(0.04*np.prod(maze.size))):
                cell = maze.find_random_accessible()
                z[cell] = -2
        self.underlying_maze = z
        m = max(maze.size)
        tck = interpolate.bisplrep(x_edges, y_edges, z, nxest=factor_grid*m, nyest=factor_grid*m, task=-1,
                                   tx=self.grid_x[:, 0], ty=self.grid_y[0, :])
        self.energies = interpolate.bisplev(self.grid_x[:, 0], self.grid_y[0, :], tck)
        self.energy_cutoff = 5
        self.spline = tck
        self.size = self.energies.shape
        self.deltas = np.ones(len(self.size), dtype=int)

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
        if not np.any(self.energies):
            raise ValueError("No energies present! First, create an energy surface (e.g. from a maze).")
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
        self.explorer.explore_and_animate()
        ex_d = DFSExplorer(self)
        ex_d.explore_and_animate()
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
        if not np.any(self.energies):
            raise ValueError("No energies present! First, create an energy surface (e.g. from a maze).")
        if not self.explorer:
            self._calculate_rates_matrix()
        return self.rates_matrix

    ############################################################################
    # --------------------------   GETTERS  -----------------------------------
    ############################################################################

    def get_spline(self):
        if not np.any(self.energies):
            raise ValueError("No splines present! The interpolation never occured.")
        return self.spline

    def get_boltzmann(self) -> np.ndarray:
        """
        Obtain a Boltzmann distribution for all accessible cells (ordered as nodes in the graph, meaning in the
        order they were discovered by the explorer.

        Returns:
            an array of floats, each containing Boltzmann distribution at the energy of one accessible cell

        Raises:
            ValueError if no self.energies
        """
        if not np.any(self.energies):
            raise ValueError("No energies present! First, create an energy surface (e.g. from a maze).")
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
        if not np.any(self.energies):
            raise ValueError("No energies present! First, create an energy surface (e.g. from a maze).")
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

    def visualize_underlying_maze(self):
        """
        Visualization of the maze (with eventually added noise) from which the Energy object was created.

        Raises:
            Value error: if there is no self.underlying_maze (if self.from_maze has not been used).
        """
        if not np.any(self.underlying_maze):
            raise ValueError("No underlying maze present! This is only available for surfaces created from mazes.")
        with plt.style.context(['Stylesheets/maze_style.mplstyle', 'Stylesheets/not_animation.mplstyle']):
            lims = dict(cmap='RdBu_r', norm=colors.TwoSlopeNorm(vcenter=0))
            fig, ax = plt.subplots(1, 1)
            im = plt.imshow(self.underlying_maze, **lims)
            self._add_colorbar(fig, ax, im)
            ax.figure.savefig(self.images_path + f"underlying_maze_{self.images_name}.png")
            plt.close()

    def visualize(self):
        """
        Visualizes the array self.energies.

        Raises:
            ValueError: if there are no self.energies
        """
        if not np.any(self.energies):
            raise ValueError("No energies present! First, create an energy surface (e.g. from a maze).")
        with plt.style.context(['Stylesheets/maze_style.mplstyle', 'Stylesheets/not_animation.mplstyle']):
            lims = dict(cmap='RdBu_r')
            fig, ax = plt.subplots(1, 1)
            im = plt.imshow(self.energies, **lims)
            self._add_colorbar(fig, ax, im)
            ax.figure.savefig(self.images_path + f"energy_{self.images_name}.png")
            plt.close()
            return ax

    def visualize_3d(self):
        """
        Visualizes the array self.energies in 3D.

        Raises:
            ValueError: if there are no self.energies
        """
        if not np.any(self.energies):
            raise ValueError("No energies present! First, create an energy surface (e.g. from a maze).")
        with plt.style.context('Stylesheets/not_animation.mplstyle'):
            ax = plt.axes(projection='3d')
            ax.plot_surface(self.grid_x, self.grid_y, self.energies, rstride=1, cstride=1,
                            cmap='RdBu_r', edgecolor='none')
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
            show: bool, whether to show the image
            num: int, how many eigenvectors of rates matrix to show

        Returns:

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
            for i in range(1, num+1):
                array = np.full(self.size, np.max(eigenvec[:, i-1])+1)
                index = 0
                for j in range(self.size[0]):
                    for k in range(self.size[1]):
                        array[j, k] = eigenvec[index, i-1]
                        index += 1
                ax[i].imshow(array, cmap=cmap, vmax=np.max(eigenvec[:, :num+1]), vmin=np.min(eigenvec[:, :num+1]))
                ax[i].set_title(f"Eigenvector {i}", fontsize=7)
            plt.savefig(self.images_path + f"eigenvectors_in_maze_{self.images_name}.png")
            plt.close()

    def visualize_eigenvalues(self):
        """
        Visualize the eigenvalues of rate matrix.

        Args:
            show: bool, whether to display the image
        """
        if not np.any(self.energies):
            raise ValueError("No energies present! First, create an energy surface (e.g. from a maze).")
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
        if not np.any(self.energies):
            raise ValueError("No energies present! First, create an energy surface (e.g. from a maze).")
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


if __name__ == '__main__':
    img_path = "images/"
    my_energy = Energy(images_path=img_path)
    my_maze = Maze((24, 20), images_path=img_path, no_branching=False, edge_is_wall=False, animate=True)
    #my_maze.visualize()
    my_energy.from_potential(size=(10, 10))
    #my_energy.from_maze(my_maze, add_noise=True)
    #my_energy.visualize_underlying_maze()
    #my_energy.visualize_boltzmann()
    my_energy.visualize()
    #my_energy.visualize_3d()
    #my_energy.get_rates_matix()
    my_energy.visualize_rates_matrix()
    my_energy.visualize_eigenvectors(num=6, which="SM")
    my_energy.visualize_eigenvectors_in_maze(num=6, which="SM")
    #my_energy.visualize_eigenvalues()


