from create_mazes import Maze, AbstractEnergy
from explore_mazes import BFSExplorer
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib import colors
#from scipy.constants import k
from mpl_toolkits import mplot3d  # a necessary import

# for now we are unitless, I will decide on k = 1
# WARNING! REDEFINING BOLTZMANN CONSTANT!
k = 1


class Energy(AbstractEnergy):

    def __init__(self, images_path: str = "./", images_name: str = "energy"):
        # for now assume uniform, square cells so that geom. parameters = 1 and also diffusion coeff is 1
        super().__init__(None, 0.5, None, None, images_path, images_name)
        self.D = 1
        self.h = 1
        self.S = 1
        self.V = 1
        # and let´s assume room temperature (does that make sense?)
        self.T = 293  # 293K <==> 20°C
        # energy cutoff - let's see if it should be changed
        # will only have a value if energy created from maze
        self.underlying_maze = None
        self.rates_matrix = None
        # in preparation
        self.grid_x = None
        self.grid_y = None

    def from_maze(self, maze: Maze, add_noise: bool = True, factor_grid: int = 2):
        """
        A function for creating a energy surface from a Maze object.

        Args:
            maze: a Maze object that should be changed into an energy surface.
            add_noise: boolean, if False, the maze is not changed, if True, some of 0s in the maze -> -1 or -2
            factor_grid: int, how many times more points for interpolation than in original maze
                         (note: factor_grid > 2 produces very localized min/max)
        """
        # for now let's assume the maze to be 2D TODO: change this to enable any dimensionality
        # sparse grid
        size_x, size_y = complex(maze.size[0]), complex(maze.size[1])
        x_edges, y_edges = np.mgrid[-1:1:size_x, -1:1:size_y]
        # dense grid
        self.grid_x, self.grid_y = np.mgrid[-1:1:factor_grid*size_x, -1:1:factor_grid*size_y]
        z = maze.energies
        # change some random zeroes into -1 and -2
        if add_noise:
            for _ in range(round(0.05*maze.size[0]*maze.size[1])):
                cell = maze.find_random_accessible()
                z[cell] = -1
            for _ in range(round(0.01*maze.size[0]*maze.size[1])):
                cell = maze.find_random_accessible()
                z[cell] = -2
        self.underlying_maze = z
        m = max(maze.size)
        tck = interpolate.bisplrep(x_edges, y_edges, z, nxest=factor_grid*m, nyest=factor_grid*m, task=-1,
                                   tx=self.grid_x[:, 0], ty=self.grid_y[0, :])
        self.energies = interpolate.bisplev(self.grid_x[:, 0], self.grid_y[0, :], tck)
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
        energy_i = self.energies[cell_i]
        energy_j = self.energies[cell_j]
        return self.D * self.S / self.h / self.V * np.sqrt(np.exp((-energy_j + energy_i)/(k*self.T)))

    def _calculate_rates_matrix(self):
        """
        Explores the self.energies matrix using breadth-first search, t.i. starting in a random accessible
        (energy < energy_cutoff) cell and then exploring accessible neighbours of the cell. This creates an adjacency
        matrix. For every 1 in adj_matrix, the SqRA formula is applied to calculate the rate of adjacent cells and this
        value is saved in the ij-position of the rates_matrix. The diagonal elements of rates_matrix are determined
        so that the rowsum of rates_matrix = 0.
        """
        bfs_explorer = BFSExplorer(self)
        adj_matrix = bfs_explorer.get_adjacency_matrix()
        self.rates_matrix = np.zeros(adj_matrix.shape)
        #TODO: this can definitely be more efficient. Consider using sparse matrices and/or calculating rates
        # immediately during the exploration of the matrix.

        # get the adjacent elements
        for i in range(len(adj_matrix)):
            cell_i = bfs_explorer.get_cell_from_adj(i)
            for j in range(len(adj_matrix)):
                if adj_matrix[i, j] == 1:
                    cell_j = bfs_explorer.get_cell_from_adj(j)
                    self.rates_matrix[i, j] = self._calculate_rates_matrix_ij(cell_i, cell_j)
        # get the i == j elements
        for i, row in enumerate(self.rates_matrix):
            self.rates_matrix[i, i] = - np.sum(row)

    def get_rates_matix(self):
        """
        Get (and create if not yet created) the rate matrix of the energy surface.

        Returns:
            np.ndarray, rates matrix Q

        Raises:
            ValueError: if there are no self.energies
        """
        if not np.any(self.energies):
            raise ValueError("No energies present! First, create an energy surface (e.g. from a maze).")
        if not np.any(self.rates_matrix):
            self._calculate_rates_matrix()
        return self.rates_matrix

    ############################################################################
    # -----------------------   VISUALIZATION  ---------------------------------
    ############################################################################

    def visualize_underlying_maze(self, show: bool = True):
        """
        Visualization of the maze (with eventually added noise) from which the Energy object was created.

        Raises:
            Value error: if there is no self.underlying_maze (if self.from_maze has not been used).
        """
        if not np.any(self.underlying_maze):
            raise ValueError("No underlying maze present! This is only available for surfaces created from mazes.")
        lims = dict(cmap='RdBu_r', norm=colors.TwoSlopeNorm(vcenter=0), shading='auto')
        size_x, size_y = complex(self.underlying_maze.shape[0]), complex(self.underlying_maze.shape[1])
        x_edges, y_edges = np.mgrid[-1:1:size_x, -1:1:size_y]
        ax = plt.pcolormesh(x_edges, y_edges, self.underlying_maze, **lims)
        plt.colorbar()
        ax.figure.savefig(self.images_path + f"underlying_maze_{self.images_name}.png", bbox_inches='tight', dpi=1200)
        if show:
            plt.show()
        else:
            plt.close()

    def visualize(self, show: bool = True):
        """
        Visualizes the array self.energies.

        Raises:
            ValueError: if there are no self.energies
        """
        if not np.any(self.energies):
            raise ValueError("No energies present! First, create an energy surface (e.g. from a maze).")
        lims = dict(cmap='RdBu_r', norm=colors.TwoSlopeNorm(vcenter=0), shading='auto')
        ax = plt.pcolormesh(self.grid_x, self.grid_y, self.energies, **lims)
        plt.colorbar()
        ax.figure.savefig(self.images_path + f"energy_{self.images_name}.png", bbox_inches='tight', dpi=1200)
        if show:
            plt.show()
        else:
            plt.close()
        return ax

    def visualize_3d(self, show: bool = True):
        """
        Visualizes the array self.energies in 3D.

        Raises:
            ValueError: if there are no self.energies
        """
        if not np.any(self.energies):
            raise ValueError("No energies present! First, create an energy surface (e.g. from a maze).")
        ax = plt.axes(projection='3d')
        ax.plot_surface(self.grid_x, self.grid_y, self.energies, rstride=1, cstride=1,
                        cmap='RdBu_r', edgecolor='none')
        ax.figure.savefig(self.images_path + f"3D_energy_{self.images_name}.png", bbox_inches='tight', dpi=1200)
        if show:
            plt.show()
        else:
            plt.close()
        return ax

    def visualize_eigenvectors(self, show: bool = True, num: int = 3):
        if not np.any(self.energies):
            raise ValueError("No energies present! First, create an energy surface (e.g. from a maze).")
        if not np.any(self.rates_matrix):
            self._calculate_rates_matrix()
        w, v = np.linalg.eig(self.rates_matrix)
        fig, ax = plt.subplots(1, num, sharey="row")
        xs = np.linspace(-0.5, 0.5, num=len(v[0]))
        for i in range(num):
            ax[i].plot(xs, v[i], "black")
            ax[i].set_title(f"Eigenvector {i}")
        plt.savefig(self.images_path + f"eigenvectors_{self.images_name}.png", bbox_inches='tight', dpi=1200)
        if show:
            plt.show()
        else:
            plt.close()

    def visualize_rates_matrix(self, show: bool = True):
        """
        Visualizes the array self.rates_matrix.

        Raises:
            ValueError: if there are no self.energies
        """
        if not np.any(self.energies):
            raise ValueError("No energies present! First, create an energy surface (e.g. from a maze).")
        if not np.any(self.rates_matrix):
            self._calculate_rates_matrix()
        norm = colors.TwoSlopeNorm(vcenter=0)
        ax = plt.imshow(self.rates_matrix, cmap="RdBu_r", norm=norm)
        plt.colorbar()
        plt.title("Rates matrix")
        ax.figure.savefig(self.images_path + f"rates_matrix_{self.images_name}.png", bbox_inches='tight', dpi=1200)
        if show:
            plt.show()
        else:
            plt.close()

    def visualize_boltzmann(self, show: bool = True):
        """
        Visualizes both the energies and the Boltzmann distribution on that energy surface.

        Raises:
            ValueError: if there are no self.energies
        """
        if not np.any(self.energies):
            raise ValueError("No energies present! First, create an energy surface (e.g. from a maze).")
        if not np.any(self.rates_matrix):
            self._calculate_rates_matrix()
        fig, ax = plt.subplots(1, 2, sharey="row")
        boltzmanns = np.exp(-self.energies/(k*self.T))
        norm = colors.TwoSlopeNorm(vcenter=0)
        ax[0].imshow(self.energies, cmap="RdBu_r", norm=norm)
        ax[0].set_title("Energy")
        ax[1].imshow(boltzmanns, cmap="RdBu_r")
        ax[1].set_title("Boltzmann distribution")
        plt.savefig(self.images_path + f"boltzmann_{self.images_name}.png", bbox_inches='tight', dpi=1200)
        if show:
            plt.show()
        else:
            plt.close()


if __name__ == '__main__':
    img_path = "Images/"
    my_energy = Energy(images_path=img_path)
    my_maze = Maze((12, 15))
    my_energy.from_maze(my_maze, add_noise=True)
    my_energy.visualize_underlying_maze()
    #my_energy.visualize_boltzmann()
    my_energy.visualize()
    my_energy.visualize_3d()
    my_energy.get_rates_matix()
    my_energy.visualize_rates_matrix()
    my_energy.visualize_eigenvectors()

