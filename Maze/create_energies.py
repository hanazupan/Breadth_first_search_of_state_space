from abc import ABC
import networkx as nx
from create_mazes import Maze
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from collections.abc import Sequence
from mpl_toolkits import mplot3d


class AbstractEnergy(ABC):
    """
    NOT USED YET
    An object with energy data saved in nodes that are connected with edges of possibly different lengths.
    Each node has a property of energy. There is some energy cutoff that makes some cells accessible and others not.
    """

    def __init__(self, graph: nx.Graph, energy_cutoff: float):
        """
        Initialize some properties of all Energy objects.

        Args:
            graph: stores cells with unique IDs, properties of energy and edges between them
            energy_cutoff: cells with energy strictly below that value are accessible
        """
        self.graph = graph
        self.energy_cutoff = energy_cutoff

    def get_energy(self, node: int) -> float:
        """Given a node identifier, get or calculate its energy."""
        return self.graph[node]["energy"]

    def get_neighbours(self, node: int) -> Sequence:
        """Given a node identifier, get identifiers of its neighbours"""
        neig_iterator = self.graph.neighbors(node)
        for neig in neig_iterator:
            yield neig

    def get_accessible_neighbours(self, node: int) -> Sequence:
        """Same as get_neighbours but filters out non-accessible neighbours."""
        for n in self.get_neighbours(node):
            if self.is_accessible(n):
                yield n

    def is_accessible(self, node: int) -> bool:
        """Determine whether a specific node of the graph is accessible."""
        return self.graph[node]["energy"] < self.energy_cutoff


class Energy:

    def __init__(self):
        # for now assume uniform, square cells so that geom. parameters = 1 and also diffusion coeff is 1
        self.D = 1
        self.h = 1
        self.S = 1
        self.V = 1
        # and let´s assume room temperature (does that make sense?)
        self.T = 293  # 293K <==> 20°C
        # energy cutoff - let's see if it should be changed
        self.energy_cutoff = 0.5
        # will only have a value if energy created from maze
        self.underlying_maze = None
        # in preparation
        self.grid_x = None
        self.grid_y = None
        self.energies = None
        self.size = None

    def from_maze(self, maze: Maze, add_noise=True, factor_grid=2):
        # for now let's assume the maze to be 2D TODO: change this to enable any dimensionality
        # sparse grid
        size_x, size_y = complex(maze.size[0]), complex(maze.size[1])
        x_edges, y_edges = np.mgrid[-1:1:size_x, -1:1:size_y]
        # dense grid
        self.grid_x, self.grid_y = np.mgrid[-1:1:factor_grid*size_x, -1:1:factor_grid*size_y]
        z = maze.maze
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
        tck = interpolate.bisplrep(x_edges, y_edges, z, nxest=10*m, nyest=10*m, task=-1, tx=self.grid_x[:, 0], ty=self.grid_y[0, :])
        znew = interpolate.bisplev(self.grid_x[:, 0], self.grid_y[0, :], tck)
        self.energies = znew
        self.size = znew.shape

    def visualize_underlying_maze(self):
        if not np.any(self.underlying_maze):
            raise ValueError("No underlying maze present! This is only available for surfaces created from mazes.")
        lims = dict(cmap='RdBu_r', vmin=-2, vmax=1)
        size_x, size_y = complex(maze.size[0]), complex(maze.size[1])
        x_edges, y_edges = np.mgrid[-1:1:size_x, -1:1:size_y]
        plt.pcolormesh(x_edges, y_edges, self.underlying_maze, **lims)
        plt.show()

    def visualize(self):
        if not np.any(self.energies):
            raise ValueError("No energies present! First, create an energy surface (e.g. from a maze).")
        lims = dict(cmap='RdBu_r', vmin=-2, vmax=1)
        plt.pcolormesh(self.grid_x, self.grid_y, self.energies, **lims)
        plt.colorbar()
        plt.show()

    def visualize_3d(self):
        if not np.any(self.energies):
            raise ValueError("No energies present! First, create an energy surface (e.g. from a maze).")
        ax = plt.axes(projection='3d')
        ax.plot_surface(self.grid_x, self.grid_y, self.energies, rstride=1, cstride=1,
                        cmap='RdBu_r', edgecolor='none')
        plt.show()


if __name__ == '__main__':
    energy = Energy()
    maze = Maze((12, 16))
    energy.from_maze(maze, add_noise=True)
    energy.visualize()
    #energy.visualize_underlying_maze()
    energy.visualize_3d()