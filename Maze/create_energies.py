from abc import ABC
import networkx as nx
from create_mazes import Maze
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from collections.abc import Sequence
import cmath


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

    def from_maze(self, maze: Maze, add_noise=True):
        # for now let's assume the maze to be 2D
        # sparse grid
        size_x, size_y = complex(maze.size[0]), complex(maze.size[1])
        x_edges, y_edges = np.mgrid[-1:1:size_x, -1:1:size_y]
        # dense grid
        x_dense, y_dense = np.mgrid[-1:1:5*size_x, -1:1:5*size_y]
        z = maze.maze
        # change some random zeroes into -1 and -2
        if add_noise:
            for _ in range(round(0.05*maze.size[0]*maze.size[1])):
                cell = maze.find_random_accessible()
                z[cell] = -1
            for _ in range(round(0.01*maze.size[0]*maze.size[1])):
                cell = maze.find_random_accessible()
                z[cell] = -2
        lims = dict(cmap='RdBu_r', vmin=-2, vmax=1)
        plt.pcolormesh(x_edges, y_edges, z, **lims)
        plt.show()
        m = maze.size[0]
        print(m - np.sqrt(2 * m), m + np.sqrt(2 * m))
        tck = interpolate.bisplrep(x_edges, y_edges, z, s=0)
        znew = interpolate.bisplev(x_dense[:, 0], y_dense[0, :], tck)
        plt.pcolormesh(x_dense, y_dense, znew, **lims)
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    energy = Energy()
    maze = Maze((22, 26))
    energy.from_maze(maze, add_noise=False)