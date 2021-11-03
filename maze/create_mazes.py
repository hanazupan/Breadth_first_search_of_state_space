"""
In this file, class AbstractEnergy, which implements general functions for finding neighbours, accessible cells etc.
is introduced. The simplest derived class, Maze, can be generated using Prim's algorithm or a random distribution of
cells. The class MazeAnimation enables producing videos of maze or energy surface exploration with maze traversal
algorithms (DFS, BFS, Dijkstra).
"""

from abc import ABC
from collections.abc import Sequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from matplotlib import colors, cm


class AbstractEnergy(ABC):
    """
    An object with energy data saved in a numpy array self.energies.
    There is some energy cutoff that makes some cells accessible and others not.
    """

    def __init__(self, energies: np.ndarray or None, energy_cutoff: float, deltas: np.ndarray or None,
                 size: tuple or None, images_path: str = "./", images_name: str = "abst_energy"):
        """
        Initialize some properties of all Energy objects.

        Args:
            energies: stores an array with energies
            energy_cutoff: cells with energy strictly below that value are accessible
            deltas: distances between cells in all dimensions
            size: tuple containing the size of all dimensions
            images_path: string, path where any generated images/videos will be saved
            images_name: string, identifier of all images/videos generated from this maze object
        """
        self.energies = energies
        self.deltas = deltas
        self.size = size
        self.energy_cutoff = energy_cutoff
        self.pbc = True                      # whether to use periodic boundary conditions
        # prepare for saving images/gifs
        self.images_path = images_path
        self.images_name = images_name

    def cell_to_node(self, cell: tuple) -> int:
        """
        Get the index of the graph node from the coordinates of the cell in the maze. Works with arrays of any
        number of dimensions. The index is simply the consecutive number of the cell in the maze array.

        Args:
            cell: (int, int, ...), representing the coordinates of the cell in the maze

        Returns:
            the index of the corresponding node
        """
        index = cell[0]
        for i in range(1, len(cell)):
            index = index * self.size[i] + cell[i]
        return index

    def node_to_cell(self, node: int) -> tuple:
        """
        Get the coordinates of the cell in the maze from the index of the graph node. Works with arrays of any
        number of dimensions. The index is simply the consecutive number of the cell in the maze array.
        Inverse function to cell_to_node.

        Args:
            node: the index of the node

        Returns:
            (int, int, ...), representing the coordinates of the corresponding cell in the maze
        """
        cell = np.zeros(len(self.size), dtype=int)
        for i in range(len(self.size) - 1, 0, -1):
            cell[i] = node % self.size[i]
            node = (node - cell[i]) // self.size[i]
        cell[0] = node
        return tuple(cell)

    def get_neighbours(self, cell: tuple) -> Sequence:
        """
        Genarator that gives neighbours of the argument cell, using periodic boundary conditions if self.pbc.

        Args:
            cell: tuple (int, int ...) of the length self.size - coordinates a cell

        Yields:
            tuple (int, int ...) of the length self.size - coordinates of a neighbouring cell
        """
        for i, coo in enumerate(cell):
            neig_cel = np.array(cell)
            neig_cel[i] = (cell[i] - self.deltas[i]) % self.size[i]
            # if no periodic boundaries, there is no -1 neighbour of the first cell
            if self.pbc or coo != 0:
                yield tuple(neig_cel)
            plus_one = (cell[i] + self.deltas[i]) % self.size[i]
            neig_cel[i] = plus_one
            # if no periodic boundaries, there is no 0 neighbour of the last cell
            if self.pbc or coo != self.size[i] - 1:
                yield tuple(neig_cel)

    def get_accessible_neighbours(self, cell: tuple) -> Sequence:
        """
        Same as get_neighbours but filters out non-accessible neighbours.

        Args:
            cell: tuple (int, int ...) of the length self.size - coordinates a cell

        Yields:
            tuple (int, int ...) of the length self.size - coordinates of an accessible neighbouring cell
        """
        for n in self.get_neighbours(cell):
            if self.is_accessible(n):
                yield n

    def get_energy(self, cell: tuple) -> float:
        """
        Returns the energy of the cell.

        Args:
            cell: tuple (int, int ...) of the length self.size - coordinates a cell

        Returns:
            float, energy taken from self.energies
        """
        return self.energies[cell]

    def is_accessible(self, cell: tuple) -> bool:
        """
        Determines whether a cell of maze is accessible. Cells strictly under energy_cutoff are accessible.

        Args:
            cell: (int, int, ...) a tuple of coordinates

        Returns:
            bool, True if cell accessible, else False
        """
        return self.energies[cell] < self.energy_cutoff

    def find_random_accessible(self) -> tuple:
        """
        Find a random cell in the maze that is accessible (is a passage).

        Returns:
            tuple, (int, int, ...) coordinates of an accessible cell
        """
        # a random integer in all dimensions of the size
        cell = tuple([np.random.randint(dim) for dim in self.size])
        # continue generating random cells until one is accessible
        while not self.is_accessible(cell):
            cell = tuple([np.random.randint(dim) for dim in self.size])
        return cell

    def determine_opposite(self, central: tuple, known_hall: tuple) -> tuple:
        """
        Determines the coordinates of the cell obtained if you start in the known_hall cell
        and jump over the central cell. E.g. if central = X, known_hall = O and opposite = ?

        |O|X|?| or |?|X|O| or

        |O|          |?|
        |X|    or    |X|
        |?|          |O|

        Args:
            central: tuple, coordinates of the central cell
            known_hall: tuple, coordinates of a cell next to the central cell

        Returns:
            tuple, coordinates of the opposite cell
        """
        neig_cel = np.array(central)
        for i, _ in enumerate(central):
            if central[i] == known_hall[i]:
                pass
            elif (central[i] + self.deltas[i]) % self.size[i] == known_hall[i]:
                neig_cel[i] = (neig_cel[i] - self.deltas[i]) % self.size[i]
            elif (central[i] - self.deltas[i]) % self.size[i] == known_hall[i]:
                neig_cel[i] = (neig_cel[i] + self.deltas[i]) % self.size[i]
            else:
                raise ValueError("Opposite cell nonexistent: central and known_hall are not neighbouring cells.")
        assert tuple(neig_cel) != known_hall
        return tuple(neig_cel)


class Maze(AbstractEnergy):

    def __init__(self, size: tuple, algorithm: str = 'Prim',
                 animate: bool = False, images_path: str = "./", images_name: str = "maze",
                 no_branching=False, edge_is_wall=False):
        """
        Creates a maze of user-defined size. A maze is represented as a numpy array with:
            - 1 to represent a wall (high energy)
            - 0 to represent a hall (low energy)
            - 2 to represent not assigned cells (should not occur in the final maze)

        Args:
            size: tuple giving the dimensions of the maze
            algorithm: string, maze generation algorithm, options ['handmade1', 'Prim', 'random']
            animate: bool, whether an animation of the maze generation should be computed and saved
            images_path: string, path where any generated images/videos will be saved
            images_name: string, identifier of all images/videos generated from this maze object
        """

        self.algorithm = algorithm
        # deltas are distances between neighbours in all directions
        deltas = np.ones(len(size), dtype=int)
        cutoff = 1
        # in the beginning all maze cells are unassigned
        energies = np.full(size, 2, dtype=int)
        # super takes care of: initializing energies, energy_cutoff, deltas, size, images_path, images_name
        super().__init__(energies, cutoff, deltas, size, images_path, images_name)
        # start the generation of a maze
        if algorithm == 'handmade1':
            self._create_handmade1()
        elif algorithm == 'Prim':
            if animate:
                ma = MazeAnimation(self)
                ma.animate_building_maze(no_branching=no_branching, edge_is_wall=edge_is_wall)
            else:
                # necessary to empty the generator
                for _ in self.create_prim(no_branching=no_branching, edge_is_wall=edge_is_wall):
                    pass
        elif algorithm == 'random':
            self.energies = np.random.randint(0, 2, size=self.size)
        else:
            raise AttributeError("Not a valid algorithm choice.")

    def __repr__(self) -> str:
        """
        When using print() on a maze object, it returns the string representation of self.energies.

        Returns:
            string representation of self.energies
        """
        return self.energies.__str__()

    def _create_handmade1(self):
        """
        Only for testing. Hand-pick some cells and turn them into halls.
        Warning: ignores height and width, the size is always (6, 6)
        """
        self.size = (6, 6)
        self.energies = np.full(self.size, 1, dtype=int)
        self.energies[0:4, 0] = 0
        self.energies[1, 1] = 0
        self.energies[3:5, 1] = 0
        self.energies[0, 3:6] = 0
        self.energies[2, 3:6] = 0
        self.energies[4:6, 3] = 0
        self.energies[5, 5] = 0

    def create_prim(self, **kwargs) -> Sequence:
        """
        Generate a maze using Prim's algorithm. From wiki (https://en.wikipedia.org/wiki/Maze_generation_algorithm):
        1. Start with a grid full of walls.
        2. Pick a cell, mark it as part of the maze. Add the walls of the cell to the wall list.
        3. While there are walls in the list:
            1. Pick a random wall from the list. If only one of the cells that the wall divides is visited, then:
                1. Make the wall a passage and mark the unvisited cell as part of the maze.
                2. Add the neighboring walls of the cell to the wall list.
            2. Remove the wall from the list.

        Yields:
            An array of current maze state while constructing the maze. Used for animation, else ignored.
        """
        wall_list = []

        def create_wall(cell):
            self.energies[cell] = 1
            wall_list.append(cell)

        if kwargs["edge_is_wall"]:
            # works only for 2D
            self.energies[0, :] = 1
            self.energies[-1, :] = 1
            self.energies[:, 0] = 1
            self.energies[:, 0] = 1
            # pick a random cell (not the edges) as a starting point and turn it into a hall
            random_cell = tuple([np.random.randint(1, dim - 1) for dim in self.size])
            self.energies[random_cell] = 0
        else:
            # pick a random cell as a starting point and turn it into a hall
            random_cell = tuple([np.random.randint(dim) for dim in self.size])
            self.energies[random_cell] = 0
        # for video
        yield self.energies
        # add neighbours of the cell to the wall list (periodic boundary conditions)
        for n in self.get_neighbours(random_cell):
            create_wall(n)
        # for video:
        yield self.energies
        # continue until you run out of walls
        while len(wall_list) > 0:
            random_wall = random.choice(wall_list)
            neighbours = []
            for n in self.get_neighbours(random_wall):
                neighbours.append(n)
            # whether neighbours are 0, 1 or 2
            values = [self.energies[cell] for cell in neighbours]
            # select only neighbours that are halls/empty
            neig_halls = [n for n in neighbours if self.energies[n] == 0]
            neig_empty = [n for n in neighbours if self.energies[n] == 2]
            for n in neig_halls:
                opposite_side = self.determine_opposite(random_wall, n)
                # possible additional conditions
                if kwargs["no_branching"]:
                    neig_of_neig = []
                    for m in self.get_accessible_neighbours(n):
                        neig_of_neig.append(m)
                    # each path cell can be surrounded by max two path cells so that there is no branching
                    branching_approved = len(neig_of_neig) < 2
                else:
                    branching_approved = True
                # values.count(0) == 1 makes sure all halls are only lines (not thicker than 1 cell)
                if self.energies[opposite_side] == 2 and values.count(0) == 1 and branching_approved:
                    # make this wall a hall
                    self.energies[random_wall] = 0
                    # add directly neighbouring empty cells to the wall_list
                    for w in neig_empty:
                        create_wall(w)
            wall_list.remove(random_wall)
            # for video
            yield self.energies
        # everything unassigned becomes a wall
        self.energies[self.energies == 2] = 1
        # to get the final image for animation with no unassigned cells
        yield self.energies

    def visualize(self):
        """
        Visualize the maze with black squares (walls) and white squares (halls).
        """
        with plt.style.context(['Stylesheets/maze_style.mplstyle', 'Stylesheets/not_animation.mplstyle']):
            ax = plt.imshow(self.energies, cmap="Greys")
            ax.figure.savefig(self.images_path + f"maze_{self.images_name}.png")
            plt.close()


class MazeAnimation:

    def __init__(self, maze_to_animate: AbstractEnergy):
        """
        MazeAnimation class enables creating an animation of a specific algorithm on a maze/energy object.

        Args:
            maze_to_animate: Maze or Energy object, the array on which we performed the search

        Raises:
            ValueError if the maze/energy is not 2D.
        """
        # iterator is the generator that gives us the arrays we want to show in each frame of the animation
        self.iterator = None
        # need to save the iterator value if the iterating function also returns the last value
        self.iterator_value = None
        self.energies = maze_to_animate
        if len(self.energies.size) != 2:
            raise ValueError("Animation only possible for 2D mazes or energies.")
        with plt.style.context('Stylesheets/maze_style.mplstyle'):
            self.fig, self.ax = plt.subplots()

    def _put_marker(self, x: int, y: int, letter: str, **kwargs):
        """
        Add a marker (e.g. a letter) to a position (x, y) in the whole animation.

        Args:
            x: int, line of the displayed array
            y: int, column of the displayed array
            letter: string, which marker to use
            **kwargs: other arguments to plt.plot()
        """
        self.ax.plot(x, y, marker=letter, **kwargs)

    def _animate(self, name_addition: str, **kwargs):
        """
        A general method for animation. Should only be called by more specific animation functions.

        Args:
            name_addition: how the gifs resulting from this animation process should be identified
            **kwargs: named arguments that can be passed to plt.imshow(), e.g. cmap
        """
        height, width = self.energies.size
        self.ax = plt.imshow(next(self.iterator), animated=True, **kwargs)

        def updatefig(i):
            self.iterator_value = i
            self.ax.set_array(i)
            return self.ax,

        # blit=True to only redraw the parts of the animation that have changed (speeds up the generation)
        # interval determines how fast the video when played (not saved)
        anim = animation.FuncAnimation(self.fig, updatefig, blit=True, frames=self.iterator,
                                       repeat=False, interval=10, save_count=height * width)
        # fps determines how fast the video when saved
        writergif = animation.PillowWriter(fps=50)
        anim.save(self.energies.images_path + f"{name_addition}_{self.energies.images_name}.gif", writer=writergif)
        plt.close()

    def animate_building_maze(self, **kwargs):
        """
        Args:
            **kwargs: optional arguments to pass to create_prim

        Creates an animation showing how the maze has been built. Colormap as follows:
            white = hall
            gray = wall
            black = unassigned
        """
        # make sure to only use this animation on Mazes, not other AbstractEnergy objects
        if type(self.energies) != Maze:
            raise TypeError("Building animation only available for Mazes, not for Energy objects!")
        else:
            self.iterator = self.energies.create_prim(**kwargs)
        self._animate("building", cmap="Greys")

    def animate_search(self, name: str, iterator, cutoff: float):
        """
        Animate solving the maze with bfs or dfs algorithm. Color map as follows:
            blue = discovered accessible cells
            white = undiscovered accessible cells
            black = walls

        Args:
            name: dfs or bfs, which search should be performed
            iterator: a generator of images
            cutoff: cells with energy higher than this cutoff will be coloured black

        Raises:
            ValueError if an inappropriate name of algorithm is provided.
        """
        self.iterator = iterator
        # self-defined color map: blue are halls that have been discovered and are negative; around
        # 0 undiscovered halls, above cutoff are the walls, which are black.
        cmap = colors.ListedColormap(['blue', 'white', 'black'])
        bounds = [-200, -100 + cutoff, cutoff, cutoff + 100]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        if name in ["bfs", "dfs"]:
            self._animate(name, cmap=cmap, norm=norm)
        else:
            raise ValueError("Only allowed names are 'bfs', 'dfs'.")

    def animate_dijkstra(self, iterator, start_cell: tuple, end_cell: tuple):
        """
        Animate finding the connection between start_cell and end_cell (both passages the in maze).
            red = visited cells
            circle = start cell
            cross = end cell

        Args:
            iterator: a generator of images
            start_cell: tuple, (x, y) coordinates of the start cell
            end_cell: tuple, (x, y) coordinates of the end cell
        """
        self.iterator = iterator
        cmap = cm.get_cmap("RdBu").copy()
        cmap.set_under("white")
        cmap.set_over("black")
        self._put_marker(end_cell[1], end_cell[0], "x", color="black", linewidth=1.5)
        self._put_marker(start_cell[1], start_cell[0], "o", color="black", linewidth=1.5)
        self._animate("dijkstra", cmap=cmap, vmin=0.5, vmax=999)


if __name__ == '__main__':
    path = "images/"
    maze = Maze((20, 20), images_path=path, images_name="style", animate=True)
    maze.visualize()
