"""
In this file, class Maze is introduced and mazes of different sizes can be created using Prim's
algorithm or a random distribution of cells. The mazes can also be solved using breadth-first search
algorithm, visualized as graphs and transformed into an adjacency matrix.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import matplotlib.image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from collections import deque
from matplotlib import colors, cm
import networkx as nx


class Energy(ABC):
    """
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


class Maze(Energy):

    def __init__(self, size: tuple, algorithm: str = 'Prim', animate: bool = False,
                 images_path: str = "./", images_name: str = "maze"):
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
        super().__init__(nx.Graph(), 1)
        self.algorithm = algorithm
        self.size = size
        # deltas are distances between neighbours in all directions
        self.deltas = np.ones(len(self.size), dtype=int)
        self.maze = np.full(self.size, 2, dtype=int)
        # prepare empty objects for solving maze
        self.adj_matrix = None
        # prepare for saving images/gifs
        self.images_path = images_path
        self.images_name = images_name
        # start the generation of a maze
        if algorithm == 'handmade1':
            self._create_handmade1()
        elif algorithm == 'Prim':
            if animate:
                ma = MazeAnimation(self)
                ma.animate_building_maze()
            else:
                # necessary to empty the generator
                for _ in self._create_prim():
                    pass
        elif algorithm == 'random':
            self.maze = np.random.randint(0, 2, size=self.size)
        else:
            raise AttributeError("Not a valid algorithm choice.")

    def __repr__(self) -> str:
        """
        When using print() on a Maze object, it returns the string representation of self.maze.

        Returns: string representation of self.maze

        """
        return self.maze.__str__()

    def _create_handmade1(self):
        """
        Only for testing. Hand-pick some cells and turn them into halls.
        Warning: ignores height and width, the size is always (6, 6)
        """
        self.size = (6, 6)
        self.maze = np.full(self.size, 1, dtype=int)
        self.maze[0:4, 0] = 0
        self.maze[1, 1] = 0
        self.maze[3:5, 1] = 0
        self.maze[0, 3:6] = 0
        self.maze[2, 3:6] = 0
        self.maze[4:6, 3] = 0
        self.maze[5, 5] = 0

    ############################################################################
    # ---------------------   PRIM'S ALGORITHM   -------------------------------
    ############################################################################

    def _create_prim(self) -> Sequence:
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
            self.maze[cell] = 1
            wall_list.append(cell)

        # pick a random cell as a starting point and turn it into a hall
        random_cell = tuple([np.random.randint(dim) for dim in self.size])
        self.maze[random_cell] = 0
        # for video
        yield self.maze
        # add walls of the cell to the wall list (periodic boundary conditions)
        for n in self.get_neighbours(random_cell):
            create_wall(n)
        # for video:
        yield self.maze
        # continue until you run out of walls
        while len(wall_list) > 0:
            #print(wall_list)
            random_wall = random.choice(wall_list)
            neighbours = []
            for n in self.get_neighbours(random_wall):
                neighbours.append(n)
            # whether neighbours are 0, 1 or 2
            values = [self.maze[l, c] for (l, c) in neighbours]
            # select only neighbours that are halls/empty
            neig_halls = [n for n in neighbours if self.maze[n] == 0]
            neig_empty = [n for n in neighbours if self.maze[n] == 2]
            for n in neig_halls:
                opposite_side = self._determine_opposite(random_wall, n)
                # values.count(0) == 1 makes sure all halls are only lines (not thicker than 1 cell)
                if self.maze[opposite_side] == 2 and values.count(0) == 1:
                    # make this wall a hall
                    self.maze[random_wall] = 0
                    # add directly neighbouring empty cells to the wall_list
                    for w in neig_empty:
                        create_wall(w)
            wall_list.remove(random_wall)
            # for video
            yield self.maze
        # everything unassigned becomes a wall
        self.maze[self.maze == 2] = 1
        # to get the final image for animation with no unassigned cells
        yield self.maze

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
        Get the index of the graph node from the coordinates of the cell in the maze. Works with arrays of any
        number of dimensions. The index is simply the consecutive number of the cell in the maze array.

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
        Overrides getting neighbours from the graph. For a maze, it makes more sense to get neighbouring cells
        than neighbouring nodes.
        Args:
            cell: tuple (int, int ...) of the length self.size - coordinates a cell

        Yields:
            tuple (int, int ...) of the length self.size - coordinates of a neighbouring cell
        """
        for i, coo in enumerate(cell):
            neig_cel = np.array(cell)
            neig_cel[i] = (cell[i] - self.deltas[i]) % self.size[i]
            yield tuple(neig_cel)
            plus_one = (cell[i] + self.deltas[i]) % self.size[i]
            neig_cel[i] = plus_one
            yield tuple(neig_cel)

    def _determine_opposite(self, central: tuple, known_hall: tuple) -> tuple:
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
            elif abs(central[i] - known_hall[i]) == self.deltas[i] or self.size[i]:
                if central[i] > known_hall[i]:
                    neig_cel[i] = (neig_cel[i] + self.deltas[i]) % self.size[i]
                else:
                    neig_cel[i] = (neig_cel[i] - self.deltas[i]) % self.size[i]
            else:
                raise ValueError("Opposite cell nonexistent: central and known_hall are not neighbouring cells.")
        return tuple(neig_cel)

    ############################################################################
    # ---------------------------   BFS    -------------------------------------
    ############################################################################

    def breadth_first_search(self, animate: bool = False) -> nx.Graph:
        """
        Perform a breadth-first search of the maze. This also generates a graph of connections and an
        adjacency matrix of the maze. The algorithm does this:
        1. Find a random accessible cell in the maze, mark it as visited and accessible
        2. Add accessible neighbours of the cell to the queue
        3. While queue not empty:
            1. Take an element from the beginning of the queue
            2. Mark all its unvisited neighbours as visited
            3. Mark all its accessible neighbours as accessible and add them to the queue


        Args:
            animate: bool, whether to generate and save the animation of solving the maze

        Returns:
            A graph of connections between nodes (cells) of the maze.
        """
        if animate:
            ma = MazeAnimation(self)
            ma.animate_bfs()
        else:
            # this needs to be done to exhaust the generator
            for _ in self._bfs():
                pass
        return self.graph

    def is_accessible(self, cell: tuple) -> bool:
        """
        Determines whether a cell of maze is accessible

        Args:
            cell: (int, int, ...) a tuple of coordinates

        Returns:
            bool, True if cell accessible, else False
        """
        return self.maze[cell] < self.energy_cutoff

    def _bfs(self) -> Sequence:
        """
        Determine the connectivity graph of accessible states (states with value 0) in the maze. Determine
        the adjacency matrix. Yield images for animation of path searching.

        Yields:
            A numpy array with 0 = passage, 1 = wall, -1 = discovered passage
        """
        self.graph = nx.Graph()
        # for video
        yield self.maze
        visited = np.zeros(self.size, dtype=int)
        accessible = np.zeros(self.size, dtype=int)
        check_queue = deque()
        height, width = self.size
        # get a random starting point that is accessible
        random_cell = self._find_random_accessible()
        visited[random_cell] = 1
        accessible[random_cell] = 1
        # for the graph we are using index of the flattened maze as the identifier
        index_rc = self.cell_to_node(random_cell)
        self.graph.add_node(index_rc, energy=1)
        # for video
        yield self.maze - accessible
        # take care of the neighbours of the first random cell
        neighbours = self.get_neighbours(random_cell)
        for n in neighbours:
            visited[n] = 1
            if self.is_accessible(n):
                index_n = n[0]*width + n[1]
                self.graph.add_node(index_n, energy=1)
                self.graph.add_edge(index_rc, index_n)
                accessible[n] = 1
                check_queue.append(n)
        # take care of all other cells
        while len(check_queue) > 0:
            cell = check_queue.popleft()
            index_cell = self.cell_to_node(cell)
            neighbours = self.get_neighbours(cell)
            # if neighbours visited already, don't need to bother with them
            unvis_neig = [n for n in neighbours if visited[n] == 0]
            for n in unvis_neig:
                visited[n] = 1
                # if accessible, add to queue
                if self.is_accessible(n):
                    index_n = self.cell_to_node(n)
                    self.graph.add_node(index_n, energy=1)
                    self.graph.add_edge(index_cell, index_n)
                    accessible[n] = 1
                    check_queue.append(n)
            # for video
            yield self.maze - accessible
        # accessible states must be the logical inverse of the maze
        assert np.all(np.logical_not(accessible) == self.maze)
        # returns adjacency matrix - ensures the order to be left-right, top-bottom
        self.adj_matrix = nx.to_numpy_matrix(self.graph,
                                             nodelist=[i for i, x in enumerate(accessible.flatten()) if x == 1])
        # the adjacency matrix must be as long as there are accessible cells in the maze
        assert len(self.adj_matrix) == np.count_nonzero(accessible)

    ############################################################################
    # ---------------------   PUBLIC METHODS    --------------------------------
    ############################################################################

    def visualize(self, show: bool = True) -> matplotlib.image.AxesImage:
        """
        Visualize the Maze with black squares (walls) and white squares (halls).

        Args:
            show: bool, should the visualization be displayed

        Returns:
            matplotlib.image.AxesImage, the plot
        """
        ax = plt.imshow(self.maze, cmap="Greys")
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.figure.savefig(self.images_path + f"maze_{self.images_name}.png", bbox_inches='tight', dpi=1200)
        if show:
            plt.show()
        else:
            plt.close()
        return ax

    def draw_connections_graph(self, show: bool = True, **kwargs):
        """
        Visualize the graph of connections between the cells of the maze.

        Args:
            show: bool, should the visualization be displayed
            **kwargs: named arguments that can be passed to nx.draw_kamada_kawai(), e.g. with_labels
        """
        if not self.graph:
            self.breadth_first_search()
        plt.figure()
        nx.draw_kamada_kawai(self.graph, **kwargs)
        plt.savefig(self.images_path + f"graph_{self.images_name}.png", bbox_inches='tight', dpi=1200)
        # causes MatplotlibDeprecationWarning
        if show:
            plt.show()
        else:
            plt.close()

    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Get (and create if not yet created) an adjacency matrix of the maze with breadth-first search.

        Returns:
            numpy array, adjacency matrix, square number the size of the number of halls

        Raises:
            ValueError if using the wrong algorithm.
        """
        if self.algorithm == 'random':
            raise ValueError("Adjacency matrix not available for a randomly created maze.")
        elif not np.any(self.adj_matrix):
            self.breadth_first_search()
            return self.adj_matrix
        else:
            return self.adj_matrix

    ############################################################################
    # ------------------   DIJKSTRA'S ALGORITHM    -----------------------------
    ############################################################################

    def find_shortest_path(self, start_cell: tuple = None, end_cell: tuple = None, animate: bool = False) -> nx.Graph:
        """
        Apply Dijkstra's algorithm to find the distance between start and end cell (both passages in maze).
        If start or end cell are None, random accessible cells will be selected.

        Args:
            start_cell: tuple or None, (int, int ..) coordinates of the cell where the path starts
            end_cell: tuple or None, (int, int ..) coordinates of the cell where the path ends
            animate: bool, whether to animate the algorithm

        Returns:
            Graph with shortest connections.
        """
        # TODO: actually return a graph
        # if no cells provided, a random start cell
        # if cells are provided, check if they are actually passages
        if not start_cell:
            start_cell = self._find_random_accessible()
        elif not self.is_accessible(start_cell):
            raise ValueError("Start cell must lie on a passage (white cell) in the maze.")
        if not end_cell:
            end_cell = self._find_random_accessible()
        elif not self.is_accessible(end_cell):
            raise ValueError("End cell must lie on a passage (white cell) in the maze.")
        # run the actual algorithm and animate if you want
        if animate:
            ma = MazeAnimation(self)
            return int(ma.animate_dijkstra(start_cell, end_cell)[end_cell])
        else:
            iterator = self._dijkstra_connect_two(start_cell=start_cell, end_cell=end_cell)
            dist = None
            # exhaust the iterator and save the last returned value
            for cv in iterator:
                dist = cv
            self._visualize_distances(dist, start_cell, end_cell)
            return int(dist[end_cell])

    def _find_random_accessible(self) -> tuple:
        """
        Find a random cell in the maze that is accessible (is a passage).

        Returns:
            tuple, (int, int, ...) coordinates of an accessible cell
        """
        cell = tuple([np.random.randint(dim) for dim in self.size])
        while not self.is_accessible(cell):
            cell = tuple([np.random.randint(dim) for dim in self.size])
        return cell

    def _visualize_distances(self, distances: np.ndarray, start_cell: tuple, end_cell: tuple):
        """
        Creates an image, visualizing distances calculated with Dijkstra's algorithm.

        Args:
            distances: np.ndarray, array of distances from start_cell
            start_cell: tuple or None, (int, int ..) coordinates of the cell where the path starts
            end_cell: tuple or None, (int, int ..) coordinates of the cell where the path ends
        """
        plt.figure()
        cmap = cm.get_cmap("plasma").copy()
        cmap.set_under("white")
        cmap.set_over("black")
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.imshow(distances, cmap=cmap, vmin=0.5, vmax=distances[end_cell])
        plt.plot(end_cell[1], end_cell[0], marker="x", color="black", linewidth=1.5)
        plt.plot(start_cell[1], start_cell[0], marker="o", color="white", linewidth=1.5)
        plt.savefig(self.images_path + f"distances_{self.images_name}.png", dpi=1200)

    def _dijkstra_connect_two(self, start_cell: tuple, end_cell: tuple) -> np.ndarray:
        """
        The description of Dijkstra's algorithm from wiki (https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm):
        1. Mark all nodes unvisited.
        2. Set initial tentative distances: 0 for start cell, infinity for all others.
        3. Add accessible neighbours of start cell to the queue. Set their distances to 1. Mark start cell as visited.
        4. While queue not empty:
            1. Set current cell to be the first element from the queue and remove from the queue
            2. Calculate tentative distances of unvisited, accesible neighbours of the current cell.
            3. If tentative distance < value of the neighbour:
                1. Replace value of the neighbour with the tentative distance.
            4. Mark current cell as visited.
            5. If end cell has been visited:
                6. Return value of end cell.

        Args:
            start_cell: tuple, (x, y) coordinates of the cell where the path starts
            end_cell: tuple, (x, y) coordinates of the cell where the path ends

        Returns:
            np.ndarray, an array of distances to start cell for all visited cells
        """
        # create empty objects
        visited = np.zeros(self.size, dtype=int)
        distances = np.full(self.size, np.inf)
        for_plotting = np.zeros(self.size, dtype=int)
        for_plotting[start_cell] = 1
        check_queue = deque()
        current_cell = start_cell
        distances[current_cell] = 0
        # here first frame of the animation
        yield np.where(visited != 0, distances, self.maze*1000) + for_plotting
        # determine accessible neighbours - their distances to
        for an in self.get_accessible_neighbours(current_cell):
            check_queue.append(an)
        for n in check_queue:
            distances[n] = 1
        # here snapshot for animation
        yield np.where(visited != 0, distances, self.maze*1000) + for_plotting
        visited[current_cell] = 1
        while check_queue:
            current_cell = check_queue.popleft()
            # unvisited, accessible neighbours
            for n in self.get_accessible_neighbours(current_cell):
                if not visited[n]:
                    tent_dist = distances[current_cell] + 1
                    if tent_dist < distances[n]:
                        distances[n] = tent_dist
                    check_queue.append(n)
            visited[current_cell] = 1
            # here snapshot for animation
            yield np.where(visited != 0, distances, self.maze * 1000) + for_plotting
            if visited[end_cell] == 1:
                return distances


class MazeAnimation:

    def __init__(self, maze_to_animate: Maze):
        """
        MazeAnimation class enables creating an animation of a specific algorithm on a Maze object.

        Args:
            maze_to_animate: Maze object, a maze that we want to animate.
        """
        self.iterator = None
        self.iterator_value = None
        self.maze = maze_to_animate
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
        height, width = self.maze.size
        self.ax = plt.imshow(next(self.iterator), animated=True, **kwargs)

        def updatefig(i):
            self.iterator_value = i
            self.ax.set_array(i)
            return self.ax,

        self.ax.axes.get_xaxis().set_visible(False)
        self.ax.axes.get_yaxis().set_visible(False)
        # blit=True to only redraw the parts of the animation that have changed (speeds up the generation)
        # interval determines how fast the video when played (not saved)
        anim = animation.FuncAnimation(self.fig, updatefig, blit=True, frames=self.iterator,
                                       repeat=False, interval=10, save_count=height * width)
        writergif = animation.PillowWriter(fps=50)
        anim.save(self.maze.images_path + f"{name_addition}_{self.maze.images_name}.gif", writer=writergif)

    def animate_building_maze(self):
        """
        Creates an animation showing how the maze has been built. Colormap as follows:
            white = hall
            gray = wall
            black = unassigned
        """
        self.iterator = self.maze._create_prim()
        self._animate("building", cmap="Greys")

    def animate_bfs(self):
        """
        Animate solving the maze with bfa. Color map as follows:
            blue = discovered accessible cells
            white = undiscovered accessible cells
            black = walls
        """
        self.iterator = self.maze._bfs()
        # self-defined color map: -1 are halls that have been discovered and are blue; 0 undiscovered halls,
        # 1 are the walls.
        cmap = colors.ListedColormap(['blue', 'white', 'black'])
        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        self._animate("solving", cmap=cmap, norm=norm)

    def animate_dijkstra(self, start_cell: tuple, end_cell: tuple) -> np.ndarray:
        """
        Animate finding the connection between start_cell and end_cell (both passages the in maze).
            red = visited cells
            circle = start cell
            cross = end cell

        Args:
            start_cell: tuple, (x, y) coordinates of the start cell
            end_cell: tuple, (x, y) coordinates of the end cell

        Returns:
            np.ndarray, the array of distances from start_cell
        """
        self.iterator = self.maze._dijkstra_connect_two(start_cell, end_cell)
        cmap = cm.get_cmap("RdBu").copy()
        cmap.set_under("white")
        cmap.set_over("black")
        self._put_marker(end_cell[1], end_cell[0], "x", color="black", linewidth=1.5)
        self._put_marker(start_cell[1], start_cell[0], "o", color="black", linewidth=1.5)
        self._animate("dijkstra", cmap=cmap, vmin=0.5, vmax=999)
        self.maze._visualize_distances(self.iterator_value, start_cell, end_cell)
        return self.iterator_value


if __name__ == '__main__':
    path = "Images/"
    maze = Maze((20, 20), images_path=path, images_name="new", animate=True)
    maze.visualize(show=False)
    maze.breadth_first_search(animate=True)
    adjacency = maze.get_adjacency_matrix()
    maze.draw_connections_graph(show=False, with_labels=True)
    length = maze.find_shortest_path(animate=True)
