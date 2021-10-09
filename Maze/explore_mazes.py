from create_mazes import Maze, MazeAnimation
from abc import ABC, abstractmethod
import networkx as nx
import numpy as np
from collections import deque
from collections.abc import Sequence
import matplotlib.pyplot as plt
from matplotlib import cm


class Explorer(ABC):

    def __init__(self, maze: Maze, explorer_name: str):
        """
        Abstract class for algorithms exploring mazes.

        Args:
            maze: the Maze object that will be explored
            explorer_name: for the purposes of saving images/gifs, each subclass will have a unique name.
        """
        self.maze = maze
        self.graph = nx.Graph()
        self.adj_matrix = None
        self.explorer_name = explorer_name

    @abstractmethod
    def explore(self) -> nx.Graph:
        """
        Every Explorer implementation uses their own algorithm here.

        Returns:
            A graph of all nodes (cells) of the maze, indexed by their consecutive index in the maze
            and with property of energy. Edges between nodes suggests which cells are neighbours.
        """

    @abstractmethod
    def explore_and_animate(self) -> nx.Graph:
        """
        Same as explore but also provides an animation using MazeAnimation class.

        Returns:
            A graph of all nodes (cells) of the maze, indexed by their consecutive index in the maze
            and with property of energy. Edges between nodes suggests which cells are neighbours.
        """

    def draw_connections_graph(self, show: bool = True, **kwargs):
        """
        Visualize the graph of connections between the cells of the maze.

        Args:
            show: bool, should the visualization be displayed
            **kwargs: named arguments that can be passed to nx.draw_kamada_kawai(), e.g. with_labels
        """
        if not self.graph:
            self.explore()
        plt.figure()
        nx.draw_kamada_kawai(self.graph, **kwargs)
        plt.savefig(self.maze.images_path +
                    f"{self.explorer_name}_graph_{self.maze.images_name}.png", bbox_inches='tight', dpi=1200)
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
        """
        if not np.any(self.adj_matrix):
            self.explore()
        return self.adj_matrix


class BFSExplorer(Explorer):
    """
    Implements Breadth-first search algorithm for mazes.
    """

    def __init__(self, maze):
        super().__init__(maze, "bfs")

    def explore(self) -> nx.Graph:
        for _ in self._bfs_algorithm():
            pass
        return self.graph

    def explore_and_animate(self) -> nx.Graph:
        ma = MazeAnimation(self.maze)
        ma.animate_bfs(self._bfs_algorithm())
        return self.graph

    def _bfs_algorithm(self) -> Sequence:
        """
        Perform a breadth-first search of the maze. This also generates a graph of connections and an
        adjacency matrix of the maze. The algorithm does this:
        1. Find a random accessible cell in the maze, mark it as visited and accessible
        2. Add accessible neighbours of the cell to the queue
        3. While queue not empty:
            1. Take an element from the beginning of the queue
            2. Mark all its unvisited neighbours as visited
            3. Mark all its accessible neighbours as accessible and add them to the queue

        Yields:
            A numpy array with 0 = undiscovered passage, 1 = wall, -1 = discovered passage
        """
        # for video
        yield self.maze.maze
        visited = np.zeros(self.maze.size, dtype=int)
        accessible = np.zeros(self.maze.size, dtype=int)
        check_queue = deque()
        # get a random starting point that is accessible
        random_cell = self.maze.find_random_accessible()
        visited[random_cell] = 1
        accessible[random_cell] = 1
        # for the graph we are using index of the flattened maze as the identifier
        index_rc = self.maze.cell_to_node(random_cell)
        self.graph.add_node(index_rc, energy=self.maze.get_energy(random_cell))
        # for video
        yield self.maze.maze - accessible
        # take care of the neighbours of the first random cell
        neighbours = self.maze.get_neighbours(random_cell)
        for n in neighbours:
            visited[n] = 1
            if self.maze.is_accessible(n):
                index_n = self.maze.cell_to_node(n)
                self.graph.add_node(index_n, energy=self.maze.get_energy(n))
                self.graph.add_edge(index_rc, index_n)
                accessible[n] = 1
                check_queue.append(n)
        # take care of all other cells
        while len(check_queue) > 0:
            cell = check_queue.popleft()
            index_cell = self.maze.cell_to_node(cell)
            neighbours = self.maze.get_neighbours(cell)
            # if neighbours visited already, don't need to bother with them
            unvis_neig = [n for n in neighbours if visited[n] == 0]
            for n in unvis_neig:
                visited[n] = 1
                # if accessible, add to queue
                if self.maze.is_accessible(n):
                    index_n = self.maze.cell_to_node(n)
                    self.graph.add_node(index_n, energy=self.maze.get_energy(n))
                    self.graph.add_edge(index_cell, index_n)
                    accessible[n] = 1
                    check_queue.append(n)
            # for video
            yield self.maze.maze - accessible
        # accessible states must be the logical inverse of the maze
        assert np.all(np.logical_not(accessible) == self.maze.maze)
        # returns adjacency matrix - ensures the order to be left-right, top-bottom
        self.adj_matrix = nx.to_numpy_matrix(self.graph,
                                             nodelist=[i for i, x in enumerate(accessible.flatten()) if x == 1])
        # the adjacency matrix must be as long as there are accessible cells in the maze
        assert len(self.adj_matrix) == np.count_nonzero(accessible)


class DijkstraExplorer(Explorer):

    def __init__(self, maze: Maze, start_cell: tuple = None, end_cell: tuple = None):
        super().__init__(maze, "dijkstra")
        self.distances = None
        self.start_cell = None
        self.end_cell = None
        self.set_start_end_cell(start_cell, end_cell)

    def set_start_end_cell(self, start_cell: tuple = None, end_cell: tuple = None):
        """
        Set up start and/or end cell - if they are provided, check whether they are valid (must be passages),
        else select random passages.

        Args:
            start_cell: tuple or None, (int, int ..) coordinates of the cell where the path starts
            end_cell: tuple or None, (int, int ..) coordinates of the cell where the path ends

        Raises:
            ValueError if trying to set start or end cell that is not accessible
        """
        self.start_cell = start_cell
        self.end_cell = end_cell
        if not self.start_cell:
            self.start_cell = self.maze.find_random_accessible()
        elif not self.maze.is_accessible(start_cell):
            raise ValueError("Start cell must lie on a passage (white cell) in the maze.")
        if not self.end_cell:
            self.end_cell = self.maze.find_random_accessible()
        elif not self.maze.is_accessible(end_cell):
            raise ValueError("End cell must lie on a passage (white cell) in the maze.")

    def get_distance(self, start_cell: tuple = None, end_cell: tuple = None) -> int:
        """
        Get the number of cells between the start and end cell.

        Args:
            start_cell: tuple or None, (int, int ..) coordinates of the cell where the path starts
            end_cell: tuple or None, (int, int ..) coordinates of the cell where the path ends

        Returns:
            the distance in number of cells
        """
        self.set_start_end_cell(start_cell, end_cell)
        if not np.any(self.distances):
            self.explore()
        return self.distances[self.end_cell]

    def explore(self) -> nx.Graph:
        for _ in self._dijkstra_algorithm():
            pass
        return self.graph

    def explore_and_animate(self) -> nx.Graph:
        ma = MazeAnimation(self.maze)
        ma.animate_dijkstra(self._dijkstra_algorithm(), self.start_cell, self.end_cell)
        return self.graph

    def visualize_distances(self, show: bool = True):
        """
        Creates an image, visualizing distances calculated with Dijkstra's algorithm.

        Arg:
            show: whether to show the image
        """
        if not np.any(self.distances):
            for _ in self.explore():
                pass
        for_plotting = np.zeros(self.maze.size, dtype=int)
        for_plotting[self.start_cell] = 1
        array_to_plot = np.where(self.visited != 0, self.distances, self.maze.maze * 1000) + for_plotting
        my_cmap = cm.get_cmap("plasma").copy()
        my_cmap.set_under("white")
        my_cmap.set_over("black")
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.plot(self.start_cell[1], self.start_cell[0], marker="o", color="white", linewidth=1.5)
        plt.plot(self.end_cell[1], self.end_cell[0], marker="x", color="black", linewidth=1.5)
        plt.imshow(array_to_plot, cmap=my_cmap, vmin=0.5, vmax=array_to_plot[self.end_cell])
        plt.savefig(self.maze.images_path + f"distances_{self.maze.images_name}.png", dpi=1200)
        if show:
            plt.show()
        else:
            plt.close()

    def _dijkstra_algorithm(self) -> Sequence:
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

        Yields:
            an image for the animation
        """
        # create empty objects
        self.visited = np.zeros(self.maze.size, dtype=int)
        self.distances = np.full(self.maze.size, np.inf)
        for_plotting = np.zeros(self.maze.size, dtype=int)
        for_plotting[self.start_cell] = 1
        check_queue = deque()
        current_cell = self.start_cell
        self.distances[current_cell] = 0
        # here first frame of the animation
        yield np.where(self.visited != 0, self.distances, self.maze.maze*1000) + for_plotting
        # determine accessible neighbours - their distances to
        for an in self.maze.get_accessible_neighbours(current_cell):
            check_queue.append(an)
        for n in check_queue:
            self.distances[n] = 1
        # here snapshot for animation
        yield np.where(self.visited != 0, self.distances, self.maze.maze*1000) + for_plotting
        self.visited[current_cell] = 1
        while check_queue:
            current_cell = check_queue.popleft()
            # unvisited, accessible neighbours
            for n in self.maze.get_accessible_neighbours(current_cell):
                if not self.visited[n]:
                    tent_dist = self.distances[current_cell] + 1
                    if tent_dist < self.distances[n]:
                        self.distances[n] = tent_dist
                    check_queue.append(n)
            self.visited[current_cell] = 1
            # here snapshot for animation
            yield np.where(self.visited != 0, self.distances, self.maze.maze * 1000) + for_plotting
            if self.visited[self.end_cell] == 1:
                return np.where(self.visited != 0, self.distances, self.maze.maze * 1000) + for_plotting


if __name__ == '__main__':
    path = "Images/"
    my_maze = Maze((15, 12), images_path=path, images_name="explore", animate=False)
    bfs_explorer = BFSExplorer(my_maze)
    bfs_explorer.explore_and_animate()
    bfs_explorer.draw_connections_graph(show=False, with_labels=True)
    bfs_explorer.get_adjacency_matrix()
    dijkstra_exp = DijkstraExplorer(my_maze)
    dijkstra_exp.explore()
    dijkstra_exp.explore_and_animate()
    dijkstra_exp.visualize_distances()
