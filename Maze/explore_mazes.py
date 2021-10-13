from create_mazes import Maze, MazeAnimation, AbstractEnergy
from abc import ABC, abstractmethod
import networkx as nx
import numpy as np
from collections import deque
from collections.abc import Sequence
import matplotlib.pyplot as plt
from matplotlib import cm
import heapq


class Explorer(ABC):

    def __init__(self, energy: AbstractEnergy, explorer_name: str):
        """
        Abstract class for algorithms exploring mazes.

        Args:
            maze: the Maze object that will be explored
            explorer_name: for the purposes of saving images/gifs, each subclass will have a unique name.
        """
        self.maze = energy
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

    def get_adjacency_matrix(self, save=False) -> np.ndarray:
        """
        Get (and create if not yet created) an adjacency matrix of the maze with breadth-first search.

        Args:
            save: whether to save the produced adjacency matrix

        Returns:
            numpy array, adjacency matrix, square number the size of the number of halls
        """
        if not np.any(self.adj_matrix):
            self.explore()
        if save:
            np.save(f"{self.maze.images_path}{self.explorer_name}_adj_matrix_{self.maze.images_name}", self.adj_matrix)
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
        ma.animate_search("bfs", self._bfs_algorithm())
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
        yield self.maze.energies
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
        yield self.maze.energies - accessible
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
            yield self.maze.energies - accessible
        # accessible states must be the logical inverse of the maze - but only true of pure mazes (not energies)
        #assert np.all(np.logical_not(accessible) == self.maze.energies)
        # returns adjacency matrix - ensures the order to be left-right, top-bottom
        self.adj_matrix = nx.to_numpy_matrix(self.graph,
                                             nodelist=[i for i, x in enumerate(accessible.flatten()) if x == 1])
        # the adjacency matrix must be as long as there are accessible cells in the maze
        assert len(self.adj_matrix) == np.count_nonzero(accessible)


class DFSExplorer(Explorer):
    """
    Implements Depth-first search algorithm for mazes.
    """

    def __init__(self, maze):
        super().__init__(maze, "dfs")

    def explore(self) -> nx.Graph:
        for _ in self._dfs_algorithm():
            pass
        return self.graph

    def explore_and_animate(self) -> nx.Graph:
        ma = MazeAnimation(self.maze)
        ma.animate_search("dfs", self._dfs_algorithm())
        return self.graph

    def _dfs_algorithm(self) -> Sequence:
        """
        Perform a depth-first search of the maze. This also generates a graph of connections and an
        adjacency matrix of the maze. The algorithm does this:
        TODO: edit this
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
        yield self.maze.energies
        visited = np.zeros(self.maze.size, dtype=int)
        accessible = np.zeros(self.maze.size, dtype=int)
        check_queue = []
        # get a random starting point that is accessible
        random_cell = self.maze.find_random_accessible()
        visited[random_cell] = 1
        accessible[random_cell] = 1
        # for the graph we are using index of the flattened maze as the identifier
        index_rc = self.maze.cell_to_node(random_cell)
        self.graph.add_node(index_rc, energy=self.maze.get_energy(random_cell))
        # for video
        yield self.maze.energies - accessible
        # take care of the neighbours of the first random cell
        neighbours = self.maze.get_neighbours(random_cell)
        for n in neighbours:
            visited[n] = 1
            index_n = self.maze.cell_to_node(n)
            self.graph.add_node(index_n, energy=self.maze.get_energy(n))
            self.graph.add_edge(index_rc, index_n)
            if self.maze.is_accessible(n):
                accessible[n] = 1
            check_queue.append(n)
        # take care of all other cells
        while len(check_queue) > 0:
            cell = check_queue.pop()
            index_cell = self.maze.cell_to_node(cell)
            neighbours = self.maze.get_neighbours(cell)
            # if neighbours visited already, don't need to bother with them
            unvis_neig = [n for n in neighbours if visited[n] == 0 and self.maze.is_accessible(n)]
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
            yield self.maze.energies - accessible
        # accessible states must be the logical inverse of the maze
        assert np.all(np.logical_not(accessible) == self.maze.energies)
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
        self.path = []

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
        self.explore()
        return self.distances[self.end_cell]

    def get_path(self, start_cell: tuple = None, end_cell: tuple = None) -> list:
        """
        Get the shortest possible path (a list of cells) between the start and end cell.

        Args:
            start_cell: tuple or None, (int, int ..) coordinates of the cell where the path starts
            end_cell: tuple or None, (int, int ..) coordinates of the cell where the path ends

        Returns:
            a list of cells starting with start_cell and ending with end_cell
        """
        self.set_start_end_cell(start_cell, end_cell)
        self.explore()
        return self.path

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
        array_to_plot = np.where(self.visited != 0, self.distances, self.maze.energies * 1000) + for_plotting
        max_value = np.max(array_to_plot[array_to_plot < 1000])
        my_cmap = cm.get_cmap("plasma")
        my_cmap.set_under("white")
        my_cmap.set_over("black")
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.plot(self.start_cell[1], self.start_cell[0], marker="o", color="white", linewidth=1.5)
        plt.plot(self.end_cell[1], self.end_cell[0], marker="x", color="black", linewidth=1.5)
        for x, y in self.path[1:-1]:
            plt.plot(y, x, marker="o", color="white", markeredgecolor="k", linewidth=0.5, markersize=4)
        plt.imshow(array_to_plot, cmap=my_cmap, vmin=0.5, vmax=max_value+1)
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
        check_queue = []
        # start with start cell
        current_cell = self.start_cell
        self.graph.add_node(self.maze.cell_to_node(current_cell), energy=self.maze.get_energy(current_cell),
                            distance=0, cell=current_cell)
        self.distances[current_cell] = 0
        # here first frame of the animation
        yield np.where(self.visited != 0, self.distances, self.maze.energies*1000) + for_plotting
        # determine accessible neighbours - their distances to
        for an in self.maze.get_accessible_neighbours(current_cell):
            # element in the priority queue must start with priority marker, here tentative distance
            # so we pack (tentative_dist, coordinates) in a tuple
            self.distances[an] = 1
            self.graph.add_node(self.maze.cell_to_node(an), energy=self.maze.get_energy(an),
                                distance=1, cell=an)
            self.graph.add_edge(self.maze.cell_to_node(current_cell), self.maze.cell_to_node(an))
            priority_an = (self.distances[an], an)
            heapq.heappush(check_queue, priority_an)
        # here snapshot for animation
        yield np.where(self.visited != 0, self.distances, self.maze.energies*1000) + for_plotting
        self.visited[current_cell] = 1
        while check_queue:
            priority_cell, current_cell = heapq.heappop(check_queue)
            # unvisited, accessible neighbours
            for n in self.maze.get_accessible_neighbours(current_cell):
                if not self.visited[n]:
                    tent_dist = self.distances[current_cell] + 1
                    if tent_dist < self.distances[n]:
                        if n in self.graph:
                            self.graph[n]["distance"] = tent_dist
                        self.distances[n] = tent_dist
                    self.graph.add_node(self.maze.cell_to_node(n), energy=self.maze.get_energy(n),
                                        distance=self.distances[n], cell=n)
                    self.graph.add_edge(self.maze.cell_to_node(current_cell), self.maze.cell_to_node(n))
                    priority_n = (self.distances[n], n)
                    heapq.heappush(check_queue, priority_n)
            self.visited[current_cell] = 1
            # here snapshot for animation
            yield np.where(self.visited != 0, self.distances, self.maze.energies * 1000) + for_plotting
            if self.visited[self.end_cell] == 1:
                # create a path by going backwards: start with end cell and always decrease in distance
                path = [self.end_cell]
                node_distances = nx.get_node_attributes(self.graph, "distance")
                while self.start_cell not in path:
                    current_cell = path[-1]
                    current_node = self.maze.cell_to_node(current_cell)
                    connected = self.graph.neighbors(current_node)
                    # add to the path a connected cell with distance 1 smaller than the distance of current node
                    for el in connected:
                        dist = node_distances[el]
                        if dist == node_distances[current_node] - 1:
                            path.append(self.maze.node_to_cell(el))
                            break
                self.path = path[::-1]
                self.adj_matrix = nx.to_numpy_matrix(self.graph)
                return


if __name__ == '__main__':
    img_path = "Images/"
    my_maze = Maze((30, 30), images_path=img_path, images_name="explore", animate=False)
    dfs_explorer = DFSExplorer(my_maze)
    dfs_explorer.explore_and_animate()
    dfs_explorer.draw_connections_graph(show=False, with_labels=True)
    dfs_explorer.get_adjacency_matrix()
    bfs_explorer = BFSExplorer(my_maze)
    bfs_explorer.explore_and_animate()
    bfs_explorer.draw_connections_graph(show=False, with_labels=True)
    bfs_explorer.get_adjacency_matrix()
    dijkstra_exp = DijkstraExplorer(my_maze)
    dijkstra_exp.explore()
    dijkstra_exp.draw_connections_graph(show=False, with_labels=True)
    dijkstra_exp.explore_and_animate()
    dijkstra_exp.visualize_distances()
    dijkstra_exp.get_adjacency_matrix()
