"""
In this file, class Maze is introduced and mazes of different sizes can be created using Prim's
algorithm or a random distribution of cells. The mazes can also be solved using breadth-first search
algorithm, visualized as graphs and transformed into an adjacency matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from collections import deque
from matplotlib import colors
import networkx as nx


class Maze:

    def __init__(self, height, width, algorithm='Prim', animate=False, images_path="./", images_name="maze"):
        """
        Creates a maze of user-defined size. A maze is represented as a numpy array with:
        - 1 to represent a wall (high energy)
        - 0 to represent a hall (low energy)
        - 2 to represent not assigned cells (should not occur in the final maze)

        :param height: int, number of rows
        :param width: int, number of columns
        :param algorithm: string, maze generation algorithm, options ['handmade1', 'Prim', 'random']
        :param animate: bool, whether an animation of the maze generation should be computed and saved
        :param images_path: string, path where any generated images/videos will be saved
        :param images_name: string, identifier of all images/videos generated from this maze object
        """
        self.algorithm = algorithm
        self.size = (height, width)
        self.maze = np.full(self.size, 2, dtype=int)
        # prepare empty objects for solving maze
        self.graph = None
        self.adj_matrix = None
        # prepare for saving images/gifs
        self.images_path = images_path
        self.images_name = images_name
        # start the generation of a maze
        if algorithm == 'handmade1':
            self._create_handmade1()
        elif algorithm == 'Prim':
            if animate:
                self._animate_building_maze()
            else:
                # necessary to empty the generator
                for _ in self._create_prim():
                    pass
        elif algorithm == 'random':
            self.maze = np.random.randint(0, 2, size=(height, width))
        else:
            raise AttributeError("Not a valid algorithm choice.")

    def __repr__(self):
        """
        When using print() on a Maze object, it returns the string representation of self.maze.
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

    def _create_prim(self):
        """
        Generate a maze using Prim's algorithm. From wiki (https://en.wikipedia.org/wiki/Maze_generation_algorithm):
        1. Start with a grid full of walls.
        2. Pick a cell, mark it as part of the maze. Add the walls of the cell to the wall list.
        3. While there are walls in the list:
            1. Pick a random wall from the list. If only one of the cells that the wall divides is visited, then:
                1. Make the wall a passage and mark the unvisited cell as part of the maze.
                2. Add the neighboring walls of the cell to the wall list.
            2. Remove the wall from the list.
        """
        wall_list = []

        def create_wall(cell):
            self.maze[cell] = 1
            wall_list.append(cell)

        # pick a random cell as a starting point and turn it into a hall
        height, width = self.size
        random_cell = np.random.randint(height), np.random.randint(width)
        self.maze[random_cell] = 0
        # for video
        yield self.maze.copy()
        # add walls of the cell to the wall list (periodic boundary conditions)
        neighbours = self.determine_neighbours_periodic(random_cell)
        for n in neighbours:
            create_wall(n)
        # continue until you run out of walls
        while len(wall_list) > 0:
            random_wall = random.choice(wall_list)
            neighbours = self.determine_neighbours_periodic(random_wall)
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
            yield self.maze.copy()
        # everything unassigned becomes a wall
        self.maze[self.maze == 2] = 1
        # to get the final image for animation with no unassigned cells
        yield self.maze.copy()

    def determine_neighbours_periodic(self, cell):
        """
        Given the cell (coordinate x, coordinate y) calculates the direct 4 neighbours of that cell
        in self.maze using periodic boundary conditions.

        :param cell: tuple (coo_x, coo_y), position of the cell whose neighbours we search for
        :return: list, a list of tuples with coordinates of the four neighbouring cells.
        """
        height, width = self.size
        line, column = cell
        neighbours = [
            ((line - 1) % height, column),
            (line, (column + 1) % width),
            ((line + 1) % height, column),
            (line, (column - 1) % width)
        ]
        return neighbours

    def _determine_opposite(self, central, known_hall):
        """
        Determines the coordinates of the cell obtained if you start in the known_hall cell
        and jump over the central cell. E.g. if central = X, known_hall = O and opposite = ?

        |O|X|?| or |?|X|O| or

        |O|          |?|
        |X|    or    |X|
        |?|          |O|

        :param central: tuple, coordinates of the central cell
        :param known_hall: tuple, coordinates of a cell next to the central cell
        :return: tuple, coordinates of the opposite cell
        """
        height, width = self.size
        if central == known_hall:
            raise ValueError("Opposite cell nonexistent: central and known_hall are the same cell.")
        elif central[0] == known_hall[0]:
            return central[0], (2*central[1] - known_hall[1]) % width
        elif central[1] == known_hall[1]:
            return (2*central[0] - known_hall[0]) % height, central[1]
        else:
            raise ValueError("Opposite cell nonexistent: central and known_hall are not neighbouring cells.")

    ############################################################################
    # ---------------------------   BFS    -------------------------------------
    ############################################################################

    def breadth_first_search(self, animate=False):
        """
        Perform a breadth-first search of the maze. This also generates a graph of connections and an
        adjacency matrix of the maze. The algorithm does this:
        1. Find a random accessible cell in the maze, mark it as visited and accessible
        2. Add accessible neighbours of the cell to the queue
        3. While queue not empty:
            1. Take an element from the beginning of the queue
            2. Mark all its unvisited neighbours as visited
            3. Mark all its accessible neighbours as accessible and add them to the queue

        :param animate: bool, whether to generate and save the animation of solving the maze
        """
        if animate:
            self._animate_solving_maze()
        else:
            # this needs to be done to exhaust the generator
            for _ in self._bfs():
                pass

    def accessible(self, cell):
        return not self.maze[cell]

    def _bfs(self):
        """
        Determine the connectivity graph of accessible states (states with value 0) in the maze. Determine
        the adjacency matrix. Yield images for animation of path searching.
        """
        self.graph = nx.Graph()
        # for video
        yield self.maze.copy()
        visited = np.zeros(self.size, dtype=int)
        accessible = np.zeros(self.size, dtype=int)
        check_queue = deque()
        height, width = self.size
        # get a random starting point that is accessible
        random_cell = self._find_random_accessible()
        visited[random_cell] = 1
        accessible[random_cell] = 1
        # for the graph we are using index of the flattened maze as the identifier
        index_rc = random_cell[0]*width + random_cell[1]
        self.graph.add_node(index_rc)
        # for video
        yield self.maze.copy() - accessible.copy()
        # take care of the neighbours of the first random cell
        neighbours = self.determine_neighbours_periodic(random_cell)
        for n in neighbours:
            visited[n] = 1
            if self.accessible(n):
                index_n = n[0]*width + n[1]
                self.graph.add_node(index_n)
                self.graph.add_edge(index_rc, index_n)
                accessible[n] = 1
                check_queue.append(n)
        # take care of all other cells
        while len(check_queue) > 0:
            cell = check_queue.popleft()
            index_cell = cell[0]*width + cell[1]
            neighbours = self.determine_neighbours_periodic(cell)
            # if neighbours visited already, don't need to bother with them
            unvis_neig = [n for n in neighbours if visited[n] == 0]
            for n in unvis_neig:
                visited[n] = 1
                # if accessible, add to queue
                if self.accessible(n):
                    index_n = n[0] * width + n[1]
                    self.graph.add_node(index_n)
                    self.graph.add_edge(index_cell, index_n)
                    accessible[n] = 1
                    check_queue.append(n)
            # for video
            yield self.maze.copy() - accessible.copy()
        # accessible states must be the logical inverse of the maze
        assert np.all(np.logical_not(accessible) == self.maze)
        # returns adjacency matrix - ensures the order to be left-right, top-bottom
        self.adj_matrix = nx.to_numpy_matrix(self.graph,
                                             nodelist=[i for i, x in enumerate(accessible.flatten()) if x == 1])
        # the adjacency matrix must be as long as there are accessible cells in the maze
        assert len(self.adj_matrix) == np.count_nonzero(accessible)

    ############################################################################
    # -----------------------   ANIMATIONS    ----------------------------------
    ############################################################################

    def _animate(self, iterator, name_addition="animate", **kwargs):
        """
        A general method for animation. Should only be called by more specific animation functions.

        :param iterator: an iterator that yields numpy arrays that should be visualized
        :param name_addition: how the gifs resulting from this animation process should be identified
        :param kwargs: named arguments that can be passed to plt.imshow(), e.g. cmap
        """
        height, width = self.size
        fig = plt.figure()
        im = plt.imshow(next(iterator), animated=True, **kwargs)

        def updatefig(i):
            im.set_array(i)
            return im,

        im.axes.get_xaxis().set_visible(False)
        im.axes.get_yaxis().set_visible(False)
        # blit=True to only redraw the parts of the animation that have changed (speeds up the generation)
        # interval determines how fast the video when played (not saved)
        anim = animation.FuncAnimation(fig, updatefig, blit=True, frames=iterator,
                                       repeat=False, interval=10, save_count=height * width)
        writergif = animation.PillowWriter(fps=50)
        anim.save(self.images_path + f"{name_addition}_{self.images_name}.gif", writer=writergif)

    def _animate_building_maze(self):
        """
        Creates an animation showing how the maze has been built. Colormap as follows:
            white = hall
            gray = wall
            black = unassigned
        """
        iterator = self._create_prim()
        self._animate(iterator, name_addition="building", cmap='Greys')

    def _animate_solving_maze(self):
        """
        Animate solving the maze with bfa. Color map as follows:
            blue = discovered accessible cells
            white = undiscovered accessible cells
            black = walls
        """
        iterator = self._bfs()
        # self-defined color map: -1 are halls that have been discovered and are blue; 0 undiscovered halls,
        # 1 are the walls.
        cmap = colors.ListedColormap(['blue', 'white', 'black'])
        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        self._animate(iterator, name_addition="solving", cmap=cmap, norm=norm)

    ############################################################################
    # ---------------------   PUBLIC METHODS    --------------------------------
    ############################################################################

    def visualize(self, show=True):
        """
        Visualize the Maze with black squares (walls) and white squares (halls).

        :param show: bool, should the visualization be displayed
        :return: matplotlib.image.AxesImage
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

    def draw_connections_graph(self, show=True, **kwargs):
        """
        Visualize the graph of connections between the cells of the maze.

        :param show: bool, should the visualization be displayed
        :param kwargs: named arguments that can be passed to nx.draw_kamada_kawai(), e.g. with_labels
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

    def get_adjacency_matrix(self):
        """
        Get (and create if not yet created) an adjacency matrix of the maze with breadth-first search.

        :return: numpy array, adjacency matrix, square number the size of the number of halls
        """
        if self.algorithm == 'random':
            print("Adjacency matrix not available for a randomly created maze.")
            return
        elif not np.any(self.adj_matrix):
            self.breadth_first_search()
            return self.adj_matrix
        else:
            return self.adj_matrix

    ############################################################################
    # ------------------   DIJKSTRA'S ALGORITHM    -----------------------------
    ############################################################################

    def _find_random_accessible(self):
        height, width = self.size
        cell = np.random.randint(height), np.random.randint(width)
        while not self.accessible(cell):
            cell = np.random.randint(height), np.random.randint(width)
        return cell

    def find_shortest_path(self, start_cell=None, end_cell=None):
        # if no cells provided, a random start cell
        if not start_cell:
            start_cell = self._find_random_accessible()
        elif not self.accessible(start_cell):
            raise ValueError("Start cell must lie on a passage (white cell) in the maze.")
        if not end_cell:
            end_cell = self._find_random_accessible()
        elif not self.accessible(end_cell):
            raise ValueError("End cell must lie on a passage (white cell) in the maze.")
        return self._dijkstra_connect_two(start_cell, end_cell)

    def _dijkstra_connect_two(self, start_cell, end_cell):
        print(start_cell, end_cell)
        # create empty objects
        visited = np.zeros(self.size, dtype=int)
        distances = np.full(self.size, np.inf)
        check_queue = deque()
        current_cell = start_cell
        distances[current_cell] = 0
        # here first frame of the animation
        # determine accessible neighbours - their distances to
        check_queue.extend([n for n in maze.determine_neighbours_periodic(current_cell) if self.accessible(n)])
        for n in check_queue:
            distances[n] = 1
        # here snapshot for animation
        visited[current_cell] = 1
        while check_queue:
            current_cell = check_queue.popleft()
            # unvisited, accessible neighbours
            neig = [n for n in maze.determine_neighbours_periodic(current_cell)
                    if self.accessible(n) and not visited[n]]
            for n in neig:
                tent_dist = distances[current_cell] + 1
                if tent_dist < distances[n]:
                    distances[n] = tent_dist
            check_queue.extend(neig)
            # here snapshot for animation
            visited[current_cell] = 1
            if visited[end_cell] == 1:
                return distances[end_cell]
        return distances


if __name__ == '__main__':
    path = "Images/"
    maze = Maze(11, 11, animate=False, images_path=path)
    maze.visualize(show=False)
    #maze.breadth_first_search(animate=False)
    #adjacency = maze.get_adjacency_matrix()
    #maze.draw_connections_graph(show=False, with_labels=True)
    length = maze.find_shortest_path()
    print(length)
