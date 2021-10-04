"""
In this file, class Maze is introduced and mazes of different sizes can be created using Prim's
algorithm or a random distribution of cells.
"""

# imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from collections import deque
from matplotlib import colors


class Maze:

    def __init__(self, height, width, algorithm='Prim'):
        """
        Creates a maze of user-defined size. A maze is represented as a numpy array with:
        - 1 to represent a wall (high energy)
        - 0 to represent a hall (low energy)
        - 2 to represent not assigned cells (should not occur in the final maze)

        :param height: int, number of rows
        :param width: int, number of columns
        :param algorithm: string, maze generation algorithm, options ['handmade', 'Prim', 'random']
        """
        self.algorithm = algorithm
        self.size = (height, width)
        self.maze = np.full(self.size, 2)
        # list of images for animation
        self.image_list = []
        self.search_image_list = []
        if algorithm == 'handmade':
            self._create_handmade()
        elif algorithm == 'Prim':
            self._create_prim()
        elif algorithm == 'random':
            self.maze = np.random.rand(height, width)
            self.maze = np.round(self.maze)
        else:
            raise AttributeError("Not a valid algorithm choice.")

    def __repr__(self):
        """
        When using print() on a Maze object, it returns the string representation of self.maze.
        """
        return self.maze.__str__()

    def _create_handmade(self):
        """
        Only for testing. Hand-pick some cells and turn them into walls.
        :return: None
        """
        try:
            self.maze.fill(0)
            self.maze[:, 0] = 1
            self.maze[:, -1] = 1
            self.maze[0, :-3] = 1
            self.maze[0, :-2] = 1
            self.maze[2, 1:4] = 1
            self.maze[-5:, 3] = 1
        except IndexError:
            print("Please, initialize a larger maze.")

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
        self.image_list.append(self.maze.copy())
        # add walls of the cell to the wall list (periodic boundary conditions)
        neighbours = self.determine_neighbours_periodic(random_cell)
        for n in neighbours:
            create_wall(n)
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
            self.image_list.append(self.maze.copy())
        # everything unassigned becomes a wall
        self.maze[self.maze == 2] = 1
        # to get the final image for animation with no unassigned cells
        self.image_list.append(self.maze.copy())

    def animation_building_maze(self, save_as=None):
        """
        Creates an animation showing how the maze has been built.

        :param save_as: string, path and name of file where you want to save the animation
        :return: None
        """
        if self.algorithm != 'Prim':
            print("Animation only available for Prim algorithm.")
            return

        fig = plt.figure()
        im = plt.imshow(self.image_list[0], cmap='Greys', animated=True)

        def updatefig(i):
            im.set_array(self.image_list[i])
            return im,

        im.axes.get_xaxis().set_visible(False)
        im.axes.get_yaxis().set_visible(False)
        # blit=True to only redraw the parts of the animation that have changed (speeds up the generation)
        # interval determines how fast the video when played (not saved)
        anim = animation.FuncAnimation(fig, updatefig, blit=True, frames=len(self.image_list),
                                       repeat=False, interval=20)
        plt.show()
        if save_as:
            writergif = animation.PillowWriter(fps=30)
            anim.save(save_as, writer=writergif)

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

    def accessible(self, cell):
        return not self.maze[cell]

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

    def visualize(self, save_as=None):
        """
        Visualize the Maze with black squares (walls) and white squares (halls).

        :param save_as: string, path and name of file where you want to save the image of the maze
        :return: matplotlib.image.AxesImage
        """
        ax = plt.imshow(self.maze, cmap="Greys")
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if save_as:
            ax.figure.savefig(save_as, bbox_inches='tight', dpi=1200)
        plt.show()
        return ax

    def breadth_first_search(self):
        """
        Determine the connectivity graph of accessible states (states with value 0) in the maze.

        :return:
        """
        # for video
        self.search_image_list.append(self.maze.copy())
        visited = np.zeros(self.size, dtype=int)
        accessible = np.zeros(self.size, dtype=int)
        check_queue = deque()
        height, width = self.size
        # get a random starting point that is accessible
        random_cell = np.random.randint(height), np.random.randint(width)
        while self.maze[random_cell] != 0:
            random_cell = np.random.randint(height), np.random.randint(width)
        # determine neighbours of the random cell and add them to queue
        visited[random_cell] = 1
        accessible[random_cell] = 1
        # for video
        self.search_image_list.append(self.maze.copy() - accessible.copy())
        neighbours = self.determine_neighbours_periodic(random_cell)
        for n in neighbours:
            visited[n] = 1
            if self.accessible(n):
                accessible[n] = 1
                check_queue.append(n)
        while len(check_queue) > 0:
            # next point
            cell = check_queue.popleft()
            # print(cell)
            neighbours = self.determine_neighbours_periodic(cell)
            # if neighbours visited already, don't need to bother with them
            unvis_neig = [n for n in neighbours if visited[n] == 0]
            # if accessible, add to queue
            for n in unvis_neig:
                visited[n] = 1
                if self.accessible(n):
                    accessible[n] = 1
                    check_queue.append(n)
            # for video
            self.search_image_list.append(self.maze.copy() - accessible.copy())
        # accessible states are the logical inverse of the maze
        assert np.all(np.logical_not(accessible) == self.maze)

    def animation_solving_maze(self, save_as=None):
        if not self.search_image_list:
            self.breadth_first_search()
        #print(self.search_image_list[3], self.search_image_list[15])
        fig = plt.figure()
        cmap = colors.ListedColormap(['blue', 'white', 'black'])
        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        im = plt.imshow(self.search_image_list[0], animated=True, cmap=cmap, norm=norm)

        def updatefig(i):
            im.set_array(self.search_image_list[i])
            return im,

        im.axes.get_xaxis().set_visible(False)
        im.axes.get_yaxis().set_visible(False)
        # blit=True to only redraw the parts of the animation that have changed (speeds up the generation)
        # interval determines how fast the video when played (not saved)
        anim = animation.FuncAnimation(fig, updatefig, blit=True, frames=len(self.search_image_list),
                                       repeat=False, interval=20)
        plt.show()
        if save_as:
            writergif = animation.PillowWriter(fps=30)
            anim.save(save_as, writer=writergif)


if __name__ == '__main__':
    images_path = "Images/"
    maze = Maze(40, 50)
    print(maze)
    maze.visualize()
    maze.breadth_first_search()
    maze.animation_solving_maze(save_as=images_path+"animation_solve.gif")
    #maze.animation_building_maze()
