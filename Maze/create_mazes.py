"""
In this file, class Maze is introduced and random mazes of different sizes can be created.
"""

# imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random


class Maze:

    def __init__(self, height, width, algorithm='Prim'):
        """

        :param height:
        :param width:
        :param algorithm:
        """
        self.size = (height, width)
        self.maze = np.full(self.size, 2)
        # list of images for animation
        self.image_list = []
        if algorithm == 'handmade':
            self._create_handmade()
        elif algorithm == 'Prim':
            self._create_prim()
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
        # pick a random cell as a starting point and turn it into a hall
        height, width = self.size
        random_cell = np.random.randint(height), np.random.randint(width)
        self.maze[random_cell] = 0
        self.image_list.append(self.maze.copy())
        # add walls of the cell to the wall list (periodic boundary conditions)
        wall_list = self._determine_neighbours_periodic(random_cell)
        for w in wall_list:
            self.maze[w] = 1
        assert len(wall_list) == 4
        while len(wall_list) > 0:
            random_wall = random.choice(wall_list)
            neighbours = self._determine_neighbours_periodic(random_wall)
            values = [self.maze[l, c] for (l, c) in neighbours]
            known_hall = neighbours[values.index(0.0)]
            assert len(neighbours) == len(values) == 4
            # if the opposite site 3 walls
            opposite_side = self._determine_opposite(random_wall, known_hall)
            if self.maze[opposite_side] == 2:
                # make this wall a hall
                self.maze[random_wall[0], random_wall[1]] = 0
                # add directly neighbouring walls (not halls) to the wall_list
                new_walls = [n for n, v in zip(neighbours, values) if v]
                for w in new_walls:
                    self.maze[w] = 1
                wall_list.extend(new_walls)
            wall_list.remove(random_wall)
            self.image_list.append(self.maze.copy())


    def animation_building_maze(self):
        # for producing a video
        fig = plt.figure()
        im = plt.imshow(self.image_list[0], animated=True) #cmap='Greys',

        def updatefig(i):
            im.set_array(self.image_list[i])
            return im,

        im.axes.get_xaxis().set_visible(False)
        im.axes.get_yaxis().set_visible(False)
        anim = animation.FuncAnimation(fig, updatefig, blit=True, frames=len(self.image_list),
                                       repeat=False, interval=50)
        plt.show()
        writergif = animation.PillowWriter(fps=10)
        anim.save("Images/animation.gif", writer=writergif)

    def _determine_neighbours_periodic(self, cell):
        height, width = self.size
        line, column = cell
        neighbours = [
            ((line - 1) % height, column),
            (line, (column + 1) % width),
            ((line + 1) % height, column),
            (line, (column - 1) % width)
            #((line - 1) % height, (column - 1) % width),
            #((line - 1) % height, (column + 1) % width),
            #((line + 1) % height, (column + 1) % width),
            #((line + 1) % height, (column - 1) % width),

        ]
        return neighbours

    def _determine_opposite(self, central, known_hall):
        height, width = self.size
        if central[0] == known_hall[0]:
            return (central[0], (2*central[1] - known_hall[1]) % width)
        elif central[1] == known_hall[1]:
            return ((2*central[0] - known_hall[0]) % height, central[1])
        else:
            raise ValueError("They are not neighbouring cells.")

    def _determine_direct_neighbours(self, line, column):
        height, width = self.size
        neighbours = [
            ((line - 1) % height, column),
            (line, (column + 1) % width),
            ((line + 1) % height, column),
            (line, (column + 1) % width)
        ]
        return neighbours


    def visualize(self, save_as=None):
        """
        Visualize the Maze with black squares (walls) and white squares (halls).

        :param save_as: path and name of file where you want to save the image of the maze
        :return: matplotlib.image.AxesImage
        """
        ax = plt.imshow(self.maze, cmap="Greys")
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if save_as:
            ax.figure.savefig(save_as, bbox_inches='tight')
        plt.show()
        return ax


if __name__ == '__main__':
    maze = Maze(20, 30)
    #print(maze)
    #maze.visualize()
    maze.animation_building_maze()