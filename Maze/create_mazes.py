"""
In this file, class Maze is introduced and random mazes of different sizes can be created.
"""

# imports
import numpy as np
import matplotlib.pyplot as plt


class Maze:

    def __init__(self, height, width, algorithm='Prim'):
        """

        :param height:
        :param width:
        :param algorithm:
        """
        self.size = (height, width)
        self.maze = np.zeros(self.size)
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
        pass

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
    maze = Maze(7, 8)
    print(maze)
    maze.visualize()