import unittest
from create_mazes import Maze


class MazeTestCase(unittest.TestCase):
    def test_init(self):
        test_maze = Maze(17, 8)
        self.assertEqual(test_maze.maze.shape, (17, 8))


if __name__ == '__main__':
    unittest.main()
