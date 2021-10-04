import unittest
from create_mazes import Maze

all_algorithms = ["Prim", "random"]


class MazeTestCase(unittest.TestCase):

    def test_init(self):

        for alg in all_algorithms:
            test_maze = Maze(17, 8, algorithm=alg)
            # shape correct
            self.assertEqual(test_maze.maze.shape, (17, 8))
            # only 0 and 1 in the final maze
            heig, wid = test_maze.maze.shape
            for x in range(heig):
                for y in range(wid):
                    self.assertIn(test_maze.maze[x, y], [0, 1])

    def test_neighbours(self):
        test_maze = Maze(20, 30)
        # test a cell in the middle
        chosen_cell = (5, 3)
        neig = test_maze._determine_neighbours_periodic(chosen_cell)
        assert len(neig) == 4
        correct_neig = [(5, 2), (5, 4), (4, 3), (6, 3)]
        for el in correct_neig:
            self.assertIn(el, neig)
        # also test periodic boundary
        chosen_cell = (0, 29)
        neig = test_maze._determine_neighbours_periodic(chosen_cell)
        assert len(neig) == 4
        correct_neig = [(0, 28), (0, 0), (1, 29), (19, 29)]
        for el in correct_neig:
            self.assertIn(el, neig)

    def test_opposite(self):
        test_maze = Maze(6, 10)
        # hall left
        central = (5, 3)
        known_hall = (5, 2)
        correct_opposite = (5, 4)
        opposite = test_maze._determine_opposite(central, known_hall)
        self.assertEqual(opposite, correct_opposite)
        # hall below
        known_hall = (6, 3)
        correct_opposite = (4, 3)
        opposite = test_maze._determine_opposite(central, known_hall)
        self.assertEqual(opposite, correct_opposite)
        # edge cases - hall above
        central = (0, 2)
        known_hall = (5, 2)
        correct_opposite = (1, 2)
        opposite = test_maze._determine_opposite(central, known_hall)
        self.assertEqual(opposite, correct_opposite)
        # edge cases - hall right
        central = (5, 9)
        known_hall = (5, 0)
        correct_opposite = (5, 8)
        opposite = test_maze._determine_opposite(central, known_hall)
        self.assertEqual(opposite, correct_opposite)


if __name__ == '__main__':
    unittest.main()
