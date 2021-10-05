import unittest

import numpy as np

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
        neig = test_maze.determine_neighbours_periodic(chosen_cell)
        assert len(neig) == 4
        correct_neig = [(5, 2), (5, 4), (4, 3), (6, 3)]
        for el in correct_neig:
            self.assertIn(el, neig)
        # also test periodic boundary
        chosen_cell = (0, 29)
        neig = test_maze.determine_neighbours_periodic(chosen_cell)
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

    def test_adjacency(self):
        test_maze = Maze(6, 6, algorithm="handmade1")
        correct_adj = np.zeros((16, 16), dtype=int)
        correct_adj[0, 3] = 1
        correct_adj[0, 4] = 1
        correct_adj[1, 2] = 1
        correct_adj[1, 14] = 1
        correct_adj[2, 3] = 1
        correct_adj[3, 15] = 1
        correct_adj[4, 5] = 1
        correct_adj[4, 6] = 1
        correct_adj[6, 9] = 1
        correct_adj[6, 10] = 1
        correct_adj[7, 8] = 1
        correct_adj[8, 9] = 1
        correct_adj[10, 11] = 1
        correct_adj[11, 12] = 1
        correct_adj[13, 14] = 1
        # below diagonal
        correct_adj[3, 0] = 1
        correct_adj[4, 0] = 1
        correct_adj[2, 1] = 1
        correct_adj[14, 1] = 1
        correct_adj[3, 2] = 1
        correct_adj[15, 3] = 1
        correct_adj[5, 4] = 1
        correct_adj[6, 4] = 1
        correct_adj[9, 6] = 1
        correct_adj[10, 6] = 1
        correct_adj[8, 7] = 1
        correct_adj[9, 8] = 1
        correct_adj[11, 10] = 1
        correct_adj[12, 11] = 1
        correct_adj[14, 13] = 1
        # assert diagonally symmetrical
        assert (correct_adj == correct_adj.T).all()
        # compare to the created one
        adj = test_maze.breadth_first_search()
        np.testing.assert_array_equal(correct_adj, adj)


if __name__ == '__main__':
    unittest.main()
