import numpy as np
from create_mazes import Maze

all_algorithms = ["Prim", "random"]


def test_init():
    for alg in all_algorithms:
        test_maze = Maze((17, 8), algorithm=alg)
        # shape correct
        assert test_maze.maze.shape == (17, 8)
        # only 0 and 1 in the final maze
        shape = test_maze.maze.shape
        for x in range(shape[0]):
            for y in range(shape[1]):
                assert test_maze.maze[x, y] in [0, 1]


def test_run_everything():
    for animate in [True, False]:
        height = np.random.randint(5, 15)
        width = np.random.randint(6, 20)
        test_maze = Maze((height, width), algorithm='Prim', animate=animate,
                         images_name="test")
        test_maze.breadth_first_search(animate=animate)
        test_maze.get_adjacency_matrix()
        test_maze.visualize(show=False)
        test_maze.draw_connections_graph(show=False, with_labels=True)
        test_maze.find_shortest_path(animate=animate)


def test_neighbours():
    test_maze = Maze((20, 30))
    # test a cell in the middle
    chosen_cell = (5, 3)
    correct_neig = [(5, 2), (5, 4), (4, 3), (6, 3)]
    for el in test_maze.get_neighbours(chosen_cell):
        assert el in correct_neig
    # also test periodic boundary
    chosen_cell = (0, 29)
    correct_neig = [(0, 28), (0, 0), (1, 29), (19, 29)]
    for el in test_maze.get_neighbours(chosen_cell):
        assert el in correct_neig


def test_neighbours_multidimensional():
    pass


def test_opposite():
    test_maze = Maze((6, 10))
    # hall left
    central = (5, 3)
    known_hall = (5, 2)
    correct_opposite = (5, 4)
    opposite = test_maze.determine_opposite(central, known_hall)
    assert opposite == correct_opposite
    # hall below
    known_hall = (6, 3)
    correct_opposite = (4, 3)
    opposite = test_maze.determine_opposite(central, known_hall)
    assert opposite == correct_opposite
    # edge cases - hall above
    central = (0, 2)
    known_hall = (5, 2)
    correct_opposite = (1, 2)
    opposite = test_maze.determine_opposite(central, known_hall)
    assert opposite == correct_opposite
    # edge cases - hall right
    central = (5, 9)
    known_hall = (5, 0)
    correct_opposite = (5, 8)
    opposite = test_maze.determine_opposite(central, known_hall)
    assert opposite == correct_opposite


def test_adjacency():
    test_maze = Maze((6, 6), algorithm="handmade1")
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
    adj = test_maze.get_adjacency_matrix()
    np.testing.assert_array_equal(correct_adj, adj)


def test_distances():
    test_maze = Maze((6, 6), algorithm="handmade1")
    start_cell = (0, 3)
    end_cell = (3, 1)
    corr_dist = 7
    dist = test_maze.find_shortest_path(start_cell, end_cell)
    assert corr_dist == dist
    # one more example
    start_cell = (4, 3)
    end_cell = (5, 5)
    corr_dist = 5
    dist = test_maze.find_shortest_path(start_cell, end_cell)
    assert corr_dist == dist
