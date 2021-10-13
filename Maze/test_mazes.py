import numpy as np
from create_mazes import Maze
from explore_mazes import BFSExplorer, DijkstraExplorer, DFSExplorer

all_algorithms = ["Prim", "random"]
img_path = "Images/"

def test_init():
    for alg in all_algorithms:
        test_maze = Maze((17, 8), algorithm=alg, images_path=img_path)
        # shape correct
        assert test_maze.energies.shape == (17, 8)
        # only 0 and 1 in the final maze
        shape = test_maze.energies.shape
        for x in range(shape[0]):
            for y in range(shape[1]):
                assert test_maze.energies[x, y] in [0, 1]


def test_run_everything():
    for animate in [True, False]:
        height = np.random.randint(5, 15)
        width = np.random.randint(6, 20)
        test_maze = Maze((height, width), algorithm='Prim', animate=animate,
                         images_name="test", images_path=img_path)
        test_maze.visualize(show=False)
        bfs_explorer = BFSExplorer(test_maze)
        bfs_explorer.draw_connections_graph(show=False)
        bfs_explorer.explore()
        bfs_explorer.explore_and_animate()
        dfs_explorer = DFSExplorer(test_maze)
        dfs_explorer.draw_connections_graph(show=False)
        dfs_explorer.explore()
        dfs_explorer.explore_and_animate()
        np.testing.assert_array_almost_equal(bfs_explorer.get_adjacency_matrix(), dfs_explorer.get_adjacency_matrix())
        d_explorer = DijkstraExplorer(test_maze)
        d_explorer.explore_and_animate()
        d_explorer.get_adjacency_matrix()
        d_explorer.explore()
        d_explorer.draw_connections_graph(show=False)


def test_neighbours():
    test_maze = Maze((20, 30), images_path=img_path)
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


def test_multidimensional():
    test_maze = Maze((4, 15, 3), images_path=img_path)
    cell = (3, 5, 0)
    # neighbours
    corr_neig = [(0, 5, 0), (2, 5, 0), (3, 6, 0), (3, 4, 0), (3, 5, 1), (3, 5, 2)]
    for el in test_maze.get_neighbours(cell):
        assert el in corr_neig
    # opposite
    known_hall1 = (3, 5, 1)
    corr_opposite = (3, 5, 2)
    assert test_maze.determine_opposite(cell, known_hall1) == corr_opposite
    known_hall2 = (3, 4, 0)
    corr_opposite = (3, 6, 0)
    assert test_maze.determine_opposite(cell, known_hall2) == corr_opposite


def test_cell_node():
    test_maze = Maze((4, 5, 3), images_path=img_path)
    cell1 = (1, 0, 2)
    corr_index1 = 17
    assert test_maze.cell_to_node(cell1) == corr_index1
    assert test_maze.node_to_cell(corr_index1) == cell1
    cell2 = (2, 1, 2)
    corr_index2 = 35
    assert test_maze.cell_to_node(cell2) == corr_index2
    assert test_maze.node_to_cell(corr_index2) == cell2


def test_opposite():
    test_maze = Maze((6, 10), images_path=img_path)
    # hall left
    central = (5, 3)
    known_hall = (5, 2)
    correct_opposite = (5, 4)
    opposite = test_maze.determine_opposite(central, known_hall)
    assert opposite == correct_opposite
    # hall below
    known_hall = (0, 3)
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
    test_maze = Maze((6, 6), algorithm="handmade1", images_path=img_path)
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
    bfs_explorer = BFSExplorer(test_maze)
    adj = bfs_explorer.get_adjacency_matrix()
    np.testing.assert_array_equal(correct_adj, adj)
    # depth-first search should behave in the same way
    dfs_explorer = BFSExplorer(test_maze)
    adj = dfs_explorer.get_adjacency_matrix()
    np.testing.assert_array_equal(correct_adj, adj)


def test_path():
    test_maze = Maze((6, 6), algorithm="handmade1", images_path=img_path)
    start_cell = (0, 3)
    end_cell = (3, 1)
    corr_path = [(0, 3), (0, 4), (0, 5), (0, 0), (1, 0), (2, 0), (3, 0), (3, 1)]
    dijkstra_ex = DijkstraExplorer(test_maze)
    assert dijkstra_ex.get_path(start_cell=start_cell, end_cell=end_cell) == corr_path
    start_cell = (2, 3)
    end_cell = (4, 3)
    corr_path = [(2, 3), (2, 4), (2, 5), (2, 0), (1, 0), (0, 0), (0, 5), (0, 4), (0, 3), (5, 3), (4, 3)]
    assert dijkstra_ex.get_path(start_cell=start_cell, end_cell=end_cell) == corr_path


def test_distances():
    test_maze = Maze((6, 6), algorithm="handmade1", images_path=img_path)
    start_cell = (0, 3)
    end_cell = (3, 1)
    corr_dist = 7
    d_explorer = DijkstraExplorer(test_maze)
    dist = d_explorer.get_distance(start_cell, end_cell)
    assert corr_dist == dist
    # one more example
    start_cell = (4, 3)
    end_cell = (5, 5)
    corr_dist = 5
    d_explorer = DijkstraExplorer(test_maze)
    dist = d_explorer.get_distance(start_cell, end_cell)
    assert corr_dist == dist
