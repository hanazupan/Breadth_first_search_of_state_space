import numpy as np
from .create_mazes import Maze
from .explore_mazes import BFSExplorer, DijkstraExplorer, DFSExplorer
from .create_energies import Energy, kB

# TODO: test adjacency for potentials
# TODO: test run everything
# TODO: test if still works after changing orientations

def formula(x, y):
    return 5 * (x ** 2 - 0.3) ** 2 + 10 * (y ** 2 - 0.5) ** 2


def test_creation_from_potential():
    my_energy = Energy(images_path="images/", images_name="test")
    size = (8, 10)
    dx = 2 / size[0]
    dy = 2 / size[1]
    my_energy.from_potential(size=size)

    # shapes of grid and energies correct
    assert my_energy.grid_x[0, 0] == -1
    assert my_energy.grid_x[-1, -1] == 1
    assert my_energy.grid_x.shape == size
    assert my_energy.grid_y[0, 0] == -1
    assert my_energy.grid_y[-1, -1] == 1
    assert my_energy.grid_y.shape == size
    assert my_energy.energies.shape == size

    # energies symmetric (x = -x, y = -y) and cells have the energies of their middle points
    assert my_energy.energies[0, 0] == formula(-1+dx/2, -1+dy/2)
    assert my_energy.energies[1, 0] == formula(-1 + (dx/2) + dx, -1 + dy / 2)
    assert my_energy.energies[2, 3] == formula(-1 + (dx/2) + 2*dx, -1 + (dy/2) + 3*dy)
    assert np.allclose(my_energy.energies[:, 0], my_energy.energies[:, -1])
    assert np.allclose(my_energy.energies[0, :], my_energy.energies[-1, :])

    # derivatives are correct, even outside the boundaries
    assert my_energy.dV_dx(0.3) == 4 * 5 * 0.3 * (0.3 ** 2 - 0.3)
    assert my_energy.dV_dx(-15) == 4 * 5 * (-15) * ((-15) ** 2 - 0.3)
    assert my_energy.dV_dy(-0.7) == 4*10*(-0.7)*((-0.7)**2 - 0.5)
    assert my_energy.dV_dy(42) == 4 * 10 * 42 * (42 ** 2 - 0.5)

    # no periodic boundaries in use

    assert len([x for x in my_energy.get_neighbours((3, 0))]) == 3
    assert len([x for x in my_energy.get_neighbours((7, 9))]) == 2
    assert len([x for x in my_energy.get_neighbours((1, 1))]) == 4


def test_q_ij():
    my_energy = Energy(images_path="images/", images_name="test")
    size = (10, 10)
    dx = 2 / size[0]
    my_energy.from_potential(size=size)
    cell_i = (2, 3)
    cell_j = (6, 1)
    q_ij = my_energy._calculate_rates_matrix_ij(cell_i, cell_j)
    sigma = np.sqrt(2*my_energy.D)
    correct_rate = sigma ** 2 / 2 / dx ** 2 * np.sqrt(np.exp(-1/(kB*my_energy.T) * (formula(0.3, -0.7) - formula(-0.5, -0.3))))
    assert np.allclose(q_ij, correct_rate)


def test_acessible():
    my_energy = Energy(images_path="images/", images_name="test")
    size = (10, 10)
    my_energy.from_potential(size=size)
    my_energy.energy_cutoff = 100
    for i in range(size[0]):
        for j in range(size[1]):
            assert my_energy.is_accessible((i, j))
    my_energy.energy_cutoff = 3
    assert my_energy.is_accessible((0, 4))
    assert not my_energy.is_accessible((4, 0))
