import numpy as np
from .create_mazes import Maze
from .explore_mazes import BFSExplorer, DijkstraExplorer, DFSExplorer
from .create_energies import EnergyFromAtoms, EnergyFromPotential, EnergyFromMaze, kB, Atom
from plotting.plotting_energies import plot_everything_energy, plot_maze

PATH = "images/tests/"


def formula(x, y):
    return 5 * (x ** 2 - 0.3) ** 2 + 10 * (y ** 2 - 0.5) ** 2


def test_geometry():
    size = (8, 10)
    my_energy = EnergyFromPotential(size, images_path=PATH, images_name="test", grid_start=(0, 0),
                                    grid_end=(1, 2))
    assert np.allclose(np.array([1/8, 2/10]), np.array(my_energy.hs))
    assert np.allclose(np.array([2 / 10, 1 / 8]), np.array(my_energy.Ss))
    assert np.isclose(my_energy.V, 1/8 * 2/10)


def test_are_neighbours():
    size = (8, 10)
    my_energy = EnergyFromPotential(size, images_path=PATH, images_name="test", grid_start=(0, 0),
                                    grid_end=(1, 2))
    my_energy.pbc = True

    cell1 = (2, 3)
    cell2 = (2, 4)
    assert my_energy.are_neighbours(cell1, cell2)
    assert my_energy.are_neighbours(cell1, cell2, axis=1)
    assert not my_energy.are_neighbours(cell1, cell2, axis=0)

    cell1 = (0, 5)
    cell2 = (7, 5)
    assert my_energy.are_neighbours(cell1, cell2)
    assert my_energy.are_neighbours(cell1, cell2, axis=0)
    assert not my_energy.are_neighbours(cell1, cell2, axis=1)

    cell1 = (0, 9)
    cell2 = (0, 0)
    assert my_energy.are_neighbours(cell1, cell2)
    assert my_energy.are_neighbours(cell1, cell2, axis=1)
    assert not my_energy.are_neighbours(cell1, cell2, axis=0)


def test_creation_from_potential():
    size = (8, 10)
    my_energy = EnergyFromPotential(size, images_path=PATH, images_name="test", grid_start=(-1, -1),
                                    grid_end=(1, 1))
    dx = 2 / size[0]
    dy = 2 / size[1]

    # shapes of grid and energies correct
    assert my_energy.grid_x[0, 0] == -1 + dx/2
    assert my_energy.grid_x[-1, -1] == 1 - dx/2
    assert my_energy.grid_x.shape == size
    assert my_energy.grid_y[0, 0] == -1 + dy/2
    assert my_energy.grid_y[-1, -1] == 1 - dy/2
    assert my_energy.grid_y.shape == size
    assert my_energy.energies.shape == size

    # energies symmetric (x = -x, y = -y) and cells have the energies of their middle points
    assert my_energy.energies[0, 0] == formula(-1+dx/2, -1+dy/2)
    assert my_energy.energies[1, 0] == formula(-1 + (dx/2) + dx, -1 + dy / 2)
    assert my_energy.energies[2, 3] == formula(-1 + (dx/2) + 2*dx, -1 + (dy/2) + 3*dy)
    assert np.allclose(my_energy.energies[:, 0], my_energy.energies[:, -1])
    assert np.allclose(my_energy.energies[0, :], my_energy.energies[-1, :])

    # derivatives are correct, even outside the boundaries
    assert my_energy.get_x_derivative((0.3, 5)) == 4 * 5 * 0.3 * (0.3 ** 2 - 0.3)
    assert my_energy.get_x_derivative((-15, 2)) == 4 * 5 * (-15) * ((-15) ** 2 - 0.3)
    assert my_energy.get_y_derivative((0, -0.7)) == 4*10*(-0.7)*((-0.7)**2 - 0.5)
    assert my_energy.get_y_derivative((55, 42)) == 4 * 10 * 42 * (42 ** 2 - 0.5)

    # no periodic boundaries in use

    assert len([x for x in my_energy.get_neighbours((3, 0))]) == 3
    assert len([x for x in my_energy.get_neighbours((7, 9))]) == 2
    assert len([x for x in my_energy.get_neighbours((1, 1))]) == 4


def test_q_ij():
    size = (10, 10)
    my_energy = EnergyFromPotential(size, images_path=PATH, images_name="test", grid_start=(-1, -1),
                                    grid_end=(1, 1))
    dx = 2 / size[0]
    cell_i = (6, 3)
    cell_j = (6, 2)
    q_ij = my_energy._calculate_rates_matrix_ij(cell_i, cell_j)
    sigma = np.sqrt(2*my_energy.D)
    correct_rate = sigma ** 2 / 2 / dx ** 2 * np.sqrt(np.exp(-1/(kB*my_energy.temperature) * (formula(0.3, -0.5) - formula(0.3, -0.3))))
    assert np.allclose(q_ij, correct_rate)


def test_acessible():
    size = (10, 10)
    my_energy = EnergyFromPotential(size, images_path=PATH, images_name="test", grid_start=(-1, -1),
                                    grid_end=(1, 1))
    my_energy.energy_cutoff = 100
    for i in range(size[0]):
        for j in range(size[1]):
            assert my_energy.is_accessible((i, j))
    my_energy.energy_cutoff = 3
    assert not my_energy.is_accessible((0, 4))
    assert my_energy.is_accessible((4, 0))


# def test_adj_energy():
#     # NOT WORKING example too small
#     # TODO: handmade2 that is bigger
#     maze = Maze((6, 6), algorithm="handmade1", images_name="test_handmade", images_path="images/tests/")
#     energy = EnergyFromMaze(maze, images_name="test_handmade", images_path="images/tests/")
#     energy.energy_cutoff = 5
#     energy.get_rates_matix()
#     correct = np.array([0, 1, 0, 0, 0, 0, 0, 1, 1])
#     toarry = energy.adj_matrix[0, :9].astype(int).toarray()[0]
#     assert np.all([x == y for x, y in zip(correct, toarry)])
#     assert np.all([x == 0 for x in energy.adj_matrix[0, 9:].toarray()])


def test_atoms():
    epsilon = 3.18*1.6022e-22
    sigma = 5.928
    atom_pos = (3.3, 20.5)
    atom_1 = Atom(atom_pos, epsilon, sigma)
    point_pos = (4.7, 8.3)
    calc_potential = atom_1.get_potential(point_pos, (-5, 50, -5, 50))
    r = np.linalg.norm(np.array(atom_pos)-np.array(point_pos))
    # test potential
    assert np.isclose(calc_potential, 4*epsilon*((sigma/r)**12 - (sigma/r)**6))
    calc_der_x = atom_1.get_dV_dx(point_pos, (-5, 50, -5, 50))
    calc_der_y = atom_1.get_dV_dy(point_pos, (-5, 50, -5, 50))
    x, y = point_pos
    assert np.isclose(calc_der_x, 4 * epsilon * (-12 * sigma ** 12 * r ** (-13) + 6 * sigma ** 6 * r ** (-7)) * x / r)
    assert np.isclose(calc_der_y, 4 * epsilon * (-12 * sigma ** 12 * r ** (-13) + 6 * sigma ** 6 * r ** (-7)) * y / r)


def test_potential_and_derivatives_pbc():
    epsilon = 3.18 * 1.6022e-22
    sigma = 5.928
    atom_1 = Atom((3.3, 7.5), epsilon, sigma)
    atom_2 = Atom((4.3, 9.3), epsilon, sigma - 2)
    my_energy = EnergyFromAtoms((8, 10), (atom_1, atom_2), grid_start=(3.2, -5.5), grid_end=(4.5, 11.5),
                                images_name="atoms", images_path=PATH)
    # potential
    in_real_box = my_energy.get_full_potential((4.5, 5.5))
    in_imaginary_box = my_energy.get_full_potential((4.5+17*1.4857142857142849, 5.5-4*18.88888888888889))
    assert np.isclose(in_real_box, in_imaginary_box)
    in_real_box = my_energy.get_full_potential((-0.3, -7.2))
    in_imaginary_box = my_energy.get_full_potential((-0.3-22*1.4857142857142849, -7.2-8*18.88888888888889))
    assert np.isclose(in_real_box, in_imaginary_box)
    # x derivative
    in_real_box = my_energy.get_x_derivative((4.5, 5.5))
    in_imaginary_box = my_energy.get_x_derivative((4.5+17*1.4857142857142849, 5.5-4*18.88888888888889))
    assert np.isclose(in_real_box, in_imaginary_box)
    in_real_box = my_energy.get_x_derivative((4.5, 5.5))
    in_imaginary_box = my_energy.get_x_derivative((4.5 - 3 * 1.4857142857142849, 5.5 + 7 * 18.88888888888889))
    assert np.isclose(in_real_box, in_imaginary_box)
    # y derivative
    in_real_box = my_energy.get_y_derivative((12.3, 88.6))
    in_imaginary_box = my_energy.get_y_derivative((12.3 + 8 * 1.4857142857142849, 88.6 - 4 * 18.88888888888889))
    assert np.isclose(in_real_box, in_imaginary_box)


def test_closest_mirror():
    epsilon = 3.18 * 1.6022e-22
    sigma = 5.928
    atom_1 = Atom((3.3, 7.5), epsilon, sigma)
    atom_2 = Atom((4.3, 9.3), epsilon, sigma - 2)
    my_energy = EnergyFromAtoms((8, 10), (atom_1, atom_2), grid_start=(3.2, -5.5), grid_end=(4.5, 11.5),
                                images_name="atoms", images_path=PATH)
    atom_3 = Atom((3.3, 7.5 - 18.88888888888889), epsilon, sigma)
    atom_4 = Atom((4.3, 9.3 - 18.88888888888889), epsilon, sigma - 2)
    second_energy = EnergyFromAtoms((8, 10), (atom_3, atom_4), grid_start=(3.2, -5.5), grid_end=(4.5, 11.5),
                                images_name="atoms", images_path=PATH)
    point = np.array((4.4, -5))
    assert np.isclose(my_energy.get_full_potential(tuple(point)), second_energy.get_full_potential(tuple(point)))
    assert np.isclose(my_energy.get_x_derivative(tuple(point)), second_energy.get_x_derivative(tuple(point)))
    assert np.isclose(my_energy.get_y_derivative(tuple(point)), second_energy.get_y_derivative(tuple(point)))


def test_run_everything():
    # ------------------- ATOMS -----------------------
    epsilon = 3.18*1.6022e-22
    sigma = 5.928
    atom_1 = Atom((0.3, 20.5), epsilon, sigma)
    atom_2 = Atom((14.3, 9.3), epsilon, sigma-2)
    atom_3 = Atom((5.3, 45.3), epsilon/5, sigma)
    my_energy = EnergyFromAtoms((9, 8), (atom_1, atom_2, atom_3), grid_start=(0, 0), grid_end=(20, 50),
                                images_name="atoms_test", images_path=PATH)
    my_energy.get_rates_matix()
    my_energy.get_eigenval_eigenvec()
    print(my_energy.images_path)
    # ------------------- EXPLORERS -----------------------
    me = BFSExplorer(my_energy)
    me.explore_and_animate()
    me = DFSExplorer(my_energy)
    me.explore_and_animate()
    # ------------------- GENERAL FUNCTIONS -----------------------
    plot_everything_energy(my_energy.images_name, plot_rates=True)
    # ------------------- MAZES -----------------------
    my_maze = Maze((15, 12), images_path=PATH, images_name="mazes_test", no_branching=False, edge_is_wall=False)
    my_energy = EnergyFromMaze(my_maze, images_path=PATH, images_name="mazes_test", friction=10)
    my_energy.get_rates_matix()
    my_energy.get_eigenval_eigenvec()
    print(my_energy.images_path)
    plot_maze(my_maze.images_name)
    # ------------------- EXPLORERS -----------------------
    me = BFSExplorer(my_energy)
    me.explore_and_animate()
    me = DFSExplorer(my_energy)
    me.explore_and_animate()
    # ------------------- GENERAL FUNCTIONS -----------------------
    plot_everything_energy(my_energy.images_name)
    # ------------------- POTENTIAL -----------------------
    my_energy = EnergyFromPotential((12, 10), images_path=PATH, images_name="potential_test", friction=10)
    my_energy.get_rates_matix()
    my_energy.get_eigenval_eigenvec()
    print(my_energy.images_path)
    # ------------------- EXPLORERS -----------------------
    me = BFSExplorer(my_energy)
    me.explore_and_animate()
    me = DFSExplorer(my_energy)
    me.explore_and_animate()
    # ------------------- GENERAL FUNCTIONS -----------------------
    plot_everything_energy(my_energy.images_name)
