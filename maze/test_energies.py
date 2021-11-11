import numpy as np
from .create_mazes import Maze
from .explore_mazes import BFSExplorer, DijkstraExplorer, DFSExplorer
from .create_energies import EnergyFromAtoms, EnergyFromPotential, EnergyFromMaze, kB, Atom


def formula(x, y):
    return 5 * (x ** 2 - 0.3) ** 2 + 10 * (y ** 2 - 0.5) ** 2


def test_creation_from_potential():
    size = (8, 10)
    my_energy = EnergyFromPotential(size, images_path="images/", images_name="test")
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
    my_energy = EnergyFromPotential(size, images_path="images/", images_name="test")
    dx = 2 / size[0]
    cell_i = (2, 3)
    cell_j = (6, 1)
    q_ij = my_energy._calculate_rates_matrix_ij(cell_i, cell_j)
    sigma = np.sqrt(2*my_energy.D)
    correct_rate = sigma ** 2 / 2 / dx ** 2 * np.sqrt(np.exp(-1/(kB*my_energy.T) * (formula(0.3, -0.7) - formula(-0.5, -0.3))))
    assert np.allclose(q_ij, correct_rate)


def test_acessible():
    size = (10, 10)
    my_energy = EnergyFromPotential(size, images_path="images/", images_name="test")
    my_energy.energy_cutoff = 100
    for i in range(size[0]):
        for j in range(size[1]):
            assert my_energy.is_accessible((i, j))
    my_energy.energy_cutoff = 3
    assert not my_energy.is_accessible((0, 4))
    assert my_energy.is_accessible((4, 0))


def test_adj_energy():
    maze = Maze((6, 6), algorithm="handmade1", images_name="test_handmade", images_path="images/")
    energy = EnergyFromMaze(maze, images_name="test_handmade", images_path="images/")
    energy.get_rates_matix()
    assert energy.adj_matrix[0, :9].astype(int).tolist()[0] == [0, 1, 0, 0, 0, 0, 0, 1, 1]
    assert np.all(energy.adj_matrix[0, 9:] == 0)


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
    my_energy = EnergyFromAtoms((8, 10), (atom_1, atom_2), grid_edges=(3.2, 4.5, -5.5, 11.5),
                                images_name="atoms", images_path="images/")
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
    my_energy = EnergyFromAtoms((8, 10), (atom_1, atom_2), grid_edges=(3.2, 4.5, -5.5, 11.5),
                                images_name="atoms", images_path="images/")
    atom_3 = Atom((3.3, 7.5 - 18.88888888888889), epsilon, sigma)
    atom_4 = Atom((4.3, 9.3 - 18.88888888888889), epsilon, sigma - 2)
    second_energy = EnergyFromAtoms((8, 10), (atom_3, atom_4), grid_edges=(3.2, 4.5, -5.5, 11.5),
                                images_name="atoms", images_path="images/")
    point = np.array((4.4, -5))
    atom_x, atom_y = my_energy.atoms[1].position
    moved_atom = np.array((atom_x, atom_y - 18.88888888888889))
    correct = np.linalg.norm(moved_atom - point)
    closest_mir = atom_2.get_closest_mirror(tuple(point), my_energy.grid_edges)
    assert np.linalg.norm(point-np.array(closest_mir)) == correct
    assert np.isclose(my_energy.get_full_potential(tuple(point)), second_energy.get_full_potential(tuple(point)))
    assert np.isclose(my_energy.get_x_derivative(tuple(point)), second_energy.get_x_derivative(tuple(point)))
    assert np.isclose(my_energy.get_y_derivative(tuple(point)), second_energy.get_y_derivative(tuple(point)))


def test_run_everything():
    img_path = "images/"
    # ------------------- ATOMS -----------------------
    epsilon = 3.18*1.6022e-22
    sigma = 5.928
    atom_1 = Atom((0.3, 20.5), epsilon, sigma)
    atom_2 = Atom((14.3, 9.3), epsilon, sigma-2)
    atom_3 = Atom((5.3, 45.3), epsilon/5, sigma)
    my_energy = EnergyFromAtoms((9, 8), (atom_1, atom_2, atom_3), grid_edges=(0, 20, 5, 50),
                                images_name="test_atoms", images_path=img_path)
    # ------------------- EXPLORERS -----------------------
    me = BFSExplorer(my_energy)
    me.explore_and_animate()
    me = DFSExplorer(my_energy)
    me.explore_and_animate()
    # ------------------- GENERAL FUNCTIONS -----------------------
    my_energy.visualize_boltzmann()
    my_energy.visualize()
    my_energy.visualize_3d()
    my_energy.visualize_rates_matrix()
    my_energy.visualize_eigenvectors(num=6, which="SR", sigma=0)
    my_energy.visualize_eigenvectors_in_maze(num=6, which="SR", sigma=0)
    my_energy.visualize_eigenvalues()
    # ------------------- MAZES -----------------------
    my_maze = Maze((15, 12), images_path=img_path, images_name="test_mazes", no_branching=False, edge_is_wall=False)
    my_energy = EnergyFromMaze(my_maze, images_path=img_path, images_name="test_mazes", friction=10)
    my_maze.visualize()
    my_energy.visualize_underlying_maze()
    # ------------------- EXPLORERS -----------------------
    me = BFSExplorer(my_energy)
    me.explore_and_animate()
    me = DFSExplorer(my_energy)
    me.explore_and_animate()
    # ------------------- GENERAL FUNCTIONS -----------------------
    my_energy.visualize_boltzmann()
    my_energy.visualize()
    my_energy.visualize_3d()
    my_energy.visualize_rates_matrix()
    my_energy.visualize_eigenvectors(num=6, which="SR", sigma=0)
    my_energy.visualize_eigenvectors_in_maze(num=6, which="SR", sigma=0)
    my_energy.visualize_eigenvalues()
    # ------------------- POTENTIAL -----------------------
    my_energy = EnergyFromPotential((12, 10), images_path=img_path, images_name="test_potential", friction=10)
    # ------------------- EXPLORERS -----------------------
    me = BFSExplorer(my_energy)
    me.explore_and_animate()
    me = DFSExplorer(my_energy)
    me.explore_and_animate()
    # ------------------- GENERAL FUNCTIONS -----------------------
    my_energy.visualize_boltzmann()
    my_energy.visualize()
    my_energy.visualize_3d()
    my_energy.visualize_rates_matrix()
    my_energy.visualize_eigenvectors(num=6, which="SR", sigma=0)
    my_energy.visualize_eigenvectors_in_maze(num=6, which="SR", sigma=0)
    my_energy.visualize_eigenvalues()
