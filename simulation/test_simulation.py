from .create_simulation import Simulation
from maze.create_energies import EnergyFromPotential, EnergyFromAtoms, Atom


def test_point2cell():
    img_pat = "images/"
    my_energy = EnergyFromPotential((10, 10), images_path=img_pat, images_name="energy", m=1, friction=10)
    my_energy.pbc = True
    my_simulation = Simulation(my_energy, images_path=img_pat)
    assert my_simulation._point_to_cell((-0.5, -0.9)) == (2, 0)
    assert my_simulation._point_to_cell((1.05, -0.7)) == (0, 1)
    assert my_simulation._point_to_cell((-6.15, 1.1)) == (4, 0)
    epsilon = 3.18 * 1.6022e-22
    sigma = 5.928
    atom_1 = Atom((3.3, 7.5), epsilon, sigma)
    atom_2 = Atom((4.3, 9.3), epsilon, sigma - 2)
    my_energy = EnergyFromAtoms((8, 10), (atom_1, atom_2), grid_edges=(3.2, 4.5, -5.5, 11.5),
                                images_name="atoms", images_path=img_pat)
    my_simulation = Simulation(my_energy, images_path=img_pat)
    # within the sim box
    assert my_simulation._point_to_cell((3.15, -6)) == (0, 0)
    assert my_simulation._point_to_cell((3.4, -1.5)) == (1, 2)
    assert my_simulation._point_to_cell((4.1, 5.5)) == (5, 6)
    # with pbc
    assert my_simulation._point_to_cell((3.1, 12.5)) == (7, 0)
    assert my_simulation._point_to_cell((3.4, -9.5)) == (1, 8)
    assert my_simulation._point_to_cell((4.5+17*1.4857142857142849, 5.5-4*18.88888888888889)) == (7, 6)
    assert my_simulation._point_to_cell((4.5 - 17 * 1.4857142857142849, 5.5 + 4 * 18.88888888888889)) == (7, 6)


def test_cell2index():
    img_pat = "images/"
    my_energy = EnergyFromPotential((10, 10), images_path=img_pat, images_name="energy", m=1, friction=10)
    my_simulation = Simulation(my_energy, images_path=img_pat)
    assert my_simulation._cell_to_index((0, 2)) == 2
    assert my_simulation._cell_to_index((1, 7)) == 17
    assert my_simulation._cell_to_index((3, 0)) == 30
