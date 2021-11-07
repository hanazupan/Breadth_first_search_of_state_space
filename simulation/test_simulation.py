from .create_simulation import Simulation
from maze.create_energies import EnergyFromPotential


def test_point2cell():
    img_pat = "images/"
    my_energy = EnergyFromPotential((10, 10), images_path=img_pat, images_name="energy", m=1, friction=10)
    my_simulation = Simulation(my_energy, images_path=img_pat)
    assert my_simulation._point_to_cell((-0.5, -0.9)) == (2, 0)
    assert my_simulation._point_to_cell((1.05, -0.7)) == (0, 1)
    assert my_simulation._point_to_cell((-6.15, 1.1)) == (4, 0)


def test_cell2index():
    img_pat = "images/"
    my_energy = EnergyFromPotential((10, 10), images_path=img_pat, images_name="energy", m=1, friction=10)
    my_simulation = Simulation(my_energy, images_path=img_pat)
    assert my_simulation._cell_to_index((0, 2)) == 2
    assert my_simulation._cell_to_index((1, 7)) == 17
    assert my_simulation._cell_to_index((3, 0)) == 30
