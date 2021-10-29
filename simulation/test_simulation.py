from .create_simulation import Simulation
from maze.create_energies import Energy

def test_point2cell():
    img_pat = "images/"
    my_energy = Energy(images_path=img_pat, images_name="energy", m=1, friction=10)
    my_energy.from_potential(size=(10, 10))
    my_simulation = Simulation(my_energy, images_path=img_pat, m=my_energy.m, friction=my_energy.friction)
    assert my_simulation._point_to_cell((-0.5, -0.9)) == (0, 2)
    assert my_simulation._point_to_cell((1.05, -0.7)) == (1, 0)
    assert my_simulation._point_to_cell((-6.15, 1.1)) == (0, 4)

def test_cell2index():
    img_pat = "images/"
    my_energy = Energy(images_path=img_pat, images_name="energy", m=1, friction=10)
    my_energy.from_potential(size=(10, 10))
    my_simulation = Simulation(my_energy, images_path=img_pat, m=my_energy.m, friction=my_energy.friction)
    assert my_simulation._cell_to_index((0, 2)) == 2
    assert my_simulation._cell_to_index((1, 7)) == 17
    assert my_simulation._cell_to_index((3, 0)) == 30