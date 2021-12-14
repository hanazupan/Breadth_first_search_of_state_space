import argparse
from plotting.plotting_energies import plot_everything_energy
from plotting.plotting_simulations import plot_everything_simulation
from simulation.create_msm import MSM
from constants import *

parser = argparse.ArgumentParser()
parser.add_argument('--name', metavar='n', type=str, nargs='?',
                    default="example000", help='Select the name id of the file that should be changed.')
parser.add_argument('--type', metavar='t', type=str, nargs='?',
                    default="simulation", help='Select what kind of plotting should be redone.')
parser.add_argument('--redo_msm', metavar='msm', type=str, nargs='?',
                    default="n", help='Select if MSM should also be re-done.')


def find_img_path(name_id: str):
    if name_id.startswith("potential"):
        return PATH_IMG_POTENTIALS
    elif name_id.startswith("maze"):
        return PATH_IMG_MAZES
    elif name_id.startswith("atoms"):
        return PATH_IMG_ATOMS
    else:
        return IMG_PATH


if __name__ == '__main__':
    my_args = parser.parse_args()
    print(f"Replotting files associated with {my_args.name} ...")
    if my_args.type == "energy" or my_args.type == "simulation":
        if my_args.redo_msm == "y":
            try:
                msm = MSM(my_args.name, images_path=find_img_path(my_args.name))
                msm.get_transitions_matrix()
                msm.get_eigenval_eigenvec(num_eigv=20, which="LR")
            except FileNotFoundError:
                print("Files required to construct MSM not found.")
        try:
            plot_everything_energy(my_args.name)
        except FileNotFoundError:
            print("This ID not associated with energy plots or data.")
    if my_args.type == "simulation":
        try:
            plot_everything_simulation(my_args.name)
        except FileNotFoundError:
            print("This ID not associated with simulation plots or data.")
    print("Done.")
