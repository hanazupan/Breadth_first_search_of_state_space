import argparse
from plotting.plotting_energies import plot_everything_energy
from plotting.plotting_simulations import plot_everything_simulation
import time
from os.path import exists

parser = argparse.ArgumentParser()
parser.add_argument('--name', metavar='n', type=str, nargs='?',
                    default="example000", help='Select the name id of the file that should be changed.')
parser.add_argument('--type', metavar='t', type=str, nargs='?',
                    default="simulation", help='Select what kind of plotting should be redone.')


if __name__ == '__main__':
    my_args = parser.parse_args()
    print(f"Replotting files associated with {my_args.name} ...")
    if my_args.type == "energy" or my_args.type == "simulation":
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
