from constants import *
from ast import literal_eval
import numpy as np
from scipy.sparse import load_npz


def read_summary_file(file_summary_id, summary_type="maze"):
    if summary_type == "maze":
        file_summary = PATH_MAZES_SUMMARY + file_summary_id + "_summary.txt"
    elif summary_type == "energy":
        if file_summary_id.startswith("potential"):
            file_summary = PATH_ENERGY_SUMMARY + "potentials/" + file_summary_id + "_summary.txt"
        elif file_summary_id.startswith("maze"):
            file_summary = PATH_ENERGY_SUMMARY + "mazes/" + file_summary_id + "_summary.txt"
        elif file_summary_id.startswith("atoms"):
            file_summary = PATH_ENERGY_SUMMARY + "atoms/" + file_summary_id + "_summary.txt"
        else:
            file_summary = PATH_ENERGY_SUMMARY + file_summary_id + "_summary.txt"
    else:
        raise ValueError("Not an existing summary_type.")
    dict_properties = dict()
    with open(file_summary, "r") as f:
        for line in f:
            if not line.startswith("#"):
                key, value = line.strip().split(" = ")
                dict_properties[key] = value
    dict_properties["size"] = literal_eval(dict_properties["size"])
    if summary_type == "maze":
        dict_properties["edge_is_wall"] = bool(dict_properties["edge_is_wall"])
        dict_properties["no_branching"] = bool(dict_properties["no_branching"])
    else:
        dict_properties["energy cutoff"] = float(dict_properties["energy cutoff"])
        dict_properties["grid_start"] = literal_eval(dict_properties["grid_start"])
        dict_properties["grid_end"] = literal_eval(dict_properties["grid_end"])
        dict_properties["mass"] = float(dict_properties["mass"])
        dict_properties["friction"] = float(dict_properties["friction"])
        dict_properties["temperature"] = float(dict_properties["temperature"])
        dict_properties["D"] = float(dict_properties["D"])
        dict_properties["hs"] = literal_eval(dict_properties["hs"])
        dict_properties["Ss"] = literal_eval(dict_properties["Ss"])
        dict_properties["V"] = float(dict_properties["V"])
    if "epsilons" in dict_properties.keys():
        dict_properties["epsilons"] = literal_eval(dict_properties["epsilons"])
        dict_properties["sigmas"] = literal_eval(dict_properties["sigmas"])
        dict_properties["atom_positions"] = literal_eval(dict_properties["atom_positions"])
    if "accessible cells" in dict_properties.keys():
        dict_properties["accessible cells"] = literal_eval(dict_properties["accessible cells"])
    if "factor" in dict_properties.keys():
        dict_properties["factor"] = int(dict_properties["factor"])
    return dict_properties


def read_everything_energies(file_id: str):
    dict_properties = read_summary_file(file_id, summary_type="energy")
    energies = np.load(PATH_ENERGY_SURFACES + "surface_" + file_id + ".npy")
    grids = np.load(f"data/energy_grids/grid_x_y_{file_id}.npz")
    grid_x = grids["x"]
    grid_y = grids["y"]
    if file_id.startswith("maze"):
        underlying_maze = np.load(PATH_ENERGY_SURFACES + f"underlaying_maze_{file_id}.npy")
    rates_matrix = load_npz(PATH_ENERGY_RATES + f"rates_{file_id}.npz")
    eigvs = np.load(PATH_ENERGY_EIGEN + f"eigv_{file_id}.npz")
    eigenvec = eigvs["eigenvec"]
    eigenval = eigvs["eigenval"]
    if file_id.startswith("maze"):
        return dict_properties, energies, grid_x, grid_y, rates_matrix, eigenvec, eigenval, underlying_maze
    else:
        return dict_properties, energies, grid_x, grid_y, rates_matrix, eigenvec, eigenval


