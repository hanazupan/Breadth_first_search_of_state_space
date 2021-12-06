from ast import literal_eval

path_data_mazes = "data/mazes/"
path_data_summary = "data/mazes_summaries/"
path_energy_summary = "data/energy_summaries/"
path_energy_surfaces = "data/energy_surfaces/"
path_energy_rates = "data/sqra_rates_matrices/"
path_energy_eigen = "data/sqra_eigenvectors_eigenvalues/"


def read_summary_file(file_summary_id, summary_type="maze"):
    if summary_type == "maze":
        file_summary = path_data_summary + file_summary_id + "_summary.txt"
    elif summary_type == "energy":
        if file_summary_id.startswith("potential"):
            file_summary = path_energy_summary + "potentials/" + file_summary_id + "_summary.txt"
        elif file_summary_id.startswith("maze"):
            file_summary = path_energy_summary + "mazes/" + file_summary_id + "_summary.txt"
        elif file_summary_id.startswith("atoms"):
            file_summary = path_energy_summary + "atoms/" + file_summary_id + "_summary.txt"
        else:
            file_summary = path_energy_summary + file_summary_id + "_summary.txt"
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
    if "epsilons" in dict_properties.keys():
        dict_properties["epsilons"] = literal_eval(dict_properties["epsilons"])
        dict_properties["sigmas"] = literal_eval(dict_properties["sigmas"])
        dict_properties["atom_positions"] = literal_eval(dict_properties["atom_positions"])
    return dict_properties
