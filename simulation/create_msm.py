# internal imports
from constants import *
from plotting.read_files import read_summary_file
# standard library
from os.path import exists
# external imports
from scipy.sparse.linalg import eigs
import numpy as np
from tqdm import tqdm


class MSM:

    def __init__(self, name_id: str, images_path: str = IMG_PATH, change_tau=None):
        if change_tau:
            self.tau_array = np.array(change_tau)
        elif name_id.startswith("potential"):
            self.tau_array = np.array([5, 7, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 130, 150, 180, 200])
        elif name_id.startswith("maze"):
            self.tau_array = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 85, 100, 150, 200, 500, 700, 1000,
                                       1500, 2000, 2500, 3000, 5000, 8000, 10000])
        else:
            self.tau_array = np.array([5, 7, 10, 20, 30, 50, 70, 100, 150, 200, 300, 350, 400])
        self.images_name = name_id
        self.images_path = images_path
        self.histogram = np.load(PATH_HISTOGRAMS + f"histogram_{self.images_name}.npy")
        self.traj_cell = np.load(PATH_TRAJECTORIES + f"cell_trajectory_{self.images_name}.npy")
        try:
            self.accessible = read_summary_file(name_id, summary_type="energy")["accessible cells"]
        except KeyError:
            self.accessible = [(i, j) for i in range(self.histogram.shape[0]) for j in range(self.histogram.shape[1])]
        with open(self.path_to_summary() + f"{self.images_name}_summary.txt", "a+") as f:
            f.write(f"tau_array = {tuple(self.tau_array)}\n")
            f.write(f"len_tau = {len(self.tau_array)}\n")

    def get_transitions_matrix(self, tau_array: np.ndarray = None, noncorr: bool = False):
        """
        Obtain a set of transition matrices for different tau-s specified in tau_array.

        Args:
            tau_array: 1D array of tau values for which the transition matrices should be constructed
            noncorr: bool, should only every tau-th frame be used for MSM construction
                     (if False, use sliding window - much more expensive but throws away less data)
        Returns:
            an array of transition matrices
        """
        if tau_array:
            self.tau_array = tau_array

        def window(seq, len_window, step=1):
            # in this case always move the window by step and use all points in simulations to count transitions
            return [seq[k: k + len_window:len_window-1] for k in range(0, (len(seq)+1)-len_window, step)]

        def noncorr_window(seq, len_window):
            # in this case, only use every tau-th element for MSM. Faster but loses a lot of data
            cut_seq = seq[0:-1:len_window]
            return [[a, b] for a, b in zip(cut_seq[0:-2], cut_seq[1:])]

        cells = [(i, j) for i in range(self.histogram.shape[0]) for j in range(self.histogram.shape[1])
                 if (i, j) in self.accessible]
        all_cells = len(cells)
        for tau_i, tau in enumerate(tqdm(self.tau_array)):
            transition_matrix = np.zeros(shape=(all_cells, all_cells))
            count_per_cell = {(i, j, m, n): 0 for i, j in cells for m, n in cells}
            if not noncorr:
                window_cell = window(self.traj_cell, int(tau))
            else:
                window_cell = noncorr_window(self.traj_cell, int(tau))
            for cell_slice in window_cell:
                start_cell = cell_slice[0]
                end_cell = cell_slice[1]
                try:
                    count_per_cell[(start_cell[0], start_cell[1], end_cell[0], end_cell[1])] += 1
                except KeyError:
                    pass
            for key, value in count_per_cell.items():
                a, b, c, d = key
                start_cell = (a, b)
                end_cell = (c, d)
                i = cells.index(start_cell)
                j = cells.index(end_cell)
                transition_matrix[i, j] += value
                # enforce detailed balance
                transition_matrix[j, i] += value
            # divide each row of each matrix by the sum of that row
            sums = transition_matrix.sum(axis=-1, keepdims=True)
            sums[sums == 0] = 1
            transition_matrix = transition_matrix / sums
            np.save(PATH_MSM_TRANSITION_MATRICES + f"transition_matrix_{tau_i}_{self.images_name}", transition_matrix)

    def get_eigenval_eigenvec(self, num_eigv: int = 6, **kwargs):
        """
        Obtain eigenvectors and eigenvalues of the transition matrices.

        Args:
            num_eigv: how many eigenvalues/vectors pairs
            **kwargs: named arguments to forward to eigs()
        Returns:
            (eigenval, eigenvec) a tuple of eigenvalues and eigenvectors, first num_eigv given for all tau-s
        """
        if not exists(PATH_MSM_TRANSITION_MATRICES + f"transition_matrix_0_{self.images_name}.npy"):
            self.get_transitions_matrix()
        for tau_i, tau in enumerate(self.tau_array):
            tm = np.load(PATH_MSM_TRANSITION_MATRICES + f"transition_matrix_{tau_i}_{self.images_name}.npy")
            tm = tm.T
            eigenval, eigenvec = eigs(tm, num_eigv, maxiter=100000, tol=0, **kwargs)
            if eigenvec.imag.max() == 0 and eigenval.imag.max() == 0:
                eigenvec = eigenvec.real
                eigenval = eigenval.real
            # sort eigenvectors according to their eigenvalues
            idx = eigenval.argsort()[::-1]
            eigenval = eigenval[idx]
            eigenvec = eigenvec[:, idx]
            np.savez(PATH_MSM_EIGEN + f"eigv_{tau_i}_{self.images_name}", eigenval=eigenval, eigenvec=eigenvec)

    def path_to_summary(self):
        data_path = DATA_PATH + "simulation_summaries/"
        if self.images_name.startswith("potential"):
            data_path += "potentials/"
        elif self.images_name.startswith("maze"):
            data_path += "mazes/"
        elif self.images_name.startswith("atom"):
            data_path += "atoms/"
        return data_path
