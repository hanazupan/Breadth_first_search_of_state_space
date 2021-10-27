import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

example_list = [2, 33.5, 1, 8, 7, 9, 23, 11, 276, 999, 12.3, 0.7]


def window(seq, len_window):
    return [seq[k: k + len_window:len_window-1] for k in range(0, (len(seq) + 1) - len_window)]

print(window(example_list, 3))
print(window(example_list, 4))