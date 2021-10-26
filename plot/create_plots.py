import matplotlib.pyplot as plt
import numpy as np

DIM_LANDSCAPE = (7.25, 4.45)
DIM_PORTRAIT = (3.45, 4.45)
DIM_SQUARE = (4.45, 4.45)

images_path = "images/"


def create_maze_plot():
    pass


def create_multiple_ax(nitems: int, max_per_row: int):
    pass


def create_multiple_maze_plots(data, max_per_row=6, name="example", **kwargs):
    """

    Args:
        data: must be a numpy array

    Returns:

    """
    dimensions = len(data.shape)
    if dimensions == 2:
        nitems = 1
        data = data[np.newaxis, :]
    elif dimensions == 3:
        nitems = len(data)
    else:
        raise ValueError("Only 2- or 3-dimensional arrays can be plotted with create_multiple_maze_plots.")
    with plt.style.context(['plot/not_animation.mplstyle', 'plot/maze_style.mplstyle']):
        full_width = DIM_LANDSCAPE[0]
        nrows = (max_per_row - 1 + nitems)//max_per_row
        ncol = min(nitems, max_per_row)
        fig = plt.figure(figsize=(full_width, nrows * full_width / ncol))
        for i in range(nitems):
            ax = fig.add_subplot(nrows, ncol, i+1)
            ax.imshow(data[i], **kwargs)

        # ax[0].set_title("Energy surface", fontsize=7)
        # for i in range(1, num + 1):
        #     array = np.full(self.size, np.max(eigenvec[:, i - 1]) + 1)
        #     for j, cell in enumerate(cell_order):
        #         array[cell] = eigenvec[j, i - 1]
        #     ax[i].imshow(array, cmap=cmap, vmax=np.max(eigenvec[:, :num + 1]), vmin=np.min(eigenvec[:, :num + 1]))
        #     ax[i].set_title(f"Eigenvector {i}", fontsize=7)
        fig.savefig(images_path + f"multiple_mazes_{name}.png")
        plt.close()


if __name__ == '__main__':
    my_data = np.random.rand(1, 30, 20)
    extra_arg = dict(cmap='RdBu_r')
    create_multiple_maze_plots(my_data, **extra_arg)