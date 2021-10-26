import timeit
import numpy as np
import matplotlib.pyplot as plt
from explore_mazes import DFSExplorer, BFSExplorer
from create_mazes import Maze


def compare_dfs_to_bfs():
    side_len = list(range(10, 151, 10))
    sizes = [(i, i) for i in side_len]
    Vs = np.zeros(len(sizes))
    bfs_times = np.zeros(len(sizes))
    dfs_times = np.zeros(len(sizes))
    for i, size in enumerate(sizes):
        number = 10
        maze = Maze(size)
        explorer = BFSExplorer(maze)
        G = explorer.explore()
        bfs_times[i] = timeit.timeit(lambda: explorer.explore(), number=number)/number
        Vs[i] = G.number_of_nodes()
        explorer = DFSExplorer(maze)
        explorer.explore()
        dfs_times[i] = timeit.timeit(lambda: explorer.explore(), number=number)/number
    plt.loglog(Vs, bfs_times, label="BFS")
    plt.loglog(Vs, dfs_times, label="DFS")
    # V is number of nodes
    plt.loglog(Vs, [v/Vs[0]*bfs_times[0] for v in Vs], "-.", label="O(V)", color="k")
    plt.loglog(Vs, [(v**2)/(Vs[0]**2)*bfs_times[0] for v in Vs], "--", label="O(V²)", color="k")
    plt.title("Time of execution: BFS vs DFS")
    plt.legend()
    plt.savefig("execution_time_bfs_dfs.png", dpi=1200)
    plt.show()


if __name__ == '__main__':
    compare_dfs_to_bfs()
