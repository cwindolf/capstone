'''
Store a bunch of graphs for use in figure-making.
'''

domain = [2 ** i for i in range(5, 15)]
import pickle
from lib.lsrg import lsrg
from lib.common import discrete_def
from lib.prefatt import sparse_price_tree
from lib.molloy_reed import gcc_of_mr_targeting


def load_the_graphs():
    with open('graphs.dat', 'rb') as g:
        return pickle.load(g)

# *************************************************************************** #
if __name__ == '__main__':
    # *********************************************************************** #
    f = discrete_def([0.0, 0.5, 0.25, 0.125])
    domain = [2 ** i for i in range(5, 15)]
    Ts, Gs, Tps, Gps = [], [], [], []
    for N in domain:
        print(N)
        # ******************************************************************* #
        # Empirical  ******************************************************** #
        # ******************************************************************* #

        # choosing alpha=2 for a power-law with neighbor deg dist that
        # sums up nice
        T = sparse_price_tree(N, 2)
        # Now, make Molloy&Reed graph targeting @ds, but only take its GC
        G = gcc_of_mr_targeting(T)
        # now, apply the transformation
        Gp = lsrg(G, f)
        Tp = lsrg(T, f)

        Ts.append(T)
        Gs.append(G)
        Gps.append(Gp)
        Tps.append(Tp)

    with open('graphs.dat', 'wb') as g:
        pickle.dump((domain, Ts, Gs, Tps, Gps, 3), g)

    print('Done!')
