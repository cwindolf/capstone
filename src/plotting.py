import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statistics import dd, jdd

figstyle = { 'dpi': 150, 'bbox_inches':'tight', 'frameon': False }


def plot_dd_and_jdd(A, title, save_to=None):
    d = dd(A)
    nbins = np.amax(d) + 1
    # Degree Distribution
    f, x = np.histogram(d, bins=np.arange(0, nbins, 1))
    # plot histogram
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey='none', sharex='none')
    ax1.bar(x[:-1], f, width=1, color='k')
    ax1.set_xlabel('d_i'); ax1.set_ylabel('count'); ax1.set_title('Degree Distribution')
    # plot loglog, fit regression
    ax2.plot(x[:-1], f, 'k.')
    ax2.set_xlabel('d_i'); ax2.set_title('LogLog')
    ax2.set_xscale('log'); ax2.set_yscale('log')
    # plot jdd
    # divider = make_axes_locatable(ax3)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    jd = jdd(A)
    im = ax3.imshow(np.rot90(jd), interpolation='nearest', 
                    cmap=plt.cm.jet, extent=(0, jd.shape[0], 0, jd.shape[1]))
    # fig.colorbar(im, cax=cax, orientation='vertical')
    ax3.set_xlabel('d_i'); ax3.set_ylabel('d_j'); ax3.set_title('Joint Degree Distribution')
    fig.suptitle(title, fontsize=18)
    if save_to is not None:
        plt.savefig(save_to, **figstyle)


def plot_dd(A):
    d = dd(A)
    nbins = np.amax(d) + 1
    # Degree Distribution
    f, x = np.histogram(d, bins=np.arange(0, nbins, 1))
    # plot histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.bar(x[:-1], f, width=1, color='gray')
    ax1.set_xlabel('d_i'); ax1.set_ylabel('count'); ax1.set_title('Degree Distribution')
    # plot loglog, fit regression
    ax2.plot(x[:-1], f, 'o', c='gray')
    ax2.set_xlabel('d_i'); ax2.set_title('LogLog')
    ax2.set_xscale('log'); ax2.set_yscale('log')


def plot_jdd(A):
    jd = jdd(A)
    # Plot 2d histogram
    plt.figure()
    plt.imshow(np.rot90(jd), interpolation='nearest', 
               cmap=plt.cm.binary, extent=(0, jd.shape[0], 0, jd.shape[1]))
    plt.xlabel('d_i'); plt.ylabel('d_j'); plt.title('Joint Degree Distribution')

def distogram(A):
    dists = dijkstra(A)
    distarr = []
    total = 0.0
    lt4 = 0.0
    for i, jd in dists.items():
        for j, d in jd.items():
            distarr.append(d)
            if d <= 4:
                lt4 += 1.0
            total += 1.0
    print(lt4 / total)
    distarr = np.array(distarr)
    print(distarr)
    nbins = np.amax(distarr) + 1
    print(nbins)
    f, x = np.histogram(distarr, bins=np.arange(0, nbins, 1))
    print(f)

    plt.figure()
    plt.bar(x[:-1], f)
    plt.show()


