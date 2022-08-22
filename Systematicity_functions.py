import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

""" HEATMAP """

"""
Sort columns

in the following we will sort the columns; we will try different metrics \
to see which one works for the data
"""

""" sort matrix with specific metric """


def cluster_mtx(mtx, metric="euclidean"):
    """[summary]
    Args:
        mtx (2d array):
    Returns:
        2d array: clustered matrix
    """

    clustergrid = sns.clustermap(
        mtx, method="average", col_cluster=True, row_cluster=False, metric=metric
    )  # metric = correlation, euclidean
    plt.close()
    col_idc = clustergrid.dendrogram_col.reordered_ind

    conf_mtx_sorted = mtx[col_idc]
    conf_mtx_sorted = conf_mtx_sorted[:, col_idc]

    return conf_mtx_sorted, col_idc


""" normalise rows and cols in matrix """


def norm_conf_mtx(conf_mtx, norm_axis):
    """[summary]
    Args:
        conf_mtx: np.ndarray
        norm_axis: Optional[Union[str, Iterable[str]]] = None
    Returns:
        2d array: clustered matrix
    """

    if norm_axis is None:
        return conf_mtx

    elif type(norm_axis) is str:
        norm_axis = [norm_axis]

    for norm_axis in norm_axis:
        if norm_axis == "row":
            conf_mtx = conf_mtx / conf_mtx.max(axis=1)[:, None]

        elif norm_axis == "column":
            conf_mtx = conf_mtx / conf_mtx.max(axis=0)[None, :]

        elif norm_axis == "column_diag":
            conf_mtx = conf_mtx / conf_mtx.diagonal()[None, :]

        elif norm_axis == "row_diag":
            conf_mtx = conf_mtx / conf_mtx.diagonal()[:, None]

        elif norm_axis == "row_l1":
            conf_mtx = conf_mtx / np.sum(np.abs(conf_mtx), axis=1)[:, None]

        else:
            raise ValueError(f"Invalid axis: {norm_axis}. Must be 'row', 'column', 'row_diag', or 'column_diag'.")

    return conf_mtx


def cm2inch(x: float) -> float:
    return x / 2.54


""" plot results """


def plot_conf_mtx(conf_mtx, annot, xticklabels="auto", yticklabels="auto", vmin=None, vmax=None, center=None,
                  plt_title=None, **kwargs):
    """Plot confusion matrix.
    Args:
        conf_mtx (np.ndarray): Confusion matrix.
        center (float, optional): Value at which to center colormap to plot diverging data. Defaults to None."""

    # fig, ax = plt.subplots(figsize=(cm2inch(0.5) * len(conf_mtx), cm2inch(0.5) * len(conf_mtx)))
    fig, ax = plt.subplots(figsize=(30, 40))

    sns.heatmap(
        data=conf_mtx,
        annot=annot,
        fmt="",
        ax=ax,
        square=True,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        vmin=vmin,
        vmax=vmax,
        center=center,
        cmap='YlGnBu',
        annot_kws={"fontsize": 20},
        cbar_kws={"shrink": 0.6},
        **kwargs,
    )  # , annot_kws={"fontsize":30})
    #ax.set_ylabel("CDI words", fontsize=0)
    #ax.set_xlabel("CDI words", fontsize=0)
    #ax.set_title(plt_title, fontsize=0)
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=50)
    #ax.set_yticklabels(yticklabels, rotation=0, size=0)
    #ax.set_xticklabels(xticklabels, rotation=90, size=0)
    # plt.show()
    fig.savefig('./' + 'plots/' + plt_title + ".png", dpi=300)
    return fig

def keys_of_value(d, value):
    return [key for key, val in d.items() if val == value]