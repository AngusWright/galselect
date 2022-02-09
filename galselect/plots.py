from re import S
import warnings

import astropandas as apd
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns


sns.set_theme(style="whitegrid")
sns.color_palette()
rc = matplotlib.rc_params()
NBINS = 100
grid_kws = dict(marginal_ticks=False,)
marginal_kws = dict(bins=NBINS, element="step", stat="density",)
joint_kws = dict(bins=NBINS,)
statline_kws = dict(color="k",)
cifill_kws = dict(color=statline_kws["color"], alpha=0.2,)
refline_kws = dict(color="0.5", lw=rc["grid.linewidth"],)


def get_ax(seaborn_fig, idx=0):
    return seaborn_fig.figure.axes[idx]


def make_bins(data, nbins=NBINS, log=False):
    low, *_, high = np.histogram_bin_edges(data, nbins)
    if log:
        return np.logspace(np.log10(low), np.log10(high), nbins)
    else:
        return np.linspace(low, high, nbins)


def make_equal_n(data, nbins=NBINS, dtype=np.float64):
    qc = pd.qcut(data, q=nbins, precision=6, duplicates="drop")
    edges = np.append(qc.categories.left, qc.categories.right[-1])
    edges = np.unique(edges.astype(dtype))
    return edges


def stats_along_xaxis(ax, df, xlabel, ylabel, bins=NBINS//2, xlog=False):
    def qlow(x):
        return x.quantile(0.1587)

    def qhigh(x):
        return x.quantile(0.8413)

    if np.isscalar(bins):
        bins = make_bins(df[xlabel], log=xlog, nbins=bins)
    centers = (bins[1:] + bins[:-1]) / 2.0
    stats = df.groupby(pd.cut(df[xlabel], bins)).agg([
        np.median, qlow, qhigh])
    y = stats[ylabel]["median"].to_numpy()
    ylow = stats[ylabel]["qlow"].to_numpy()
    yhigh = stats[ylabel]["qhigh"].to_numpy()
    ax.plot(centers, y, **statline_kws)
    ax.fill_between(centers, ylow, yhigh, **cifill_kws)


def make_figure(nrows, ncols, size=2.5):
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(0.5 + size*ncols, 0.5 + size*ncols),
        sharex=False, sharey=False)
    for i, ax in enumerate(axes.flatten()):
        for pos in ["top", "right"]:
            ax.spines[pos].set_visible(False)
        ax.grid(alpha=0.33)
    return fig, axes


class BasePlotter:

    def __init__(self, fpath):
        self.fpath = fpath

    def __enter__(self, *args, **kwargs):
        self._backend = PdfPages(self.fpath)
        return self

    def __exit__(self, *args, **kwargs):
        self._backend.close()

    def add_fig(self, fig):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.tight_layout()
        self._backend.savefig(fig)


class Plotter(BasePlotter):

    def __init__(self, fpath, mock):
        super().__init__(fpath)
        self.mock = mock

    @staticmethod
    def make_cbar_ax(fig):
        ax = fig.add_axes([0.86, 0.82, 0.02, 0.16])
        return ax

    @staticmethod
    def add_refline(ax, which, value=None):
        if which == "diag":
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            lo_hi = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
            ax.plot(lo_hi, lo_hi, **refline_kws)
            # restore original limits
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        elif which == "vert":
            ax.axvline(value, **refline_kws)
        elif which == "hor":
            ax.axhline(value, **refline_kws)

    def redshifts(self, zmock, zdata):
        log = False
        xlabel = "Redshift"
        df = pd.DataFrame({
            "type": np.append(
                ["mock"]*len(self.mock), ["data"]*len(self.mock)),
            xlabel: np.append(self.mock[zmock], self.mock[zdata])})
        g = sns.histplot(
            data=df, x=xlabel, log_scale=log,
            hue="type", legend=True, **marginal_kws)
        sns.despine()
        self.add_fig(g.figure)

    def redshift_redshift(self, zdata, zmock):
        log = [False, False]
        xlabel = "Redshift data"
        ylabel = "Redshift mock"
        df = pd.DataFrame({
            xlabel: self.mock[zdata],
            ylabel: self.mock[zmock]})
        g = sns.JointGrid(data=df, x=xlabel, y=ylabel, **grid_kws)
        cax = self.make_cbar_ax(g.figure)
        g.plot_joint(
            sns.histplot, log_scale=log,
            cbar=True, cbar_ax=cax, **joint_kws)
        g.plot_marginals(
            sns.histplot,
            **marginal_kws)
        self.add_refline(get_ax(g), which="diag")
        stats_along_xaxis(get_ax(g), df, xlabel, ylabel, xlog=log[0])
        self.add_fig(g.figure)


    def distances(self):
        log = [True, True]
        xlabel = "Match distance"
        ylabel = "Internal neighbour distance"
        df = pd.DataFrame({
            xlabel: self.mock["dist_data"],
            ylabel: self.mock["dist_mock"]})
        g = sns.JointGrid(data=df, x=xlabel, y=ylabel, **grid_kws)
        cax = self.make_cbar_ax(g.figure)
        g.plot_joint(
            sns.histplot, log_scale=log,
            cbar=True, cbar_ax=cax, **joint_kws)
        g.plot_marginals(
            sns.histplot,
            **marginal_kws)
        self.add_fig(g.figure)

    def distance_redshift(self, zmock):
        log = [False, True]
        xlabel = "Redshift"
        ylabel = "Match distance"
        df = pd.DataFrame({
            xlabel: self.mock[zmock],
            ylabel: self.mock["dist_data"]})
        g = sns.JointGrid(data=df, x=xlabel, y=ylabel, **grid_kws)
        cax = self.make_cbar_ax(g.figure)
        g.plot_joint(
            sns.histplot, log_scale=log,
            cbar=True, cbar_ax=cax, **joint_kws)
        g.plot_marginals(
            sns.histplot,
            **marginal_kws)
        stats_along_xaxis(get_ax(g), df, xlabel, ylabel, xlog=log[0])
        self.add_fig(g.figure)

    def distance_neighbours(self):
        log=[False, True]
        xlabel = "Number of available neighbours"
        ylabel = "Match distance"
        df = pd.DataFrame({
            xlabel: self.mock["n_neigh"],
            ylabel: self.mock["dist_data"]})
        g = sns.JointGrid(data=df, x=xlabel, y=ylabel, **grid_kws)
        cax = self.make_cbar_ax(g.figure)
        g.plot_joint(
            sns.histplot, log_scale=log,
            cbar=True, cbar_ax=cax, **joint_kws)
        g.plot_marginals(
            sns.histplot,
            **marginal_kws)
        stats_along_xaxis(
            get_ax(g), df, xlabel, ylabel, xlog=log[0],
            bins=make_equal_n(df[xlabel].to_numpy(), NBINS//2, np.int_))
        self.add_fig(g.figure)

    def delta_redshift_neighbours(self, zmock, zdata):
        log = [False, False]
        xlabel = "Number of available neighbours"
        ylabel = r"$z_\mathrm{mock} - z_\mathrm{data}$"
        df = pd.DataFrame({
            xlabel: self.mock["n_neigh"],
            ylabel: self.mock[zmock] - self.mock[zdata]})
        g = sns.JointGrid(data=df, x=xlabel, y=ylabel, **grid_kws)
        cax = self.make_cbar_ax(g.figure)
        g.plot_joint(
            sns.histplot, log_scale=log,
            cbar=True, cbar_ax=cax, **joint_kws)
        g.plot_marginals(
            sns.histplot,
            **marginal_kws)
        self.add_refline(get_ax(g), which="y", value=0.0)
        stats_along_xaxis(
            get_ax(g), df, xlabel, ylabel, xlog=log[0],
            bins=make_equal_n(df[xlabel].to_numpy(), NBINS//2, np.int_))
        self.add_fig(g.figure)


class Catalogue:

    def __init__(self, fpath, specname, photname, *features, fields=None):
        print(f"reading catalogue: {fpath}")
        self._data = apd.read_fits(fpath)
        self.z_spec = self._data[specname]
        self.z_phot = self._data[photname]
        if fields is None:
            self.fields = None
        else:
            self.fields = self._data[fields]
        self.features = []
        for name in features:
            self.features.append(None if name is None else self._data[fields])


class RedshiftStats(BasePlotter):

    def __init__(self, fpath):
        super().__init__(fpath)
        self._cats = []
        self.n_feat = None

    def add_catalogue(self, fpath, specname, photname, *features, fields=None):
        if self.n_feat is None:
            self.n_feat = len(features)
        else:
            assert(len(features) == self.n_feat)
        self._cats.append(Catalogue(
            fpath, specname, photname, *features, fields))

    def plot(self, labels, outlier_threshold=0.15):
        assert(len(labels) == self.n_feat)
        fig, axes = make_figure(3, 2 + self.n_feat, size=2.5)
        # iterate through catalogues
        for cat in self._cats:
            dz = (cat.z_phot - cat.z_spec) / (1.0 + cat.z_spec)
            bin_data_sets = [cat.z_spec, cat.z_phot, *cat.features]
            for i, bin_data in enumerate(bin_data_sets):
                # row 1: photo-z scatter (nMAD)
                # row 2: photo-z bias
                # row 3: outlier fraction (dz > outlier_threshold)
                # row 4: binning data distribution
                pass
