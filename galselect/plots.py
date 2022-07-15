import collections
import warnings

import astropandas as apd
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import median_abs_deviation as nMAD
import seaborn as sns


Statistic = collections.namedtuple("Statistic", ["name", "func", "label"])

# seaborn configuration
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
        if self.fpath is not None:
            self._backend = PdfPages(self.fpath)
        return self

    def __exit__(self, *args, **kwargs):
        if self.fpath is not None:
            self._backend.close()

    def add_fig(self, fig):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.tight_layout()
        if self.fpath is not None:
            self._backend.savefig(fig)
        return fig

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
        else:
            raise ValueError(f"invalid mode (which): {which}")


class Plotter(BasePlotter):

    def __init__(self, fpath, mock):
        super().__init__(fpath)
        self.mock = mock

    @staticmethod
    def make_cbar_ax(fig):
        ax = fig.add_axes([0.86, 0.82, 0.02, 0.16])
        return ax

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
        return self.add_fig(g.figure)

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
        return self.add_fig(g.figure)


    def distances(self):
        if "dist_mock" not in self.mock:
            print("internal neighbour distance not available - skipping ...")
            return
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
        return self.add_fig(g.figure)

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
        return self.add_fig(g.figure)

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
        return self.add_fig(g.figure)

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
        self.add_refline(get_ax(g), which="hor", value=0.0)
        stats_along_xaxis(
            get_ax(g), df, xlabel, ylabel, xlog=log[0],
            bins=make_equal_n(df[xlabel].to_numpy(), NBINS//2, np.int_))
        return self.add_fig(g.figure)


class Catalogue:

    def __init__(
            self, name, fpath, specname, photname, *features, fields=None):
        print(f"reading catalogue: {fpath}")
        self.name = name
        self._data = apd.read_fits(fpath)
        self.z_spec = self._data[specname]
        self.z_phot = self._data[photname]
        if fields is None:
            self.fields = pd.Series(
                data=np.zeros(len(self._data)), name="fields")
        else:
            self.fields = self._data[fields]
            self.fields.name = "fields"
        self.features = []
        for name in features:
            self.features.append(None if name is None else self._data[name])

    def __len__(self):
        return len(self._data)


def outlier_frac(delta_z, threshold=0.15):
    return np.count_nonzero(delta_z > threshold) / len(delta_z)


class RedshiftStats(BasePlotter):

    _key_dset = "Data set"
    _key_field = "fields"
    _key_dz = "dz"

    def __init__(self, fpath=None, out_thresh=0.15):
        super().__init__(fpath)
        self.cats = []
        self.n_feat = None
        self.labels = [r"$z_\mathsf{spec}$", r"$z_\mathsf{phot}$"]
        dz = r"\left( \frac{{{zs:} - {zp:}}}{{1 + {zs:}}} \right)".format(
            zs=self.labels[0].strip("$"), zp=self.labels[1].strip("$"))
        self.stats = (  # calculated on the redshift bias
            # row 1: photo-z scatter (nMAD)
            Statistic("nmad", nMAD, rf"$\sigma_\mathsf{{mad}}\,{dz}$"),
            # row 2: photo-z bias
            Statistic("mean", np.mean, rf"$\mu_{{\delta z}}\,{dz}$"),
            # row 3: outlier fraction (dz > out_thresh)
            Statistic("fout", outlier_frac, rf"$\xi_{{{out_thresh}}}\,{dz}$"))
        self.is_log = [False, False]

    def add_catalogue(
            self, name, fpath, specname, photname, features, fields=None):
        """
        features is a dictionary where the key is the column name and the value
        is a string used as x-axis label. If this label is used to match the
        features between the provided catalogues.
        """
        if type(fpath) is str:
            print(f"reading catalogue: {fpath}")
            data = apd.read_fits(fpath)
        else:
            data = fpath
        # collect the data
        zspec = data[specname]
        zphot = data[photname]
        df = pd.DataFrame({
            self._key_dset: name,
            self._key_field: 0 if fields is None else data[fields],
            # NOTE: check with value of dz in __init__
            self._key_dz: (zspec - zphot) / (zspec + 1.0),
            self.labels[0]: zspec,
            self.labels[1]: zphot})
        # add features
        for colname, label in features.items():
            df[label] = data[colname]
            # collect list of all labels
            if label not in self.labels:
                self.labels.append(label)
        # register
        self.cats.append(df)

    def get_stacked_column(self, label):
        cats = [
            cat[[self._key_dset, label]] for cat in self.cats if label in cat]
        return pd.concat(cats, ignore_index=True)

    def set_binscale(self, *is_log):
        if len(self.cats) is None:
            raise ValueError("binning scale must best after adding catalogue")
        elif len(is_log) != len(self.labels) - 2:
            raise ValueError(
                "number of scale flags does not match the number of optional "
                "features")
        self.is_log = [False, False, *is_log]

    @staticmethod
    def make_bins(data, nbins=30, log=False):
        lims = np.percentile(data, q=[0.5, 99.5])
        if log:
            return np.logspace(*np.log10(lims), nbins)
        else:
            return np.linspace(*lims, nbins)

    def plot(self, nbins=30):
        fig, axes = make_figure(4, len(self.labels), size=3.5)
        if len(self.is_log) == 2:
            self.is_log = [False] * len(self.labels)
        # compute the global binning and plot
        # row 4: distribution of data used for binning
        bins = {}
        for i, label in enumerate(self.labels):
            # compute the binning
            stacked = self.get_stacked_column(label)
            bins[label] = self.make_bins(
                stacked[label], nbins, log=self.is_log[i])
            # plot the histogram
            ax = axes[3, i]
            sns.histplot(
                data=stacked, x=label, hue=self._key_dset,
                bins=bins[label], #log_scale=self.is_log[i],
                common_norm=False, element="step",
                stat="density", ax=ax)
            if self.is_log[i]:
                ax.set_xscale("log")
            if i != 0:  # remove all y-axis labels except for first column
                ax.set_ylabel("")
        # plot statistics of redshift bias
        # row 1: photo-z scatter (nMAD)
        # row 2: photo-z bias
        # row 3: outlier fraction (dz > out_thresh)
        for n, data in enumerate(self.cats):
            for i, label in enumerate(self.labels):
                if label not in data:
                    continue
                binning = pd.cut(data[label], bins[label])
                # group per field and feature bin, compute main statistics
                stats = data.groupby([self._key_field, binning]).agg(**{
                    "x": (label, np.mean),
                    **{stat.name: ("dz", stat.func) for stat in self.stats}})
                # compute the statistics in each bin over all fields, if no
                # fields are provided these are all identical
                low = stats.groupby(label).quantile(0.1587)
                mid = stats.groupby(label).quantile(0.5)
                high = stats.groupby(label).quantile(0.8413)
                # plot median and 68% uncertainty of the field statistics
                for j, stat in enumerate(self.stats):
                    ax = axes[j, i]
                    ax.fill_between(
                        mid["x"], low[stat.name], high[stat.name],
                        color=f"C{n}", alpha=0.2)
                    ax.plot(mid["x"], mid[stat.name], color=f"C{n}")
                    # add missing y-axis labels
                    if i == 0:
                        ax.set_ylabel(stat.label)
                    # draw a reference line for a bias of zero
                    if j == 1:
                        self.add_refline(ax, "hor", value=0.0)
        return self.add_fig(fig)


def quantiles_mask_outliers(quantiles, n_drop=6):
    idx_extreme = np.argsort(np.diff(quantiles))[-n_drop:]
    idx_extreme.sort()
    mid = len(quantiles) // 2
    i_low = idx_extreme[idx_extreme <= mid] + 1
    i_high = idx_extreme[idx_extreme > mid]
    mask = np.zeros(len(quantiles), dtype="bool")
    mask[i_low.max():i_high.min()] = True
    return mask


def QQ_plot(
        quantiles, x_quantiles, y_quantiles, median=True, deciles=True,
        n_drop=6, ax=None):
    if ax is None:
        ax = plt.gca()
    l = ax.plot(x_quantiles, y_quantiles, lw=2)[0]
    if median:
        i_median = np.searchsorted(quantiles, 0.5)
        ax.plot(
            x_quantiles[i_median], y_quantiles[i_median],
            ls="none", color=l.get_color(), marker="o", markersize=8)
    if deciles:
        i_deciles = np.searchsorted(quantiles, np.arange(0.1, 1.0, 0.1))
        ax.plot(
            x_quantiles[i_deciles], y_quantiles[i_deciles],
            ls="none", color=l.get_color(), marker="o", markersize=5)
    ax.set_aspect("equal")
    # limits
    mask = (
        quantiles_mask_outliers(x_quantiles, n_drop) &
        quantiles_mask_outliers(y_quantiles, n_drop))
    lims = (
        min(x_quantiles[mask][0], y_quantiles[mask][0]),
        max(x_quantiles[mask][-1], y_quantiles[mask][-1]))
    ax.plot(lims, lims, color="k", lw=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)


def CDF_plot(
        quantiles, data_quantiles, mock_quantiles,
        median=True, deciles=True, n_drop=4, ax=None):
    if ax is None:
        ax = plt.gca()
    gmask = np.ones(len(quantiles), dtype="bool")
    for x, label in zip([data_quantiles, mock_quantiles], ["data", "mock"]):
        l = ax.plot(x, quantiles, lw=2, label=label)[0]
        if median:
            i_median = np.searchsorted(quantiles, 0.5)
            ax.plot(
                x[i_median], quantiles[i_median], ls="none",
                color=l.get_color(), marker="o", markersize=8)
        if deciles:
            i_deciles = np.searchsorted(quantiles, np.arange(0.1, 1.0, 0.1))
            ax.plot(
                x[i_deciles], quantiles[i_deciles], ls="none",
                color=l.get_color(), marker="o", markersize=5)
        gmask &= quantiles_mask_outliers(x, n_drop)
    ax.legend(loc="upper left")
    # limits
    lims = x[gmask][0], x[gmask][-1]
    ax.set_xlim(lims)
    ax.set_ylabel("CDF")


def PDF_plot(
        quantiles, data_quantiles, mock_quantiles,
        median=True, deciles=True, n_drop=2, smooth=2, ax=None):
    if ax is None:
        ax = plt.gca()
    y = quantiles[::smooth]
    dy = np.diff(y)
    gmask = np.ones(len(y), dtype="bool")
    for dset, label in zip([data_quantiles, mock_quantiles], ["data", "mock"]):
        x = dset[::smooth]
        xmean = (x[1:] + x[:-1]) / 2.0
        dx = np.diff(x)
        l = ax.plot(xmean, dy/dx, lw=2, label=label)[0]
        if median:
            i_median = np.searchsorted(y, 0.5)
            ax.plot(
                xmean[i_median], (dy/dx)[i_median], ls="none",
                color=l.get_color(), marker="o", markersize=8)
        if deciles:
            i_deciles = np.searchsorted(y, np.arange(0.1, 1.0, 0.1))
            ax.plot(
                xmean[i_deciles], (dy/dx)[i_deciles], ls="none",
                color=l.get_color(), marker="o", markersize=5)
        gmask &= quantiles_mask_outliers(x, n_drop)
    ax.legend(loc="upper left")
    # limits
    lims = x[gmask][0], x[gmask][-1]
    ax.set_xlim(lims)
    ax.set_ylabel("PDF")
