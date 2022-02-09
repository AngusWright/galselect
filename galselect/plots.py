import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns


NBINS = 30
marginal_kws = dict(bins=NBINS, element="step")
joint_kws = dict(bins=NBINS)



def stats_along_xaxis(ax, df, xlabel, ylabel, nbins=NBINS):
    def qlow(x):
        return x.quantile(0.1587)

    def qhigh(x):
        return x.quantile(0.8413)

    if np.isscalar(nbins):
        bins = np.histogram_bin_edges(df[xlabel], nbins)
    centers = (bins[1:] + bins[:-1]) / 2.0
    stats = df.groupby(pd.cut(df[xlabel], bins)).agg([
        np.median, qlow, qhigh])
    y = stats[ylabel]["median"].to_numpy()
    ylow = y - stats[ylabel]["qlow"].to_numpy()
    yhigh = y + stats[ylabel]["qhigh"].to_numpy()
    ax.plot(centers, y, color="k", lw=0.7)
    ax.plot(centers, ylow, color="k", ls="--", lw=0.7)
    ax.plot(centers, yhigh, color="k", ls="--", lw=0.7)


class Plotter:

    def __init__(self, fpath, mock):
        self.fpath = fpath
        self.mock = mock

    @staticmethod
    def make_cbar_ax(fig):
        return fig.add_axes([0.88, 0.82, 0.02, 0.16])

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

    def redshifts(self, zmock, zdata):
        xlabel = "Redshift"
        df = pd.DataFrame({
            "type": np.append(
                ["mock"]*len(self.mock), ["data"]*len(self.mock)),
            xlabel: np.append(self.mock[zmock], self.mock[zdata])})
        g = sns.histplot(
            data=df, x=xlabel, hue="type", legend=True, **marginal_kws)
        sns.despine()
        self.add_fig(g.figure)

    def distances(self):
        xlabel = "Match distance"
        ylabel = "Internal neighbour distance"
        df = pd.DataFrame({
            xlabel: self.mock["dist_data"],
            ylabel: self.mock["dist_mock"]})
        g = sns.JointGrid(data=df, x=xlabel, y=ylabel, marginal_ticks=True)
        cax = self.make_cbar_ax(g.figure)
        g.plot_joint(
            sns.histplot, log_scale=True, cbar=True, cbar_ax=cax, **joint_kws)
        g.plot_marginals(
            sns.histplot, log_scale=True, **marginal_kws)
        self.add_fig(g.figure)

    def distance_redshift(self, zmock):
        xlabel = "Redshift"
        ylabel = "Match distance"
        df = pd.DataFrame({
            xlabel: self.mock[zmock],
            ylabel: self.mock["dist_data"]})
        g = sns.JointGrid(data=df, x=xlabel, y=ylabel, marginal_ticks=True)
        cax = self.make_cbar_ax(g.figure)
        g.plot_joint(
            sns.histplot, log_scale=[False, True],
            cbar=True, cbar_ax=cax, **joint_kws)
        g.plot_marginals(
            sns.histplot, log_scale=False, **marginal_kws)
        stats_along_xaxis(g.figure.axes[0], df, xlabel, ylabel)
        self.add_fig(g.figure)

    def delta_redshift(self, zmock, zdata):
        xlabel = "Redshift"
        ylabel = r"$z_\mathrm{mock} - z_\mathrm{data}$"
        df = pd.DataFrame({
            xlabel: self.mock[zmock],
            ylabel: self.mock[zmock] - self.mock[zdata]})
        g = sns.JointGrid(data=df, x=xlabel, y=ylabel, marginal_ticks=True)
        cax = self.make_cbar_ax(g.figure)
        g.plot_joint(
            sns.histplot, log_scale=False, cbar=True, cbar_ax=cax, **joint_kws)
        g.plot_marginals(
            sns.histplot, log_scale=False, **marginal_kws)
        stats_along_xaxis(g.figure.axes[0], df, xlabel, ylabel)
        self.add_fig(g.figure)

    def distance_neighbours(self):
        xlabel = "Number of available neighbours"
        ylabel = "Match distance"
        df = pd.DataFrame({
            xlabel: self.mock["n_neigh"],
            ylabel: self.mock["dist_data"]})
        g = sns.JointGrid(data=df, x=xlabel, y=ylabel, marginal_ticks=True)
        cax = self.make_cbar_ax(g.figure)
        g.plot_joint(
            sns.histplot, log_scale=[True, True],
            cbar=True, cbar_ax=cax, **joint_kws)
        g.plot_marginals(
            sns.histplot, log_scale=False, **marginal_kws)
        #stats_along_xaxis(g.figure.axes[0], df, xlabel, ylabel)
        self.add_fig(g.figure)

    def delta_redshift_neighbours(self, zmock, zdata):
        xlabel = "Number of available neighbours"
        ylabel = r"$z_\mathrm{mock} - z_\mathrm{data}$"
        df = pd.DataFrame({
            xlabel: self.mock["n_neigh"],
            ylabel: self.mock[zmock] - self.mock[zdata]})
        g = sns.JointGrid(data=df, x=xlabel, y=ylabel, marginal_ticks=True)
        cax = self.make_cbar_ax(g.figure)
        g.plot_joint(
            sns.histplot, log_scale=[True, False], cbar=True,
            cbar_ax=cax, **joint_kws)
        g.plot_marginals(
            sns.histplot, log_scale=False, **marginal_kws)
        #stats_along_xaxis(g.figure.axes[0], df, xlabel, ylabel)
        self.add_fig(g.figure)
