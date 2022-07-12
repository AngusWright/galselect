from statistics import quantiles
import warnings
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import tabeval
import tqdm
import matplotlib.pyplot as plt

from .data import MatchingCatalogue, FeaturesIncompatibleError


def euclidean_distance(point, other):
    """
    Compute the Euclidean distance between two points, a point and a set of
    points or two set of points with equal size.

    Parameters:
    -----------
    point : array_like
    other : array_like

    Returns:
    --------
    dist : array_like
    """
    dist_squares = (np.atleast_2d(other) - np.atleast_2d(point)) ** 2
    dist = np.sqrt(dist_squares.sum(axis=1))
    return dist


class DataMatcher:
    """
    Match galaxy mock data to a galaxy data catalogue based on features such as
    magnitudes. The matching is done within a window of nearest mock data
    redshift around the input data redshifts. The matching is unique such that
    each mock objects is used at most once. This essentially allows cloning the
    data catalogue using mock data with very similar properties.

    Parameters:
    -----------
    data : MatchingCatalogue
        Galaxy mock data such as redshifts and the features used for matching.
    redshift_warning : float (optional)
        Issue warnings if the redshift window of the selected mock data
        exceeds this threshold, e.g. due to a lock of objects in the redshift
        range or exhaustion of objects due to the uniqueness of the matching.
    """
    
    def __init__(
            self,
            mockdata: MatchingCatalogue,
            redshift_warning: Optional[float] = 0.05):
        if not isinstance(mockdata, MatchingCatalogue):
            raise TypeError(
                f"input data must be of type {type(MatchingCatalogue)}")
        self.mock = mockdata
        self.z_warn = redshift_warning
        # sort the mock redshift data in ascending order and create an index
        # mapping from the sorted to originally ordered mock data
        self.redshifts = self.mock.get_redshifts()
        self.z_sort_idx = np.argsort(self.redshifts)
        self.z_sorted = self.redshifts[self.z_sort_idx]

    def redshift_window(
        self,
        redshift: float,
        d_idx: int
    ):
        """
        Find a number of objects in the mock data that are closest to a given
        input redshift.

        Parameters:
        -----------
        redshift : float
            Find the closest mock objects around this redshift.
        d_idx : int
            Number of mock objects closest to the input redshift (above and
            below).
        
        Returns:
        --------
        idx_mock : array_like of int
            Indices of entries in the mock data that selects the d_idx closest
            objects around the input redshift.
        dz : float
            Redshift range covered by the selected mock objects.
        """
        d_idx //= 2
        # find the appropriate range of indices in the sorted redshift array
        # around the target redshift
        idx = np.searchsorted(self.z_sorted, redshift)
        idx_lo = np.maximum(idx - d_idx, 0)
        idx_hi = np.minimum(idx + d_idx, len(self.z_sorted) - 1)
        # map the selected indices back to the order in the mock data table
        idx_mock = self.z_sort_idx[idx_lo:idx_hi+1]
        dz = self.z_sorted[idx_hi] - self.z_sorted[idx_lo]
        return idx_mock, dz

    def _single_match(
        self,
        redshift: float,
        data_features: npt.NDArray,
        d_idx: int,
        duplicates: bool
    ):
        """
        Match a single data point in the feature space to the mock data with
        similar redshift. The method will fail if all mock objects are
        exhausted in the redshift window if duplicates are not permitted.

        Parameters:
        -----------
        redshift : float
            Match only the mock data closest to this redshift.
        data_features : array_like
            Features of the data that are compared to the mock data features.
        d_idx : int
            Size of the redshift window, total number of mock objects
            that are considered for the feature space matching.
        duplicates : bool
            Whether data duplication is allowed.

        Returns:
        --------
        match_idx : int
            Index in the mock data table of the best match to the input data.
        meta : dict
            Meta data for the match such as the number of neighbours after
            masking (n_neigh), the distance in feature space between the data
            and the match (dist_data), and the width of the redshift window.
        """
        # select the nearest objects in the mock data and its features used for
        # matching to the data
        window_idx, z_range = self.redshift_window(redshift, d_idx)
        if z_range > self.z_warn:
            warnings.warn(
                f"redshift range of window exceeds dz={z_range:.3f}")
        mock_features = self.features[window_idx]

        # check the assignment count, remove initial mask (values: -1)
        if not duplicates:
            # use only entries that are not matched yet
            mask = self.match_count[window_idx] == 0
            n_candidates = np.count_nonzero(mask)
            if n_candidates == 0:
                raise ValueError(f"no unmasked entries within d_idx={d_idx:d}")
        else:
            n_candidates = len(mock_features)
            mask = np.ones(n_candidates, dtype="bool")

        # find nearest unmasked mock entry
        data_dist = euclidean_distance(
            data_features, mock_features[mask])
        idx = np.argmin(data_dist)  # nearest neighbour in feature space
        match_data_dist = data_dist[idx]
        match_idx = window_idx[mask][idx]

        # increment the assignment counts
        self.match_count[match_idx] += 1
        meta = {
            "n_neigh": n_candidates,
            "dist_data": match_data_dist,
            "z_range": z_range}
        return match_idx, meta

    def match_catalog(
        self,
        data,
        d_idx=10000,
        duplicates=False,
        normalise=True,
        progress=False,
        return_quantiles=False
    ):
        """
        Create data catalouge by matching a data catalogue in the feature space
        to the mock data with in a window of similar redshifts. Matches
        are unique and every consequtive match is guaranteed to be a different
        mock data entry. The method will fail if the data redshift range
        exceeds the range present in the mock data or all mock objects are
        exhausted within the redshift window if no duplicates are permitted.

        Parameters:
        -----------
        data : MatchingCatalogue
            Data catalgue with data such as redshifts and the features used for
            matching.
        d_idx : int
            Size of the redshift window, total number of mock objects that are
            considered for the feature space matching.
        duplicates : bool
            Whether data duplication is allowed.
        normalise : bool
            Normalise (whiten) the feature space.
        progress : bool
            Show a progressbar for the matching operation.
        return_quantiles : bool
            Compute quantiles of the feature space distributions.

        Returns:
        --------
        catalogue : pd.DataFrame
            Catalogue of matches from the mock data that are matched to the
            input data. Additional columns with match statistics are appended
            which contain the number of neighbours after masking (n_neigh), the
            distance in feature space between the data and the match
            (dist_data), and, if return_mock_distance is given, the distance of
            the match to the next point in the mock feature space (dist_mock).
        """
        if not isinstance(data, MatchingCatalogue):
            raise TypeError(
                f"input data must be of type {type(MatchingCatalogue)}")
        # check the redshift range
        mock_lim = self.mock.get_redshift_limit()
        data_lim = data.get_redshift_limit()
        if data_lim[0] < mock_lim[0] or data_lim[1] > mock_lim[1]:
            raise ValueError("data redshift range exceeds the mock range")
        # check the feature compatibility
        if not self.mock.is_compatible(data):
            raise FeaturesIncompatibleError

        # compute the feature space
        if normalise is True:
            normalise = self.mock  # compute normalisation from mock data
        self.features = self.mock.get_features(normalise)
        data_features = data.get_features(normalise)
        # initialise the output columns
        idx_match = np.empty(len(data), dtype=int)
        match_stats = {
            "n_neigh": np.empty(len(data), dtype=int),
            "dist_data": np.empty(len(data), dtype=np.float),
            "z_range": np.empty(len(data), dtype=np.float)}
        self.match_count = np.zeros(len(self.mock), dtype=int)

        # iterate the input catalogue and collect match index and statistics
        data_iter = zip(data.get_redshifts(), data_features)
        try:
            if progress:
                pbar = tqdm.tqdm(total=len(data))
            step = 500
            for i, (redshift, entry) in enumerate(data_iter):
                idx, meta = self._single_match(
                    redshift, entry, d_idx, duplicates)
                # collect the output data
                idx_match[i] = idx
                for key, val in meta.items():
                    match_stats[key][i] = val
                if progress and i % step == 0:
                    pbar.update(step)
        finally:
            if progress:
                pbar.close()

        # construct the matched output catalogue
        if self.mock.has_extra_columns():
            mock = self.mock.get_extra_columns()
        else:
            mock = self.mock.data
        catalogue = mock.iloc[idx_match].copy(deep=True)
        for colname, values in match_stats.items():
            catalogue[colname] = values
        catalogue["match_count"] = self.match_count[idx_match]

        # clone additional columns from the data
        clone_cols = data.get_extra_columns()
        for col in clone_cols.columns:
            colname = f"{col}_data" if col in catalogue else col
            catalogue[colname] = clone_cols[col].to_numpy()

        if return_quantiles:
            mock = self.mock.copy()
            mock.data = catalogue
            quantiles = Quantiles(
                mock=mock, mock_features=mock.get_features(normalise),
                data=data, data_features=data_features)
            return catalogue, quantiles
        else:
            return catalogue


def q_mask_outliers(data, n_drop=6):
    idx_extreme = np.argsort(np.diff(data))[-n_drop:]
    idx_extreme.sort()
    mid = len(data) // 2
    i_low = idx_extreme[idx_extreme <= mid] + 1
    i_high = idx_extreme[idx_extreme > mid]
    mask = np.zeros(len(data), dtype="bool")
    mask[i_low.max():i_high.min()] = True
    return mask


class Quantiles:

    q = np.linspace(0.0, 1.0, 101)

    def __init__(self, mock, mock_features, data, data_features):
        self.mock_labels = mock.labels
        self.mock_features = [
            np.quantile(vals, q=self.q) for vals in mock_features.T]
        self.data_labels = data.labels
        self.data_features = [
            np.quantile(vals, q=self.q) for vals in data_features.T]

    def qq_plot(self, i, median=True, deciles=True, ax=None, n_drop=6):
        if ax is None:
            ax = plt.gca()
        x = self.data_features[i]
        y = self.mock_features[i]
        l = ax.plot(x, y, lw=2)[0]
        if median:
            i_median = np.searchsorted(self.q, 0.5)
            ax.plot(
                x[i_median], y[i_median], ls="none",
                color=l.get_color(), marker="o", markersize=8)
        if deciles:
            i_deciles = np.searchsorted(self.q, np.arange(0.1, 1.0, 0.1))
            ax.plot(
                x[i_deciles], y[i_deciles], ls="none",
                color=l.get_color(), marker="o", markersize=5)
        ax.set_aspect("equal")
        ax.set_xlabel(self.data_labels[i])
        ax.set_ylabel(self.mock_labels[i])
        # limits
        mask = q_mask_outliers(x, n_drop) & q_mask_outliers(y, n_drop)
        lims = min(x[mask][0], y[mask][0]), max(x[mask][-1], y[mask][-1])
        ax.plot(lims, lims, color="k", lw=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    def q_plot(self, i, median=True, deciles=True, ax=None, n_drop=4):
        if ax is None:
            ax = plt.gca()
        y = self.q
        gmask = np.ones(len(y), dtype="bool")
        for dset, label in zip(
                [self.data_features, self.mock_features], ["data", "mock"]):
            x = dset[i]
            l = ax.plot(x, y, lw=2, label=label)[0]
            if median:
                i_median = np.searchsorted(y, 0.5)
                ax.plot(
                    x[i_median], y[i_median], ls="none",
                    color=l.get_color(), marker="o", markersize=8)
            if deciles:
                i_deciles = np.searchsorted(y, np.arange(0.1, 1.0, 0.1))
                ax.plot(
                    x[i_deciles], y[i_deciles], ls="none",
                    color=l.get_color(), marker="o", markersize=5)
            gmask &= q_mask_outliers(x, n_drop)
        ax.legend(loc="upper left")
        # limits
        lims = x[gmask][0], x[gmask][-1]
        ax.set_xlim(lims)
        ax.set_xlabel(f"{self.data_labels[i]} / {self.mock_labels[i]}")
        ax.set_ylabel("CDF")

    def p_plot(self, i, median=True, deciles=True, ax=None, n_drop=2):
        if ax is None:
            ax = plt.gca()
        y = self.q[::2]
        dy = np.diff(y)
        gmask = np.ones(len(y), dtype="bool")
        for dset, label in zip(
                [self.data_features, self.mock_features], ["data", "mock"]):
            x = dset[i][::2]
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
            gmask &= q_mask_outliers(x, n_drop)
        ax.legend(loc="upper left")
        # limits
        lims = x[gmask][0], x[gmask][-1]
        ax.set_xlim(lims)
        ax.set_xlabel(f"{self.data_labels[i]} / {self.mock_labels[i]}")
        ax.set_ylabel("PDF")