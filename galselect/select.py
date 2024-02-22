import warnings
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import tqdm

from .data import MatchingCatalogue, FeaturesIncompatibleError, Quantiles


def euclidean_distance(
    point: npt.ArrayLike,
    other: npt.ArrayLike
) -> npt.NDArray:
    """
    Compute the Euclidean distance between two points, a point and a set of
    points or two set of points with equal length. A point is defined as a
    vector of features (coordinates).

    Parameters:
    -----------
    point : array-like
        Input coordinates, can be a single vector or an array of vectors.
    other : array-like
        Input coordinates, can be a single vector or an array of vectors. If
        both `point` and `other` are arrays, they must have the same length.

    Returns:
    --------
    dist : array-like
        Eucleadean distance between points in both inputs. The output is an
        array of length equal to the longest of the two inputs.
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
        redshift_warning: Optional[float] = 0.05
    ) -> None:
        if not isinstance(mockdata, MatchingCatalogue):
            raise TypeError(
                f"input data must be of type {type(MatchingCatalogue)}")
        self.mock = mockdata
        self.z_warn = redshift_warning
        # initialise and extract data
        self.redshifts = self.mock.get_redshifts()
        self.match_count = np.zeros(len(self.mock), dtype=int)

    def redshift_window(
        self,
        redshift: float,
        d_idx: int
    ):
        """
        Compute the index range of objects closest to a given refence redshift.

        Parameters:
        -----------
        redshift : float
            Find the closest mock objects around this redshift.
        d_idx : int
            Number of mock objects closest to the input redshift (above and
            below).
        
        Returns:
        --------
        window : slice
            Slice that selects the data falling into the window.
        """
        #d_idx //= 2
        ## find the appropriate range of indices in the sorted redshift array
        ## around the target redshift
        #idx = np.searchsorted(self.redshifts, redshift)
        #idx_lo = np.maximum(idx - d_idx, 0)
        #idx_hi = np.minimum(idx + d_idx, len(self.redshifts) - 1)
        #return slice(idx_lo, idx_hi)
        if d_idx==0: 
            #Assume redshift axis is discrete and use all matching values 
            #Left returns the lowest index of all matching values 
            idx_lo = np.searchsorted(self.redshifts, redshift,side='left')
            #Right returns the highest index of all matching values 
            idx_hi = np.searchsorted(self.redshifts, redshift,side='right')
            if idx_lo == idx_hi: 
                #The current 'redshift' is not an entry in the reference list of redshifts! 
                warnings.warn(
                    f"current redshift is not in the set of reference redshifts! {redshift}")
                #print("Index returned same value for both hi and lo!")
                #print(self.redshifts)
                #print(redshift)


                #Define the lower index
                idx_lo = np.maximum(idx_lo - 100, 0)
                #Define the upper index
                idx_hi = np.minimum(idx_hi + 100, len(self.redshifts) - 1)
                #print(self.redshifts[idx_lo:idx_hi])
                #Define the new discrete redshift 
                new_z = self.redshifts[idx_lo:idx_hi][np.argmin(np.abs(self.redshifts[idx_lo:idx_hi]-redshift))]  # nearest neighbour in feature space
                #Assume redshift axis is discrete and use all matching values 
                #Left returns the lowest index of all matching values 
                idx_lo = np.searchsorted(self.redshifts, new_z,side='left')
                #Right returns the highest index of all matching values 
                idx_hi = np.searchsorted(self.redshifts, new_z,side='right')
                if idx_lo == idx_hi: 
                    raise ValueError(f"Cannot make list of reference redshifts from target redshift")
                    
        else: 
            #Split the n by 2
            d_idx //= 2
            # find the appropriate range of indices in the sorted redshift array
            # around the target redshift
            idx = np.searchsorted(self.redshifts, redshift)
            #Define the lower index
            idx_lo = np.maximum(idx - d_idx, 0)
            #Define the upper index
            idx_hi = np.minimum(idx + d_idx, len(self.redshifts) - 1)
        return slice(idx_lo, idx_hi)

    def _single_match(
        self,
        redshift: float,
        data_features: npt.NDArray,
        d_idx: int,
        duplicates: bool
    ) -> Tuple[int, dict]:
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
        match_data_dist : float
            Tha distance between the data object and the best match.
        n_candidates : int
            The number of candidates available for matching.
        z_range : float
            The width of the redshift window in which the matching occured.
        """
        # select the nearest objects in the mock data and its features used for
        # matching to the data
        window = self.redshift_window(redshift, d_idx)
        z_range = self.redshifts[window.stop-1] - self.redshifts[window.start]
        if z_range > self.z_warn:
            warnings.warn(
                f"redshift range of window exceeds dz={z_range:.3f}")
        mock_features = self.features[window]

        if not duplicates:
            # use only entries that are not matched yet
            mask = self.match_count[window] == 0
            n_candidates = np.count_nonzero(mask)
            if n_candidates == 0:
                raise ValueError(f"no unmasked entries within d_idx={d_idx:d}")
        else:
            mask = ...  # selects every entry
            n_candidates = len(mock_features)
            if n_candidates == 0:
                raise ValueError(f"no entries in the simulation that match the data redshift?")


        # find nearest unmasked mock entry
        data_dist = euclidean_distance(data_features, mock_features[mask])
        #print(data_features)
        #print(mock_features)
        #print(mock_features[mask])
        #print(data_dist)

        idx = np.argmin(data_dist)  # nearest neighbour in feature space
        match_idx = window.start + idx

        self.match_count[match_idx] += 1
        match_data_dist = data_dist[idx]
        return match_idx, match_data_dist, n_candidates, z_range

    def match_catalog(
        self,
        data: MatchingCatalogue,
        d_idx: Optional[int] = 10000,
        duplicates: Optional[bool] = False,
        normalise: Optional[bool] = True,
        progress: Optional[bool] = False
    ) -> MatchingCatalogue:
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

        Returns:
        --------
        result : MatchingCatalogue
            Catalogue of matches from the mock data that are matched to the
            input data. Additional columns with match statistics are appended
            which contain the number of neighbours after masking (n_neigh), the
            distance in feature space between the data and the match
            (dist_data), and, if return_mock_distance is given, the distance of
            the match to the next point in the mock feature space (dist_mock).
            The tabular data can be accessed through the `.data` attribute.
        quantiles : Quantiles
            Get the quantiles of the feature distributions of the data and the
            matched catalogue from the latest matching run.
        """
        if not isinstance(data, MatchingCatalogue):
            raise TypeError(
                f"input data must be of type {type(MatchingCatalogue)}")
        # check the redshift range
        mock_zlo, mock_zhi = self.mock.get_redshift_limit()
        data_zlo, data_zhi = data.get_redshift_limit()
        if data_zlo < mock_zlo or data_zhi > mock_zhi:
            raise ValueError("data redshift range exceeds the mock range")
        # check the feature compatibility
        if not self.mock.is_compatible(data):
            raise FeaturesIncompatibleError

        # compute the feature space
        if normalise is True:
            normalise = self.mock  # compute normalisation from mock data
        self.features = self.mock.get_features(normalise)
        data_features = data.get_features(normalise)

        # initialise the statistics columns
        match_stats = pd.DataFrame(index=data.data.index)
        matchcols_tmp = ["idx_match", "match_dist", "n_neigh", "z_range"]
        matchdtype_tmp = [int, float, int, float]
        coliter = zip(
            matchcols_tmp,
            matchdtype_tmp)
        for name, dtype in coliter:
            match_stats[name] = np.empty(len(data), dtype)

        # iterate the input catalogue and collect match index and statistics
        data_iter = zip(data.get_redshifts(), data_features)
        self.match_count[:] = 0  # reset counts
        try:
            if progress:
                pbar = tqdm.tqdm(total=len(data))
            for i, (redshift, entry) in enumerate(data_iter):
                match_stats.iloc[i] = self._single_match(
                    redshift, entry, d_idx, duplicates)
                if progress:
                    pbar.update()
        finally:
            if progress:
                pbar.close()
        match_stats["match_count"] = self.match_count[match_stats["idx_match"]]

        # collect data from the mock catalogue
        if self.mock.has_extra_columns():
            mockcols = self.mock.get_extra_columns()
        else:
            mockcols = self.mock.data
        mockcols = mockcols.iloc[match_stats["idx_match"]]
        mockcols.set_index(data.data.index, inplace=True)
        # collect cloned columns and rename duplicates
        datacols = data.get_extra_columns().copy()
        datacols.rename(
            columns={
                col: f"{col}_data" for col in datacols.columns
                if col in mockcols},
            inplace=True)

        mockcols.rename(
            columns={
                col: f"{col}_mock" for col in mockcols.columns
                if col in match_stats},
            inplace=True)
        mockcols.rename(
            columns={
                col: f"{col}_mock" for col in mockcols.columns
                if col in datacols.columns},
            inplace=True)
        datacols.rename(
            columns={
                col: f"{col}_data" for col in datacols.columns
                if col in match_stats},
            inplace=True)
        datacols.rename(
            columns={
                col: f"{col}_data" for col in datacols.columns
                if col in mockcols},
            inplace=True)
        # construct the matched output catalogue
        result = self.mock.template()  # copy all column attributes
        result.data = pd.concat([mockcols, match_stats, datacols], axis=1)
        print(result.data)
        print(result.data.keys())

        # store quantiles of the feature distributions for comparison
        quantiles = Quantiles(
            mock=result, mock_features=result.get_features(normalise),
            data=data, data_features=data_features)

        return result, quantiles
