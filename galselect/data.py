import fnmatch
from typing import List, Optional, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import tabeval  # evaluate expressions for data features
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation

from .plots import QQ_plot, CDF_plot, PDF_plot


MCType = TypeVar("MCType", bound="MatchingCatalogue")
NormaliseType = Union[bool, MCType]


def compute_norm(feature_array) -> Tuple[npt.NDArray, npt.NDArray]:
    offset = np.median(feature_array, axis=0)
    scale = median_abs_deviation(feature_array, axis=0)
    return offset, scale


def column_numeric(column: pd.Series) -> None:
    """
    Verify that a DataFrame column contains numerical data.

    Parameters:
    -----------
    column : pd.Series
        Column to check.
    
    Raises:
    -------
    TypeError
        If the column data is not numerical.
    """
    if not np.issubdtype(column, np.number):
        raise TypeError(
            f"type of column '{column.name}' is not numeric: {column.dtype}")


class FeaturesIncompatibleError(Exception):
    pass


class MatchingCatalogue(object):
    """
    Container class for tabular data for matching catalogues. This container
    provides all the required metadata and provides methods to check the
    data validity.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        Table with galaxy data.
    redshift : str
        Name of the redshift column in the input dataframe.
    feature_expressions : list of str
        List of mathematical expressions that defines a (multidimensional)
        feature space that is used to match two catalogues. The expressions
        may include constonstants and column names of the dataframes as
        variables and standard set of mathematical operators (see `tabeval`).
    sort : bool
        Sort the data by redshift increasingly.
    """

    _weights = None

    def __init__(
        self,
        dataframe: pd.DataFrame,
        redshift: str,
        feature_expressions: List[str],
    ) -> None:
        # check the redshift data
        column_numeric(dataframe[redshift])
        self._redshift = redshift
        # check the feature data
        if len(feature_expressions) == 0:
            raise ValueError("empty list of feature (expressions) provided")
        # parse the feature expressions and get a list of all required columns
        self._feature_terms = []
        self.labels = []
        columns = set()
        for expression in feature_expressions:
            term = tabeval.MathTerm.from_string(expression)
            columns.update(term.list_variables())
            self._feature_terms.append(term)
            self.labels.append(expression)
        # check the feature column data types
        for col in columns:
            column_numeric(dataframe[col])
        # initialise the optional feature weights
        self._weights = np.ones(self.n_features)
        self._extra_cols = []
        # apply redshift sorting
        self.data = dataframe.sort_values(by=redshift)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def n_features(self) -> int:
        """
        Returns the dimensionality of the feature space.
        """
        return len(self._feature_terms)

    def set_feature_weights(
        self,
        feature_weights: Union[List[float], np.ndarray]
    ) -> None:
        """
        Set a fixed weight for each dimension of the feature space. The feature
        data in each dimension is scaled by this weight.

        Parameters:
        -----------
        feature_weights : array-like
            List of weights for each feature space dimension. 
        """
        if len(feature_weights) != self.n_features:
            raise ValueError(
                f"number of features ({self.n_features}) and weights "
                f"({len(feature_weights)}) does not match")
        else:
            self._weights = feature_weights

    def set_extra_columns(
        self,
        pattern_list: List[str]
    ) -> None:
        """
        Define a list of additional columns that must exist in the internal
        tabular data. These columns can be specified using globbing * and ?
        wildcards.

        Parameters:
        -----------
        pattern_list : list of str
            List of globbing patterns that are matched against the table
            columns to create a list of column names.

        Raises:
        -------
        KeyError:
            If any of the provided patterns does not have a matching column in
            the tabular data.
        """
        # iterate all patterns find at least one match with the column data
        columns = []
        for pattern in pattern_list:
            matches = fnmatch.filter(self.data.columns, pattern)
            if len(matches) == 0:
                raise KeyError(
                    f"could not match any column to pattern '{pattern}'")
            columns.extend(m for m in matches if m not in columns)
        self._extra_cols = columns

    def is_compatible(
        self,
        other: MCType
    ) -> bool:
        """
        Check whether two MatchingCatalogue are compatible for matching by
        requiring the same number of dimenions and the same weights in each
        feature dimension.

        Parameters:
        -----------
        other : MatchingCatalogue
            Instance to check the compatibility with.

        Returns:
        --------
        compatible : bool
            Whether the two instances are compatible.
        """
        if self.n_features != other.n_features:
            return False
        if not np.array_equal(self._weights, other._weights):
            return False
        return True

    def get_redshift_limit(self) -> Tuple[float, float]:
        """
        Get the minimum and maximum redshift in the catalogue.

        Returns:
        --------
        zmin : float
            Minimum redshift.
        zmax : float
            Maximum redshift.
        """
        # make use of the fact that data is sorted by redshift
        return self.data[self._redshift].iloc[0], self.data[self._redshift].iloc[-1]

    def template(self) -> MCType:
        """
        Return a new instance of the catalogue wrapper without any data. The
        data frame in the data attribute has the same columns but is empty.
        """
        # create a new instance of the containter without duplicating the data
        new = MatchingCatalogue.__new__(MatchingCatalogue)
        attr_list = (
            "_redshift", "_feature_terms", "labels", "_weights", "_extra_cols")
        for attr in attr_list:
            setattr(new, attr, getattr(self, attr))
        new.data = pd.DataFrame(columns=self.data.columns)
        return new

    def copy(self) -> MCType:
        """
        Return a copy of the instance and the underlying tabular data.
        """
        # create a new instance of the containter
        new = self.template()
        new.data = self.data.copy()
        return new

    def apply_redshift_limit(
        self,
        lower: float,
        upper: float
    ) -> MCType:
        """
        Return a new instance of this instance that is masked to the specified
        redshift limits.

        Parameters:
        -----------
        zmin : float
            Minimum redshift.
        zmax : float
            Maximum redshift.

        Returns:
        --------
        new : MatchingCatalogue
            Instance in which entries outside the redshift limits are removed.
        """
        mask = (
            (self.data[self._redshift] >= lower) &
            (self.data[self._redshift] <= upper))
        # create a new instance of the containter with redshift mask applied
        new = self.copy()
        new.data = self.data[mask]
        return new

    def get_redshifts(self) -> npt.NDArray:
        """
        Get the data of the redshift column.

        Returns:
        --------
        z : array-like
            Data redshifts.
        """
        return self.data[self._redshift].to_numpy()

    def has_extra_columns(self) -> bool:
        """
        Check if any extra columns are specified.
        """
        return len(self._extra_cols) > 0

    def get_extra_columns(self) -> pd.DataFrame:
        """
        Get the data of the extra columns.

        Returns:
        --------
        extra : pd.DataFrame
            Subset of table columns as specified in extra columns.
        """
        return self.data[self._extra_cols]

    @property
    def features(self) -> npt.NDArray:
        """
        Get the feature space data as contiguous array. Evaluates the feature
        expressions on the tabular data and apply the optional feature weights.

        Returns:
        --------
        features : array-like
            Contiguous array of feature space data with dimension
            [features x objects], includes weights.
        """
        # evaluate the terms and convert to a numpy array
        features = [
            term(self.data) * weight
            for term, weight in zip(self._feature_terms, self._weights)]
        return np.column_stack(features)

    def get_norm(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Compute the normalisation of the feature space once and cache the
        result. The normalisation (whitening) is computed by subtracting the
        median of the feature distribution and rescaling it with the
        median absolute deviation.
        """
        if not hasattr(self, "_norm"):
            self._norm = compute_norm(self.features)
        return self._norm

    def get_features(
        self,
        normalise: Optional[NormaliseType] = None
    ) -> npt.NDArray:
        """
        Compute the feature space data including the feature weights and the
        normalisation. The normalisation is optional and can be calculated from
        an external data set.

        Parameters:
        -----------
        normalise : bool or MatchingCatalogue
            If true, compute the feature space normalisation internally, else
            provide a different instance of a MatchingCatalogue from which the
            normalisation is computed and applied to the feature data in this
            instance.
        
        Returns:
        --------
        features : array-like
            Contiguous array of feature space data with dimension
            [features x objects], includes weights and normalisation.
        """
        features = self.features
        # apply no normalisation
        if normalise is None or normalise is False:
            return features
        if normalise is not None:
            if normalise is True:
                offset, scale = self.get_norm()
            elif isinstance(normalise, MatchingCatalogue):
                if not self.is_compatible(normalise):
                    raise FeaturesIncompatibleError
                offset, scale = normalise.get_norm()
            else:
                raise TypeError(f"invalid normalisation '{type(normalise)}'")
            return (features - offset) / scale


class Quantiles:

    q = np.linspace(0.0, 1.0, 101)

    def __init__(self, mock, mock_features, data, data_features):
        self.mock_labels = mock.labels
        self.mock_features = [
            np.quantile(vals, q=self.q) for vals in mock_features.T]
        self.data_labels = data.labels
        self.data_features = [
            np.quantile(vals, q=self.q) for vals in data_features.T]

    def QQ_plot(self, i, median=True, deciles=True, n_drop=6, ax=None):
        QQ_plot(
            self.q, self.data_features[i], self.mock_features[i],
            median=median, deciles=deciles, n_drop=n_drop, ax=ax)
        if ax is None:
            ax = plt.gca()
        ax.set_xlabel(self.data_labels[i])
        ax.set_ylabel(self.mock_labels[i])

    def CDF_plot(self, i, median=True, deciles=True, n_drop=4, ax=None):
        CDF_plot(
            self.q, self.data_features[i], self.mock_features[i],
            median=median, deciles=deciles, n_drop=n_drop, ax=ax)
        if ax is None:
            ax = plt.gca()
        ax.set_xlabel(f"{self.data_labels[i]} / {self.mock_labels[i]}")

    def PDF_plot(self, i, median=True, deciles=True, ax=None, n_drop=2):
        PDF_plot(
            self.q, self.data_features[i], self.mock_features[i],
            median=median, deciles=deciles, n_drop=n_drop, ax=ax)
        if ax is None:
            ax = plt.gca()
        ax.set_xlabel(f"{self.data_labels[i]} / {self.mock_labels[i]}")
