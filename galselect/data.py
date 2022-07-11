import fnmatch
from pyexpat import features
from typing import List, Optional, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import tabeval  # evaluate expressions for data features
from scipy.stats import median_abs_deviation


MCType = TypeVar("MCType", bound="MatchingCatalogue")
NormaliseType = Union[bool, MCType]


def compute_norm(feature_array) -> Tuple[npt.NDArray, npt.NDArray]:
    offset = np.median(feature_array, axis=0)
    scale = median_abs_deviation(feature_array, axis=0)
    return offset, scale


class FeaturesIncompatibleError(Exception):
    pass


class MatchingCatalogue(object):
    """
    Container class for tabular data for matching catalogues. This container
    provides all the required metadata and provides methods to check the
    data validity.
    """

    _weights = None

    def __init__(
        self,
        dataframe: pd.DataFrame,
        redshift: str,
        feature_expressions: List[str],
    ) -> None:
        self.data = dataframe
        # check the redshift data
        self._check_column(redshift)
        self._redshift = redshift
        # check the feature data
        if len(feature_expressions) == 0:
            raise ValueError("empty list of feature (expressions) provided")
        # parse the feature expressions and get a list of all required columns
        self._feature_terms = []
        columns = set()
        for expression in feature_expressions:
            term = tabeval.MathTerm.from_string(expression)
            columns.update(term.list_variables())
            self._feature_terms.append(term)
        # check the feature column data types
        for col in columns:
            self._check_column(col)
        # initialise the optional feature weights
        self._weights = np.ones(self.n_features)
        self._extra_cols = []

    def _check_column(
        self,
        colname: str
    ) -> None:
        if not np.issubdtype(self.data[colname], np.number):
            raise TypeError(
                f"type of column '{colname}' is not numeric")

    def __len__(self) -> int:
        return len(self.data)

    @property
    def n_features(self) -> int:
        return len(self._feature_terms)

    def set_feature_weights(
        self,
        feature_weights: Union[List[float], np.ndarray]
    ):
        if len(feature_weights) != self.n_features:
            raise ValueError("number of features and weights does not match")
        else:
            self._weights = feature_weights

    def set_extra_columns(
        self,
        pattern_list: List[str]
    ):
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
        if self.n_features != other.n_features:
            return False
        if not np.array_equal(self._weights, other._weights):
            return False
        return True

    def get_redshift_limit(self) -> Tuple[float, float]:
        return self.data[self._redshift].min(), self.data[self._redshift].max()

    def apply_redshift_limit(
        self,
        lower: float,
        upper: float
    ) -> 'MatchingCatalogue':
        mask = (
            (self.data[self._redshift] >= lower) &
            (self.data[self._redshift] <= upper))
        # create a new instance of the containter with redshift mask applied
        new = MatchingCatalogue.__new__(MatchingCatalogue)
        new.data = self.data[mask]
        for attr in ("_redshift", "_feature_terms", "_weights", "_extra_cols"):
            setattr(new, attr, getattr(self, attr))
        return new

    def get_redshifts(self) -> npt.NDArray:
        return self.data[self._redshift].to_numpy()

    def has_extra_columns(self) -> bool:
        return len(self._extra_cols) > 0

    def get_extra_columns(self) -> pd.DataFrame:
        return self.data[self._extra_cols]

    @property
    def features(self) -> npt.NDArray:
        # evaluate the terms and convert to a numpy array
        features = [
            term(self.data) * weight
            for term, weight in zip(self._feature_terms, self._weights)]
        return np.column_stack(features)

    def get_norm(self) -> Tuple[float, float]:
        if not hasattr(self, "_norm"):
            self._norm = compute_norm(self.features)
        return self._norm

    def get_features(
        self,
        normalise: Optional[NormaliseType] = None
    ) -> npt.NDArray:
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
