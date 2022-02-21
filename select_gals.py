#!/usr/bin/env python3
import argparse

import astropandas as apd
import numpy as np

import galselect


parser = argparse.ArgumentParser()

data = parser.add_argument_group(
    "Catalogues", description="Specify in- and output data catalogues")
data.add_argument(
    "-d", "--data", metavar="path", required=True,
    help="input FITS catalogue to which the mock data is matched")
data.add_argument(
    "-m", "--mock", metavar="path", required=True,
    help="input FITS catalogue which is matched to the data")
data.add_argument(
    "-o", "--output", metavar="path", required=True,
    help="matched FITS output mock data catalogue")

_map_kwargs = dict(nargs=2, metavar=("data", "mock"))
features = parser.add_argument_group(
    "Feature names",
    description="Specify mapping between data and mock catalogue columns")
features.add_argument(
    "-z", "--z-name", **_map_kwargs, required=True,
    help="names of the redshift column in data and mock catalogues")
features.add_argument(
    "-f", "--feature", **_map_kwargs, action="append", required=True,
    help="name of a feature column in data and mock catalogues, repeat "
         "argument for every feature required for the matching")
features.add_argument(
    "--norm", action="store_true",
    help="normalise the feature space (centered on zero, scaled by 1/stdev)")
features.add_argument(
    "--weights", nargs="*", type=float,
    help="rescale the feature data (after normalising), must be one weight "
         "per --feature if provided")

data = parser.add_argument_group(
    "Configuration", description="Optional parameters")
data.add_argument(
    "--duplicates", action="store_true",
    help="allow matching one mock object to many data objects")
data.add_argument(
    "--clone", nargs="*",
    help="columns to clone from the data into the matched mock data (data "
         "redshift is always cloned)")
data.add_argument(
    "--z-warn", metavar="float", type=float,
    help="issue a warning if the redshift difference of a match exceeds this "
         "threshold (default: no warning")
data.add_argument(
    "--idx-interval", metavar="int", type=int, default=10000,
    help="number of nearest neighbours in redshift used to find a match "
         "(default: %(default)s")
data.add_argument(
    "--distances", action="store_true",
    help="store the distance in feature space of the matches in the output "
         "catalogue")
data.add_argument(
    "--progress", action="store_true",
    help="display a progress bar")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.z_warn is None:
        args.z_warn = 1e9  # arbitrary large value to supress warnings
    # check the weights argument
    if args.weights is not None:
        if len(args.weights) != len(args.feature):
            parser.error(
                f"number of --feature dimensions ({len(args.feature)}) and "
                f"--weights ({len(args.weights)}) do not match")

    # unpack the redshift column name parameter
    z_name_data, z_name_mock = args.z_name

    # unpack the mapping for feature/column names from mock to data
    feature_names_data, feature_names_mock = [], []
    for feature_data, feature_mock in args.feature:
        feature_names_data.append(feature_data)
        feature_names_mock.append(feature_mock)

    # read the mock and data input catalogues
    print(f"reading data file: {args.data}")
    data_columns = [z_name_data, *feature_names_data]
    if args.clone is not None:
        data_columns.extend(args.clone)
    data = apd.read_fits(args.data, columns=data_columns)
    print(f"reading simulation file: {args.mock}")
    mock = apd.read_fits(args.mock)

    # match the catalogues
    selector = galselect.DataMatcher(
        mock, z_name_mock, [f for f in feature_names_mock],
        normalise=args.norm, weights=args.weights, duplicates=args.duplicates,
        redshift_warning=args.z_warn)
    # mask to redshift range
    mask = (
        (data[z_name_data] > selector.z_min) &
        (data[z_name_data] < selector.z_max))
    nbad = len(mask) - np.count_nonzero(mask)
    data = data[mask]
    if nbad != 0:
        print(
            f"WARNING: removed {nbad}/{len(mask)} data objects outside the mock"
            f" redshift range of {selector.z_min:.3f} to {selector.z_max:.3f}")
    if args.clone is None:
        args.clone = []
    if z_name_data not in args.clone:
        if z_name_data == z_name_mock:
            args.clone.append(f"{z_name_data}_data")
        else:
            args.clone.append(z_name_data)
    matched = selector.match_catalog(
        data[z_name_data], data[[f for f in feature_names_data]].to_numpy(),
        d_idx=args.idx_interval, clonecols=data[args.clone],
        return_mock_distance=args.distances, progress=args.progress)

    # write
    print(f"writing matched data: {args.output}")
    apd.to_fits(matched, args.output)
