#!/usr/bin/env python3
import argparse

import numpy as np
import pandas as pd

import galselect


def load_input(fpath):
    if fpath.endswith(".csv"):
        data = pd.read_csv(fpath)
    if fpath.endswith(".fits"):
        data = galselect.read_fits(fpath)
    else:
        raise ValueError(f"fileformat not supported: {fpath}")
    return data


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

data = parser.add_argument_group(
    "Configuration", description="Optional parameters")
data.add_argument(
    "--clone", nargs="*",
    help="columns to clone from the data into the matched mock data")
data.add_argument(
    "--z-warn", metavar="float", type=float, default=0.05,
    help="issue a warning if the redshift difference of a match exceeds this "
         "threshold (default: %(default)s")
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

    # read the mock and data input catalogues
    data = load_input(args.data)
    mock = load_input(args.mock)

    # unpack the redshift column name parameter
    z_name_data, z_name_mock = args.z_name

    # unpack the mapping for feature/column names from mock to data
    feature_names_data, feature_names_mock = [], []
    for feature_data, feature_mock in args.feature:
        feature_names_data.append(feature_data)
        feature_names_mock.append(feature_mock)

    # match the catalogues
    selector = galselect.DataMatcher(
        mock, z_name_mock, [f for f in feature_names_mock],
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
    matched = selector.match_catalog(
        data[z_name_data], data[[f for f in feature_names_data]].to_numpy(),
        d_idx=args.idx_interval, clonecols=data[args.clone],
        return_mock_distance=args.distances, progress=args.progress)

    # write 
    galselect.write_fits(matched, args.output)
