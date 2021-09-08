#!/usr/bin/env python3
import argparse

import pandas as pd

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
    "-o", "--out", metavar="path", required=True,
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
    "--z-warn", metavar="float", default=0.05,
    help="issue a warning if the redshift difference of a match exceeds this "
         "threshold (default: %(default)s")
data.add_argument(
    "--idx-interval", metavar="int", default=10000,
    help="number of nearest neighbours in redshift used to find a match "
         "(default: %(default)s")
data.add_argument(
    "--distances", action="store_true",
    help="store the distance in feature space of the matches in the output "
         "catalogue")


if __name__ == "__main__":
    args = parser.parse_args()

    # read the mock and data input catalogues
    if args.datapath.endswith(".csv"):
        data = pd.read_csv(args.datapath)
    if args.datapath.endswith(".fits"):
        data = galselect.read_fits(args.mockpath)
    else:
        raise ValueError(f"fileformat not supported: {args.mockpath}")
    if args.mockpath.endswith(".csv"):
        mock = pd.read_csv(args.mockpath)
    if args.mockpath.endswith(".fits"):
        mock = galselect.read_fits(args.mockpath)
    else:
        raise ValueError(f"fileformat not supported: {args.mockpath}")

    # unpack the redshift column name parameter
    z_name_data, z_name_mock = args.z_name

    # unpack the mapping for feature/column names from mock to data
    feature_names_data, feature_names_mock = [], []
    for feature_data, feature_mock in args.features:
        feature_names_data.append(feature_data)
        feature_names_mock.append(feature_mock)

    # match the catalogues
    selector = galselect.DataMatcher(
        data, z_name_data, [f for f in feature_names_data], args.z_warn)
    matched = selector.match_catalog(
        mock[args.z_name_mock], mock[[f for f in feature_names_mock]],
        args.idx_interval, args.distances)

    # write 
    galselect.write_fits(matched, args.matchpath)
