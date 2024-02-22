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
    "--progress", action="store_true",
    help="display a progress bar")
data.add_argument(
    "--z-cut", action="store_true",
    help="apply a redshift cut on the data to remove objects that exceed the "
         "range of redshifts in the mock data")


if __name__ == "__main__":
    args = parser.parse_args()

    # unpack the redshift column name parameter
    z_name_data, z_name_mock = args.z_name
    # unpack the mapping of feature/column expresions for mock and data
    feature_expr_data, feature_expr_mock = [], []
    for feature_data, feature_mock in args.feature:
        feature_expr_data.append(feature_data)
        feature_expr_mock.append(feature_mock)

    # load the mocks
    print(f"reading mock sample: {args.mock}")
    mock = galselect.MatchingCatalogue(
        apd.read_fits(args.mock),
        redshift=z_name_mock,
        feature_expressions=feature_expr_mock)
    if args.weights is not None:
        mock.set_feature_weights(args.weights)
    zmin, zmax = mock.get_redshift_limit()

    # load the data
    print(f"reading data sample: {args.data}")
    base=apd.read_fits(args.data)
    if z_name_data not in base.keys(): 
        print(f"z name {z_name_data} not found in extension 1: trying extension 2")
        #Try the next extension 
        base=apd.read_fits(args.data,hdu=2)
        if z_name_data not in base.keys(): 
            print(f"z name {z_name_data} not found in extension 2: trying extension 3")
            #Try the next extension 
            base=apd.read_fits(args.data,hdu=3)
            if z_name_data not in base.keys(): 
                raise Exception(f"z name {z_name_data} was not found in any of the first 3 data file extensions!")
    data = galselect.MatchingCatalogue(
        base, 
        redshift=z_name_data,
        feature_expressions=feature_expr_data)
    if args.weights is not None:
        data.set_feature_weights(args.weights)
    # add the columns to be cloned
    if len(args.clone) > 0:
        data.set_extra_columns(args.clone)
    if args.z_cut:
        data = data.apply_redshift_limit(lower=zmin, upper=zmax)

    # initialise and run the matching
    print("matching on:")
    mterm_len = max(len(s) for s in feature_expr_mock)
    dterm_len = max(len(s) for s in feature_expr_data)
    for feature_data, feature_mock in args.feature:
        print(f"    {feature_mock:{mterm_len}s} -> {feature_data:{dterm_len}}")

    if args.z_warn is None:
        args.z_warn = 1e9  # arbitrary large value to supress warnings
    selector = galselect.DataMatcher(
        mock,
        redshift_warning=args.z_warn)
    matched, _ = selector.match_catalog(
        data,
        d_idx=args.idx_interval,
        duplicates=args.duplicates,
        normalise=args.norm,
        progress=args.progress)

    # write
    print(f"writing matched data: {args.output}")
    apd.to_fits(matched.data, args.output)
