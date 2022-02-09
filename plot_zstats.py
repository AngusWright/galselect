#!/usr/bin/env python3
import argparse
import os
from tkinter import E

import astropandas as apd

import galselect


parser = argparse.ArgumentParser()
parser.add_argument(
    "input_files",
    help="input FITS files")
parser.add_argument(
    "-o", "--output", metavar="path", required=True
    help="path for output PDF with plots (default: [match].pdf")
parser.add_argument(
    "--z-spec", nargs="*", required=True,
    help="list of names of the spectroscopic/true redshift column in each of "
         "the input catalogues")
parser.add_argument(
    "--z-phot", nargs="*", required=True,
    help="list of names of the photometric redshift column in each of the "
         "input catalogues")
parser.add_argument(
    "--feature", nargs="*", action="append",
    help="name of additional feature column in each of the input catalogues, "
         "repeat for each additional feature to plot (if not present in a "
         "catalogue, omit with - (hyphen)")
parser.add_argument(
    "--labels", nargs="*",
    help="optional TEX labels for each of the additional features")
parser.add_argument(
    "--field", nargs="*",
    help="list of names of columns that can be used to identify different "
         "realisations of the data (if not present in a catalogue, omit with "
         "- (hyphen)")
parser.add_argument(
    "--z-thresh", type=float, default=0.15,
    help="threshold in normalised redshift difference that is considered an "
         "outlier (default: %(default)s)")


if __name__ == "__main__":
    args = parser.parse_args()
    # check the provided inputs
    n_cats = len(args.input_files)
    if args.fields is None:
        args.fields = [None] * n_cats
    for name in ["z_spec", "z_phot", "field"]:
        n_param = len(getattr(args, name))
        if n_cats != n_param:
            parser.exit(
                f"number of input catalogues ({n_cats}) does not match the "
                f"provided columns for --{name.replace('_', '-')} ({n_param})")
        # indicate omitted entries with None
        setattr(args, name, [
            None if c == "-" else c for c in getattr(args, name)])
    # check the variable features
    n_feat = len(args.features)
    for i, feat_set in enumerate(args.features):
        if n_cats != len(feat_set):
            parser.exit(
                f"number of input catalogues ({n_cats}) does not match the "
                f"provided feature #{i} ({len(feat_set)})")
        args.features[i] = [None if c == "-" else c for c in args.features[i]]
    if args.labels is not None:
        if n_cats != len(args.labels):
            parser.exit(
                f"number of input catalogues ({n_cats}) does not match the "
                f"provided labels ({len(args.labels)})")

    with galselect.RedshiftStats(args.output, labels=args.labels) as plt:
        for i in range(n_cats):
            plt.add_catalogue(
                args.input_files[i], args.z_spec[i], args.z_phot[i],
                *[args.features[i][n] for n in range(n_feat)],
                fields=args.field[i])
        print(f"plotting to: {args.output}")
        plt.plot(args.fields, outlier_threshold=args.z_thresh)
