#!/usr/bin/env python3
import argparse
import os

import astropandas as apd

import galselect


parser = argparse.ArgumentParser()
parser.add_argument(
    "match",
    help="matched FITS catalogue produced with select_gals.py")
parser.add_argument(
    "-o", "--output", metavar="path",
    help="path for output PDF with plots (default: [match].pdf")
parser.add_argument(
    "-z", "--z-name", nargs=2, metavar=("data", "mock"), required=True,
    help="names of the redshift column in original data and matched "
         "catalogues")


if __name__ == "__main__":
    args = parser.parse_args()
    # read the mock and data input catalogues
    print(f"reading matched file: {args.match}")
    match = apd.read_fits(args.match)

    if args.output is None:
        args.output = os.path.splitext(args.match)[0] + ".pdf"
    print(f"plotting to: {args.output}")
    with galselect.Plotter(args.output, match) as plt:
        plt.redshift_redshift(*args.z_name)
        plt.distances()
        plt.distance_neighbours()
        plt.distance_redshift(args.z_name[1])
        plt.delta_redshift_neighbours(*args.z_name)
