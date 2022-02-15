#!/usr/bin/env python3
import argparse

import galselect


empty_token = "-"


parser = argparse.ArgumentParser(
    description="Make a photo-z statistic plot by comparing to known galaxy "
                "redshifts.")
parser.add_argument(
    "input_file", nargs="+",
    help="input FITS files")
parser.add_argument(
    "-o", "--output", metavar="path", required=True,
    help="path for output PDF with plots")
parser.add_argument(
    "--name", nargs="+", required=True,
    help="list label names, one for each input catalogue")

features = parser.add_argument_group(
    "Feature names",
    description="Specify the column names in each catalogue of redshifts and "
                "optional additional features. All of the optional arguments "
                "allow omitting a column in a catalogue by inserting the "
                f"special value '{empty_token}'.")
features.add_argument(
    "--z-spec", nargs="+", required=True,
    help="list of names of the spectroscopic/true redshift column in each of "
         "the input catalogues")
features.add_argument(
    "--z-phot", nargs="+", required=True,
    help="list of names of the photometric redshift column in each of the "
         "input catalogues")
features.add_argument(
    "--feature", nargs="*", action="append",
    help="name of additional feature column in each of the input catalogues, "
         "repeat for each additional feature to plot")
features.add_argument(
    "--field", nargs="*",
    help="list of names of columns that can be used to identify different "
         "realisations of the data (if not present in a catalogue, omit with "
         "- (hyphen)")

config = parser.add_argument_group(
    "Plot configuration",
    description="Options to configure the plot layout.")
config.add_argument(
    "--labels", nargs="*",
    help="optional TEX labels for each of the additional features")
config.add_argument(
    "--log", nargs="*",
    help="List of characters specifying the binning for each optional feature "
         "(F: linear, T: logarithmic, default: all linear)")
config.add_argument(
    "--bins", type=int, default=25,
    help="number of bins (default: %(default)s)")
config.add_argument(
    "--z-thresh", type=float, default=0.15,
    help="threshold in normalised redshift difference that is considered an "
         "outlier (default: %(default)s)")


if __name__ == "__main__":
    args = parser.parse_args()

    # check the required arguments
    n_cats = len(args.input_file)
    if args.field is None:
        args.field = [None] * n_cats
    for name in ["name", "z_spec", "z_phot", "field"]:
        n_param = len(getattr(args, name))
        if n_cats != n_param:
            parser.error(
                f"number of columns ({n_param}) provided for "
                f"--{name.replace('_', '-')} does not match the number of "
                f"input catalogues ({n_cats})")
        # indicate omitted entries with None
        setattr(args, name, [
            None if c == empty_token else c for c in getattr(args, name)])

    # check number of labels and features
    n_feat = len(args.feature) if args.feature is not None else 0
    n_label = len(args.labels) if args.labels is not None else 0
    if n_label == 0 and n_feat > 0:
        parser.error("there are labels but no features provided")
    elif n_label != n_feat:
        parser.error(
            f"number of labels ({n_label}) does not match the number of "
            f"optional features ({n_feat})")

    # check the optional features
    if n_feat > 0:
        if n_label == 0:
            args.labels = [None] * n_feat
        for i, feat_set in enumerate(args.feature):
            # check that there is one feature column name for each catalogue
            if n_cats != len(feat_set):
                parser.error(
                    f"number of column names ({len(feat_set)}) for feature "
                    f"#{i} does not match the number of catalogues ({n_cats})")
            # indicate omitted entries with None
            args.feature[i] = [
                None if c == empty_token else c for c in feat_set]
            if all(c is None for c in feat_set):
                parser.error(f"feature #{i} has no valid arguments")
            # set labels if none are provided
            if args.labels[i] is None:
                args.labels[i] = tuple(  # use the first which is not None
                    c for c in feat_set if c is not None)[0]

    # check the binning scale
    if args.log is None:
        args.log = [False] * n_feat
    else:
        if len(args.log) != n_feat:
            parser.error(
                f"number of --log entries ({len(args.log)}) does not match "
                f"the number of features ({n_feat})")
        for i, log in enumerate(args.log):
            if log not in "FT":
                parser.error(
                    f"invalid value '{log}' for --log, must be T or F")
            else:
                args.log[i] = True if log == "T" else False

    with galselect.RedshiftStats(args.output, out_thresh=args.z_thresh) as plt:
        for i in range(n_cats):
            name = args.name[i]
            fpath = args.input_file[i]
            z_spec = args.z_spec[i]
            z_phot = args.z_phot[i]
            features = {}
            if n_feat > 0:
                for label, feat_set in zip(args.labels, args.feature):
                    colname = feat_set[i]
                    if colname is not None:
                        features[colname] = label
            field = args.field[i]
            plt.add_catalogue(name, fpath, z_spec, z_phot, features, field)
        plt.set_binscale(*args.log)

        print(f"plotting to: {args.output}")
        plt.plot(nbins=args.bins)
