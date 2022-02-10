#!/usr/bin/env python3
import argparse

import galselect


empty_token = "-"


parser = argparse.ArgumentParser(
    description="Make a photo-z statistic plot by comparing to known galaxy "
                "redshifts.",
    epilog="All of the non-positional arguments above support omitting values "
           f"by inserting the special value '{empty_token}'")
parser.add_argument(
    "input_file", nargs="+",
    help="input FITS files")
parser.add_argument(
    "-o", "--output", metavar="path", required=True,
    help="path for output PDF with plots (default: [match].pdf")
parser.add_argument(
    "--name", nargs="+", required=True,
    help="list label names, one for each input catalogue")
parser.add_argument(
    "--z-spec", nargs="+", required=True,
    help="list of names of the spectroscopic/true redshift column in each of "
         "the input catalogues")
parser.add_argument(
    "--z-phot", nargs="+", required=True,
    help="list of names of the photometric redshift column in each of the "
         "input catalogues")
parser.add_argument(
    "--feature", nargs="*", action="append",
    help="name of additional feature column in each of the input catalogues, "
         "repeat for each additional feature to plot")
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
    n_cats = len(args.input_file)
    if args.field is None:
        args.field = [None] * n_cats
    for name in ["name", "z_spec", "z_phot", "field"]:
        n_param = len(getattr(args, name))
        if n_cats != n_param:
            parser.exit(
                f"number of input catalogues ({n_cats}) does not match the "
                f"provided columns for --{name.replace('_', '-')} ({n_param})")
        # indicate omitted entries with None
        setattr(args, name, [
            None if c == empty_token else c for c in getattr(args, name)])
    # check the variable features
    if args.feature is not None:
        for i, feat_set in enumerate(args.feature):
            if n_cats != len(feat_set):
                parser.exit(
                    f"number of input catalogues ({n_cats}) does not match "
                    f"the provided feature #{i} ({len(feat_set)})")
            args.feature[i] = [
                None if c == empty_token else c for c in args.feature[i]]
    if args.labels is not None:
        n_feat = 0 if args.feature is None else len(args.feature)
        if n_feat != len(args.labels):
            parser.exit(
                f"number of additional features ({n_feat}) does not match the "
                f"provided labels ({len(args.labels)})")
    else:
        args.labels = [f[0] for f in args.feature]

    with galselect.RedshiftStats(args.output) as plt:
        for i in range(n_cats):
            name = args.name[i]
            fpath = args.input_file[i]
            z_spec = args.z_spec[i]
            z_phot = args.z_phot[i]
            field = args.field[i]
            if args.feature is None:
                plt.add_catalogue(name, fpath, z_spec, z_phot, fields=field)
            else:
                plt.add_catalogue(
                    name, fpath, z_spec, z_phot, *[f[i] for f in args.feature],
                    fields=field)
        plt.set_labels(args.labels)
        print(f"plotting to: {args.output}")
        plt.plot(outlier_threshold=args.z_thresh)
