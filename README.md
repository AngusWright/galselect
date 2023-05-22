# galselect

This package implements an algorithm that allows to replicate galaxy sample selection functions in simulated data sets through nearest neighbour matching. The matching is performed within a sliding redshift window on an arbitrary feature space, typically however using photometric observables, such as colors or magnitudes.

## Installation

The code requires `python>=3.6` and can be installed using `pip` after cloning and switching into the repository:

    pip install .

If the package is installed via pip, the main executables in the `bin/` directory are directly accessible from the terminal environment.

## Usage

The easiest way to use this package is through the executable script `bin/select_gals.py`. Type

    select_gals.py -h

for an overview of the commandline interface.

### Example

The script allows to replicate a data set with known redshift in a simulated (mock) galaxy catalog without any prior selections applied, e.g. to replicate a spectroscopic dataset. A simple example is given below where the data and simulation are provided as FITS files, `data.fits` and `mock.fits` accordingly. The matched output file is written to the `output.fits` file.

    select_gals.py \
        -d data.fits \
        -m mock.fits \
        -o output.fits \
        -z redshift_data redshift_mock \
        -f feature1_data feature1_mock \
        -f feature2_data feature2_mock \
        [...]
        --norm \
        --duplicates \
        --clone redshift_data [...] \
        --idx-interval 1000 \
        --z-cut

Meaning of the arguments:

 - `-z`: two arguments, the column name of the redshift column in the data catalogue, followed by the column name in the mock catalogue.
 - `-f`: two arguments, the column name of the first feature to use for the matching in the data catalogue, followed by the corresponding name in the mock catalogue. This argument can be repeated as many times as necessary to construct the matching feature space.  
   *Note:* Expressions are allowed, such as `col1_data-col2_data col1_mock-col2_mock` (e.g. to match on the difference between two columns).
 - `--norm` (flag): Recommended, normalises the feature space in both cataloges by subtracting the median and rescaling by the inverse nMAD (static taken from the data catalogue).
 - `--duplicates` (flag): Allows to assign one mock object to many object of the data catalogue. Otherwise, matches are unique and may cause the script to stop, if there are no more unmatched partners left at a given redshift (see `--idx-interval` below).
 - `--clone`: Provide a list of column names in the data catalogue that are copied into the matched mock output catalogue. If the column already exists in the data, the suffix `_data` is appended to the name of the cloned column. Useful to copy data attributes for direct comparison of the data and matched mock objects (e.g. redshift).
 - `--idx-interval`: This parameter determines the redshift window in which the matching is performed. Mock objects with redshifts outside this window are ignored during the matching. The window is centered on the redshift of the data object `z_data` to be matched and is limited to the `2*N` objects with redshifts just below and above `z_data` in the mock catalogue. I.e. if `--idx-interval 1000`, the 2000 closest objects in redshift are considered as candidates for the matching.  
   *Note:* If `--duplicates` is not provided, the script will interrupt if there are no more sufficient unmatched objects left in the current redshift window. Either increase the `--idx-interval` value, use a larger simulation box, or use the `--duplicates` flag.
 - `--z-cut` (flag): Automatically cut the data catalogue to the redshift range of the mock cataloge, e.g. reject data objects with `z < z_min_mock`or `z > z_max_mock`, which otherwise cannot be matched.

### Troubleshooting

The `select_gals.py` script adds a few columns to the output file that can provide additional insights into the matching. These are:

 - `idx_match` (int): The running index of matched mock object in the input mock catalogue.
 - `match_dist` (float): The total Euclidean distance in the feature space. Note that this may include the normalisation if `--norm` is used.
 - `n_neigh` (int): The total number of neighbours available in the redshift window. Always equal to `--idx-match` if `--duplicates` is used, otherwise depends on the number density and matching order of objects in the catalogue.
 - `z_range` (float): The total width of the redshift window `z_max - z_min` for this match.
