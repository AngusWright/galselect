import sys

import numpy as np
import pandas as pd
try:
    import fitsio
    _FITSIO = True
except ImportError:
    import astropy.io
    _FITSIO = False


class FitsTable:
    """
    Class to access a single table extension of FITS file, optionally selecting
    a subset of table columns. Data is accessed via a context manager and is
    read at once to memory. The preferred implementation uses fitsio.FITS, the
    fallback implementation uses astropy.io.fits.

    Parameters:
    -----------
    fpath : str
        Path to the FITS file.
    hdu : int (optional)
        Index of the extension to read from the FITS file, defaults to 1.
    columns : list of str (optional)
        Subset of columns to read from the table. All are read if not specified.
    """

    def __init__(self, fpath, hdu=1, columns=None):
        self.fpath = fpath
        self.hdu = hdu
        self.cols = columns

    def __enter__(self):
        """
        Returns:
        --------
        data : array_like
            Data from the specified file, hdu and optional column selection
            applied.
        """
        if _FITSIO:
            self.fits = fitsio.FITS(self.fpath)
            if self.cols is None:
                data = self.fits[self.hdu]
            else:
                data = self.fits[self.hdu][self.cols][:]
        else:
            self.fits = astropy.io.fits.open(self.fpath)
            if self.cols is None:
                data = self.fits[self.hdu]
            else:
                data = self.fits[self.hdu][self.cols]
        return data

    def __exit__(self, exc_type, exc_val, exc_tb):
        if _FITSIO:
            self.fits.close()


def load_fits_to_df(fpath, cols=None, hdu=1):
    """
    Convert a FITS table to a pandas.DataFrame instance, ensuring the correct
    byte-ordering.

    Parameters:
    -----------
    fpath : str
        Path to the FITS file.
    hdu : int (optional)
        Index of the extension to read from the FITS file, defaults to 1.
    columns : list of str (optional)
        Subset of columns to read from the table. All are read if not specified.
    
    Returns:
        df : pandas.DataFrame
            Table data converted to a DataFrame instance.
    """
    with FitsTable(fpath, hdu, cols) as data:
        df = pd.DataFrame()
        for colname, (dtype, offset) in data.dtype.fields.items():
            if dtype.str.startswith(("<", ">")):
                if sys.byteorder == "little":
                    dtype = np.dtype("<" + dtype.base.str.strip("><"))
                elif sys.byteorder == "big":
                    dtype = np.dtype(">" + dtype.base.str.strip("><"))
            df[colname] = data[colname].astype(dtype)
    return df
