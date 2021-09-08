import sys

import numpy as np
import pandas as pd
try:
    import fitsio
    _FITSIO = True
except ImportError:
    import astropy.io
    _FITSIO = False


def convert_byteorder(data):
    dtype = data.dtype
    # check if the byte order matches the native order, identified by the
    # numpy dtype string representation: little endian = "<" and
    # big endian = ">"
    if dtype.str.startswith(("<", ">")):
        if sys.byteorder == "little":
            dtype = np.dtype("<" + dtype.base.str.strip("><"))
        elif sys.byteorder == "big":
            dtype = np.dtype(">" + dtype.base.str.strip("><"))
    return data.astype(dtype, casting="equiv", copy=False)


def read_fits(fpath, cols=None, hdu=1):
    """
    Read a FITS table into a pandas.DataFrame, ensuring the correct byte-order.

    Parameters:
    -----------
    fpath : str
        Path to the FITS file.
    hdu : int (optional)
        Index of the extension to read from the FITS file, defaults to 1.
    columns : list of str (optional)
        Subset of columns to read from the table, defaults to all.
    
    Returns:
        df : pandas.DataFrame
            Table data converted to a DataFrame instance.
    """
    # load the FITS data
    if _FITSIO:
        fits = fitsio.FITS(fpath)
        if cols is None:
            data = fits[hdu]
        else:
            data = fits[hdu][cols][:]
        fits.close()
    else:
        with astropy.io.fits.open(fpath) as fits:
            if cols is None:
                data = fits[hdu]
            else:
                data = fits[hdu][cols]
    # construct the data frame
    df = pd.DataFrame()
    for colname, (dtype, offset) in data.dtype.fields.items():
        df[colname] = convert_byteorder(data[colname])
    return df


def read_fits(fpath, cols=None, hdu=1):
    """
    Read a FITS table into a pandas.DataFrame, ensuring the correct byte-order.

    Parameters:
    -----------
    fpath : str
        Path to the FITS file.
    hdu : int (optional)
        Index of the extension to read from the FITS file, defaults to 1.
    columns : list of str (optional)
        Subset of columns to read from the table, defaults to all.
    
    Returns:
        df : pandas.DataFrame
            Table data converted to a DataFrame instance.
    """
    # load the FITS data
    if _FITSIO:
        fits = fitsio.FITS(fpath)
        if cols is None:
            data = fits[hdu]
        else:
            data = fits[hdu][cols][:]
        fits.close()
    else:
        with astropy.io.fits.open(fpath) as fits:
            if cols is None:
                data = fits[hdu]
            else:
                data = fits[hdu][cols]
    # construct the data frame
    df = pd.DataFrame()
    for colname, (dtype, offset) in data.dtype.fields.items():
        df[colname] = convert_byteorder(data[colname])
    return df


def write_fits(data, fpath):
    """
    Write a pandas.DataFrame as FITS table file.

    Parameters:
    -----------
    data : pandas.DataFrame
        Data to write as FITS table.
    fpath : str
        Path to the FITS file.
    """
    # load the FITS data
    if _FITSIO:
        array = np.empty(len(data), dtype=np.dtype(list(data.dtypes.items())))
        fits = fitsio.FITS(fpath, "rw")
        fits.write(array)
        fits.close()
    else:
        columns = [
            astropy.io.fits.Column(name=col, array=data[col])
            for col in data.columns]
        hdu = astropy.io.fits.BinTableHDU.from_columns(columns)
        hdu.writeto(fpath)
