import healsparse
import numpy as np
import healpy as hp
import tqdm
import time

from .gaia_stars import make_gaia_mask

MASK_NOSLICE = 2**0
MASK_GAIA_STAR = 2**1
MASK_SLICEDUPE = 2**2
MASK_TILEDUPE = 2**3


def in_unique_coadd_tile_region(
    *, ra, dec, crossra0, udecmin, udecmax, uramin, uramax,
):
    """Compute a boolean mask of the same shape as the input ra, dec indicating
    if the given position is in the unique region for a coadd tile.

    Parameters
    ----------
    ra : array-like
        The array of ra values for the objects.
    dec : array-like
        The array of dec values for the objects.
    crossra0 : str
        A string that is either 'Y' or 'N' indicating if the tile cross ra=0.
    udecmin, udecmax, uramin, uramax : float
        The min and max values for ra and dec indicating the unique coadd tile
        region.

    Returns
    -------
    flag : bool array
        The boolean flag array with True indicating the (ra,dec) is in the
        unique coadd tile region.
    """
    ra = ra.copy()

    msk = (ra < 0) & np.isfinite(ra)
    while np.any(msk):
        ra[msk] = ra[msk] + 360.0
        msk = (ra < 0) & np.isfinite(ra)

    msk = (ra > 360) & np.isfinite(ra)
    while np.any(msk):
        ra[msk] = ra[msk] - 360.0
        msk = (ra > 360) & np.isfinite(ra)

    if crossra0 == 'Y':
        uramin = uramin - 360.0
        msk = ra > 180.0
        ra[msk] -= 360

    in_coadd = (
        (ra > uramin)
        & (ra <= uramax)
        & (dec > udecmin)
        & (dec <= udecmax)
    )
    msk = np.isfinite(ra) & np.isfinite(dec)
    in_coadd[~msk] = False

    return in_coadd


def get_slice_bounds(
    *, orig_start_col, orig_start_row, central_size, buffer_size, coadd_dims,
):
    """Get the unique pixel bounds for each slice.

    Typically, this will simply cut the buffers. However, on the edge of a tile,
    the buffer closest to the edge will be kept.

    Note that all coordinates are 0-indexed pixel coordinates.

    Parameters
    ----------
    orig_start_col : int
        The starting col for the slice in the coadd coordinates.
    orig_start_row : int
        The starting row for the slice in the coadd coordinates.
    central_size : int
        The size of the central region of the slice.
    buffer_size : int
        The size of the buffer around each slice.
    coadd_dims : 2-tuple of ints
        The dimension of the coadd image from which the slice was made.

    Returns
    -------
    slice_bnds : dict
        A directory with keys 'min_row', 'max_row', 'min_col', and 'max_col'.
    """
    slice_dim = central_size + 2*buffer_size
    if orig_start_row == 0:
        min_slice_row = 0
        max_slice_row = slice_dim - buffer_size
    elif orig_start_row + slice_dim == coadd_dims[0]:
        min_slice_row = buffer_size
        max_slice_row = slice_dim
    else:
        min_slice_row = buffer_size
        max_slice_row = slice_dim - buffer_size

    if orig_start_col == 0:
        min_slice_col = 0
        max_slice_col = slice_dim - buffer_size
    elif orig_start_col + slice_dim == coadd_dims[1]:
        min_slice_col = buffer_size
        max_slice_col = slice_dim
    else:
        min_slice_col = buffer_size
        max_slice_col = slice_dim - buffer_size

    return {
        "min_row": min_slice_row,
        "max_row": max_slice_row,
        "min_col": min_slice_col,
        "max_col": max_slice_col,
    }


def make_mask(
    *,
    preconfig,
    missing_slice_inds,
    obj_data,
    central_size,
    buffer_size,
    wcs,
    position_offset,
    coadd_dims,
    info,
    gaia_stars=None,
    healpix_nisde=131072,
):
    """Make the healsparse mask indicating where measurements were made.

    Parameters
    ----------
    preconfig : dict
        Proprocessing configuration. Must contain gaia_star_masks
        entry if gaia_stars is not None.
    missing_slice_inds : list of int
        The indices into `obj_data` indicating slices that did not get measurements.
    obj_data : structure numpy array
        The object data from the MEDS file. Used for getting the slice locations
        orig_start_col/orig_start_row. These values are the same across bands and
        so the data from any of the bands is fine.
    central_size : int
        The size of the central region of the slice.
    buffer_size : int
        The size of the buffer around each slice.
    wcs : esutil.wcsutil.WCS
        A WCS object for the coadd tile.
    position_offset : int
        The position offset to convert from 0-indexed pixel coordinates into
        the convention expected by the `wcs`.
    coadd_dims : 2-tuple of ints
        The dimension of the coadd image.
    info : dict
        The coadd tile info produced by the pizze cutter and stored in the MEDS
        metadata as the 'tile_info' entry. This dict is used to get the coadd
        tile unique ra/dec bounds (uramin, uramax, udecmin, udecmax) and if the
        tile cross ra = 0 (crossra0).
    gaia_stars : structured numpy array, optional
        The GAIA star data indicating where the GAIA star mask holes are. The default
        of None will result in no GAIA star mask holes.
    healpix_nisde : int, optional
        The nisde of the healsparse mask. The default is 131072.

    Returns
    -------
    hs_msk : healsparse map
        The healsparse boolean map indicating what areas were masked and why. The
        possible flag values are

            MASK_NOSLICE = 2**0 - indicates no data for a given coadd slice
            MASK_GAIA_STAR = 2**1 - indicates regions where a GAIA star hole exists
    """
    slice_dim = buffer_size*2 + central_size
    msk_img = np.zeros(coadd_dims, dtype=np.int32)

    # first do GAIA star masks
    if gaia_stars is not None:
        for slice_ind in tqdm.trange(len(obj_data), desc='making GAIA masks', ncols=79):
            scol = obj_data["orig_start_col"][slice_ind, 0]
            srow = obj_data["orig_start_row"][slice_ind, 0]

            slice_msk = make_gaia_mask(
                gaia_stars,
                dims=(slice_dim, slice_dim),
                start_row=srow,
                start_col=scol,
                symmetrize=preconfig["gaia_star_masks"]["symmetrize"],
            )
            msk = slice_msk != 0
            slice_msk[:, :] = 0
            slice_msk[msk] = MASK_GAIA_STAR

            slice_bnds = get_slice_bounds(
                orig_start_col=scol,
                orig_start_row=srow,
                central_size=central_size,
                buffer_size=buffer_size,
                coadd_dims=coadd_dims,
            )

            msk_img[
                slice_bnds["min_row"]+srow:slice_bnds["max_row"]+srow,
                slice_bnds["min_col"]+scol:slice_bnds["max_col"]+scol,
            ] |= slice_msk[
                slice_bnds["min_row"]:slice_bnds["max_row"],
                slice_bnds["min_col"]:slice_bnds["max_col"],
            ]

    # then do the slice masks for missing slices
    for slice_ind in tqdm.tqdm(missing_slice_inds, desc='making slice masks', ncols=79):
        scol = obj_data["orig_start_col"][slice_ind, 0]
        srow = obj_data["orig_start_row"][slice_ind, 0]
        slice_bnds = get_slice_bounds(
            orig_start_col=scol,
            orig_start_row=srow,
            central_size=central_size,
            buffer_size=buffer_size,
            coadd_dims=coadd_dims,
        )
        msk_img[
            slice_bnds["min_row"]+srow:slice_bnds["max_row"]+srow,
            slice_bnds["min_col"]+scol:slice_bnds["max_col"]+scol,
        ] |= MASK_NOSLICE

    # now lets flatten to bad pixels so far
    msk = msk_img != 0
    x, y = np.meshgrid(np.arange(coadd_dims[0]), np.arange(coadd_dims[1]))
    x = x[msk].ravel()
    y = y[msk].ravel()
    vals = msk_img[msk].ravel()
    ra, dec = wcs.image2sky(x+position_offset, y+position_offset)

    # we do not want to mark masks outside of the unique coadd region
    t0 = time.time()
    print("cutting to the unique tile region...", end="", flush=True)
    msk = in_unique_coadd_tile_region(
        ra=ra,
        dec=dec,
        crossra0=info['crossra0'],
        udecmin=info['udecmin'],
        udecmax=info['udecmax'],
        uramin=info['uramin'],
        uramax=info['uramax'],
    )
    x = x[msk]
    y = y[msk]
    vals = vals[msk]
    ra = ra[msk]
    dec = dec[msk]
    print("done (%0.2f seconds)" % (time.time() - t0), flush=True)

    # now convert to healsparse
    t0 = time.time()
    print("converting mask to healsparse...", end="", flush=True)
    hs_msk = healsparse.HealSparseMap.make_empty(
        32, healpix_nisde, np.int32, sentinel=0
    )
    nest_inds = hp.ang2pix(
        healpix_nisde, ra, dec,  nest=True, lonlat=True,
    )
    hs_msk.update_values_pix(nest_inds, vals, nest=True, operation='or')
    print("done (%0.2f seconds)" % (time.time() - t0), flush=True)

    return hs_msk
