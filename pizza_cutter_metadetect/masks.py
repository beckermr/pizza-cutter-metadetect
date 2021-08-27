import healsparse
import numpy as np
import tqdm

from .gaia_stars import make_gaia_mask

MASK_INTILE = 2**0
MASK_NOSLICE = 2**1
MASK_GAIA_STAR = 2**2
MASK_SLICEDUPE = 2**3
MASK_TILEDUPE = 2**4


def _wrap_ra(ra):
    fnt = np.isfinite(ra)
    msk = (ra < 0) & fnt
    while np.any(msk):
        ra[msk] = ra[msk] + 360.0
        msk = (ra < 0) & fnt

    msk = (ra > 360) & fnt
    while np.any(msk):
        ra[msk] = ra[msk] - 360.0
        msk = (ra > 360) & fnt

    return ra


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

    ra = _wrap_ra(ra)

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


def _mask_one_slice_for_gaia_stars(
    *,
    buffer_size,
    central_size,
    gaia_stars,
    symmetrize,
    coadd_dims,
    msk_img,
    scol,
    srow,
):
    # compute the gaia mask for one slice
    slice_dim = buffer_size*2 + central_size
    slice_msk = make_gaia_mask(
        gaia_stars,
        dims=(slice_dim, slice_dim),
        start_row=srow,
        start_col=scol,
        symmetrize=symmetrize,
    )

    # reset the bits to match our output bits
    msk = slice_msk != 0
    slice_msk[:, :] = 0
    slice_msk[msk] = MASK_GAIA_STAR

    # get the final bounds for this slice
    # these bounds are in local slice coords
    slice_bnds = get_slice_bounds(
        orig_start_col=scol,
        orig_start_row=srow,
        central_size=central_size,
        buffer_size=buffer_size,
        coadd_dims=coadd_dims,
    )

    # set the right entries in the global mask image
    msk_img[
        slice_bnds["min_row"]+srow:slice_bnds["max_row"]+srow,
        slice_bnds["min_col"]+scol:slice_bnds["max_col"]+scol,
    ] |= slice_msk[
        slice_bnds["min_row"]:slice_bnds["max_row"],
        slice_bnds["min_col"]:slice_bnds["max_col"],
    ]


def _mask_one_slice_for_missing_data(
    *,
    buffer_size,
    central_size,
    coadd_dims,
    msk_img,
    scol,
    srow,
):
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
    healpix_nside=131072,
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
    healpix_nside : int, optional
        The nisde of the healsparse mask. The default is 131072.

    Returns
    -------
    msk_img : array-like
        An image containing the mask bits.
    hs_msk : healsparse map
        The healsparse boolean map indicating what areas were masked and why. The
        possible flag values are

            MASK_INTILE    (2**0) - indicates data is inside the unique tile boundaries
            MASK_NOSLICE   (2**1) - indicates no data for a given coadd slice
            MASK_GAIA_STAR (2**2) - indicates regions where a GAIA star hole exists
    """
    # We will build a coadd image of the mask bits and then convert them to
    # healsparse at the end. We do things this way for a few reasons.
    # 1. The coadd image of mask bits is the mask actually used by the code.
    # 2. We do not have pure geometric primitives that are easy to write down
    #    in healsparse due to various factors.
    # 3. it makes the code a bit simpler since we don't have to combine more
    #    than one representation of the mask to get the final one

    # this image holds the mask bits
    msk_img = np.zeros(coadd_dims, dtype=np.int32)

    # first do GAIA star masks
    # we don't union healsparse.Circle objects because we may need to
    # apply a 90 degree rotation to the mask holes within each slice
    # this is done in the GAIA masking functions we have and so we use those here
    if gaia_stars is not None:
        if preconfig["gaia_star_masks"]["mask_expand_rad"] > 0:
            _gaia_stars = gaia_stars.copy()
            _gaia_stars['radius_pixels'] += \
                preconfig["gaia_star_masks"]["mask_expand_rad"]
        else:
            _gaia_stars = gaia_stars

        for slice_ind in tqdm.trange(
            len(obj_data), desc='making GAIA masks', ncols=120
        ):
            _mask_one_slice_for_gaia_stars(
                buffer_size=buffer_size,
                central_size=central_size,
                gaia_stars=_gaia_stars,
                symmetrize=preconfig["gaia_star_masks"]["symmetrize"],
                coadd_dims=coadd_dims,
                msk_img=msk_img,
                scol=obj_data["orig_start_col"][slice_ind, 0],
                srow=obj_data["orig_start_row"][slice_ind, 0],
            )

    # then do the slice masks for missing slices
    # these are formally defined in pixels, not ra-dec, though we could likely
    # use healsparse convex polygons instead of the pixels
    for slice_ind in tqdm.tqdm(
        missing_slice_inds, desc='making slice masks', ncols=120
    ):
        _mask_one_slice_for_missing_data(
            buffer_size=buffer_size,
            central_size=central_size,
            coadd_dims=coadd_dims,
            msk_img=msk_img,
            scol=obj_data["orig_start_col"][slice_ind, 0],
            srow=obj_data["orig_start_row"][slice_ind, 0],
        )

    # we need to cut out the parts of the mask that are outside of the unique
    # coadd tile boundaries
    # this part of the mask will be built by the adjacent tiles
    # to do this, we set a flag for pixels inside the boundaries
    # since we need to convert to ra-dec for the cut and ra-dec is needed for
    # making the healsparse files, we do that at the same time as well.
    hs_msk = healsparse.HealSparseMap.make_empty(
        128, healpix_nside, np.int32, sentinel=0
    )
    for yind in tqdm.trange(
        coadd_dims[0],
        desc='cutting tile buffers and making healsparse',
        ncols=120,
    ):
        y = np.zeros(coadd_dims[1]) + yind
        x = np.arange(coadd_dims[1])
        ra, dec = wcs.image2sky(x+position_offset, y+position_offset)
        umsk = in_unique_coadd_tile_region(
            ra=ra,
            dec=dec,
            crossra0=info['crossra0'],
            udecmin=info['udecmin'],
            udecmax=info['udecmax'],
            uramin=info['uramin'],
            uramax=info['uramax'],
        )
        msk_img[yind, umsk] |= MASK_INTILE
        # set the mask to zero outside the unique tile bounds
        msk_img[yind, ~umsk] = 0

        # now add to healsparse - keep only non-zero entries
        msk = msk_img[yind, :] != 0
        if np.any(msk):
            hs_msk.update_values_pos(
                ra[msk],
                dec[msk],
                msk_img[yind, msk],
                operation='or',
            )

    return msk_img, hs_msk
