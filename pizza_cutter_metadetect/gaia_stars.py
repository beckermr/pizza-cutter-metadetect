from numba import njit
import numpy as np
import esutil as eu
from pizza_cutter.des_pizza_cutter import (
    GAIA_STARS_EXTNAME, BMASK_GAIA_STAR, BMASK_SPLINE_INTERP,
)
from pizza_cutter.slice_utils.symmetrize import symmetrize_bmask
from pizza_cutter.slice_utils.interpolate import _grid_interp


def load_gaia_stars(
    mbmeds,
    poly_coeffs,
    max_g_mag,
):
    """
    load the gaia stars from the input meds

    Parameters
    -----------
    mbmeds: MultiBandNGMixMEDS
        multi band meds file
    poly_coeffs: tuple
        Tuple describing a np.poly1d for log10(radius) vs mag.
        E.g. (0.00443223, -0.22569131, 2.99642999)
    max_g_mag: float, optional
        Maximum g mag to mask.  Default 18
    """

    # gaia stars are same in all bands for a given tile, just read one
    gaia_stars = mbmeds.mlist[0]._fits[GAIA_STARS_EXTNAME].read()

    w, = np.where(gaia_stars['phot_g_mean_mag'] <= max_g_mag)
    gaia_stars = gaia_stars[w]

    add_dt = [('radius_pixels', 'f4')]
    gaia_stars = eu.numpy_util.add_fields(gaia_stars, add_dt)

    ply = np.poly1d(poly_coeffs)
    log10_radius_pixels = ply(gaia_stars['phot_g_mean_mag'])
    gaia_stars['radius_pixels'] = 10.0**log10_radius_pixels

    return gaia_stars


def mask_gaia_stars(mbobs, gaia_stars, config):
    """
    mask gaia stars, setting a bit in the bmask, interpolating image and noise,
    setting weight to zero, and setting mfrac=1

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        The observations to mask
    gaia_stars: array
        The gaia star catalog.
    config: dict
        A dictionary of config parameters.
    """

    # masking is same for all, just take the first
    obs0 = mbobs[0][0]
    ormask = obs0.ormask
    gaia_bmask = make_gaia_mask(
        gaia_stars,
        dims=ormask.shape,
        start_row=obs0.meta['orig_start_row'],
        start_col=obs0.meta['orig_start_col'],
        symmetrize=config['symmetrize'],
    )

    # now modify the masks, weight maps, and interpolate in all
    # bands
    bad_logic = gaia_bmask != 0
    wbad = np.where(bad_logic)
    if wbad[0].size > 0:

        for obslist in mbobs:
            for obs in obslist:
                # the pixels list will be reset upon exiting
                with obs.writeable():
                    obs.bmask |= gaia_bmask

                    obs.mfrac[wbad] = 1.0
                    obs.weight[wbad] = 0.0

                    interp_image = _grid_interp(
                        image=obs.image, bad_msk=bad_logic, maxfrac=1.0,
                    )
                    interp_noise = _grid_interp(
                        image=obs.noise, bad_msk=bad_logic, maxfrac=1.0,
                    )
                    if interp_image is None or interp_noise is None:
                        obs.bmask |= BMASK_GAIA_STAR
                        obs.mfrac[:, :] = 1.0
                        obs.weight[:, :] = 0.0
                        # this is ok here, but we need to add a way to
                        # expose this in ngmix
                        obs._ignore_zero_weight = False
                    else:
                        obs.image = interp_image
                        obs.noise = interp_noise
                        obs.bmask[wbad] |= BMASK_SPLINE_INTERP


def make_gaia_mask(
    gaia_stars,
    dims,
    start_row,
    start_col,
    symmetrize,
):
    """
    mask gaia stars, setting a bit in the bmask, interpolating image and noise,
    setting weight to zero, and setting mfrac=1

    Parameters
    ----------
    gaia_stars: array
        The gaia star catalog
    dims: array
        The shape of the image; this is all that is needed to determine
        the masking
    start_row: int
        Row in original larger image frame corresponding to row=0
    start_col: int
        Column in original larger image frame corresponding to col=0
    symmetrize: bool
        If True, symmetrize the mask.
    """

    # must be native byte order for numba
    x = gaia_stars['x'].astype('f8')
    y = gaia_stars['y'].astype('f8')
    radius_pixels = gaia_stars['radius_pixels'].astype('f8')

    gaia_bmask = np.zeros(dims, dtype='i4')
    do_mask_gaia_stars(
        rows=y - start_row,
        cols=x - start_col,
        radius_pixels=radius_pixels,
        bmask=gaia_bmask,
        flag=BMASK_GAIA_STAR,
    )

    if symmetrize:
        symmetrize_bmask(bmask=gaia_bmask)

    return gaia_bmask


@njit
def intersects(row, col, radius_pixels, nrows, ncols):
    """
    low level routine to check if the mask intersects the image.
    For simplicty just check the bounding rectangle

    Parameters
    ----------
    row, col: float
        The row and column of the mask center
    radius_pixels: float
        The radius for the star mask
    nrows, ncols: int
        Shape of the image

    Returns
    -------
    True if it intersects, otherwise False
    """

    low_row = -radius_pixels
    high_row = nrows + radius_pixels - 1
    low_col = -radius_pixels
    high_col = ncols + radius_pixels - 1

    if (
        row > low_row and row < high_row and
        col > low_col and col < high_col
    ):
        return True
    else:
        return False


@njit
def do_mask_gaia_stars(rows, cols, radius_pixels, bmask, flag):
    """
    low level code to mask stars

    Parameters
    ----------
    rows, cols: arrays
        Arrays of rows/cols of star locations in the "local" pixel frame of the
        slice, not the overall pixels of the big coadd. These positions may be
        off the image.
    radius_pixels: array
        The radius for each star mask.
    bmask: array
        The bmask to modify.
    flag: int
        The flag value to "or" into the bmask.
    """
    nrows, ncols = bmask.shape
    nmasked = 0

    for istar in range(rows.size):
        row = rows[istar]
        col = cols[istar]

        rad = radius_pixels[istar]
        rad2 = rad * rad

        if not intersects(row, col, rad, nrows, ncols):
            continue

        for irow in range(nrows):
            rowdiff2 = (row - irow)**2
            for icol in range(ncols):

                r2 = rowdiff2 + (col - icol)**2
                if r2 < rad2:
                    bmask[irow, icol] |= flag
                    nmasked += 1

    return nmasked
