from numba import njit
import numpy as np
import esutil as eu
from pizza_cutter.des_pizza_cutter import (
    GAIA_STARS_EXTNAME,
    BMASK_GAIA_STAR,
    BMASK_SPLINE_INTERP,
    BMASK_EXPAND_GAIA_STAR,
)
from pizza_cutter.slice_utils.symmetrize import symmetrize_bmask
from pizza_cutter.slice_utils.interpolate import interpolate_image_at_mask


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

    if 'interp' in config and 'apodize' not in config:
        _mask_gaia_stars_interp(mbobs, gaia_stars, config)
    elif 'apodize' in config and 'interp' not in config:
        _mask_gaia_stars_apodize(mbobs, gaia_stars, config)
    elif 'interp' in config and 'apodize' in config:
        raise RuntimeError(
            "Can only do one of 'interp' or 'apodize' for handling GAIA stars!"
        )

    if config["mask_expand_rad"] > 0:
        expanded_gaia_stars = gaia_stars.copy()
        expanded_gaia_stars['radius_pixels'] += config['mask_expand_rad']
        gaia_bmask = make_gaia_mask(
            gaia_stars=expanded_gaia_stars,
            dims=mbobs[0][0].ormask.shape,
            start_row=mbobs[0][0].meta['orig_start_row'],
            start_col=mbobs[0][0].meta['orig_start_col'],
            symmetrize=config['symmetrize'],
            flag=BMASK_EXPAND_GAIA_STAR,
        )
        for obslist in mbobs:
            for obs in obslist:
                # the pixels list will be reset upon exiting
                with obs.writeable():
                    obs.bmask |= gaia_bmask


def _mask_gaia_stars_interp(mbobs, gaia_stars, config):
    # masking is same for all, just take the first
    obs0 = mbobs[0][0]
    ormask = obs0.ormask
    gaia_bmask = make_gaia_mask(
        gaia_stars=gaia_stars,
        dims=ormask.shape,
        start_row=obs0.meta['orig_start_row'],
        start_col=obs0.meta['orig_start_col'],
        symmetrize=config['symmetrize'],
        flag=BMASK_GAIA_STAR,
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

                    if hasattr(obs, "mfrac"):
                        obs.mfrac[wbad] = 1.0
                    if np.all(bad_logic):
                        obs.ignore_zero_weight = False
                    obs.weight[wbad] = 0.0

                    if not np.all(bad_logic):
                        interp_image = interpolate_image_at_mask(
                            image=obs.image, bad_msk=bad_logic, maxfrac=1.0,
                            **config["interp"],
                        )
                        interp_noise = interpolate_image_at_mask(
                            image=obs.noise, bad_msk=bad_logic, maxfrac=1.0,
                            **config["interp"],
                        )
                    else:
                        interp_image = None
                        interp_noise = None

                    if interp_image is None or interp_noise is None:
                        obs.bmask |= BMASK_GAIA_STAR
                        if hasattr(obs, "mfrac"):
                            obs.mfrac[:, :] = 1.0
                        obs.ignore_zero_weight = False
                        obs.weight[:, :] = 0.0
                    else:
                        obs.image = interp_image
                        obs.noise = interp_noise
                        obs.bmask[wbad] |= BMASK_SPLINE_INTERP


def make_gaia_mask(
    *,
    gaia_stars,
    dims,
    start_row,
    start_col,
    symmetrize,
    flag=BMASK_GAIA_STAR,
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
    flag: int, optional
        The bit flag value to use. Default is BMASK_GAIA_STAR from pizza_cutter.
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
        flag=flag,
    )

    if symmetrize:
        symmetrize_bmask(bmask=gaia_bmask)

    return gaia_bmask


def _mask_gaia_stars_apodize(mbobs, gaia_stars, config):
    obs0 = mbobs[0][0]
    ap_mask = make_gaia_apodization_mask(
        gaia_stars=gaia_stars,
        dims=obs0.image.shape,
        start_row=obs0.meta['orig_start_row'],
        start_col=obs0.meta['orig_start_col'],
        symmetrize=config['symmetrize'],
        ap_rad=config["apodize"]["ap_rad"],
    )

    msk = ap_mask < 1
    if np.any(msk):
        for obslist in mbobs:
            for obs in obslist:
                # the pixels list will be reset upon exiting
                with obs.writeable():
                    obs.image *= ap_mask
                    obs.noise *= ap_mask
                    obs.bmask[msk] |= BMASK_GAIA_STAR
                    if hasattr(obs, "mfrac"):
                        obs.mfrac[msk] = 1.0
                    if np.all(msk):
                        obs.ignore_zero_weight = False
                    obs.weight[msk] = 0.0


def make_gaia_apodization_mask(
    *,
    gaia_stars,
    dims,
    start_row,
    start_col,
    symmetrize,
    ap_rad,
):
    """
    mask gaia stars by computing an apodization mask

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
    ap_rad: float
        The scale in pixels for the apodization transition from 1 to 0.
    """

    # must be native byte order for numba
    x = gaia_stars['x'].astype('f8')
    y = gaia_stars['y'].astype('f8')
    radius_pixels = gaia_stars['radius_pixels'].astype('f8')

    ap_mask = np.ones(dims)
    do_apodization_mask(
        rows=y-start_row,
        cols=x-start_col,
        radius_pixels=radius_pixels,
        ap_mask=ap_mask,
        ap_rad=ap_rad,
    )

    if symmetrize:
        ap_mask *= np.rot90(ap_mask)

    return ap_mask


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
def _ap_kern_kern(x, m, h):
    # cumulative triweight kernel
    y = (x - m) / h + 3
    if y < -3:
        return 0
    elif y > 3:
        return 1
    else:
        val = (
            -5 * y ** 7 / 69984
            + 7 * y ** 5 / 2592
            - 35 * y ** 3 / 864
            + 35 * y / 96
            + 1 / 2
        )
        return val


@njit
def do_apodization_mask(*, rows, cols, radius_pixels, ap_mask, ap_rad):
    """low-level code to make the apodization mask

    Parameters
    ----------
    rows, cols: arrays
        Arrays of rows/cols of star locations in the "local" pixel frame of the
        slice, not the overall pixels of the big coadd. These positions may be
        off the image.
    radius_pixels: array
        The radius for each star mask.
    ap_mask: array
        The array to fill with the apodization fraction.
    ap_rad: float
        The scale in pixels for the apodization transition from 1 to 0.
    """
    ny, nx = ap_mask.shape
    ns = cols.shape[0]

    nmasked = 0
    for i in range(ns):
        x = cols[i]
        y = rows[i]
        rad = radius_pixels[i]
        rad2 = rad**2

        if not intersects(y, x, rad, ny, nx):
            continue

        for _y in range(ny):
            dy2 = (_y - y)**2
            for _x in range(nx):
                dr2 = (_x - x)**2 + dy2
                if dr2 < rad2:
                    ap_mask[_y, _x] *= _ap_kern_kern(np.sqrt(dr2), rad, ap_rad)
                    nmasked += 1

    return nmasked


@njit
def do_mask_gaia_stars(*, rows, cols, radius_pixels, bmask, flag):
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
