import numpy as np
import esutil as eu
from pizza_cutter.des_pizza_cutter import (
    GAIA_STARS_EXTNAME,
    BMASK_GAIA_STAR,
    BMASK_SPLINE_INTERP,
    BMASK_EXPAND_GAIA_STAR,
)
from metadetect.masking import apply_foreground_masking_corrections


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


def mask_gaia_stars(mbobs, gaia_stars, config, rng):
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
    rng: np.random.RandomState
        An RNG to use for filling with noise.
    """

    obs0 = mbobs[0][0]
    xm = gaia_stars['x'].astype('f8') - obs0.meta['orig_start_col']
    ym = gaia_stars['y'].astype('f8') - obs0.meta['orig_start_row']
    rm = gaia_stars['radius_pixels'].astype('f8')

    if 'interp' in config and 'apodize' not in config:
        apply_foreground_masking_corrections(
            mbobs=mbobs,
            xm=xm,
            ym=ym,
            rm=rm,
            method=(
                'interp-noise'
                if config['interp']['fill_isolated_with_noise']
                else 'interp'
            ),
            mask_expand_rad=config["mask_expand_rad"],
            mask_bit_val=BMASK_GAIA_STAR,
            expand_mask_bit_val=BMASK_EXPAND_GAIA_STAR,
            interp_bit_val=BMASK_SPLINE_INTERP,
            symmetrize=config['symmetrize'],
            ap_rad=None,
            iso_buff=config['interp']['iso_buff'],
            rng=rng,
        )
    elif 'apodize' in config and 'interp' not in config:
        apply_foreground_masking_corrections(
            mbobs=mbobs,
            xm=xm,
            ym=ym,
            rm=rm,
            method='apodize',
            mask_expand_rad=config["mask_expand_rad"],
            mask_bit_val=BMASK_GAIA_STAR,
            expand_mask_bit_val=BMASK_EXPAND_GAIA_STAR,
            interp_bit_val=BMASK_SPLINE_INTERP,
            symmetrize=config['symmetrize'],
            ap_rad=config['apodize']['ap_rad'],
            iso_buff=None,
            rng=rng,
        )
    elif 'interp' in config and 'apodize' in config:
        raise RuntimeError(
            "Can only do one of 'interp' or 'apodize' for handling GAIA stars!"
        )
