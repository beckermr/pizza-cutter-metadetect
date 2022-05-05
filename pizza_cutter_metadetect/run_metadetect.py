import os
import json
import time
import datetime
import logging
import subprocess

import yaml
import joblib
import numpy as np
import esutil as eu
import fitsio
import healpy as hp

from esutil.pbar import PBar
from metadetect.metadetect import do_metadetect
from metadetect.masking import apply_apodization_corrections
from pizza_cutter.files import expandpath
from pizza_cutter.des_pizza_cutter import BMASK_SLICE_APODIZED
from .gaia_stars import (
    load_gaia_stars, mask_gaia_stars, BMASK_GAIA_STAR, BMASK_EXPAND_GAIA_STAR,
)
from .masks import (
    make_mask, get_slice_bounds, in_unique_coadd_tile_region,
    MASK_TILEDUPE, MASK_SLICEDUPE, MASK_GAIA_STAR,
    MASK_NOSLICE, MASK_MISSING_BAND, MASK_MISSING_NOSHEAR_DET,
    MASK_MISSING_BAND_PREPROC, MASK_MISSING_MDET_RES,
    MASK_MDET_FAILED,
)
from pizza_cutter.des_pizza_cutter import get_coaddtile_geom
from metadetect.fitting import MAX_NUM_SHEAR_BANDS

LOGGER = logging.getLogger(__name__)


def split_range(meds_range):
    """
    Parameters
    ----------
    meds_range: str
        e.g. '3:7' is like a python slice

    Returns
    -------
    start, end_plus_one, num
        Start index, end+1 from the slice, and number to process
    """
    start, end_plus_one = meds_range.split(':')
    start = int(start)
    end_plus_one = int(end_plus_one)
    num = end_plus_one - start  # last element is the end of the range + 1
    return start, end_plus_one, num


def make_output_filename(directory, config_fname, meds_fname, part, meds_range):
    """
    make the output name

    Parameters
    ----------
    directory: str
        The directory for the outputs.
    config_fname: str
        The config file name.
    meds_fname: str
        Example meds file name
    part: int
        The part of the file processed
    meds_range: str
        The slice to process, as as string, e.g. '3:7'

    Returns
    -------
    file basename
    """
    mdetrun = os.path.basename(config_fname)
    if mdetrun.endswith(".yaml") or mdetrun.endswith(".yml"):
        mdetrun = mdetrun.rsplit(".", 1)[0]

    fname = os.path.basename(meds_fname)
    fname = fname.replace('.fz', '').replace('.fits', '')

    items = fname.split('_')
    # keep real DES data names short. By convention, the first part
    # is the tilename
    parts = [items[0], mdetrun, "mdetcat"]

    if part is None and meds_range is None:
        part = 0

    if part is not None:
        tail = 'part%04d.fits.fz' % part
    else:
        start, end_plus_one, num = split_range(meds_range)
        end = end_plus_one - 1

        tail = 'range%04d-%04d.fits.fz' % (start, end)

    parts.append(tail)

    fname = '_'.join(parts)

    fname = os.path.join(directory, fname)
    fname = expandpath(fname)
    return fname


def _make_output_dtype(*, nbands, filename_len, tilename_len, band_names):
    new_dt = [
        ('slice_id', 'i8'),
        ('mdet_step', 'U7'),
        ('ra', 'f8'),
        ('dec', 'f8'),
        ('ra_noshear', 'f8'),
        ('dec_noshear', 'f8'),
        ('y_noshear', 'f8'),
        ('x_noshear', 'f8'),
        ('y', 'f8'),
        ('x', 'f8'),
        ('slice_y', 'f8'),
        ('slice_x', 'f8'),
        ('slice_y_noshear', 'f8'),
        ('slice_x_noshear', 'f8'),
        ('hpix_16384', 'i8'),
        ('hpix_16384_noshear', 'i8'),
        ('filename', 'U%d' % filename_len),
        ('tilename', 'U%d' % tilename_len),

        # columns from mdet
        # we flatten shears and matrices
        ("flags", 'i4'),
        ("shear_bands", "U%d" % MAX_NUM_SHEAR_BANDS),

        ('psf_flags', 'i4'),
        ('psf_g_1', 'f8'),
        ('psf_g_2', 'f8'),
        ('psf_T', 'f8'),

        ("mdet_flags", 'i4'),
        ("mdet_s2n", "f8"),
        ("mdet_g_1", "f8"),
        ("mdet_g_2", "f8"),
        ("mdet_g_cov_1_1", "f8"),
        ("mdet_g_cov_1_2", "f8"),
        ("mdet_g_cov_2_2", "f8"),

        ("mdet_T", "f8"),
        ("mdet_T_err", "f8"),
        ("mdet_T_ratio", "f8"),
        ("mdet_T_flags", "i4"),

        ('ormask', 'i4'),
        ('mfrac', 'f4'),
        ('bmask', 'i4'),
        ('mask_flags', 'i4'),

        ('ormask_noshear', 'i4'),
        ('mfrac_noshear', 'f4'),
        ('bmask_noshear', 'i4'),
        ('mask_flags_noshear', 'i4'),

        # this is the original PSF
        ('psfrec_flags', 'i4'),
        ('psfrec_g_1', 'f8'),
        ('psfrec_g_2', 'f8'),
        ('psfrec_T', 'f8'),
    ]

    new_dt += [
        ("mdet_%s_flux_flags" % b, "i4")
        for b in band_names
    ]
    new_dt += [
        ("mdet_%s_flux" % b, "f8")
        for b in band_names
    ]
    new_dt += [
        ("mdet_%s_flux_err" % b, "f8")
        for b in band_names
    ]
    new_dt += [
        ("nepoch_%s" % b, "i4")
        for b in band_names
    ]
    new_dt += [
        ("nepoch_eff_%s" % b, "i4")
        for b in band_names
    ]

    return new_dt


def _make_output_array(
    *,
    data, slice_id, mdet_step,
    orig_start_row, orig_start_col, position_offset, wcs, buffer_size,
    central_size, coadd_dims, model, info, output_file, band_names,
    nepoch_per_band, nepoch_eff_per_band,
):
    """
    Add columns to the output data array. These include the slice id, metacal
    step (e.g. '1p'), ra, dec in the sheared coordinates as well as unsheared
    coordinates

    Parameters
    ----------
    data: array with fields
        The data, to be augmented
    slice_id: int
        The slice id
    mdet_step: str
        e.g. 'noshear', '1p', '1m', '2p', '2m'
    orig_start_row: int
        Start row of origin of slice
    orig_start_col: int
        Start col of origin of slice
    position_offset: int
        Often 1 for wcs
    wcs: world coordinate system object
        wcs for converting image positions to sky positions
    buffer_size: int
        The size of the buffer region to cut for each slice.
    central_size: int
        The size of the central region.
    coadd_dims: tuple of ints
        The dimension of the full image that the slices tile.
    model: str
        The model used for metadetect. This is used to rename columns starting
        with '{model}_' to 'mdet_'.
    info : dict
        Dict of tile geom information for getting detections in the tile boundaries.
    output_file : str
        The output filename.
    band_names : list of str
        If given, the names of the bands as single strings to use in generating the
        output data.
    nepoch_per_band : list of int
        The number of coadded epochs per band.
    nepoch_eff_per_band : list of int
        The effective number of coadded epochs per band.

    Returns
    -------
    array with new fields
    """
    mpre = model + '_'

    # get # of bands
    name = mpre + "band_flux"
    if len(data[name].shape) == 1:
        nbands = 1
    else:
        nbands = data[name].shape[1]

    if band_names is not None:
        assert len(band_names) == nbands, (
            "The # of band names %s doesn't match the number of bands %d." % (
                band_names,
                nbands,
            )
        )
    else:
        band_names = ["b%d" % i for i in range(nbands)]

    filename = os.path.basename(output_file)
    if filename.endswith(".fz"):
        filename = filename[:-len(".fz")]
    tilename = filename.split('_')[0]
    new_dt = _make_output_dtype(
        nbands=nbands,
        filename_len=len(filename),
        tilename_len=len(tilename),
        band_names=band_names,
    )

    arr = np.zeros(data.shape, dtype=new_dt)
    for name in arr.dtype.names:
        if arr[name].dtype.kind == "f":
            arr[name] = np.nan

    # fill simple columns
    for col in [
        "flags",
        "shear_bands",
        "psf_flags",
        "psf_T",
        "ormask",
        "mfrac",
        "bmask",
        "ormask_noshear",
        "mfrac_noshear",
        "bmask_noshear",
        "psfrec_flags",
        "psfrec_T",
    ]:
        arr[col] = data[col]

    # now fill the model dependent ones
    for col in [
        "mdet_flags",
        "mdet_s2n",
        "mdet_T",
        "mdet_T_err",
        "mdet_T_ratio",
        "mdet_T_flags",
    ]:
        data_col = mpre + col[len("mdet_"):]
        arr[col] = data[data_col]

    # do the shears
    arr["psf_g_1"] = data["psf_g"][:, 0]
    arr["psf_g_2"] = data["psf_g"][:, 1]
    arr["psfrec_g_1"] = data["psfrec_g"][:, 0]
    arr["psfrec_g_2"] = data["psfrec_g"][:, 1]

    arr["mdet_g_1"] = data[mpre + "g"][:, 0]
    arr["mdet_g_2"] = data[mpre + "g"][:, 1]

    arr["mdet_g_cov_1_1"] = data[mpre + "g_cov"][:, 0, 0]
    arr["mdet_g_cov_1_2"] = data[mpre + "g_cov"][:, 0, 1]
    arr["mdet_g_cov_2_2"] = data[mpre + "g_cov"][:, 1, 1]

    # fluxes
    if nbands == 1:
        arr["mdet_%s_flux_flags" % band_names[0]] = data[mpre + "band_flux_flags"][:]
        arr["mdet_%s_flux" % band_names[0]] = data[mpre + "band_flux"][:]
        arr["mdet_%s_flux_err" % band_names[0]] \
            = data[mpre + "band_flux_err"][:]
    else:
        for i, b in enumerate(band_names):
            arr["mdet_%s_flux_flags" % b] = data[mpre + "band_flux_flags"][:, i]
            arr["mdet_%s_flux" % b] = data[mpre + "band_flux"][:, i]
            arr["mdet_%s_flux_err" % b] = data[mpre + "band_flux_err"][:, i]

    assert len(nepoch_per_band) == nbands, (
        "The length of the band nepochs list %s doesn't match the "
        "number of bands %d." % (
            nepoch_per_band,
            nbands,
        )
    )
    assert len(nepoch_eff_per_band) == nbands, (
        "The length of the effective band nepochs list %s doesn't match the "
        "number of bands %d." % (
            nepoch_eff_per_band,
            nbands,
        )
    )
    for b, ne in zip(band_names, nepoch_per_band):
        arr["nepoch_%s" % b] = ne
    for b, ne in zip(band_names, nepoch_eff_per_band):
        arr["nepoch_eff_%s" % b] = ne

    arr['slice_id'] = slice_id
    arr['mdet_step'] = mdet_step
    arr['filename'] = filename
    arr['tilename'] = tilename

    # deal with positions
    arr['slice_y_noshear'] = data['sx_row_noshear']
    arr['slice_x_noshear'] = data['sx_col_noshear']
    arr['slice_y'] = data['sx_row']
    arr['slice_x'] = data['sx_col']

    # these are in global coadd coords
    arr['y'] = orig_start_row + data['sx_row']
    arr['x'] = orig_start_col + data['sx_col']
    arr['y_noshear'] = orig_start_row + data['sx_row_noshear']
    arr['x_noshear'] = orig_start_col + data['sx_col_noshear']

    arr['ra'], arr['dec'] = _get_radec(
        row=arr['slice_y'],
        col=arr['slice_x'],
        orig_start_row=orig_start_row,
        orig_start_col=orig_start_col,
        position_offset=position_offset,
        wcs=wcs,
    )
    arr['hpix_16384'] = hp.ang2pix(
        16384, arr['ra'], arr['dec'], nest=True, lonlat=True
    )
    arr['ra_noshear'], arr['dec_noshear'] = _get_radec(
        row=arr['slice_y_noshear'],
        col=arr['slice_x_noshear'],
        orig_start_row=orig_start_row,
        orig_start_col=orig_start_col,
        position_offset=position_offset,
        wcs=wcs,
    )
    arr['hpix_16384_noshear'] = hp.ang2pix(
        16384, arr['ra_noshear'], arr['dec_noshear'], nest=True, lonlat=True
    )

    slice_bnds = get_slice_bounds(
        orig_start_col=orig_start_col,
        orig_start_row=orig_start_row,
        central_size=central_size,
        buffer_size=buffer_size,
        coadd_dims=coadd_dims
    )

    for tail in ["", "_noshear"]:
        msk = (
            (arr['slice_y' + tail] >= slice_bnds["min_row"])
            & (arr['slice_y' + tail] < slice_bnds["max_row"])
            & (arr['slice_x' + tail] >= slice_bnds["min_col"])
            & (arr['slice_x' + tail] < slice_bnds["max_col"])
        )
        arr["mask_flags" + tail][~msk] |= MASK_SLICEDUPE
        msk = in_unique_coadd_tile_region(
            ra=arr['ra' + tail],
            dec=arr['dec' + tail],
            crossra0=info['crossra0'],
            udecmin=info['udecmin'],
            udecmax=info['udecmax'],
            uramin=info['uramin'],
            uramax=info['uramax'],
        )
        arr["mask_flags" + tail][~msk] |= MASK_TILEDUPE
        msk = (
            ((arr['bmask' + tail] & BMASK_EXPAND_GAIA_STAR) != 0)
            | ((arr['bmask' + tail] & BMASK_GAIA_STAR) != 0)
        )
        arr["mask_flags" + tail][msk] |= MASK_GAIA_STAR

    return arr


def _get_radec(*,
               row,
               col,
               orig_start_row,
               orig_start_col,
               position_offset,
               wcs):
    """
    Convert image positions to sky positions

    Parameters
    ----------
    row: array
        array of rows in slice coordinates
    col: array
        array of columns in slice coordinates
    orig_start_row: float
        Start row of origin of slice
    orig_start_col: float
        Start col of origin of slice
    position_offset: int
        Always 1 for DES WCS transforms
    wcs: world coordinate system object
        wcs for converting image positions to sky positions

    Returns
    -------
    ra, dec arrays
    """

    trow = np.array(row + orig_start_row + position_offset).astype(np.float64)
    tcol = np.array(col + orig_start_col + position_offset).astype(np.float64)
    ra, dec = wcs.image2sky(x=tcol, y=trow)
    return ra, dec


def _post_process_results(
    *, outputs, obj_data_list, image_info, buffer_size, central_size, config, info,
    output_file, band_names,
):
    # post process results
    wcs_cache = {}

    obj_data = obj_data_list[0]

    output = []
    dt = 0
    missing_slice_inds = []
    missing_slice_flags = []
    for res, i, _dt, flags in outputs:
        dt += _dt
        if res is None or res["noshear"] is None or res["noshear"].size == 0:
            if res is None:
                flags |= MASK_MISSING_MDET_RES
            else:
                flags |= MASK_MISSING_NOSHEAR_DET
            missing_slice_inds.append(i)
            missing_slice_flags.append(flags)
            continue

        for mdet_step, data in res.items():
            if data is not None and data.size > 0:
                file_id = max(obj_data['file_id'][i, 0], 0)
                if file_id in wcs_cache:
                    wcs, position_offset = wcs_cache[file_id]
                else:
                    wcs = eu.wcsutil.WCS(json.loads(image_info['wcs'][file_id]))
                    position_offset = image_info['position_offset'][file_id]
                    wcs_cache[file_id] = (wcs, position_offset)

                coadd_dims = (wcs.get_naxis()[0], wcs.get_naxis()[1])
                assert coadd_dims == (10000, 10000), (
                    "Wrong coadd dims %s computed!" % (coadd_dims,)
                )

                output.append(_make_output_array(
                    data=data,
                    slice_id=obj_data['id'][i],
                    mdet_step=mdet_step,
                    orig_start_col=obj_data['orig_start_col'][i, 0],
                    orig_start_row=obj_data['orig_start_row'][i, 0],
                    wcs=wcs,
                    position_offset=position_offset,
                    buffer_size=buffer_size,
                    central_size=central_size,
                    coadd_dims=coadd_dims,
                    model=config['model'],
                    info=info,
                    output_file=output_file,
                    band_names=band_names,
                    nepoch_per_band=[od["nepoch"][i] for od in obj_data_list],
                    nepoch_eff_per_band=[od["nepoch_eff"][i] for od in obj_data_list],
                ))

    if len(output) > 0:
        # concatenate once since generally more efficient
        output = np.concatenate(output)
        assert len(wcs_cache) == 1
    else:
        output = None
        # default to first slice if we find nothing
        i = 0
        file_id = max(obj_data['file_id'][i, 0], 0)
        wcs = eu.wcsutil.WCS(json.loads(image_info['wcs'][file_id]))
        position_offset = image_info['position_offset'][file_id]
        coadd_dims = (wcs.get_naxis()[0], wcs.get_naxis()[1])
        assert coadd_dims == (10000, 10000), (
            "Wrong coadd dims %s computed!" % (coadd_dims,)
        )

    return (
        output, dt, missing_slice_inds, missing_slice_flags, wcs,
        position_offset, coadd_dims
    )


def _truncate_negative_mfrac_weight(mbobs):
    for obslist in mbobs:
        for obs in obslist:
            with obs.writeable():
                msk = obs.mfrac < 0
                if np.any(msk):
                    LOGGER.debug(
                        "truncating negative mfrac values: min %f",
                        obs.mfrac[msk].min(),
                    )
                    obs.mfrac[msk] = 0

                msk = obs.weight < 0
                if np.any(msk):
                    LOGGER.debug(
                        "truncating negative weight values: min %f",
                        obs.weight[msk].min(),
                    )
                    obs.weight[msk] = 0


def _write_mbobs_image(viz_dir, mbobs, islice, slug):
    import proplot as pplt

    nrows = sum([1 if len(mbobs[i]) > 0 else 0 for i in range(len(mbobs))])
    ncols = 6
    cmap = "rocket"

    fig, axs = pplt.subplots(
        nrows=nrows, ncols=ncols, refaspect=1, span=False,
    )

    for i in range(nrows):
        if len(mbobs[i]) == 0:
            continue
        obs = mbobs[i][0]

        ax = axs[i, 0]
        ax.imshow(
            np.arcsinh(obs.image * np.sqrt(obs.weight)),
            cmap=cmap,
            origin='lower',
        )
        ax.grid(False)
        if i == 0:
            ax.set_title("image")
        ax.set_ylabel("band %d" % i)

        ax = axs[i, 1]
        ax.imshow(
            obs.mfrac,
            cmap=cmap,
            origin='lower',
            vmin=0,
            vmax=obs.mfrac.max(),
        )
        ax.grid(False)
        if i == 0:
            ax.set_title("mfrac")

        ax = axs[i, 2]
        ax.imshow(
            np.arcsinh(obs.bmask),
            cmap=cmap,
            origin='lower',
        )
        ax.grid(False)
        if i == 0:
            ax.set_title("bmask")

        ax = axs[i, 3]
        ax.imshow(
            np.arcsinh(obs.ormask),
            cmap=cmap,
            origin='lower',
        )
        ax.grid(False)
        if i == 0:
            ax.set_title("ormask")

        ax = axs[i, 4]
        ax.imshow(
            np.arcsinh(obs.noise),
            cmap=cmap,
            origin='lower',
        )
        ax.grid(False)
        if i == 0:
            ax.set_title("noise")

        ax = axs[i, 5]
        ax.imshow(
            np.arcsinh(obs.weight),
            cmap=cmap,
            origin='lower',
            vmin=0,
            vmax=obs.weight.max(),
        )
        ax.grid(False)
        if i == 0:
            ax.set_title("weight")

    fname = os.path.join(viz_dir, "mbobs%d%s.png" % (islice, slug))
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    fig.savefig(fname)


def _preprocess_for_metadetect(preconfig, mbobs, gaia_stars, i, rng):
    LOGGER.debug("preprocessing entry %d", i)

    _truncate_negative_mfrac_weight(mbobs)

    if gaia_stars is not None:
        LOGGER.debug("masking GAIA stars")
        mask_gaia_stars(mbobs, gaia_stars, preconfig['gaia_star_masks'], rng)

    if preconfig is None:
        return mbobs
    else:
        if "slice_apodization" in preconfig:
            apply_apodization_corrections(
                mbobs=mbobs,
                ap_rad=preconfig["slice_apodization"]["ap_rad"],
                mask_bit_val=BMASK_SLICE_APODIZED,
            )
        return mbobs


def _get_shearband_combs(nbands):
    shear_band_combs = [list(range(nbands))]
    if nbands > 2:
        shear_band_combs += [list(range(nbands))[1:]]
    if nbands > 1:
        for i in range(nbands):
            shear_band_combs += [[i]]
    return shear_band_combs


def _do_metadetect(config, mbobs, gaia_stars, seed, i, preconfig, viz_dir):
    _t0 = time.time()
    res = None
    flags = 0
    if mbobs is not None:
        if viz_dir is not None:
            _write_mbobs_image(viz_dir, mbobs, i, "_raw")

        rng = np.random.RandomState(seed=seed)
        minnum = min([len(olist) for olist in mbobs])
        if minnum > 0:
            LOGGER.debug("preprocessing entry %d", i)
            mbobs = _preprocess_for_metadetect(preconfig, mbobs, gaia_stars, i, rng)

            if viz_dir is not None:
                _write_mbobs_image(viz_dir, mbobs, i, "_preproc")

            minnum = min([len(olist) for olist in mbobs])
            if minnum > 0:
                LOGGER.debug("running mdet for entry %d", i)

                try:
                    res = do_metadetect(
                        config,
                        mbobs,
                        rng,
                        shear_band_combs=_get_shearband_combs(len(mbobs)),
                    )
                except Exception as e:
                    LOGGER.debug("metadetect failed for slice %d: %s", i, repr(e))
                    flags |= MASK_MDET_FAILED
            else:
                LOGGER.debug(
                    "mbobs has no data for entry %d in one or more "
                    "bands after pre-processing: %s",
                    i,
                    [len(olist) for olist in mbobs],
                )
                flags |= MASK_MISSING_BAND_PREPROC
        else:
            LOGGER.debug(
                "mbobs has no data for entry %d in one or more bands: %s",
                i,
                [len(olist) for olist in mbobs],
            )
            flags |= MASK_MISSING_BAND
    else:
        LOGGER.debug("mbobs is None for entry %d", i)
        flags |= MASK_NOSLICE

    return res, i, time.time() - _t0, flags


def get_part_ranges(part, n_parts, size):
    """Divide a list of things of length `size` into `n_parts` and
    retrurn the range for the given `part`.

    Parameters
    ----------
    part : int
        The 1-indexed part.
    n_parts : int
        The total number of parts.
    size : int
        The length of the list of items to split into `n_parts`.

    Returns
    -------
    start : int
        The staring location.
    num : int
        The number of items in the part.
    """
    n_per = size // n_parts
    n_extra = size - n_per * n_parts
    n_per = np.ones(n_parts, dtype=np.int64) * n_per
    if n_extra > 0:
        n_per[:n_extra] += 1
    stop = np.cumsum(n_per)
    start = stop - n_per
    return start[part-1], n_per[part-1]


def _make_meds_iterator(mbmeds, start, num):
    """This function returns a function which is used as an iterator.

    Closure closure blah blah blah.

    TLDR: Doing things this way allows the code to only read a subset of the
    images from disk in a pipelined manner.

    This works because all of the list-like things fed to joblib are actually
    generators that build their values on-the-fly.
    """
    def _func():
        for i in range(start, start+num):
            mbobs = mbmeds.get_mbobs(i)
            LOGGER.debug("read meds entry %d", i)
            yield i, mbobs

    return _func


def _load_gaia_stars(mbmeds, preconfig):
    if 'gaia_star_masks' in preconfig:
        gaia_config = preconfig['gaia_star_masks']
        gaia_stars = load_gaia_stars(
            mbmeds=mbmeds,
            poly_coeffs=gaia_config['poly_coeffs'],
            max_g_mag=gaia_config['max_g_mag'],
        )
        print("loaded GAIA star masks", flush=True)
    else:
        gaia_stars = None
    return gaia_stars


def run_metadetect(
    *,
    config,
    multiband_meds,
    output_file,
    mask_output_file,
    seed,
    preconfig,
    start=0,
    num=None,
    n_jobs=1,
    verbose=100,
    viz_dir=None,
    band_names=None,
):
    """Run metadetect on a "pizza slice" MEDS file and write the outputs to
    disk.

    Parameters
    ----------
    config : dict
        The metadetect configuration file.
    multiband_meds : `ngmix.medsreaders.MultiBandNGMixMEDS`
        A multiband MEDS data structure.
    output_file : str
        The file to which to write the outputs.
    mask_output_file : str
        The file to write the healsparse mask to.
    seed: int
        Base seed for generating seeds
    preconfig : dict
        Proprocessing configuration.  May contain gaia_star_masks
        entry.
    start : int, optional
        The first entry of the file to process. Defaults to zero.
    num : int, optional
        The number of entries of the file to process, starting at `start`.
        The default of `None` will process all entries in the file.
    n_jobs : int, optional
        The number of jobs to use.
    verbose : int, optional
        joblib logging level.
    viz_dir : str, optional
        If not None, write images of the slices to the given location.
    band_names : list of str, optional
        If given, the names of the bands as single strings to use in generating the
        output data.
    """
    t0 = time.time()

    # process each slice in a pipeline
    if num is None:
        num = multiband_meds.size

    if num + start > multiband_meds.size:
        num = multiband_meds.size - start

    print('# of slices: %d' % num, flush=True)
    print('slice range: [%d, %d)' % (start, start+num), flush=True)
    meds_iter = _make_meds_iterator(multiband_meds, start, num)

    gaia_stars = _load_gaia_stars(mbmeds=multiband_meds, preconfig=preconfig)

    if n_jobs == 1:
        outputs = [
            _do_metadetect(
                config, mbobs, gaia_stars, seed+i*256, i,
                preconfig, viz_dir
            )
            for i, mbobs in PBar(meds_iter(), total=num)]
    else:
        outputs = joblib.Parallel(
            verbose=verbose,
            n_jobs=n_jobs,
            pre_dispatch='2*n_jobs',
            max_nbytes=None,  # never memmap
        )(
            joblib.delayed(_do_metadetect)(
                config, mbobs, gaia_stars, seed+i*256, i,
                preconfig, viz_dir,
            )
            for i, mbobs in meds_iter()
        )

    # join all the outputs
    meta = multiband_meds.mlist[0].get_meta()
    if 'tile_info' in meta.dtype.names:
        info = json.loads(meta["tile_info"][0])
    else:
        try:
            info = json.loads(
                multiband_meds.mlist[0]._fits['tile_info'].read().tobytes()
            )
        except Exception:
            print(
                "WARNING: tile info not found! attempting to read from the database!",
                flush=True,
            )
            tilename = json.loads(
                multiband_meds.mlist[0].get_image_info()['wcs'][0]
            )['desfname'].split("_")[0]
            info = get_coaddtile_geom(tilename)

    pz_config = yaml.safe_load(meta['config'][0])
    (
        output, cpu_time, missing_slice_inds, missing_slice_flags,
        wcs, position_offset, coadd_dims
    ) = _post_process_results(
        outputs=outputs,
        obj_data_list=[mle.get_cat() for mle in multiband_meds.mlist],
        image_info=multiband_meds.mlist[0].get_image_info(),
        buffer_size=int(pz_config['coadd']['buffer_size']),
        central_size=int(pz_config['coadd']['central_size']),
        config=config,
        info=info,
        output_file=output_file,
        band_names=band_names,
    )

    # make the masks
    msk_img, hs_msk = make_mask(
        preconfig=preconfig,
        gaia_stars=gaia_stars,
        missing_slice_inds=missing_slice_inds,
        missing_slice_flags=missing_slice_flags,
        obj_data=multiband_meds.mlist[0].get_cat(),
        buffer_size=int(pz_config['coadd']['buffer_size']),
        central_size=int(pz_config['coadd']['central_size']),
        wcs=wcs,
        position_offset=position_offset,
        coadd_dims=coadd_dims,
        info=info,
    )

    # report and do i/o
    wall_time = time.time() - t0
    print(
        "run time:",
        str(datetime.timedelta(seconds=int(wall_time))),
        flush=True,
    )
    print(
        "CPU time:",
        str(datetime.timedelta(seconds=int(cpu_time))),
        flush=True,
    )
    print(
        "CPU seconds per slice:",
        cpu_time / num,
        flush=True,
    )

    if output is not None:
        with fitsio.FITS(output_file[:-len(".fz")], "rw", clobber=True) as fits:
            fits.write(output, extname="cat")
            fits.create_image_hdu(
                img=None,
                dtype="i4",
                dims=msk_img.shape,
                extname="msk",
                header=pz_config["fpack_pars"])
            fits["msk"].write_keys(pz_config["fpack_pars"], clean=False)
            fits["msk"].write(msk_img)

        # fpack it
        try:
            os.remove(output_file)
        except FileNotFoundError:
            pass
        cmd = 'fpack %s' % output_file[:-len(".fz")]
        print("fpack cmd:", cmd, flush=True)
        try:
            subprocess.check_call(cmd, shell=True)
        except Exception:
            pass
        else:
            try:
                os.remove(output_file[:-len(".fz")])
            except Exception:
                pass

        hs_msk.write(mask_output_file, clobber=True)
    else:
        print("WARNING: no output produced by metadetect!", flush=True)
