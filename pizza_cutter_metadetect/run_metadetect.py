import os
import json
import time
import datetime
import logging
import copy

import yaml
import joblib
import numpy as np
import esutil as eu
import fitsio
import ngmix

from metadetect.metadetect import do_metadetect
from pizza_cutter.slice_utils.pbar import PBar
from pizza_cutter.files import expandpath
from .gaia_stars import load_gaia_stars, mask_gaia_stars

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
        tail = 'part%04d.fits' % part
    else:
        start, end_plus_one, num = split_range(meds_range)
        end = end_plus_one - 1

        tail = 'range%04d-%04d.fits' % (start, end)

    parts.append(tail)

    fname = '_'.join(parts)

    fname = os.path.join(directory, fname)
    fname = expandpath(fname)
    return fname


def _make_output_array(
    *,
    data, slice_id, mdet_step,
    orig_start_row, orig_start_col, position_offset, wcs, buffer_size,
    central_size, coadd_dims, model,
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

    Returns
    -------
    array with new fields
    """
    new_dt = [
        ('slice_id', 'i8'),
        ('mdet_step', 'U7'),
        ('ra', 'f8'),
        ('dec', 'f8'),
        ('ra_det', 'f8'),
        ('dec_det', 'f8'),
        ('row_det', 'f8'),
        ('col_det', 'f8'),
        ('row', 'f8'),
        ('col', 'f8'),
        ('slice_row', 'f8'),
        ('slice_col', 'f8'),
        ('slice_row_det', 'f8'),
        ('slice_col_det', 'f8'),
    ]
    skip_cols = ["sx_row", "sx_col", "sx_row_noshear", "sx_col_noshear"]
    mpre = model + '_'
    for fld in data.dtype.descr:
        if fld[0] in skip_cols:
            continue
        elif fld[0].startswith(mpre):
            new_fld = copy.deepcopy(list(fld))
            new_fld[0] = "mdet_" + fld[0][len(mpre):]
            new_fld = tuple(new_fld)
        else:
            new_fld = fld
        new_dt += [new_fld]

    arr = np.zeros(data.shape, dtype=new_dt)
    for name in data.dtype.names:
        if name in arr.dtype.names:
            arr[name] = data[name]
        elif name.startswith(mpre):
            new_name = "mdet_" + name[len(mpre):]
            arr[new_name] = data[name]

    arr['slice_id'] = slice_id
    arr['mdet_step'] = mdet_step

    # we swap names here calling the sheared pos _det
    arr['slice_row'] = data['sx_row_noshear']
    arr['slice_col'] = data['sx_col_noshear']
    arr['slice_row_det'] = data['sx_row']
    arr['slice_col_det'] = data['sx_col']

    # these are in global coadd coords
    arr['row'] = orig_start_row + data['sx_row_noshear']
    arr['col'] = orig_start_col + data['sx_col_noshear']
    arr['row_det'] = orig_start_row + data['sx_row']
    arr['col_det'] = orig_start_col + data['sx_col']

    arr['ra'], arr['dec'] = _get_radec(
        row=arr['slice_row'],
        col=arr['slice_col'],
        orig_start_row=orig_start_row,
        orig_start_col=orig_start_col,
        position_offset=position_offset,
        wcs=wcs,
    )
    arr['ra_det'], arr['dec_det'] = _get_radec(
        row=arr['slice_row_det'],
        col=arr['slice_col_det'],
        orig_start_row=orig_start_row,
        orig_start_col=orig_start_col,
        position_offset=position_offset,
        wcs=wcs,
    )

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

    msk = (
        (arr['slice_row'] >= min_slice_row)
        & (arr['slice_row'] < max_slice_row)
        & (arr['slice_col'] >= min_slice_col)
        & (arr['slice_col'] < max_slice_col)
    )
    arr = arr[msk]
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

    trow = row + orig_start_row + position_offset
    tcol = col + orig_start_col + position_offset
    ra, dec = wcs.image2sky(x=tcol, y=trow)
    return ra, dec


def _post_process_results(
    *, outputs, obj_data, image_info, buffer_size, central_size, config
):
    # post process results
    wcs_cache = {}

    output = []
    dt = 0
    for res, i, _dt in outputs:
        if res is None:
            continue

        dt += _dt
        for mdet_step, data in res.items():
            if data.size > 0:
                file_id = obj_data['file_id'][i, 0]
                if file_id in wcs_cache:
                    wcs, position_offset = wcs_cache[file_id]
                else:
                    wcs = eu.wcsutil.WCS(json.loads(image_info['wcs'][file_id]))
                    position_offset = image_info['position_offset'][file_id]
                    wcs_cache[file_id] = (wcs, position_offset)

                # compute the dims of the full image so we can keep buffers on the edge
                slize_size = central_size + 2*buffer_size
                max_slice_row = (
                    int(np.max(obj_data['orig_start_row'][i, 0]))
                    + slize_size
                )
                max_slice_col = (
                    int(np.max(obj_data['orig_start_col'][i, 0]))
                    + slize_size
                )
                coadd_dims = (
                    max_slice_row - int(np.min(obj_data['orig_start_row'][i, 0])),
                    max_slice_col - int(np.min(obj_data['orig_start_col'][i, 0])),
                )
                if coadd_dims != (10000, 10000):
                    LOGGER.critical(
                        "Computed coadd dims of %s which is not quite right for DES!",
                        coadd_dims,
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
                ))

    # concatenate once since generally more efficient
    output = np.concatenate(output)

    return output, dt


def _preprocess_for_metadetect(preconfig, mbobs, gaia_stars, i, rng):
    LOGGER.debug("preprocessing entry %d", i)

    if gaia_stars is not None:
        mask_gaia_stars(mbobs, gaia_stars, preconfig['gaia_star_masks'])

    if preconfig is None:
        return mbobs
    else:
        # TODO do something here?
        return mbobs


def _do_metadetect(config, mbobs, gaia_stars, seed, i, preconfig, shear_bands):
    LOGGER.debug("running mdet for entry %d", i)
    _t0 = time.time()
    res = None
    if mbobs is not None:
        minnum = min([len(olist) for olist in mbobs])
        if minnum > 0:
            rng = np.random.RandomState(seed=seed)
            mbobs = _preprocess_for_metadetect(preconfig, mbobs, gaia_stars, i, rng)

            shear_mbobs = ngmix.MultiBandObsList()
            nonshear_mbobs = ngmix.MultiBandObsList()
            for obslist, is_shear_band in zip(mbobs, shear_bands):
                if is_shear_band:
                    shear_mbobs.append(obslist)
                else:
                    nonshear_mbobs.append(obslist)

            if len(nonshear_mbobs) == 0:
                nonshear_mbobs = None

            res = do_metadetect(config, shear_mbobs, rng, nonshear_mbobs=nonshear_mbobs)
    return res, i, time.time() - _t0


def get_part_ranges(part, n_parts, size):
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
    else:
        gaia_stars = None
    return gaia_stars


def run_metadetect(
    *,
    config,
    multiband_meds,
    output_file,
    seed,
    preconfig,
    start=0,
    num=None,
    n_jobs=1,
    shear_bands=None,
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
    n_jobs : int
        The number of jobs to use.
    shear_bands : list of bool or None
        If not None, this is a list of boolean values indicating if a given
        band is to be used for shear. The length must match the number of MEDS
        files used to make the `multiband_meds`.
    """
    t0 = time.time()

    # process each slice in a pipeline
    if num is None:
        num = multiband_meds.size

    if num + start > multiband_meds.size:
        num = multiband_meds.size - start

    if shear_bands is None:
        shear_bands = [True] * len(multiband_meds.mlist)

    if not any(shear_bands):
        raise RuntimeError(
            "You must have at least one band marked to be "
            "used for shear in `shear_bands`!"
        )

    print('# of slices: %d' % num, flush=True)
    print('slice range: [%d, %d)' % (start, start+num), flush=True)
    meds_iter = _make_meds_iterator(multiband_meds, start, num)

    gaia_stars = _load_gaia_stars(mbmeds=multiband_meds, preconfig=preconfig)

    if n_jobs == 1:
        outputs = [
            _do_metadetect(
                config, mbobs, gaia_stars, seed+i*256, i, preconfig, shear_bands)
            for i, mbobs in PBar(meds_iter(), total=num)]
    else:
        outputs = joblib.Parallel(
            verbose=100,
            n_jobs=n_jobs,
            pre_dispatch='2*n_jobs',
        )(
            joblib.delayed(_do_metadetect)(
                config, mbobs, gaia_stars, seed+i*256, i, preconfig, shear_bands,
            )
            for i, mbobs in meds_iter()
        )

    # join all the outputs
    meta = multiband_meds.mlist[0].get_meta()
    pz_config = yaml.safe_load(meta['config'][0])
    output, cpu_time = _post_process_results(
        outputs=outputs,
        obj_data=multiband_meds.mlist[0].get_cat(),
        image_info=multiband_meds.mlist[0].get_image_info(),
        buffer_size=int(pz_config['coadd']['buffer_size']),
        central_size=int(pz_config['coadd']['central_size']),
        config=config,
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

    fitsio.write(output_file, output, clobber=True)
