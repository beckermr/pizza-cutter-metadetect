import os
import json
import time
import datetime
import logging

import yaml
import joblib
import numpy as np
import esutil as eu
import fitsio

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
    data, slice_id, mcal_step,
    orig_start_row, orig_start_col, position_offset, wcs, buffer_size,
    central_size,
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
    mcal_step: str
        e.g. 'noshear', '1p', '1m', '2p', '2m'
    orig_start_row: float
        Start row of origin of slice
    orig_start_col: float
        Start col of origin of slice
    position_offset: int
        Often 1 for wcs
    wcs: world coordinate system object
        wcs for converting image positions to sky positions
    buffer_size: float
        The size of the buffer region to cut for each slice.
    central_size: float
        The size of the central region.

    Returns
    -------
    array with new fields
    """
    add_dt = [
        ('slice_id', 'i8'),
        ('mcal_step', 'S7'),
        ('ra', 'f8'),
        ('dec', 'f8'),
        ('ra_noshear', 'f8'),
        ('dec_noshear', 'f8'),
    ]
    arr = eu.numpy_util.add_fields(data, add_dt)
    arr['slice_id'] = slice_id
    arr['mcal_step'] = mcal_step

    arr['ra'], arr['dec'] = _get_radec(
        row=arr['sx_row'],
        col=arr['sx_col'],
        orig_start_row=orig_start_row,
        orig_start_col=orig_start_col,
        position_offset=position_offset,
        wcs=wcs,
    )
    arr['ra_noshear'], arr['dec_noshear'] = _get_radec(
        row=arr['sx_row_noshear'],
        col=arr['sx_col_noshear'],
        orig_start_row=orig_start_row,
        orig_start_col=orig_start_col,
        position_offset=position_offset,
        wcs=wcs,
    )

    min_sx = buffer_size
    max_sx = central_size + buffer_size
    msk = (
        (arr['sx_row_noshear'] >= min_sx)
        & (arr['sx_row_noshear'] < max_sx)
        & (arr['sx_col_noshear'] >= min_sx)
        & (arr['sx_col_noshear'] < max_sx)
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
        array of rows
    col: array
        array of columns
    orig_start_row: float
        Start row of origin of slice
    orig_start_col: float
        Start col of origin of slice
    position_offset: int
        Often 1 for wcs
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
    *, outputs, obj_data, image_info, buffer_size, central_size
):
    # post process results
    wcs_cache = {}

    output = []
    dt = 0
    for res, i, _dt in outputs:
        if res is None:
            continue

        dt += _dt
        for mcal_step, data in res.items():
            if data.size > 0:
                file_id = obj_data['file_id'][i, 0]
                if file_id in wcs_cache:
                    wcs, position_offset = wcs_cache[file_id]
                else:
                    wcs = eu.wcsutil.WCS(json.loads(image_info['wcs'][file_id]))
                    position_offset = image_info['position_offset'][file_id]
                    wcs_cache[file_id] = (wcs, position_offset)

                output.append(_make_output_array(
                    data=data,
                    slice_id=obj_data['id'][i],
                    mcal_step=mcal_step,
                    orig_start_col=obj_data['orig_start_col'][i, 0],
                    orig_start_row=obj_data['orig_start_row'][i, 0],
                    wcs=wcs,
                    position_offset=position_offset,
                    buffer_size=buffer_size,
                    central_size=central_size,
                ))

    # concatenate once since generally more efficient
    output = np.concatenate(output)

    return output, dt


def _preprocess_for_metadetect(preconfig, mbobs, gaia_stars, i, rng):
    LOGGER.debug("preprocessing multiband obslist %d", i)

    if gaia_stars is not None:
        mask_gaia_stars(mbobs, gaia_stars, preconfig['gaia_star_masks'])

    if preconfig is None:
        return mbobs
    else:
        # TODO do something here?
        return mbobs


def _do_metadetect(config, mbobs, gaia_stars, seed, i, preconfig):
    _t0 = time.time()
    res = None
    if mbobs is not None:
        minnum = min([len(olist) for olist in mbobs])
        if minnum > 0:
            rng = np.random.RandomState(seed=seed)
            mbobs = _preprocess_for_metadetect(preconfig, mbobs, gaia_stars, i, rng)
            res = do_metadetect(config, mbobs, rng)
    return res, i, time.time() - _t0


def _get_part_ranges(part, n_parts, size):
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
                config, mbobs, gaia_stars, seed+i*256, i, preconfig)
            for i, mbobs in PBar(meds_iter(), total=num)]
    else:
        outputs = joblib.Parallel(
            verbose=100,
            n_jobs=n_jobs,
            pre_dispatch='2*n_jobs',
            max_nbytes=None
        )(
            joblib.delayed(_do_metadetect)(
                config, mbobs, gaia_stars, seed+i*256, i, preconfig
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
        buffer_size=pz_config['coadd']['buffer_size'],
        central_size=pz_config['coadd']['central_size'],
    )

    # report and do i/o
    wall_time = time.time() - t0
    print(
        "run time: ",
        str(datetime.timedelta(seconds=int(wall_time))),
        flush=True)
    print(
        "CPU seconds per slice: ",
        cpu_time / len(outputs), flush=True)

    fitsio.write(output_file, output, clobber=True)
