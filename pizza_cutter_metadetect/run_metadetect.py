import os
import json
import time
import datetime
import logging
import copy
import subprocess

import yaml
import joblib
import numpy as np
import esutil as eu
import fitsio
import ngmix

from esutil.pbar import PBar
from metadetect.metadetect import do_metadetect
from pizza_cutter.files import expandpath
from .gaia_stars import (
    load_gaia_stars, mask_gaia_stars, BMASK_GAIA_STAR, BMASK_EXPAND_GAIA_STAR,
)
from .masks import (
    make_mask, get_slice_bounds, in_unique_coadd_tile_region,
    MASK_TILEDUPE, MASK_SLICEDUPE, MASK_GAIA_STAR
)
from pizza_cutter.des_pizza_cutter import get_coaddtile_geom

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


def _make_output_dtype(*, old_dt, model, filename_len, tilename_len):
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
        ('mask_flags', 'i4'),
        ('filename', 'U%d' % filename_len),
        ('tilename', 'U%d' % tilename_len),
    ]
    skip_cols = ["sx_row", "sx_col", "sx_row_noshear", "sx_col_noshear"]
    mpre = model + '_'
    for fld in old_dt:
        if fld[0] in skip_cols:
            continue
        elif fld[0].startswith(mpre):
            new_fld = copy.deepcopy(list(fld))
            new_fld[0] = "mdet_" + fld[0][len(mpre):]
            new_fld = tuple(new_fld)
        else:
            new_fld = fld
        new_dt += [new_fld]

    return new_dt


def _make_output_array(
    *,
    data, slice_id, mdet_step,
    orig_start_row, orig_start_col, position_offset, wcs, buffer_size,
    central_size, coadd_dims, model, info, output_file,
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

    Returns
    -------
    array with new fields
    """
    filename = os.path.basename(output_file)
    if filename.endswith(".fz"):
        filename = filename[:-len(".fz")]
    tilename = filename.split('_')[0]
    new_dt = _make_output_dtype(
        old_dt=data.dtype.descr,
        model=model,
        filename_len=len(filename),
        tilename_len=len(tilename),
    )
    mpre = model + '_'
    arr = np.zeros(data.shape, dtype=new_dt)
    for name in data.dtype.names:
        if name in arr.dtype.names:
            arr[name] = data[name]
        elif name.startswith(mpre):
            new_name = "mdet_" + name[len(mpre):]
            arr[new_name] = data[name]

    arr['slice_id'] = slice_id
    arr['mdet_step'] = mdet_step
    arr['filename'] = filename
    arr['tilename'] = tilename

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

    slice_bnds = get_slice_bounds(
        orig_start_col=orig_start_col,
        orig_start_row=orig_start_row,
        central_size=central_size,
        buffer_size=buffer_size,
        coadd_dims=coadd_dims
    )

    msk = (
        (arr['slice_row'] >= slice_bnds["min_row"])
        & (arr['slice_row'] < slice_bnds["max_row"])
        & (arr['slice_col'] >= slice_bnds["min_col"])
        & (arr['slice_col'] < slice_bnds["max_col"])
    )
    arr["mask_flags"][~msk] |= MASK_SLICEDUPE

    msk = in_unique_coadd_tile_region(
        ra=arr['ra'],
        dec=arr['dec'],
        crossra0=info['crossra0'],
        udecmin=info['udecmin'],
        udecmax=info['udecmax'],
        uramin=info['uramin'],
        uramax=info['uramax'],
    )
    arr["mask_flags"][~msk] |= MASK_TILEDUPE

    msk = (
        ((arr['bmask'] & BMASK_EXPAND_GAIA_STAR) != 0)
        | ((arr['bmask'] & BMASK_GAIA_STAR) != 0)
    )
    arr["mask_flags"][msk] |= MASK_GAIA_STAR

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
    *, outputs, obj_data, image_info, buffer_size, central_size, config, info,
    output_file,
):
    # post process results
    wcs_cache = {}

    output = []
    dt = 0
    missing_slice_inds = []
    for res, i, _dt in outputs:
        if res is None:
            missing_slice_inds.append(i)

        dt += _dt
        for mdet_step, data in res.items():
            if data.size > 0:
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
                ))

    if len(output) > 0:
        # concatenate once since generally more efficient
        output = np.concatenate(output)
    else:
        output = None

    assert len(wcs_cache) == 1

    return output, dt, missing_slice_inds, wcs, position_offset, coadd_dims


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
            max_nbytes=None,  # never memmap
        )(
            joblib.delayed(_do_metadetect)(
                config, mbobs, gaia_stars, seed+i*256, i, preconfig, shear_bands,
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
        output, cpu_time, missing_slice_inds, wcs, position_offset, coadd_dims
    ) = _post_process_results(
        outputs=outputs,
        obj_data=multiband_meds.mlist[0].get_cat(),
        image_info=multiband_meds.mlist[0].get_image_info(),
        buffer_size=int(pz_config['coadd']['buffer_size']),
        central_size=int(pz_config['coadd']['central_size']),
        config=config,
        info=info,
        output_file=output_file,
    )

    # make the masks
    msk_img, hs_msk = make_mask(
        preconfig=preconfig,
        gaia_stars=gaia_stars,
        missing_slice_inds=missing_slice_inds,
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
