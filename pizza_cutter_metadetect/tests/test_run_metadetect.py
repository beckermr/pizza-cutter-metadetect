import time
import joblib

import ngmix
import galsim
import pytest
import numpy as np
from esutil.wcsutil import WCS

from ..run_metadetect import _make_output_array, _do_metadetect
from ..masks import MASK_SLICEDUPE, MASK_GAIA_STAR
from ..gaia_stars import BMASK_GAIA_STAR, BMASK_EXPAND_GAIA_STAR


CONFIG = {
    "model": "wmom",

    'weight': {
        'fwhm': 1.2,  # arcsec
    },

    'metacal': {
        'psf': 'fitgauss',
        'types': ['noshear', '1p', '1m', '2p', '2m'],
    },

    'sx': {
        # in sky sigma
        # DETECT_THRESH
        'detect_thresh': 0.8,

        # Minimum contrast parameter for deblending
        # DEBLEND_MINCONT
        'deblend_cont': 0.00001,

        # minimum number of pixels above threshold
        # DETECT_MINAREA: 6
        'minarea': 4,

        'filter_type': 'conv',

        # 7x7 convolution mask of a gaussian PSF with FWHM = 3.0 pixels.
        'filter_kernel': [
            [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963],  # noqa
            [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388],  # noqa
            [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328],  # noqa
            [0.068707, 0.296069, 0.710525, 0.951108, 0.710525, 0.296069, 0.068707],  # noqa
            [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328],  # noqa
            [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388],  # noqa
            [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963],  # noqa
        ]
    },

    'meds': {
        'min_box_size': 32,
        'max_box_size': 32,

        'box_type': 'iso_radius',

        'rad_min': 4,
        'rad_fac': 2,
        'box_padding': 2,
    },

    # needed for PSF symmetrization
    'psf': {
        'model': 'gauss',

        'ntry': 2,

        'lm_pars': {
            'maxfev': 2000,
            'ftol': 1.0e-5,
            'xtol': 1.0e-5,
        }
    },

    # check for an edge hit
    'bmask_flags': 2**30,

    'maskflags': 2**0,
}


def make_sim(
    *,
    seed,
    nbands,
    g1,
    g2,
    dim=251,
    buff=34,
    scale=0.25,
    dens=100,
    ngrid=7,
    snr=1e6,
):
    rng = np.random.RandomState(seed=seed)

    half_loc = (dim-buff*2)*scale/2

    if ngrid is None:
        area_arcmin2 = ((dim - buff*2)*scale/60)**2
        nobj = int(dens * area_arcmin2)
        x = rng.uniform(low=-half_loc, high=half_loc, size=nobj)
        y = rng.uniform(low=-half_loc, high=half_loc, size=nobj)
    else:
        half_ngrid = (ngrid-1)/2
        x, y = np.meshgrid(np.arange(ngrid), np.arange(ngrid))
        x = (x.ravel() - half_ngrid)/half_ngrid * half_loc
        y = (y.ravel() - half_ngrid)/half_ngrid * half_loc
        nobj = x.shape[0]

    cen = (dim-1)/2
    psf_dim = 53
    psf_cen = (psf_dim-1)/2

    psf = galsim.Gaussian(fwhm=0.9)
    gals = []
    for ind in range(nobj):
        u, v = rng.uniform(low=-scale, high=scale, size=2)
        u += x[ind]
        v += y[ind]
        gals.append(galsim.Exponential(half_light_radius=0.5).shift(u, v))
    gals = galsim.Add(gals)
    gals = gals.shear(g1=g1, g2=g2)
    gals = galsim.Convolve([gals, psf])

    im = gals.drawImage(nx=dim, ny=dim, scale=scale).array
    psf_im = psf.drawImage(nx=psf_dim, ny=psf_dim, scale=scale).array

    nse = (
        np.sqrt(np.sum(
            galsim.Convolve([
                psf,
                galsim.Exponential(half_light_radius=0.5),
            ]).drawImage(scale=0.25).array**2)
        )
        / snr
    )

    mbobs = ngmix.MultiBandObsList()
    for band in range(nbands):
        im += rng.normal(size=im.shape, scale=nse)
        wgt = np.ones_like(im) / nse**2
        jac = ngmix.DiagonalJacobian(scale=scale, row=cen, col=cen)
        psf_jac = ngmix.DiagonalJacobian(scale=scale, row=psf_cen, col=psf_cen)

        obs = ngmix.Observation(
            image=im,
            weight=wgt,
            jacobian=jac,
            ormask=np.zeros_like(im, dtype=np.int32),
            bmask=np.zeros_like(im, dtype=np.int32),
            psf=ngmix.Observation(
                image=psf_im,
                jacobian=psf_jac,
            ),
        )
        obslist = ngmix.ObsList()
        obslist.append(obs)
        mbobs.append(obslist)

    return mbobs


def _shear_cuts(arr):
    msk = (
        (arr['flags'] == 0)
        & (arr['wmom_s2n'] > 10)
        & (arr['wmom_T_ratio'] > 1.2)
    )
    return msk


def _meas_shear_data(res):
    msk = _shear_cuts(res['noshear'])
    g1 = np.mean(res['noshear']['wmom_g'][msk, 0])
    g2 = np.mean(res['noshear']['wmom_g'][msk, 1])

    msk = _shear_cuts(res['1p'])
    g1_1p = np.mean(res['1p']['wmom_g'][msk, 0])
    msk = _shear_cuts(res['1m'])
    g1_1m = np.mean(res['1m']['wmom_g'][msk, 0])
    R11 = (g1_1p - g1_1m) / 0.02

    msk = _shear_cuts(res['2p'])
    g2_2p = np.mean(res['2p']['wmom_g'][msk, 1])
    msk = _shear_cuts(res['2m'])
    g2_2m = np.mean(res['2m']['wmom_g'][msk, 1])
    R22 = (g2_2p - g2_2m) / 0.02

    dt = [
        ('g1', 'f8'),
        ('g2', 'f8'),
        ('R11', 'f8'),
        ('R22', 'f8')]
    return np.array([(g1, g2, R11, R22)], dtype=dt)


def _bootstrap_stat(d1, d2, func, seed, nboot=500):
    dim = d1.shape[0]
    rng = np.random.RandomState(seed=seed)
    stats = []
    for _ in range(nboot):
        ind = rng.choice(dim, size=dim, replace=True)
        stats.append(func(d1[ind], d2[ind]))
    return stats


def meas_m_c_cancel(pres, mres):
    x = np.mean(pres['g1'] - mres['g1'])/2
    y = np.mean(pres['R11'] + mres['R11'])/2
    m = x/y/0.02 - 1

    x = np.mean(pres['g2'] + mres['g2'])/2
    y = np.mean(pres['R22'] + mres['R22'])/2
    c = x/y

    return m, c


def boostrap_m_c(pres, mres):
    m, c = meas_m_c_cancel(pres, mres)
    bdata = _bootstrap_stat(pres, mres, meas_m_c_cancel, 14324, nboot=500)
    merr, cerr = np.std(bdata, axis=0)
    return m, merr, c, cerr


def test_make_output_array():
    wcs = WCS(dict(
        naxis1=100,
        naxis2=100,
        ctype1='RA---TAN',
        ctype2='DEC--TAN',
        crpix1=50.5,
        crpix2=50.5,
        cd1_1=-7.305555555556E-05,
        cd1_2=0.0,
        cd2_1=0.0,
        cd2_2=7.305555555556E-05,
        cunit1='deg     ',
        cunit2='deg     ',
        crval1=321.417528,
        crval2=1.444444))
    position_offset = 2
    orig_start_col = 325
    orig_start_row = 330
    slice_id = 11
    mdet_step = 'blah'

    dtype = [
        ('sx_row', 'f8'),
        ('sx_col', 'f8'),
        ('sx_row_noshear', 'f8'),
        ('sx_col_noshear', 'f8'),
        ('a', 'i8'),
        ('wmom_blah', 'f8'),
        ('wmomm_blah', 'f8'),
        ('bmask', 'i4'),
    ]
    data = np.zeros(10, dtype=dtype)

    data['sx_row'] = np.arange(10) + 7
    data['sx_col'] = np.arange(10) + 12
    data['sx_row_noshear'] = np.arange(10) + 8
    data['sx_col_noshear'] = np.arange(10) + 13
    data['a'] = np.arange(10)
    data['wmomm_blah'] = np.arange(10) + 23.5
    data['wmom_blah'] = np.arange(10) + 314234.5
    data['bmask'] = [BMASK_GAIA_STAR, BMASK_EXPAND_GAIA_STAR] + [0]*8

    arr = _make_output_array(
        data=data,
        slice_id=slice_id,
        mdet_step=mdet_step,
        orig_start_row=orig_start_row,
        orig_start_col=orig_start_col,
        position_offset=position_offset,
        wcs=wcs,
        buffer_size=5,
        central_size=10,
        coadd_dims=(10000, 10000),
        model='wmomm',
        info={
            'crossra0': 'Y',
            'udecmin': -90,
            'udecmax': 90,
            'uramin': 180,
            'uramax': 180,
        },
    )

    assert np.all(arr['slice_id'] == slice_id)
    assert np.all(arr['mdet_step'] == mdet_step)

    # the bounds of the slice are [5,15) for both row and col
    # thus only first two elements of sx_col_noshear pass since they are
    # from np.arange(10) + 13 = [13, 14, 15, ...]
    assert np.array_equal(
        arr["mask_flags"],
        [MASK_GAIA_STAR] * 2 + [MASK_SLICEDUPE] * 8,
    )

    assert np.array_equal(arr['a'], data['a'])
    assert np.array_equal(arr['wmom_blah'], data['wmom_blah'])
    assert np.array_equal(arr['mdet_blah'], data['wmomm_blah'])

    ra, dec = wcs.image2sky(
        x=data['sx_col_noshear'] + orig_start_col + position_offset,
        y=data['sx_row_noshear'] + orig_start_row + position_offset,
    )
    ura, udec = wcs.image2sky(
        x=data['sx_col'] + orig_start_col + position_offset,
        y=data['sx_row'] + orig_start_row + position_offset,
    )
    assert np.all(arr['ra'] == ra)
    assert np.all(arr['dec'] == dec)
    assert np.all(arr['ra_det'] == ura)
    assert np.all(arr['dec_det'] == udec)

    assert np.all(arr['slice_row_det'] == data['sx_row'])
    assert np.all(arr['slice_col_det'] == data['sx_col'])
    assert np.all(arr['slice_row'] == data['sx_row_noshear'])
    assert np.all(arr['slice_col'] == data['sx_col_noshear'])

    assert 'sx_row_noshear' not in arr.dtype.names
    assert 'sx_col_noshear' not in arr.dtype.names
    assert 'sx_row' not in arr.dtype.names
    assert 'sx_col' not in arr.dtype.names
    assert 'wmomm_blah' not in arr.dtype.names


@pytest.mark.parametrize(
    'orig_start_col,orig_start_row,min_col,max_col,min_row,max_row',
    [
        (0, 0, 0, 15, 0, 15),
        (5, 0, 5, 15, 0, 15),
        (0, 5, 0, 15, 5, 15),
        (5, 5, 5, 15, 5, 15),
        (80, 0, 5, 20, 0, 15),
        (0, 80, 0, 15, 5, 20),
        (80, 80, 5, 20, 5, 20),
        (5, 80, 5, 15, 5, 20),
        (80, 5, 5, 20, 5, 15),
    ],
)
def test_make_output_array_bounds(
    orig_start_col, orig_start_row, min_col, max_col, min_row, max_row
):
    coadd_dims = (100, 100)
    central_size = 10
    buffer_size = 5

    wcs = WCS(dict(
        naxis1=100,
        naxis2=100,
        ctype1='RA---TAN',
        ctype2='DEC--TAN',
        crpix1=50.5,
        crpix2=50.5,
        cd1_1=-7.305555555556E-05,
        cd1_2=0.0,
        cd2_1=0.0,
        cd2_2=7.305555555556E-05,
        cunit1='deg     ',
        cunit2='deg     ',
        crval1=321.417528,
        crval2=1.444444))
    position_offset = 2
    slice_id = 11
    mdet_step = 'blah'
    output_file = "/bdsjd/tname_blah.fits.fz"

    dtype = [
        ('sx_row', 'f8'),
        ('sx_col', 'f8'),
        ('sx_row_noshear', 'f8'),
        ('sx_col_noshear', 'f8'),
        ('a', 'i8'),
        ('wmom_blah', 'f8'),
        ('wmomm_blah', 'f8'),
        ('bmask', 'i4'),
    ]
    data = np.zeros(21, dtype=dtype)

    data['sx_row'] = np.arange(21) + 2024
    data['sx_col'] = np.arange(21) + 2132
    data['sx_row_noshear'] = np.arange(21)
    data['sx_col_noshear'] = np.arange(21)
    data['a'] = np.arange(21)
    data['wmomm_blah'] = np.arange(21) + 23.5
    data['wmom_blah'] = np.arange(21) + 314234.5

    arr = _make_output_array(
        data=data,
        slice_id=slice_id,
        mdet_step=mdet_step,
        orig_start_row=orig_start_row,
        orig_start_col=orig_start_col,
        position_offset=position_offset,
        wcs=wcs,
        buffer_size=buffer_size,
        central_size=central_size,
        coadd_dims=coadd_dims,
        model='wmomm',
        info={
            'crossra0': 'Y',
            'udecmin': -90,
            'udecmax': 90,
            'uramin': 180,
            'uramax': 180,
        },
        output_file=output_file,
    )

    assert np.all(arr['slice_id'] == slice_id)
    assert np.all(arr['mdet_step'] == mdet_step)
    assert np.all(arr['tilename'] == "tname")
    assert np.all(arr['filename'] == "tname_blah.fits")

    msk = (
        (data['sx_row_noshear'] >= min_row)
        & (data['sx_row_noshear'] < max_row)
        & (data['sx_col_noshear'] >= min_col)
        & (data['sx_col_noshear'] < max_col)
    )
    flags = np.zeros(21, dtype=np.int32) + MASK_SLICEDUPE
    flags[msk] = 0
    assert np.array_equal(arr["mask_flags"], flags)

    assert np.array_equal(arr['a'], data['a'])
    assert np.array_equal(arr['wmom_blah'], data['wmom_blah'])
    assert np.array_equal(arr['mdet_blah'], data['wmomm_blah'])

    ra, dec = wcs.image2sky(
        x=data['sx_col_noshear'] + orig_start_col + position_offset,
        y=data['sx_row_noshear'] + orig_start_row + position_offset,
    )
    ura, udec = wcs.image2sky(
        x=data['sx_col'] + orig_start_col + position_offset,
        y=data['sx_row'] + orig_start_row + position_offset,
    )
    assert np.all(arr['ra'] == ra)
    assert np.all(arr['dec'] == dec)
    assert np.all(arr['ra_det'] == ura)
    assert np.all(arr['dec_det'] == udec)

    assert np.all(arr['slice_row_det'] == data['sx_row'])
    assert np.all(arr['slice_col_det'] == data['sx_col'])
    assert np.all(arr['slice_row'] == data['sx_row_noshear'])
    assert np.all(arr['slice_col'] == data['sx_col_noshear'])

    assert 'sx_row_noshear' not in arr.dtype.names
    assert 'sx_col_noshear' not in arr.dtype.names
    assert 'sx_row' not in arr.dtype.names
    assert 'sx_col' not in arr.dtype.names
    assert 'wmomm_blah' not in arr.dtype.names


def run_sim(seed, mdet_seed, shear_bands):
    gaia_stars = None
    seed = 10
    i = 10
    preconfig = None

    mbobs = make_sim(seed=seed, nbands=3, g1=0.02, g2=0.00, ngrid=7, snr=1e6)
    _pres = _do_metadetect(
        CONFIG, mbobs, gaia_stars, mdet_seed, i, preconfig, shear_bands
    )
    if _pres is None:
        return None

    mbobs = make_sim(seed=seed, nbands=3, g1=-0.02, g2=0.00, ngrid=7, snr=1e6)
    _mres = _do_metadetect(
        CONFIG, mbobs, gaia_stars, mdet_seed, i, preconfig, shear_bands
    )
    if _mres is None:
        return None

    # do some of the tests here due to joblib running
    # checking one is enough
    assert _mres[0]['noshear']['wmom_band_flux'][0].shape == (3,)

    return _meas_shear_data(_pres[0]), _meas_shear_data(_mres[0])


@pytest.mark.parametrize(
    'shear_bands',
    [
        [True, True, True],
        [True, False, False],
        [False, True, False],
    ]
)
def test_do_metadetect(shear_bands):
    ntrial = 10
    rng = np.random.RandomState(seed=116)
    seeds = rng.randint(low=1, high=2**29, size=ntrial)
    mdet_seeds = rng.randint(low=1, high=2**29, size=ntrial)

    tm0 = time.time()

    print("\n")

    jobs = [
        joblib.delayed(run_sim)(
            seeds[i], mdet_seeds[i], shear_bands,
        )
        for i in range(ntrial)
    ]
    outputs = joblib.Parallel(n_jobs=-1, verbose=100, backend='loky')(jobs)

    pres = []
    mres = []
    for out in outputs:
        if out is None:
            continue
        pres.append(out[0])
        mres.append(out[1])

    total_time = time.time()-tm0
    print("time per:", total_time/ntrial, flush=True)

    pres = np.concatenate(pres)
    mres = np.concatenate(mres)
    m, merr, c, cerr = boostrap_m_c(pres, mres)

    print(
        (
            "\n\nm [1e-3, 3sigma]: %s +/- %s"
            "\nc [1e-5, 3sigma]: %s +/- %s"
        ) % (
            m/1e-3,
            3*merr/1e-3,
            c/1e-5,
            3*cerr/1e-5,
        ),
        flush=True,
    )

    assert np.abs(m) < max(1e-3, 3*merr)
    assert np.abs(c) < 3*cerr


def test_do_metadetect_shear_bands():
    gaia_stars = None
    seed = 10
    i = 10
    preconfig = None
    mdet_seed = 12

    mbobs = make_sim(seed=seed, nbands=3, g1=0.02, g2=0.00, ngrid=7, snr=1e6)
    res_all = _do_metadetect(
        CONFIG, mbobs, gaia_stars, mdet_seed, i, preconfig, [True, True, True],
    )

    mbobs = make_sim(seed=seed, nbands=3, g1=0.02, g2=0.00, ngrid=7, snr=1e6)
    res1 = _do_metadetect(
        CONFIG, mbobs, gaia_stars, mdet_seed, i, preconfig, [True, False, False],
    )

    mbobs = make_sim(seed=seed, nbands=3, g1=0.02, g2=0.00, ngrid=7, snr=1e6)
    res2 = _do_metadetect(
        CONFIG, mbobs, gaia_stars, mdet_seed, i, preconfig, [False, True, True],
    )

    # look at a subset of keys
    for col in ['sx_row', 'sx_col', 'wmom_g', 'wmom_band_flux']:
        for key in ['noshear', '1p', '1m', '2p', '2m']:
            assert not np.array_equal(
                res_all[0][key][col],
                res1[0][key][col],
            )

            assert not np.array_equal(
                res_all[0][key][col],
                res2[0][key][col],
            )

            assert not np.array_equal(
                res1[0][key][col],
                res2[0][key][col],
            )
