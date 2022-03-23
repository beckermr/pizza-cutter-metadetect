import time
import joblib

import ngmix
import galsim
import pytest
import numpy as np
import healpy as hp
from esutil.wcsutil import WCS

from numpy.testing import assert_array_equal

from ..run_metadetect import (
    _make_output_array,
    _do_metadetect,
    _truncate_negative_mfrac_weight,
)
from ..masks import (
    MASK_SLICEDUPE, MASK_GAIA_STAR,
    MASK_MISSING_BAND, MASK_NOSLICE,
)
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
    'nodet_flags': 2**0,
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
    neg_mfrac=False,
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
            mfrac=rng.normal(size=im.shape, scale=0.1) * (
                1 if neg_mfrac else 0
            ),
        )
        obslist = ngmix.ObsList()
        obslist.append(obs)
        mbobs.append(obslist)

    return mbobs


def _shear_cuts(arr, sb):
    msk = (
        (arr['flags'] == 0)
        & (arr['wmom_s2n'] > 10)
        & (arr['wmom_T_ratio'] > 1.2)
        & (arr['shear_bands'] == sb)
    )
    return msk


def _meas_shear_data(res, sb):
    msk = _shear_cuts(res['noshear'], sb)
    g1 = np.mean(res['noshear']['wmom_g'][msk, 0])
    g2 = np.mean(res['noshear']['wmom_g'][msk, 1])

    msk = _shear_cuts(res['1p'], sb)
    g1_1p = np.mean(res['1p']['wmom_g'][msk, 0])
    msk = _shear_cuts(res['1m'], sb)
    g1_1m = np.mean(res['1m']['wmom_g'][msk, 0])
    r11 = (g1_1p - g1_1m) / 0.02

    msk = _shear_cuts(res['2p'], sb)
    g2_2p = np.mean(res['2p']['wmom_g'][msk, 1])
    msk = _shear_cuts(res['2m'], sb)
    g2_2m = np.mean(res['2m']['wmom_g'][msk, 1])
    r22 = (g2_2p - g2_2m) / 0.02

    dt = [
        ('g1', 'f8'),
        ('g2', 'f8'),
        ('R11', 'f8'),
        ('R22', 'f8')]
    return np.array([(g1, g2, r11, r22)], dtype=dt)


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


def run_sim(seed, mdet_seed):
    gaia_stars = None
    seed = 10
    i = 10
    preconfig = None

    mbobs = make_sim(seed=seed, nbands=3, g1=0.02, g2=0.00, ngrid=7, snr=1e6)
    _pres = _do_metadetect(
        CONFIG, mbobs, gaia_stars, mdet_seed, i, preconfig, None,
    )
    if _pres is None:
        return None

    mbobs = make_sim(seed=seed, nbands=3, g1=-0.02, g2=0.00, ngrid=7, snr=1e6)
    _mres = _do_metadetect(
        CONFIG, mbobs, gaia_stars, mdet_seed, i, preconfig, None,
    )
    if _mres is None:
        return None

    # do some of the tests here due to joblib running
    # checking one is enough
    assert _mres[0]['noshear']['wmom_band_flux'][0].shape == (3,)

    return _meas_shear_data(_pres[0], "012"), _meas_shear_data(_mres[0], "012")


def test_do_metadetect():
    ntrial = 10
    rng = np.random.RandomState(seed=116)
    seeds = rng.randint(low=1, high=2**29, size=ntrial)
    mdet_seeds = rng.randint(low=1, high=2**29, size=ntrial)

    tm0 = time.time()

    print("\n")

    jobs = [
        joblib.delayed(run_sim)(
            seeds[i], mdet_seeds[i],
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
    assert np.abs(c) < max(1e-6, 3*cerr)


def test_do_metadetect_pos_mfrac():
    gaia_stars = None
    seed = 10
    i = 10
    preconfig = None
    mdet_seed = 12

    mbobs = make_sim(
        seed=seed, nbands=3, g1=0.02, g2=0.00, ngrid=7, snr=1e6, neg_mfrac=True
    )
    res = _do_metadetect(
        CONFIG, mbobs, gaia_stars, mdet_seed, i, preconfig, None,
    )

    for key in ['noshear', '1p', '1m', '2p', '2m']:
        assert np.all(res[0][key]["mfrac"] >= 0)


def test_do_metadetect_flagging():
    gaia_stars = None
    seed = 10
    i = 10
    preconfig = None
    mdet_seed = 12

    res = _do_metadetect(
        CONFIG, None, gaia_stars, mdet_seed, i, preconfig, None,
    )
    assert (res[3] & MASK_NOSLICE) != 0

    mbobs = make_sim(
        seed=seed, nbands=3, g1=0.02, g2=0.00, ngrid=7, snr=1e6,
    )
    del mbobs[0][0]
    res = _do_metadetect(
        CONFIG, mbobs, gaia_stars, mdet_seed, i, preconfig, None,
    )
    assert (res[3] & MASK_MISSING_BAND) != 0

    mbobs = make_sim(
        seed=seed, nbands=3, g1=0.02, g2=0.00, ngrid=7, snr=1e6,
    )
    del mbobs[-1][0]
    res = _do_metadetect(
        CONFIG, mbobs, gaia_stars, mdet_seed, i, preconfig, None,
    )
    assert (res[3] & MASK_MISSING_BAND) != 0


@pytest.mark.parametrize("band_names,nbands", [
    (None, 3),
    (["f", "j", "p"], 3),
    (None, 1),
    (["f"], 1),
])
def test_make_output_array(band_names, nbands):
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
    output_file = "/bdsjd/tname_blah.fits.fz"

    dtype = [
        ('flags', 'i4'),
        ('wmomm_flags', 'i4'),
        ('wmomm_T_flags', 'i4'),
        ('wmomm_s2n', 'f8'),
        ('wmomm_T', 'f8'),
        ('wmomm_T_err', 'f8'),
        ('wmomm_T_ratio', 'f8'),
        ('shear_bands', 'U6'),
        ('psf_flags', 'i4'),
        ('psf_T', 'f8'),
        ('psfrec_flags', 'i4'),
        ('psfrec_T', 'f8'),
        ('ormask', 'i4'),
        ('bmask', 'i4'),
        ('mfrac', 'f8'),
        ('ormask_noshear', 'i4'),
        ('mfrac_noshear', 'f8'),
        ('sx_row', 'f8'),
        ('sx_col', 'f8'),
        ('sx_row_noshear', 'f8'),
        ('sx_col_noshear', 'f8'),
        ('a', 'i8'),
        ('wmom_blah', 'f8'),
        ('wmomm_blah', 'f8'),
        ('bmask_noshear', 'i4'),
        ("wmomm_g", "f8", (2,)),
        ("wmomm_g_cov", "f8", (2, 2)),
        ("wmomm_band_flux_flags", "i4"),
        ("psf_g", "f8", (2,)),
        ("psfrec_g", "f8", (2,)),
    ]
    if nbands > 1:
        dtype += [
            ("wmomm_band_flux", "f8", (nbands,)),
            ("wmomm_band_flux_err", "f8", (nbands,)),
        ]
    else:
        dtype += [
            ("wmomm_band_flux", "f8"),
            ("wmomm_band_flux_err", "f8"),
        ]

    data = np.zeros(10, dtype=dtype)

    data['sx_row'] = np.arange(10) + 7
    data['sx_col'] = np.arange(10) + 12
    data['sx_row_noshear'] = np.arange(10) + 8
    data['sx_col_noshear'] = np.arange(10) + 13
    data['a'] = np.arange(10)
    data['wmomm_blah'] = np.arange(10) + 23.5
    data['wmom_blah'] = np.arange(10) + 314234.5
    data['bmask_noshear'] = [BMASK_GAIA_STAR, BMASK_EXPAND_GAIA_STAR] + [0]*8
    data['wmomm_g'] = np.arange(10*2).reshape((10, 2)) + 17
    data['wmomm_g_cov'] = np.arange(10*4).reshape((10, 2, 2)) + 23
    if nbands > 1:
        bflux = np.arange(10*nbands).reshape((10, nbands)) + 37
        bfluxerr = np.arange(10*nbands).reshape((10, nbands)) + 47
        for i in range(nbands):
            data["wmomm_band_flux"][:, i] = bflux[:, i]
            data["wmomm_band_flux_err"][:, i] = bfluxerr[:, i]
    else:
        data["wmomm_band_flux"] = np.arange(10) + 37
        data["wmomm_band_flux_err"] = np.arange(10) + 47
    data["wmomm_band_flux_flags"] = np.arange(10) + 53
    data['psf_g'] = np.arange(10*2).reshape((10, 2)) + 177
    data['psfrec_g'] = np.arange(10*2).reshape((10, 2)) + 1777

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
        output_file=output_file,
        band_names=band_names,
        nepoch_per_band=[b + 10 for b in range(nbands)],
        nepoch_eff_per_band=[b + 20 for b in range(nbands)],
    )

    assert all(len(df) == 2 for df in arr.dtype.descr)

    assert np.all(arr['slice_id'] == slice_id)
    assert np.all(arr['mdet_step'] == mdet_step)
    assert np.all(arr['tilename'] == "tname")
    assert np.all(arr['filename'] == "tname_blah.fits")

    assert np.all(arr['psf_g_1'] == data['psf_g'][:, 0])
    assert np.all(arr['psf_g_2'] == data['psf_g'][:, 1])
    assert np.all(arr['psfrec_g_1'] == data['psfrec_g'][:, 0])
    assert np.all(arr['psfrec_g_2'] == data['psfrec_g'][:, 1])

    assert np.all(arr['mdet_g_1'] == data['wmomm_g'][:, 0])
    assert np.all(arr['mdet_g_2'] == data['wmomm_g'][:, 1])
    assert np.all(arr['mdet_g_cov_1_1'] == data['wmomm_g_cov'][:, 0, 0])
    assert np.all(arr['mdet_g_cov_1_2'] == data['wmomm_g_cov'][:, 0, 1])
    assert np.all(arr['mdet_g_cov_2_2'] == data['wmomm_g_cov'][:, 1, 1])

    if band_names is None:
        band_names = ["b%d" % i for i in range(nbands)]

    for i, b in enumerate(band_names):
        assert np.all(
            arr["nepoch_%s" % b]
            == np.array([b + 10 for b in range(nbands)])[i:i+1]
        )
        assert np.all(
            arr["nepoch_eff_%s" % b]
            == np.array([b + 20 for b in range(nbands)])[i:i+1]
        )

        for tail in ["flux", "flux_err"]:
            if nbands > 1:
                assert np.all(
                    arr["mdet_%s_%s" % (b, tail)]
                    == data["wmomm_band_%s" % tail][:, i]
                )
            else:
                assert np.all(
                    arr["mdet_%s_%s" % (b, tail)]
                    == data["wmomm_band_%s" % tail][:]
                )
    assert np.all(arr["mdet_flux_flags"] == data["wmomm_band_flux_flags"])

    # the bounds of the slice are [5,15) for both row and col
    # thus only first two elements of sx_col_noshear pass since they are
    # from np.arange(10) + 13 = [13, 14, 15, ...]
    assert np.array_equal(
        arr["mask_flags_noshear"],
        [MASK_GAIA_STAR] * 2 + [MASK_SLICEDUPE] * 8,
    )

    ura, udec = wcs.image2sky(
        x=data['sx_col_noshear'] + orig_start_col + position_offset,
        y=data['sx_row_noshear'] + orig_start_row + position_offset,
    )
    ra, dec = wcs.image2sky(
        x=data['sx_col'] + orig_start_col + position_offset,
        y=data['sx_row'] + orig_start_row + position_offset,
    )
    assert np.all(arr['ra'] == ra)
    assert np.all(arr['dec'] == dec)
    assert np.all(arr['hpix_16384'] == hp.ang2pix(
            16384, ra, dec, nest=True, lonlat=True
        )
    )
    assert np.all(arr['ra_noshear'] == ura)
    assert np.all(arr['dec_noshear'] == udec)
    assert np.all(arr['hpix_16384_noshear'] == hp.ang2pix(
            16384, ura, udec, nest=True, lonlat=True
        )
    )

    assert np.all(arr['slice_y'] == data['sx_row'])
    assert np.all(arr['slice_x'] == data['sx_col'])
    assert np.all(arr['slice_y_noshear'] == data['sx_row_noshear'])
    assert np.all(arr['slice_x_noshear'] == data['sx_col_noshear'])

    assert 'a' not in arr.dtype.names
    assert 'wmom_blah' not in arr.dtype.names
    assert 'mdet_blah' not in arr.dtype.names
    assert 'sx_row_noshear' not in arr.dtype.names
    assert 'sx_col_noshear' not in arr.dtype.names
    assert 'sx_row' not in arr.dtype.names
    assert 'sx_col' not in arr.dtype.names
    assert 'wmomm_blah' not in arr.dtype.names
    assert "wmomm_g" not in arr.dtype.names
    assert "wmomm_g_cov" not in arr.dtype.names
    assert "psf_g" not in arr.dtype.names
    assert "psfrec_g" not in arr.dtype.names
    assert "mdet_g" not in arr.dtype.names
    assert "mdet_g_cov" not in arr.dtype.names
    assert "wmomm_band_flux" not in arr.dtype.names
    assert "wmomm_band_flux_err" not in arr.dtype.names
    assert "wmomm_band_flux_flags" not in arr.dtype.names
    assert "mdet_band_flux" not in arr.dtype.names
    assert "mdet_band_flux_err" not in arr.dtype.names
    assert "mdet_band_flux_flags" not in arr.dtype.names

    print(arr.dtype.names)


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
        ('flags', 'i4'),
        ('wmomm_flags', 'i4'),
        ('wmomm_T_flags', 'i4'),
        ('wmomm_s2n', 'f8'),
        ('wmomm_T', 'f8'),
        ('wmomm_T_err', 'f8'),
        ('wmomm_T_ratio', 'f8'),
        ('shear_bands', 'U6'),
        ('psf_flags', 'i4'),
        ('psf_T', 'f8'),
        ('psfrec_flags', 'i4'),
        ('psfrec_T', 'f8'),
        ('ormask', 'i4'),
        ('bmask_noshear', 'i4'),
        ('mfrac', 'f8'),
        ('ormask_noshear', 'i4'),
        ('mfrac_noshear', 'f8'),
        ("wmomm_g", "f8", (2,)),
        ("wmomm_g_cov", "f8", (2, 2)),
        ("psf_g", "f8", (2,)),
        ("psfrec_g", "f8", (2,)),
        ("wmomm_band_flux_flags", "i4"),
        ('sx_row', 'f8'),
        ('sx_col', 'f8'),
        ('sx_row_noshear', 'f8'),
        ('sx_col_noshear', 'f8'),
        ('a', 'i8'),
        ('wmom_blah', 'f8'),
        ('wmomm_blah', 'f8'),
        ('bmask', 'i4'),
        ("wmomm_band_flux", "f8"),
        ("wmomm_band_flux_err", "f8"),
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
        band_names=None,
        nepoch_per_band=[10],
        nepoch_eff_per_band=[20],
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
    assert np.array_equal(arr["mask_flags_noshear"], flags)

    ura, udec = wcs.image2sky(
        x=data['sx_col_noshear'] + orig_start_col + position_offset,
        y=data['sx_row_noshear'] + orig_start_row + position_offset,
    )
    ra, dec = wcs.image2sky(
        x=data['sx_col'] + orig_start_col + position_offset,
        y=data['sx_row'] + orig_start_row + position_offset,
    )
    assert np.all(arr['ra'] == ra)
    assert np.all(arr['dec'] == dec)
    assert np.all(arr['ra_noshear'] == ura)
    assert np.all(arr['dec_noshear'] == udec)

    assert np.all(arr['slice_y'] == data['sx_row'])
    assert np.all(arr['slice_x'] == data['sx_col'])
    assert np.all(arr['slice_y_noshear'] == data['sx_row_noshear'])
    assert np.all(arr['slice_x_noshear'] == data['sx_col_noshear'])

    assert 'sx_row_noshear' not in arr.dtype.names
    assert 'sx_col_noshear' not in arr.dtype.names
    assert 'sx_row' not in arr.dtype.names
    assert 'sx_col' not in arr.dtype.names
    assert 'wmomm_blah' not in arr.dtype.names


@pytest.mark.parametrize("band_names,nbands", [
    (["f", "j", "p"], 3),
    (["f", "j", "p"], 3),
    (None, 1),
    (["f"], 1),
])
def test_make_output_array_with_sim(band_names, nbands):
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
    output_file = "/bdsjd/tname_blah.fits.fz"

    gaia_stars = None
    seed = 10
    preconfig = None
    mdet_step = "noshear"

    mbobs = make_sim(seed=seed, nbands=nbands, g1=0.02, g2=0.00, ngrid=7, snr=1e6)
    res = _do_metadetect(
        CONFIG, mbobs, gaia_stars, seed, slice_id, preconfig, None,
    )
    data = res[0][mdet_step]

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
        model='wmom',
        info={
            'crossra0': 'Y',
            'udecmin': -90,
            'udecmax': 90,
            'uramin': 180,
            'uramax': 180,
        },
        output_file=output_file,
        band_names=band_names,
        nepoch_per_band=[b + 10 for b in range(nbands)],
        nepoch_eff_per_band=[b + 20 for b in range(nbands)],
    )

    assert all(len(df) == 2 for df in arr.dtype.descr)

    assert np.all(arr['slice_id'] == slice_id)
    assert np.all(arr['mdet_step'] == mdet_step)
    assert np.all(arr['tilename'] == "tname")
    assert np.all(arr['filename'] == "tname_blah.fits")

    assert_array_equal(arr['psf_g_1'], data['psf_g'][:, 0])
    assert_array_equal(arr['psf_g_2'], data['psf_g'][:, 1])
    assert_array_equal(arr['psfrec_g_1'], data['psfrec_g'][:, 0])
    assert_array_equal(arr['psfrec_g_2'], data['psfrec_g'][:, 1])

    assert_array_equal(arr['mdet_g_1'], data['wmom_g'][:, 0])
    assert_array_equal(arr['mdet_g_2'], data['wmom_g'][:, 1])
    assert_array_equal(arr['mdet_g_cov_1_1'], data['wmom_g_cov'][:, 0, 0])
    assert_array_equal(arr['mdet_g_cov_1_2'], data['wmom_g_cov'][:, 0, 1])
    assert_array_equal(arr['mdet_g_cov_2_2'], data['wmom_g_cov'][:, 1, 1])

    if nbands == 3:
        for sb in ["012", "12", "0", "1", "2"]:
            assert np.any(arr["shear_bands"] == sb)
    else:
        assert np.all(arr["shear_bands"] == "0")

    if band_names is None:
        band_names = ["b%d" % i for i in range(nbands)]

    for i, b in enumerate(band_names):
        assert np.all(
            arr["nepoch_%s" % b]
            == np.array([b + 10 for b in range(nbands)])[i:i+1]
        )
        assert np.all(
            arr["nepoch_eff_%s" % b]
            == np.array([b + 20 for b in range(nbands)])[i:i+1]
        )

        for tail in ["flux", "flux_err"]:
            if nbands > 1:
                assert_array_equal(
                    arr["mdet_%s_%s" % (b, tail)],
                    data["wmom_band_%s" % tail][:, i]
                )
            else:
                assert_array_equal(
                    arr["mdet_%s_%s" % (b, tail)],
                    data["wmom_band_%s" % tail][:]
                )
    assert_array_equal(arr["mdet_flux_flags"], data["wmom_band_flux_flags"])

    assert np.array_equal(arr["mask_flags"], [MASK_SLICEDUPE] * len(arr))

    ura, udec = wcs.image2sky(
        x=(
            data['sx_col_noshear']
            + orig_start_col
            + position_offset
        ).astype(np.float64),
        y=(
            data['sx_row_noshear']
            + orig_start_row
            + position_offset
        ).astype(np.float64),
    )
    ra, dec = wcs.image2sky(
        x=(
            data['sx_col']
            + orig_start_col
            + position_offset
        ).astype(np.float64),
        y=(
            data['sx_row']
            + orig_start_row
            + position_offset
        ).astype(np.float64),
    )
    assert np.allclose(arr['ra'], ra, atol=1e-8)
    assert np.allclose(arr['dec'], dec, atol=1e-8)
    assert np.allclose(arr['ra_noshear'], ura, atol=1e-8)
    assert np.allclose(arr['dec_noshear'], udec, atol=1e-8)

    assert np.all(arr['slice_y'] == data['sx_row'])
    assert np.all(arr['slice_x'] == data['sx_col'])
    assert np.all(arr['slice_y_noshear'] == data['sx_row_noshear'])
    assert np.all(arr['slice_x_noshear'] == data['sx_col_noshear'])

    for col in [
        "flags",
        "shear_bands",
        "psf_flags",
        "psf_T",
        "ormask",
        "mfrac",
        "bmask",
        "psfrec_flags",
        "psfrec_T",
    ]:
        assert_array_equal(arr[col], data[col])

    for col in [
        "mdet_flags",
        "mdet_s2n",
        "mdet_T",
        "mdet_T_err",
        "mdet_T_ratio",
        "mdet_T_flags",
    ]:
        data_col = "wmom_" + col[len("mdet_"):]
        assert_array_equal(arr[col], data[data_col])

    assert 'a' not in arr.dtype.names
    assert 'wmom_blah' not in arr.dtype.names
    assert 'mdet_blah' not in arr.dtype.names
    assert 'sx_row_noshear' not in arr.dtype.names
    assert 'sx_col_noshear' not in arr.dtype.names
    assert 'sx_row' not in arr.dtype.names
    assert 'sx_col' not in arr.dtype.names
    assert 'wmom_blah' not in arr.dtype.names
    assert "wmom_g" not in arr.dtype.names
    assert "wmom_g_cov" not in arr.dtype.names
    assert "psf_g" not in arr.dtype.names
    assert "psfrec_g" not in arr.dtype.names
    assert "mdet_g" not in arr.dtype.names
    assert "mdet_g_cov" not in arr.dtype.names
    assert "wmom_band_flux" not in arr.dtype.names
    assert "wmom_band_flux_err" not in arr.dtype.names
    assert "wmom_band_flux_flags" not in arr.dtype.names
    assert "mdet_band_flux" not in arr.dtype.names
    assert "mdet_band_flux_err" not in arr.dtype.names
    assert "mdet_band_flux_flags" not in arr.dtype.names

    print(arr.dtype.names)


@pytest.mark.parametrize("wmul", [
    [1, 0, 0, 1],
    [0, 0, 0, 0],
    [0, 1, 1, 0],
])
@pytest.mark.parametrize("mmul", [
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [1, 0, 0, 1],
])
def test_truncate_negative_mfrac_weight(wmul, mmul):
    rng = np.random.RandomState(seed=10)
    mbobs = ngmix.MultiBandObsList()
    for i in range(3):
        obslist = ngmix.ObsList()
        for j in range(4):
            obs = ngmix.Observation(
                image=rng.normal(size=(10, 10)),
                weight=rng.normal(size=(10, 10)) * wmul[j],
                ignore_zero_weight=False,
                mfrac=rng.normal(size=(10, 10)) * mmul[j],
            )
            obslist.append(obs)
        mbobs.append(obslist)

    _truncate_negative_mfrac_weight(mbobs)

    rng = np.random.RandomState(seed=10)
    for obslist in mbobs:
        for j, obs in enumerate(obslist):
            if wmul[j] > 0:
                assert np.all(obs.weight >= 0)
                assert np.any(obs.weight > 0)
            else:
                assert np.all(obs.weight == 0)

            if mmul[j] > 0:
                assert np.all(obs.mfrac >= 0)
                assert np.any(obs.mfrac > 0)
            else:
                assert np.all(obs.mfrac == 0)
