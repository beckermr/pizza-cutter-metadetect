import numpy as np
import ngmix

import pytest

from pizza_cutter.des_pizza_cutter import (
    BMASK_GAIA_STAR, BMASK_SPLINE_INTERP,
    BMASK_EXPAND_GAIA_STAR,
)
from ..gaia_stars import (
    mask_gaia_stars,
    intersects,
    do_mask_gaia_stars,
    make_gaia_mask,
    do_apodization_mask,
    make_gaia_apodization_mask,
    _ap_kern_kern,
)


@pytest.mark.parametrize('row,col,radius_pixels,nrows,ncols,yes', [
    # basic
    (0, 0, 10, 10, 10, True),
    (-1000, 0, 10, 10, 10, False),
    (1000, 0, 10, 10, 10, False),
    (0, -1000, 10, 10, 10, False),
    (0, 1000, 10, 10, 10, False),
    # edge cases
    (-10, 0, 10, 10, 10, False),
    (19, 0, 10, 10, 10, False),
    (0, -10, 10, 10, 10, False),
    (0, 19, 10, 10, 10, False),
    # partial
    (2, 5, 10, 7, 7, True),
])
def test_intersects(row, col, radius_pixels, nrows, ncols, yes):
    if yes:
        assert intersects(row, col, radius_pixels, nrows, ncols)
    else:
        assert not intersects(row, col, radius_pixels, nrows, ncols)


@pytest.mark.parametrize("fac,val", [
    (1, 1),
    (0, 1),
    (-3, 0.5),
    (-6, 0),
    (-7, 0),
])
def test_ap_kern_kern(fac, val):
    h = 2.0
    m = 10.0
    assert np.allclose(_ap_kern_kern(m + fac*h, m, h), val)


def test_do_apodization_mask_all_masked():
    ap_mask = np.ones((10, 13))
    rows = np.array([0, 3])
    cols = rows = np.array([0, 5])
    radius_pixels = np.array([100, 10])

    do_apodization_mask(
        rows=rows,
        cols=cols,
        radius_pixels=radius_pixels,
        ap_mask=ap_mask,
        ap_rad=1.5,
    )

    assert np.all(ap_mask == 0)


def test_do_apodization_mask_half_masked():
    ap_mask = np.ones((10, 13))
    rows = np.array([0])
    cols = np.array([6])
    radius_pixels = np.array([5])

    do_apodization_mask(
        rows=rows,
        cols=cols,
        radius_pixels=radius_pixels,
        ap_mask=ap_mask,
        ap_rad=0.1,
    )

    assert np.all(ap_mask[6:, :] == 1)
    assert not np.all(ap_mask[:6, :] == 1)
    assert np.all(ap_mask[0:2, 6:8] == 0)

    if False:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.pcolormesh(ap_mask)
        import pdb
        pdb.set_trace()


def test_make_gaia_apodization_mask():
    start_row = 1012
    start_col = 4513
    gaia_stars = np.array(
        [(6+start_col, 0+start_row, 5)],
        dtype=[('x', 'f8'), ('y', '<f4'), ('radius_pixels', '>f4')]
    )
    dims = (10, 13)

    ap_mask = make_gaia_apodization_mask(
        gaia_stars=gaia_stars,
        dims=dims,
        start_row=start_row,
        start_col=start_col,
        symmetrize=False,
        ap_rad=0.1
    )

    assert np.all(ap_mask[6:, :] == 1)
    assert not np.all(ap_mask[:6, :] == 1)
    assert np.all(ap_mask[0:2, 6:8] == 0)


def test_make_gaia_apodization_symmetrize():
    start_row = 1012
    start_col = 4513
    gaia_stars = np.array(
        [(6+start_col, 0+start_row, 5)],
        dtype=[('x', 'f8'), ('y', '<f4'), ('radius_pixels', '>f4')]
    )
    dims = (10, 10)

    ap_mask = make_gaia_apodization_mask(
        gaia_stars=gaia_stars,
        dims=dims,
        start_row=start_row,
        start_col=start_col,
        symmetrize=False,
        ap_rad=0.1
    )

    ap_mask_sym = make_gaia_apodization_mask(
        gaia_stars=gaia_stars,
        dims=dims,
        start_row=start_row,
        start_col=start_col,
        symmetrize=True,
        ap_rad=0.1
    )

    msk = ap_mask == 0
    assert np.allclose(ap_mask[msk], ap_mask_sym[msk])

    msk = ap_mask_sym == 1
    assert np.allclose(ap_mask[msk], ap_mask_sym[msk])

    rot_ap_mask = np.rot90(ap_mask)
    msk = rot_ap_mask == 0
    assert np.allclose(ap_mask_sym[msk], rot_ap_mask[msk])

    assert np.all(ap_mask[0:2, 6:8] == 0)

    if False:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.pcolormesh(ap_mask_sym)
        import pdb
        pdb.set_trace()


def test_do_mask_gaia_stars_all_masked():
    flag = 2**5
    bmask = np.zeros((10, 13), dtype=np.int32)
    rows = np.array([0, 3])
    cols = rows = np.array([0, 5])
    radius_pixels = np.array([100, 10])
    bmask[4, 5] |= 2**3

    do_mask_gaia_stars(
        rows=rows,
        cols=cols,
        radius_pixels=radius_pixels,
        bmask=bmask,
        flag=flag,
    )

    assert np.all((bmask & flag) != 0)
    assert (bmask[4, 5] & 2**3) != 0
    assert (bmask[1, 2] & 2**3) == 0


def test_do_mask_gaia_stars_half_masked():
    flag = 2**5
    bmask = np.zeros((10, 13), dtype=np.int32)
    rows = np.array([0])
    cols = np.array([6])
    radius_pixels = np.array([5])
    bmask[4, 5] |= 2**3

    do_mask_gaia_stars(
        rows=rows,
        cols=cols,
        radius_pixels=radius_pixels,
        bmask=bmask,
        flag=flag,
    )

    assert np.all((bmask[6:, 6:] & flag) == 0)
    assert np.all((bmask[0:2, 6:8] & flag) != 0)
    assert (bmask[4, 5] & 2**3) != 0
    assert (bmask[1, 2] & 2**3) == 0


def test_make_gaia_mask():
    start_row = 1012
    start_col = 4513
    gaia_stars = np.array(
        [(6+start_col, 0+start_row, 5)],
        dtype=[('x', 'f8'), ('y', '<f4'), ('radius_pixels', '>f4')]
    )
    dims = (10, 13)

    bmask = make_gaia_mask(
        gaia_stars=gaia_stars,
        dims=dims,
        start_row=start_row,
        start_col=start_col,
        symmetrize=False,
    )

    assert np.all((bmask[6:, 6:] & BMASK_GAIA_STAR) == 0)
    assert np.all((bmask[0:2, 6:8] & BMASK_GAIA_STAR) != 0)


def test_make_gaia_mask_symmetrize():
    start_row = 1012
    start_col = 4513
    gaia_stars = np.array(
        [(6+start_col, 0+start_row, 5)],
        dtype=[('x', 'f8'), ('y', '<f4'), ('radius_pixels', '>f4')]
    )
    dims = (10, 10)

    bmask = make_gaia_mask(
        gaia_stars=gaia_stars,
        dims=dims,
        start_row=start_row,
        start_col=start_col,
        symmetrize=False,
    )

    bmask_sym = make_gaia_mask(
        gaia_stars=gaia_stars,
        dims=dims,
        start_row=start_row,
        start_col=start_col,
        symmetrize=True,
    )

    msk = bmask > 0
    assert np.all((bmask[msk] & bmask_sym[msk]) != 0)

    rot_bmask = np.rot90(bmask)
    msk = rot_bmask > 0
    assert np.all((rot_bmask[msk] & bmask_sym[msk]) != 0)

    assert np.all((bmask_sym[0:2, 6:8] & BMASK_GAIA_STAR) != 0)


@pytest.mark.parametrize("msk_exp_rad", [0, 4])
def test_mask_gaia_stars_interp(msk_exp_rad):
    nband = 2
    seed = 10
    start_row = 1012
    start_col = 4513
    gaia_stars = np.array(
        [(6+start_col, 0+start_row, 5-msk_exp_rad)],
        dtype=[('x', 'f8'), ('y', '<f4'), ('radius_pixels', '>f4')]
    )
    dims = (13, 13)
    config = dict(symmetrize=False, interp={}, mask_expand_rad=msk_exp_rad)
    mbobs = ngmix.MultiBandObsList()
    rng = np.random.RandomState(seed=seed)
    for _ in range(nband):
        image = rng.uniform(size=dims)
        noise = rng.uniform(size=dims)
        obs = ngmix.Observation(
            image=image,
            noise=noise,
            weight=rng.uniform(size=dims),
            bmask=np.zeros(dims, dtype=np.int32),
            ormask=np.zeros(dims, dtype=np.int32),
            meta={"orig_start_row": start_row, "orig_start_col": start_col},
        )
        obs.mfrac = rng.uniform(size=dims)
        obslist = ngmix.ObsList()
        obslist.append(obs)
        mbobs.append(obslist)

    mask_gaia_stars(mbobs, gaia_stars, config)

    rng = np.random.RandomState(seed=seed)
    for obslist in mbobs:
        for obs in obslist:
            msk = (obs.bmask & BMASK_GAIA_STAR) != 0
            image = rng.uniform(size=dims)
            noise = rng.uniform(size=dims)
            # we need to match these calls to the ones above
            rng.uniform(size=dims)
            rng.uniform(size=dims)

            assert np.all(obs.mfrac[msk] == 1)
            assert np.all(obs.weight[msk] == 0)
            assert np.all((obs.bmask[msk] & BMASK_SPLINE_INTERP) != 0)

            assert np.all(image[msk] != obs.image[msk])
            assert np.all(image[~msk] == obs.image[~msk])
            assert np.all(noise[msk] != obs.noise[msk])
            assert np.all(noise[~msk] == obs.noise[~msk])

            msk = (obs.bmask & BMASK_EXPAND_GAIA_STAR) != 0
            if msk_exp_rad > 0:
                assert np.sum(msk) > 0

                assert not np.all(obs.mfrac[msk] == 1)
                assert not np.all(obs.weight[msk] == 0)
                assert not np.all((obs.bmask[msk] & BMASK_SPLINE_INTERP) != 0)

                assert not np.all(image[msk] != obs.image[msk])
                assert np.all(image[~msk] == obs.image[~msk])
                assert not np.all(noise[msk] != obs.noise[msk])
                assert np.all(noise[~msk] == obs.noise[~msk])
            else:
                assert np.sum(msk) == 0


def test_mask_gaia_stars_interp_all():
    nband = 2
    seed = 10
    start_row = 1012
    start_col = 4513
    gaia_stars = np.array(
        [(6+start_col, 0+start_row, 100)],
        dtype=[('x', 'f8'), ('y', '<f4'), ('radius_pixels', '>f4')]
    )
    dims = (13, 13)
    config = dict(symmetrize=False, interp={}, mask_expand_rad=0)
    mbobs = ngmix.MultiBandObsList()
    rng = np.random.RandomState(seed=seed)
    for _ in range(nband):
        image = rng.uniform(size=dims)
        noise = rng.uniform(size=dims)
        obs = ngmix.Observation(
            image=image,
            noise=noise,
            weight=rng.uniform(size=dims),
            bmask=np.zeros(dims, dtype=np.int32),
            ormask=np.zeros(dims, dtype=np.int32),
            meta={"orig_start_row": start_row, "orig_start_col": start_col},
        )
        obs.mfrac = rng.uniform(size=dims)
        obslist = ngmix.ObsList()
        obslist.append(obs)
        mbobs.append(obslist)

    mask_gaia_stars(mbobs, gaia_stars, config)

    rng = np.random.RandomState(seed=seed)
    for obslist in mbobs:
        for obs in obslist:
            assert np.all(obs.mfrac == 1)
            assert np.all(obs.weight == 0)
            assert np.all((obs.bmask & BMASK_GAIA_STAR) != 0)


@pytest.mark.parametrize("msk_exp_rad", [0, 4])
def test_mask_gaia_stars_apodize(msk_exp_rad):
    nband = 2
    seed = 10
    start_row = 1012
    start_col = 4513
    gaia_stars = np.array(
        [(6+start_col, 0+start_row, 5-msk_exp_rad)],
        dtype=[('x', 'f8'), ('y', '<f4'), ('radius_pixels', '>f4')]
    )
    dims = (13, 13)
    config = dict(
        symmetrize=False,
        apodize={"ap_rad": 0.5},
        mask_expand_rad=msk_exp_rad,
    )
    mbobs = ngmix.MultiBandObsList()
    rng = np.random.RandomState(seed=seed)
    for _ in range(nband):
        image = rng.uniform(size=dims)
        noise = rng.uniform(size=dims)
        obs = ngmix.Observation(
            image=image,
            noise=noise,
            weight=rng.uniform(size=dims),
            bmask=np.zeros(dims, dtype=np.int32),
            ormask=np.zeros(dims, dtype=np.int32),
            meta={"orig_start_row": start_row, "orig_start_col": start_col},
        )
        obs.mfrac = rng.uniform(size=dims)
        obslist = ngmix.ObsList()
        obslist.append(obs)
        mbobs.append(obslist)

    mask_gaia_stars(mbobs, gaia_stars, config)

    rng = np.random.RandomState(seed=seed)
    for obslist in mbobs:
        for obs in obslist:
            msk = (obs.bmask & BMASK_GAIA_STAR) != 0
            image = rng.uniform(size=dims)
            noise = rng.uniform(size=dims)
            # we need to match these calls to the ones above
            rng.uniform(size=dims)
            rng.uniform(size=dims)

            assert np.all(obs.mfrac[msk] == 1)
            assert np.all(obs.weight[msk] == 0)

            assert np.all(image[msk] != obs.image[msk])
            assert np.all(image[~msk] == obs.image[~msk])
            assert np.all(noise[msk] != obs.noise[msk])
            assert np.all(noise[~msk] == obs.noise[~msk])

            msk = (obs.bmask & BMASK_EXPAND_GAIA_STAR) != 0
            if msk_exp_rad > 0:
                assert np.sum(msk) > 0

                assert not np.all(obs.mfrac[msk] == 1)
                assert not np.all(obs.weight[msk] == 0)

                assert not np.all(image[msk] != obs.image[msk])
                assert np.all(image[~msk] == obs.image[~msk])
                assert not np.all(noise[msk] != obs.noise[msk])
                assert np.all(noise[~msk] == obs.noise[~msk])
            else:
                assert np.sum(msk) == 0


def test_mask_gaia_stars_apodize_all():
    nband = 2
    seed = 10
    start_row = 1012
    start_col = 4513
    gaia_stars = np.array(
        [(6+start_col, 0+start_row, 100)],
        dtype=[('x', 'f8'), ('y', '<f4'), ('radius_pixels', '>f4')]
    )
    dims = (13, 13)
    config = dict(
        symmetrize=False,
        apodize={"ap_rad": 0.5},
        mask_expand_rad=0,
    )
    mbobs = ngmix.MultiBandObsList()
    rng = np.random.RandomState(seed=seed)
    for _ in range(nband):
        image = rng.uniform(size=dims)
        noise = rng.uniform(size=dims)
        obs = ngmix.Observation(
            image=image,
            noise=noise,
            weight=rng.uniform(size=dims),
            bmask=np.zeros(dims, dtype=np.int32),
            ormask=np.zeros(dims, dtype=np.int32),
            meta={"orig_start_row": start_row, "orig_start_col": start_col},
        )
        obs.mfrac = rng.uniform(size=dims)
        obslist = ngmix.ObsList()
        obslist.append(obs)
        mbobs.append(obslist)

    mask_gaia_stars(mbobs, gaia_stars, config)

    rng = np.random.RandomState(seed=seed)
    for obslist in mbobs:
        for obs in obslist:
            assert np.all(obs.mfrac == 1)
            assert np.all(obs.weight == 0)
            assert np.all((obs.bmask & BMASK_GAIA_STAR) != 0)
            assert np.all(obs.image == 0)
