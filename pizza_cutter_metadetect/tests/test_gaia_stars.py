import numpy as np
import ngmix

import pytest

from pizza_cutter.des_pizza_cutter import BMASK_GAIA_STAR, BMASK_SPLINE_INTERP
from ..gaia_stars import (
    mask_gaia_stars,
    intersects,
    do_mask_gaia_stars,
    make_gaia_mask,
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


def test_do_mask_gaia_stars_all_masked():
    flag = 2**5
    bmask = np.zeros((10, 13), dtype=np.int32)
    rows = np.array([0, 3])
    cols = rows = np.array([0, 5])
    radius_pixels = np.array([100, 10])
    bmask[4, 5] |= 2**3

    do_mask_gaia_stars(rows, cols, radius_pixels, bmask, flag)

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

    do_mask_gaia_stars(rows, cols, radius_pixels, bmask, flag)

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
        gaia_stars,
        dims,
        start_row,
        start_col,
        False,
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
        gaia_stars,
        dims,
        start_row,
        start_col,
        False,
    )

    bmask_sym = make_gaia_mask(
        gaia_stars,
        dims,
        start_row,
        start_col,
        True,
    )

    msk = bmask > 0
    assert np.all((bmask[msk] & bmask_sym[msk]) != 0)

    rot_bmask = np.rot90(bmask)
    msk = rot_bmask > 0
    assert np.all((rot_bmask[msk] & bmask_sym[msk]) != 0)

    assert np.all((bmask_sym[0:2, 6:8] & BMASK_GAIA_STAR) != 0)


def test_mask_gaia_stars():
    nband = 2
    seed = 10
    start_row = 1012
    start_col = 4513
    gaia_stars = np.array(
        [(6+start_col, 0+start_row, 5)],
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


def test_mask_gaia_stars_all():
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
