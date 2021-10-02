import numpy as np
import ngmix

import pytest

from pizza_cutter.des_pizza_cutter import (
    BMASK_GAIA_STAR, BMASK_SPLINE_INTERP,
    BMASK_EXPAND_GAIA_STAR,
)
from ..gaia_stars import mask_gaia_stars


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
    config = dict(
        symmetrize=False,
        interp={"fill_isolated_with_noise": False, "iso_buff": 1},
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

    mask_gaia_stars(mbobs, gaia_stars, config, np.random.RandomState(seed=seed+1))

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
    config = dict(
        symmetrize=False,
        interp={"fill_isolated_with_noise": False, "iso_buff": 1},
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

    mask_gaia_stars(mbobs, gaia_stars, config, np.random.RandomState(seed=seed+1))

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

    mask_gaia_stars(mbobs, gaia_stars, config, np.random.RandomState(seed=seed+1))

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

    mask_gaia_stars(mbobs, gaia_stars, config, np.random.RandomState(seed=seed+1))

    rng = np.random.RandomState(seed=seed)
    for obslist in mbobs:
        for obs in obslist:
            assert np.all(obs.mfrac == 1)
            assert np.all(obs.weight == 0)
            assert np.all((obs.bmask & BMASK_GAIA_STAR) != 0)
            assert np.all(obs.image == 0)
            assert np.all(obs.noise == 0)
