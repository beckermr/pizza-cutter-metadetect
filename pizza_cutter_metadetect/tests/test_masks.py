import numpy as np
import pytest

from ..masks import (
    in_unique_coadd_tile_region,
    get_slice_bounds,
    _convert_ra_dec_vals_to_healsparse,
    _mask_one_gaia_stars,
    _mask_one_slice,
    MASK_GAIA_STAR,
    MASK_NOSLICE,
)


def test_in_unique_coadd_tile_region():
    ra = np.array([
        210,
        190,
        210,
        200,
        220,
        210,
        210,
        np.nan,
        np.inf,
        210,
        210,
        210 + 360*4,
        210 - 360*4,
    ])
    dec = np.array([
        0,
        0,
        20,
        0,
        0,
        -10,
        10,
        0,
        0,
        np.nan,
        np.inf,
        0,
        0,
    ])
    truth = np.array([
        True,
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
    ], dtype=bool)
    res = in_unique_coadd_tile_region(
        ra=ra,
        dec=dec,
        crossra0='N',
        udecmin=-10,
        udecmax=10,
        uramin=200,
        uramax=220,
    )
    assert np.array_equal(res, truth)


def test_in_unique_coadd_tile_region_crossra0():
    ra = np.array([
        0,
        30,
        0,
        340,
        20,
        0,
        0,
        np.nan,
        np.inf,
        0,
        0,
        360*4,
        -360*4,
    ])
    dec = np.array([
        0,
        0,
        20,
        0,
        0,
        -10,
        10,
        0,
        0,
        np.nan,
        np.inf,
        0,
        0,
    ])
    truth = np.array([
        True,
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
    ], dtype=bool)
    res = in_unique_coadd_tile_region(
        ra=ra,
        dec=dec,
        crossra0='Y',
        udecmin=-10,
        udecmax=10,
        uramin=360-20,
        uramax=20,
    )
    assert np.array_equal(res, truth)


@pytest.mark.parametrize('col,row,truth', [
    (200, 200, {"min_row": 50, "max_row": 150, "min_col": 50, "max_col": 150}),
    (0, 200, {"min_row": 50, "max_row": 150, "min_col": 0, "max_col": 150}),
    (200, 0, {"min_row": 0, "max_row": 150, "min_col": 50, "max_col": 150}),
    (0, 0, {"min_row": 0, "max_row": 150, "min_col": 0, "max_col": 150}),
    (800, 200, {"min_row": 50, "max_row": 150, "min_col": 50, "max_col": 200}),
    (200, 800, {"min_row": 50, "max_row": 200, "min_col": 50, "max_col": 150}),
    (800, 800, {"min_row": 50, "max_row": 200, "min_col": 50, "max_col": 200}),
    (0, 800, {"min_row": 50, "max_row": 200, "min_col": 0, "max_col": 150}),
    (800, 0, {"min_row": 0, "max_row": 150, "min_col": 50, "max_col": 200}),
])
def test_get_slice_bounds(col, row, truth):
    res = get_slice_bounds(
        orig_start_col=col,
        orig_start_row=row,
        central_size=100,
        buffer_size=50,
        coadd_dims=(1000, 1000),
    )
    assert res == truth


def test_convert_ra_dec_vals_to_healsparse():
    ra = np.array([10, 20, 50, 10])
    dec = np.array([0, 65, 47, 0])
    vals = np.array([2**0, 2**2, 2**3, 2**1], dtype=np.int32)
    healpix_nisde = 4096

    hs_msk = _convert_ra_dec_vals_to_healsparse(ra, dec, vals, healpix_nisde)

    assert np.array_equal(
        hs_msk.get_values_pos(ra, dec),
        np.array([2**0 | 2**1, 2**2, 2**3, 2**0 | 2**1], dtype=np.int32),
    )


def test_mask_one_gaia_stars(show=False):
    buffer_size = 5
    central_size = 10
    coadd_dims = (100, 100)

    gaia_stars = np.array(
        [(20, 10, 5)],
        dtype=[('x', 'f4'), ('y', 'f4'), ('radius_pixels', 'f4')],
    )

    msk_img = np.zeros(coadd_dims, dtype=np.int32)
    _mask_one_gaia_stars(
        buffer_size=buffer_size,
        central_size=central_size,
        gaia_stars=gaia_stars,
        symmetrize=False,
        coadd_dims=coadd_dims,
        msk_img=msk_img,
        scol=15,
        srow=0,
    )

    assert np.any((msk_img & MASK_GAIA_STAR) != 0)
    assert np.all((msk_img[10, 20:24] & MASK_GAIA_STAR) != 0)
    assert np.all((msk_img[10, 16:20] & MASK_GAIA_STAR) == 0)
    assert np.all((msk_img[14, 23:26] & MASK_GAIA_STAR) == 0)

    if show:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(msk_img)
        import pdb
        pdb.set_trace()

    msk_img = np.zeros(coadd_dims, dtype=np.int32)
    _mask_one_gaia_stars(
        buffer_size=buffer_size,
        central_size=central_size,
        gaia_stars=gaia_stars,
        symmetrize=True,
        coadd_dims=coadd_dims,
        msk_img=msk_img,
        scol=15,
        srow=0,
    )

    if show:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(msk_img)
        import pdb
        pdb.set_trace()

    assert np.any((msk_img & MASK_GAIA_STAR) != 0)
    assert np.all((msk_img[10, 20:24] & MASK_GAIA_STAR) != 0)
    assert np.all((msk_img[10, 16:20] & MASK_GAIA_STAR) == 0)
    assert np.all((msk_img[14, 20:26] & MASK_GAIA_STAR) != 0)


def test_mask_one_slice():
    buffer_size = 5
    central_size = 10
    coadd_dims = (100, 100)

    msk_img = np.zeros(coadd_dims, dtype=np.int32)

    _mask_one_slice(
        buffer_size=buffer_size,
        central_size=central_size,
        coadd_dims=coadd_dims,
        msk_img=msk_img,
        scol=15,
        srow=0,
    )

    assert np.all((msk_img[0:15, 20:30] & MASK_NOSLICE) != 0)
    assert np.all((msk_img[15:, 30:] & MASK_NOSLICE) == 0)

    _mask_one_slice(
        buffer_size=buffer_size,
        central_size=central_size,
        coadd_dims=coadd_dims,
        msk_img=msk_img,
        scol=15,
        srow=15,
    )

    assert np.all((msk_img[20:30, 20:30] & MASK_NOSLICE) != 0)
    assert np.all((msk_img[30:, 30:] & MASK_NOSLICE) == 0)
