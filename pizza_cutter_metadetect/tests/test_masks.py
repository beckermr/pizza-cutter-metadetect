import numpy as np
import pytest

from pizza_cutter.slice_utils.locate import build_slice_locations

from ..masks import (
    in_unique_coadd_tile_region,
    get_slice_bounds,
    _mask_one_slice_for_gaia_stars,
    _mask_one_slice_for_missing_data,
    MASK_INTILE,
    MASK_GAIA_STAR,
    MASK_NOSLICE,
    _wrap_ra,
    make_mask,
)


def test_wrap_ra():
    dra = np.array([-350, -170, 0, 350, 350 + 360*10, -350 - 360*10])
    ans = np.array([10, 360-170, 0, 350, 350, 10])
    assert np.allclose(_wrap_ra(dra), ans)


def test_wrap_dra_array_nan_inf():
    dra = np.array([np.nan, np.inf, -350, -170, 0, 350, 350 + 360*10, -350 - 360*10])
    ans = np.array([np.nan, np.inf, 10, 360-170, 0, 350, 350, 10])
    msk = np.isfinite(dra)
    assert np.allclose(_wrap_ra(dra[msk]), ans[msk])
    assert np.isnan(ans[0])
    assert np.isinf(ans[1])


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


def test_mask_one_gaia_stars(show=False):
    buffer_size = 5
    central_size = 10
    coadd_dims = (100, 100)

    gaia_stars = np.array(
        [(20, 10, 5)],
        dtype=[('x', 'f4'), ('y', 'f4'), ('radius_pixels', 'f4')],
    )

    msk_img = np.zeros(coadd_dims, dtype=np.int32)
    _mask_one_slice_for_gaia_stars(
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
    _mask_one_slice_for_gaia_stars(
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
    flags = 2**9

    msk_img = np.zeros(coadd_dims, dtype=np.int32)

    _mask_one_slice_for_missing_data(
        buffer_size=buffer_size,
        central_size=central_size,
        coadd_dims=coadd_dims,
        msk_img=msk_img,
        scol=15,
        srow=0,
        flags=flags,
    )

    for f in [MASK_NOSLICE, flags]:
        assert np.all((msk_img[0:15, 20:30] & f) != 0)
        assert np.all((msk_img[15:, 30:] & f) == 0)

    _mask_one_slice_for_missing_data(
        buffer_size=buffer_size,
        central_size=central_size,
        coadd_dims=coadd_dims,
        msk_img=msk_img,
        scol=15,
        srow=15,
        flags=flags,
    )

    for f in [MASK_NOSLICE, flags]:
        assert np.all((msk_img[20:30, 20:30] & f) != 0)
        assert np.all((msk_img[30:, 30:] & f) == 0)


@pytest.mark.parametrize("msk_exp_rad", [0, 99])
def test_make_mask(coadd_image_data, msk_exp_rad):
    preconfig = {
        "gaia_star_masks": {"symmetrize": False, "mask_expand_rad": msk_exp_rad},
    }
    missing_slice_inds = [100, 768]
    missing_slice_flags = [2**9, 2**11]
    central_size = 100
    buffer_size = 50
    wcs = coadd_image_data["eu_wcs"]
    position_offset = coadd_image_data["position_offset"]
    coadd_dims = (10000, 10000)
    info = coadd_image_data
    gaia_stars = np.array(
        [(275, 275, 100-msk_exp_rad)],
        dtype=[("x", "f8"), ("y", "f8"), ("radius_pixels", "f8")],
    )
    _, _, srow, scol = build_slice_locations(
        central_size=central_size,
        buffer_size=buffer_size,
        image_width=coadd_dims[0],
    )
    obj_data = np.zeros(
        srow.shape[0],
        dtype=[('orig_start_row', 'i4', (2,)), ('orig_start_col', 'i4', (2,))]
    )
    obj_data["orig_start_row"][:, 0] = srow
    obj_data["orig_start_col"][:, 0] = scol

    msk_img, hs_msk = make_mask(
        preconfig=preconfig,
        missing_slice_inds=missing_slice_inds,
        missing_slice_flags=missing_slice_flags,
        obj_data=obj_data,
        central_size=central_size,
        buffer_size=buffer_size,
        wcs=wcs,
        position_offset=position_offset,
        coadd_dims=coadd_dims,
        info=info,
        gaia_stars=gaia_stars,
        healpix_nside=131072,
    )

    if False:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots()
        axs.imshow(msk_img[:500, :500])
        import pdb
        pdb.set_trace()

    hs_vals, hs_ra, hs_dec = hs_msk.valid_pixels_pos(return_pixels=True)
    hs_vals = hs_msk[hs_vals]

    # basic tests
    # we have some bits set
    assert np.any((msk_img & MASK_INTILE) != 0)
    assert np.any((msk_img & MASK_NOSLICE) != 0)
    assert np.any((msk_img & MASK_GAIA_STAR) != 0)
    assert np.any((hs_vals & MASK_INTILE) != 0)
    assert np.any((hs_vals & MASK_NOSLICE) != 0)
    assert np.any((hs_vals & MASK_GAIA_STAR) != 0)
    assert np.any((msk_img & 2**9) != 0)
    assert np.any((msk_img & 2**11) != 0)

    # edges are all zero
    assert np.all(msk_img[:, 0] == 0)
    assert np.all(msk_img[:, -1] == 0)
    assert np.all(msk_img[0, :] == 0)
    assert np.all(msk_img[-1, :] == 0)

    for x, y in [
        (0, np.arange(coadd_dims[0])),
        (coadd_dims[1]-1, np.arange(coadd_dims[0])),
        (np.arange(coadd_dims[1]), 0),
        (np.arange(coadd_dims[1]), coadd_dims[0]-1),
    ]:
        ra, dec = wcs.image2sky(x+position_offset, y+position_offset)
        _vals = hs_msk.get_values_pos(ra, dec)
        assert np.all(_vals == 0)

    # most of the coadd is fine
    assert np.mean((msk_img & MASK_INTILE) != 0) > 0.80
    assert np.mean((hs_vals & MASK_INTILE) != 0) > 0.80

    # slice ind at 1, 1 is fully masked except for edges
    assert np.all((msk_img[220:250, 220:250] & MASK_NOSLICE) != 0)
    assert np.all((msk_img[220:250, 220:250] & 2**9) != 0)
    assert np.all((msk_img[220:250, 220:250] & 2**11) == 0)
    x, y = np.meshgrid(np.arange(220, 250), np.arange(220, 250))
    x = x.ravel()
    y = y.ravel()
    ra, dec = wcs.image2sky(x+position_offset, y+position_offset)
    _vals = hs_msk.get_values_pos(ra, dec)
    assert np.all((_vals & MASK_NOSLICE) != 0)
    assert np.all((_vals & 2**9) != 0)
    assert np.all((_vals & 2**11) == 0)

    # slice ind at 6, 75 is fully masked except for edges
    assert np.all((msk_img[750:850, 7550:7650] & MASK_NOSLICE) != 0)
    assert np.all((msk_img[750:850, 7550:7650] & 2**9) == 0)
    assert np.all((msk_img[750:850, 7550:7650] & 2**11) != 0)
    x, y = np.meshgrid(np.arange(7550, 7650), np.arange(750, 850))
    x = x.ravel()
    y = y.ravel()
    ra, dec = wcs.image2sky(x+position_offset, y+position_offset)
    _vals = hs_msk.get_values_pos(ra, dec)
    assert np.all((_vals & MASK_NOSLICE) != 0)
    assert np.all((_vals & 2**9) == 0)
    assert np.all((_vals & 2**11) != 0)

    # there is a star so let's check that
    assert np.all((msk_img[260:290, 260:290] & MASK_GAIA_STAR) != 0)
    x, y = np.meshgrid(np.arange(260, 290), np.arange(260, 290))
    x = x.ravel()
    y = y.ravel()
    ra, dec = wcs.image2sky(x+position_offset, y+position_offset)
    _vals = hs_msk.get_values_pos(ra, dec)
    assert np.all((_vals & MASK_GAIA_STAR) != 0)

    # make sure the buffer is ok
    assert np.all(msk_img[0:20, 0:20] == 0)
    x, y = np.meshgrid(np.arange(20), np.arange(20))
    x = x.ravel()
    y = y.ravel()
    ra, dec = wcs.image2sky(x+position_offset, y+position_offset)
    _vals = hs_msk.get_values_pos(ra, dec)
    assert np.all(_vals == 0)
