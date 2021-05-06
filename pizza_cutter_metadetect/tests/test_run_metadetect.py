import numpy as np

from esutil.wcsutil import WCS

import pytest

from ..run_metadetect import _make_output_array


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
    orig_start_col = 10
    orig_start_row = 20
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
    ]
    data = np.zeros(10, dtype=dtype)

    data['sx_row'] = np.arange(10) + 324
    data['sx_col'] = np.arange(10) + 326
    data['sx_row_noshear'] = np.arange(10) + 325
    data['sx_col_noshear'] = np.arange(10) + 327
    data['a'] = np.arange(10)
    data['wmomm_blah'] = np.arange(10) + 23.5
    data['wmom_blah'] = np.arange(10) + 314234.5

    arr = _make_output_array(
        data=data,
        slice_id=slice_id,
        mdet_step=mdet_step,
        orig_start_row=orig_start_row,
        orig_start_col=orig_start_col,
        position_offset=position_offset,
        wcs=wcs,
        buffer_size=328,
        central_size=5,
        coadd_dims=(10000, 10000),
        model='wmomm',
    )

    assert np.all(arr['slice_id'] == slice_id)
    assert np.all(arr['mdet_step'] == mdet_step)

    msk = (
        (data['sx_row_noshear'] >= 328)
        & (data['sx_row_noshear'] < 333)
        & (data['sx_col_noshear'] >= 328)
        & (data['sx_col_noshear'] < 333)
    )

    assert np.array_equal(arr['a'], data['a'][msk])
    assert np.array_equal(arr['wmom_blah'], data['wmom_blah'][msk])
    assert np.array_equal(arr['mdet_blah'], data['wmomm_blah'][msk])

    ra, dec = wcs.image2sky(
        x=data['sx_col_noshear'] + orig_start_col + position_offset,
        y=data['sx_row_noshear'] + orig_start_row + position_offset,
    )
    ura, udec = wcs.image2sky(
        x=data['sx_col'] + orig_start_col + position_offset,
        y=data['sx_row'] + orig_start_row + position_offset,
    )
    assert np.all(arr['ra'] == ra[msk])
    assert np.all(arr['dec'] == dec[msk])
    assert np.all(arr['ra_det'] == ura[msk])
    assert np.all(arr['dec_det'] == udec[msk])

    assert np.all(arr['slice_row_det'] == data['sx_row'][msk])
    assert np.all(arr['slice_col_det'] == data['sx_col'][msk])
    assert np.all(arr['slice_row'] == data['sx_row_noshear'][msk])
    assert np.all(arr['slice_col'] == data['sx_col_noshear'][msk])

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

    dtype = [
        ('sx_row', 'f8'),
        ('sx_col', 'f8'),
        ('sx_row_noshear', 'f8'),
        ('sx_col_noshear', 'f8'),
        ('a', 'i8'),
        ('wmom_blah', 'f8'),
        ('wmomm_blah', 'f8'),
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
    )

    assert np.all(arr['slice_id'] == slice_id)
    assert np.all(arr['mdet_step'] == mdet_step)

    msk = (
        (data['sx_row_noshear'] >= min_row)
        & (data['sx_row_noshear'] < max_row)
        & (data['sx_col_noshear'] >= min_col)
        & (data['sx_col_noshear'] < max_col)
    )

    assert np.array_equal(arr['a'], data['a'][msk])
    assert np.array_equal(arr['wmom_blah'], data['wmom_blah'][msk])
    assert np.array_equal(arr['mdet_blah'], data['wmomm_blah'][msk])

    ra, dec = wcs.image2sky(
        x=data['sx_col_noshear'] + orig_start_col + position_offset,
        y=data['sx_row_noshear'] + orig_start_row + position_offset,
    )
    ura, udec = wcs.image2sky(
        x=data['sx_col'] + orig_start_col + position_offset,
        y=data['sx_row'] + orig_start_row + position_offset,
    )
    assert np.all(arr['ra'] == ra[msk])
    assert np.all(arr['dec'] == dec[msk])
    assert np.all(arr['ra_det'] == ura[msk])
    assert np.all(arr['dec_det'] == udec[msk])

    assert np.all(arr['slice_row_det'] == data['sx_row'][msk])
    assert np.all(arr['slice_col_det'] == data['sx_col'][msk])
    assert np.all(arr['slice_row'] == data['sx_row_noshear'][msk])
    assert np.all(arr['slice_col'] == data['sx_col_noshear'][msk])

    assert 'sx_row_noshear' not in arr.dtype.names
    assert 'sx_col_noshear' not in arr.dtype.names
    assert 'sx_row' not in arr.dtype.names
    assert 'sx_col' not in arr.dtype.names
    assert 'wmomm_blah' not in arr.dtype.names
