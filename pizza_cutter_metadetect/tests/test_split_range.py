from ..run_metadetect import split_range


def test_split_range():
    rstr = "13:16"
    start, end_plus_one, num = split_range(rstr)

    assert start == 13
    assert end_plus_one == 16
    assert num == 3
