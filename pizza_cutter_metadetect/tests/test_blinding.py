from ..blinding import generate_blinding_factor


def test_generate_blinding_factor():
    # this test ensures the code does not change so when we unblind it works
    assert generate_blinding_factor("blah") == 1.033477298
