"""Tests for prince_cr.version module."""


def test_version_exists():
    from prince_cr.version import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0
