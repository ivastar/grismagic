"""
Integration tests for aXe .conf files using GrismTrace.
Tests both a row grism (GR150R) and a column grism (GR150C).
"""

import os
import tarfile
import urllib.request

import numpy as np
import pytest

from grismagic.traces import GrismTrace


REF_X, REF_Y = 1024.0, 1024.0

_TARBALL_URL = "https://zenodo.org/api/records/7628094/files/niriss_config_221215.tar.gz/content"
_CACHE_DIR = os.path.expanduser("~/.cache/grismagic/niriss_config_221215")
_AXE_FILES = [
    "GR150R.F200W.221215.conf",
    "GR150C.F200W.221215.conf",
]


def _ensure_cache():
    """Download and extract the config tarball into the cache dir if needed."""
    if os.path.isdir(_CACHE_DIR):
        return
    os.makedirs(_CACHE_DIR, exist_ok=True)
    tarball = _CACHE_DIR + ".tar.gz"
    try:
        urllib.request.urlretrieve(_TARBALL_URL, tarball)
        with tarfile.open(tarball) as tf:
            tf.extractall(_CACHE_DIR)
    except Exception as exc:
        import shutil
        shutil.rmtree(_CACHE_DIR, ignore_errors=True)
        if os.path.exists(tarball):
            os.remove(tarball)
        raise exc
    finally:
        if os.path.exists(tarball):
            os.remove(tarball)


@pytest.fixture(scope="module", params=_AXE_FILES, ids=lambda f: f.split(".")[0])
def conf_file(request):
    try:
        _ensure_cache()
    except Exception as exc:
        pytest.skip(f"Could not fetch config tarball: {exc}")
    path = os.path.join(_CACHE_DIR, request.param)
    if not os.path.exists(path):
        pytest.skip(f"Config file not found after extraction: {request.param}")
    return path


@pytest.fixture(scope="module")
def grism_trace(conf_file):
    return GrismTrace.from_axe(conf_file)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def test_loads(grism_trace):
    assert grism_trace is not None


def test_has_orders(grism_trace):
    assert len(grism_trace.orders) > 0


def test_dx_range(grism_trace):
    order = grism_trace.orders[0]
    lo, hi = grism_trace.dx_range(order)
    assert hi > lo


def test_get_trace_shape(grism_trace):
    order = grism_trace.orders[0]
    lo, hi = grism_trace.dx_range(order)
    dx = np.linspace(lo, hi, 50)
    x_tr, y_tr, lam = grism_trace.get_trace(REF_X, REF_Y, order, dx)

    assert x_tr.shape == (50,)
    assert y_tr.shape == (50,)
    assert lam.shape == (50,)


def test_get_trace_wavelength_monotonic(grism_trace):
    order = grism_trace.orders[0]
    lo, hi = grism_trace.dx_range(order)
    dx = np.linspace(lo, hi, 200)
    _, _, lam = grism_trace.get_trace(REF_X, REF_Y, order, dx)
    dlam = np.diff(lam)
    assert np.all(dlam > 0) or np.all(dlam < 0)


def test_get_trace_x_exact(grism_trace):
    """x_trace = x + dx exactly for aXe (no inversion involved)."""
    order = grism_trace.orders[0]
    lo, hi = grism_trace.dx_range(order)
    dx = np.linspace(lo, hi, 50)
    x_tr, _, _ = grism_trace.get_trace(REF_X, REF_Y, order, dx)
    np.testing.assert_allclose(x_tr, REF_X + dx, rtol=1e-10)


def test_get_trace_at_wavelength(grism_trace):
    order = grism_trace.orders[0]
    lo, hi = grism_trace.dx_range(order)
    dx = np.linspace(lo, hi, 200)
    _, _, lam_full = grism_trace.get_trace(REF_X, REF_Y, order, dx)

    lam_test = np.linspace(lam_full.min(), lam_full.max(), 10)
    x_tr, y_tr = grism_trace.get_trace_at_wavelength(REF_X, REF_Y, order, lam_test)

    assert x_tr.shape == (10,)
    assert y_tr.shape == (10,)


def test_get_trace_at_wavelength_roundtrip(grism_trace):
    """Wavelengths recovered from get_trace_at_wavelength should match the input."""
    order = grism_trace.orders[0]
    lo, hi = grism_trace.dx_range(order)
    dx = np.linspace(lo, hi, 200)
    _, _, lam_full = grism_trace.get_trace(REF_X, REF_Y, order, dx)

    lam_test = np.linspace(lam_full.min(), lam_full.max(), 20)
    x_rt, y_rt = grism_trace.get_trace_at_wavelength(REF_X, REF_Y, order, lam_test)

    # Re-evaluate the trace at the returned x positions (via dx = x_rt - REF_X)
    _, _, lam_rt = grism_trace.get_trace(REF_X, REF_Y, order, x_rt - REF_X)
    np.testing.assert_allclose(lam_rt, lam_test, rtol=1e-3)
