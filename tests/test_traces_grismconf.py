"""
Integration test: download a GRISMCONF .conf file and exercise GrismTrace.
Compares output against the reference `grismconf` package by Pirzkal.
"""

import os
import shutil
import glob

import numpy as np
import pytest

from grismagic.traces import GrismTrace


REF_X, REF_Y = 1024.0, 1024.0
T_GRID = np.linspace(0, 1, 50)

_CONF_NAME = "NIRISS_F200W_GR150C.V5.conf"
_SENS_GLOB = "NIRISS_NIS_GR150C_F200W_*_sens_*.fits"

# Candidate directories to find the conf file (tried in order)
_CONF_SEARCH_DIRS = [
    os.path.join(os.environ.get("GRIZLI", ""), "CONF"),
    os.path.expanduser("~/dev/NGDEEP_NIRISS_CALIB"),
]
# Directory that holds the sensitivity FITS files
_SENS_DIR = os.path.expanduser("~/dev/NGDEEP_NIRISS_CALIB")


@pytest.fixture(scope="module")
def conf_file(tmp_path_factory):
    """
    Return a path to the GRISMCONF .conf file inside a temp directory that
    also contains all required sensitivity FITS files so that
    ``grismconf.Config`` can load them without error.
    """
    # Find the conf file
    src_conf = None
    for d in _CONF_SEARCH_DIRS:
        candidate = os.path.join(d, _CONF_NAME)
        if os.path.exists(candidate):
            src_conf = candidate
            break
    if src_conf is None:
        pytest.skip(f"Config file {_CONF_NAME!r} not found in any of {_CONF_SEARCH_DIRS}")

    # Build a temp dir with conf + sensitivity files co-located
    tmp_dir = str(tmp_path_factory.mktemp("grismconf"))
    shutil.copy2(src_conf, os.path.join(tmp_dir, _CONF_NAME))

    sens_files = glob.glob(os.path.join(_SENS_DIR, _SENS_GLOB))
    if not sens_files:
        pytest.skip(f"No sensitivity files matching {_SENS_GLOB!r} found in {_SENS_DIR}")
    for sf in sens_files:
        shutil.copy2(sf, os.path.join(tmp_dir, os.path.basename(sf)))

    return os.path.join(tmp_dir, _CONF_NAME)


@pytest.fixture(scope="module")
def grism_trace(conf_file):
    return GrismTrace.from_grismconf(conf_file)


@pytest.fixture(scope="module")
def grismconf_ref(conf_file):
    """Load the same file with the reference `grismconf` package."""
    grismconf = pytest.importorskip("grismconf")
    return grismconf.Config(str(conf_file))


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------

def test_loads(grism_trace):
    assert grism_trace is not None


def test_has_orders(grism_trace):
    assert len(grism_trace.orders) > 0


def test_dx_range(grism_trace):
    order = grism_trace.orders[0]
    lo, hi = grism_trace.dx_range(order, x=REF_X, y=REF_Y)
    assert hi > lo


def test_get_trace_shape(grism_trace):
    order = grism_trace.orders[0]
    lo, hi = grism_trace.dx_range(order, x=REF_X, y=REF_Y)
    dx = np.linspace(lo, hi, 50)
    x_tr, y_tr, lam = grism_trace.get_trace(REF_X, REF_Y, order, dx)

    assert x_tr.shape == (50,)
    assert y_tr.shape == (50,)
    assert lam.shape == (50,)


def test_get_trace_wavelength_monotonic(grism_trace):
    order = grism_trace.orders[0]
    lo, hi = grism_trace.dx_range(order, x=REF_X, y=REF_Y)
    dx = np.linspace(lo, hi, 200)
    _, _, lam = grism_trace.get_trace(REF_X, REF_Y, order, dx)
    dlam = np.diff(lam)
    assert np.all(dlam > 0) or np.all(dlam < 0)


def test_get_trace_at_wavelength(grism_trace):
    order = grism_trace.orders[0]
    lo, hi = grism_trace.dx_range(order, x=REF_X, y=REF_Y)
    dx = np.linspace(lo, hi, 200)
    _, _, lam_full = grism_trace.get_trace(REF_X, REF_Y, order, dx)

    lam_test = np.linspace(lam_full.min(), lam_full.max(), 10)
    x_tr, y_tr = grism_trace.get_trace_at_wavelength(REF_X, REF_Y, order, lam_test)

    assert x_tr.shape == (10,)
    assert y_tr.shape == (10,)


# ---------------------------------------------------------------------------
# Comparison against the reference `grismconf` package
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("quantity", ["DISPX", "DISPY", "DISPL"])
def test_polynomial_matches_grismconf(grism_trace, grismconf_ref, quantity):
    """DISPX/DISPY/DISPL at the same t should match the grismconf package exactly."""
    order = grism_trace.orders[0]

    ref_fn  = getattr(grismconf_ref, quantity)
    our_fn  = getattr(grism_trace.reader, quantity)

    ref  = np.array([ref_fn(order, REF_X, REF_Y, t) for t in T_GRID])
    ours = our_fn(order, REF_X, REF_Y, T_GRID)

    np.testing.assert_allclose(ours, ref, rtol=1e-6)


def test_get_trace_matches_grismconf(grism_trace, grismconf_ref):
    """
    get_trace output should match the grismconf package to within INVDISP
    interpolation error.

    Drive the trace along its primary dispersion axis (y for GR150C column grisms,
    x for row grisms).  The driven axis is exact by construction; the other axis
    and wavelength are checked to sub-pixel / sub-percent tolerance.
    """
    order = grism_trace.orders[0]

    ref_dx  = np.array([grismconf_ref.DISPX(order, REF_X, REF_Y, t) for t in T_GRID])
    ref_dy  = np.array([grismconf_ref.DISPY(order, REF_X, REF_Y, t) for t in T_GRID])
    ref_lam = np.array([grismconf_ref.DISPL(order, REF_X, REF_Y, t) for t in T_GRID])

    axis = grism_trace._primary_axis(order, REF_X, REF_Y)
    ref_offset = ref_dy if axis == 'y' else ref_dx

    x_tr, y_tr, lam = grism_trace.get_trace(REF_X, REF_Y, order, ref_offset)

    if axis == 'y':
        np.testing.assert_allclose(y_tr, REF_Y + ref_dy, rtol=1e-10)
        np.testing.assert_allclose(x_tr, REF_X + ref_dx,  atol=0.01)
    else:
        np.testing.assert_allclose(x_tr, REF_X + ref_dx, rtol=1e-10)
        np.testing.assert_allclose(y_tr, REF_Y + ref_dy,  atol=0.01)
    np.testing.assert_allclose(lam, ref_lam, rtol=1e-4)
