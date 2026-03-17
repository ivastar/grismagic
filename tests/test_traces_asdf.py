"""
Integration test: download a grism config file and exercise GrismTrace.
"""

import numpy as np
import pytest

from grismagic.traces import GrismTrace


# ---------------------------------------------------------------------------
# Reference evaluation helper (mirrors CRDSGrismConf._eval_model in grizli)
# ---------------------------------------------------------------------------

def _eval_dm_model(model, x0, y0, t):
    """
    Evaluate an astropy model list from a jwst.datamodels dispx/dispy/displ entry.

    This mirrors the logic in grizli's CRDSGrismConf._eval_model and our
    CRDSReader._eval:
      - Single model with n_inputs==1 → polynomial in t
      - Single model with n_inputs==2 → Polynomial2D in (x0, y0)
      - List of models → coefficients of a polynomial in t, where each
        coefficient is obtained by evaluating the corresponding model
    """
    if hasattr(model, "n_inputs"):
        return model(t) if model.n_inputs == 1 else model(x0, y0)
    if len(model) == 1:
        m = model[0]
        return m(t) if m.n_inputs == 1 else m(x0, y0)
    coeffs = [m(t) if m.n_inputs == 1 else m(x0, y0) for m in model]
    return np.polynomial.Polynomial(coeffs)(t)


import os

_ASDF_FILES = [
    "jwst_niriss_specwcs_0073.GR150R.F150W.asdf",
    "jwst_niriss_specwcs_0078.GR150C.F150W.asdf",
]

_CONF_SEARCH_DIRS = [
    os.path.join(os.environ.get("GRIZLI", ""), "CONF"),
    os.path.expanduser("~/dev/grizli_conf/CONF"),
]


def _find_asdf(name):
    for d in _CONF_SEARCH_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


@pytest.fixture(scope="module", params=_ASDF_FILES, ids=lambda f: f.split("_")[3])
def conf_file(request):
    path = _find_asdf(request.param)
    if path is None:
        pytest.skip(f"Config file not found: {request.param}")
    return path


@pytest.fixture(scope="module")
def grism_trace(conf_file):
    return GrismTrace.from_file(conf_file)


def test_loads(grism_trace):
    assert grism_trace is not None


def test_has_orders(grism_trace):
    assert len(grism_trace.orders) > 0


REF_X, REF_Y = 1024.0, 1024.0


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
    """Wavelength should be monotonically increasing (or decreasing) along the trace."""
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
# Comparison against jwst.datamodels
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dm_fixture(conf_file):
    """
    Load the specwcs file with jwst.datamodels and return (orders, dispx, dispy, displ).
    Skips the tests if jwst is not installed.
    """
    jwst_dm = pytest.importorskip("jwst.datamodels")
    path = str(conf_file)
    if "nircam" in path.lower():
        dm = jwst_dm.NIRCAMGrismModel(path)
    else:
        dm = jwst_dm.NIRISSGrismModel(path)

    orders = [f"+{o}" if o > 0 else str(o) for o in dm.orders]
    return orders, dm.dispx, dm.dispy, dm.displ


# t grid used for all direct polynomial comparisons (no inversion, tight tolerance)
T_GRID = np.linspace(0, 1, 50)


@pytest.mark.parametrize("quantity", ["dispx", "dispy", "displ"])
def test_polynomial_matches_datamodel(grism_trace, dm_fixture, quantity):
    """DISPX/DISPY/DISPL evaluated at the same t should match jwst.datamodels exactly."""
    dm_orders, dm_dispx, dm_dispy, dm_displ = dm_fixture
    dm_models = {"dispx": dm_dispx, "dispy": dm_dispy, "displ": dm_displ}
    our_methods = {
        "dispx": grism_trace.reader.DISPX,
        "dispy": grism_trace.reader.DISPY,
        "displ": grism_trace.reader.DISPL,
    }

    order = grism_trace.orders[0]
    io = dm_orders.index(order)

    ref = _eval_dm_model(dm_models[quantity][io], REF_X, REF_Y, T_GRID)
    ours = our_methods[quantity](order, REF_X, REF_Y, T_GRID)

    np.testing.assert_allclose(ours, ref, rtol=1e-6)


def test_get_trace_matches_datamodel(grism_trace, dm_fixture):
    """
    get_trace output should match jwst.datamodels to within INVDISP interpolation error.

    Drive the trace along its primary dispersion axis (y for column grisms like GR150C,
    x for row grisms like GR150R).  The driven axis is exact by construction; the other
    axis and wavelength are checked to sub-pixel / sub-percent tolerance.
    """
    dm_orders, dm_dispx, dm_dispy, dm_displ = dm_fixture
    order = grism_trace.orders[0]
    io = dm_orders.index(order)

    ref_dx  = _eval_dm_model(dm_dispx[io], REF_X, REF_Y, T_GRID)
    ref_dy  = _eval_dm_model(dm_dispy[io], REF_X, REF_Y, T_GRID)
    ref_lam = _eval_dm_model(dm_displ[io], REF_X, REF_Y, T_GRID)

    axis = grism_trace._primary_axis(order, REF_X, REF_Y)
    ref_offset = ref_dy if axis == 'y' else ref_dx

    x_tr, y_tr, lam = grism_trace.get_trace(REF_X, REF_Y, order, ref_offset)

    if axis == 'y':
        # y is exact by construction; x goes through INVDISPY interpolation
        np.testing.assert_allclose(y_tr, REF_Y + ref_dy, rtol=1e-10)
        np.testing.assert_allclose(x_tr, REF_X + ref_dx,  atol=0.01)
    else:
        # x is exact by construction; y goes through INVDISPX interpolation
        np.testing.assert_allclose(x_tr, REF_X + ref_dx, rtol=1e-10)
        np.testing.assert_allclose(y_tr, REF_Y + ref_dy,  atol=0.01)
    np.testing.assert_allclose(lam, ref_lam, rtol=1e-4)
