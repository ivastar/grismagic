"""
Grism configuration file readers with no dependency on the external
`grismconf` package or `jwst.datamodels`.
"""

import os
import copy
import numpy as np


# ---------------------------------------------------------------------------
# Polynomial helpers (replaces grismconf/poly.py)
# ---------------------------------------------------------------------------

def _xy_terms(x, y, n_xy):
    """
    Compute 2D polynomial basis terms in the grismconf ordering:
    [1, x, y, x^2, xy, y^2, x^3, x^2*y, ...] up to n_xy terms.
    """
    terms = []
    degree = 0
    while len(terms) < n_xy:
        for dy in range(degree + 1):
            terms.append(x ** (degree - dy) * y ** dy)
        degree += 1
    return np.array(terms[:n_xy])


def _eval_poly(coeffs, x, y, t):
    """
    Evaluate a GRISMCONF dispersion polynomial.

    Parameters
    ----------
    coeffs : np.ndarray, shape (n_t, n_xy)
        coeffs[i] are the xy-polynomial coefficients for the t^i term.
    x, y : float
        Detector position (direct image coordinates).
    t : float or array-like
        Trace parameter.

    Returns
    -------
    float or array-like
    """
    n_t, n_xy = coeffs.shape
    xy = _xy_terms(x, y, n_xy)
    return sum(t ** i * np.dot(coeffs[i], xy) for i in range(n_t))


def _inv_poly(coeffs, x, y, val, t0=np.linspace(0, 1, 128)):
    """
    Invert a dispersion polynomial via interpolation.
    For linear-in-t polynomials (n_t == 2), uses the closed-form solution.
    """
    n_t, n_xy = coeffs.shape
    if n_t == 2:
        xy = _xy_terms(x, y, n_xy)
        c0 = np.dot(coeffs[0], xy)
        c1 = np.dot(coeffs[1], xy)
        return (val - c0) / c1
    forward = _eval_poly(coeffs, x, y, t0)
    so = np.argsort(forward)
    return np.interp(val, forward[so], t0[so])


# ---------------------------------------------------------------------------
# aXe field-dependent polynomial helpers
# ---------------------------------------------------------------------------

def _axe_field_dependent(xi, yi, coeffs):
    """
    Evaluate an aXe field-dependent coefficient polynomial.

    The polynomial order n satisfies n*(n+1)/2 == len(coeffs).
    Terms are ordered: 1, xi, yi, xi^2, xi*yi, yi^2, ...

    Parameters
    ----------
    xi, yi : float or array-like
        Detector coordinates relative to reference pixel (x - REFX, y - REFY).
    coeffs : array-like
        Field-dependent polynomial coefficients.

    Returns
    -------
    float or array-like
    """
    if not hasattr(coeffs, "__len__"):
        return float(coeffs)
    order = int(-1 + np.sqrt(1 + 8 * len(coeffs))) // 2
    xy = []
    for p in range(order):
        for py in range(p + 1):
            xy.append(xi ** (p - py) * yi ** py)
    return np.sum((np.array(xy).T * coeffs).T, axis=0)


def _axe_arc_length(dx, dydx):
    """
    Compute arc length along the trace given trace polynomial coefficients.

    Parameters
    ----------
    dx : array-like
        x pixel offsets from source position.
    dydx : list of float
        Polynomial coefficients [c0, c1, c2, ...] so that
        dy = c0 + c1*dx + c2*dx^2 + ...

    Returns
    -------
    dp : array-like
        Arc length at each dx.
    """
    order = len(dydx) - 1
    if order == 2 and np.abs(dydx[2]) == 0:
        order = 1

    if order == 0:
        return np.asarray(dx, dtype=float)
    elif order == 1:
        return np.sqrt(1 + dydx[1] ** 2) * np.asarray(dx, dtype=float)
    elif order == 2:
        u0 = dydx[1] + 2 * dydx[2] * 0
        dp0 = (u0 * np.sqrt(1 + u0 ** 2) + np.arcsinh(u0)) / (4 * dydx[2])
        u = dydx[1] + 2 * dydx[2] * np.asarray(dx, dtype=float)
        return (u * np.sqrt(1 + u ** 2) + np.arcsinh(u)) / (4 * dydx[2]) - dp0
    else:
        dx = np.asarray(dx, dtype=float)
        xmin = min(dx.min(), 0)
        xmax = max(dx.max(), 0)
        xfull = np.arange(xmin, xmax)
        dyfull = sum(i * dydx[i] * (xfull - 0.5) ** (i - 1) for i in range(1, order + 1))
        dpfull = np.zeros_like(xfull)
        lt0 = xfull < 0
        if lt0.sum() > 1:
            dpfull[lt0] = -np.cumsum(np.sqrt(1 + dyfull[lt0][::-1] ** 2))[::-1]
        gt0 = xfull > 0
        if gt0.sum() > 0:
            dpfull[gt0] = np.cumsum(np.sqrt(1 + dyfull[gt0] ** 2))
        return np.interp(dx, xfull, dpfull)


# ---------------------------------------------------------------------------
# Reader 1: aXe text .conf files
# ---------------------------------------------------------------------------

class aXeConfReader:
    """
    Read an aXe-format grism configuration file.

    Dispersion runs along x. The trace shape and wavelength solution are
    encoded as field-dependent polynomials in detector position.

    Parameters
    ----------
    conf_file : str
        Path to an aXe ``.conf`` file.
    """

    def __init__(self, conf_file):
        self.conf_file = conf_file
        self._conf = self._parse(conf_file)

        self.xoff = float(self._conf.get("XOFF", 0.0))
        self.yoff = float(self._conf.get("YOFF", 0.0))
        self.fwcpos_ref = float(self._conf["FWCPOS_REF"]) if "FWCPOS_REF" in self._conf else None

        self.beams = []
        self.beam_range = {}
        for beam in "ABCDEFGHIJ":
            # Accept both BEAMA and BEAM_A conventions
            key = f"BEAM{beam}" if f"BEAM{beam}" in self._conf else f"BEAM_{beam}"
            if key in self._conf:
                self.beams.append(beam)
                self.beam_range[beam] = self._conf[key]

    @staticmethod
    def _parse(conf_file):
        """Parse key-value pairs from an aXe conf file into a dict."""
        conf = {}
        with open(conf_file) as f:
            for line in f:
                line = line.split(";")[0].split("#")[0].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                key = parts[0]
                vals = parts[1:]
                if len(vals) == 1:
                    try:
                        conf[key] = float(vals[0])
                    except ValueError:
                        conf[key] = vals[0]
                else:
                    try:
                        conf[key] = np.array(vals, dtype=float)
                    except ValueError:
                        conf[key] = vals
        return conf

    def get_beam_trace(self, x, y, dx, beam):
        """
        Compute trace offset and wavelength along a beam.

        Parameters
        ----------
        x, y : float
            Source position on the detector.
        dx : float or array-like
            x pixel offsets from source at which to evaluate the trace.
        beam : str
            Beam name (e.g. 'A').

        Returns
        -------
        dy : float or array-like
            Trace y offset from source position at each dx.
        lam : float or array-like
            Wavelength (Angstrom) at each dx.
        """
        dx = np.asarray(dx, dtype=float)
        xi, yi = x - self.xoff, y - self.yoff

        xoff_beam = _axe_field_dependent(xi, yi, self._conf[f"XOFF_{beam}"])
        yoff_beam = _axe_field_dependent(xi, yi, self._conf[f"YOFF_{beam}"])

        norder = int(self._conf.get(f"DYDX_ORDER_{beam}", 0)) + 1
        dydx = [_axe_field_dependent(xi, yi, self._conf.get(f"DYDX_{beam}_{i}", 0.0))
                for i in range(norder)]

        dy = yoff_beam + sum(dydx[i] * (dx - xoff_beam) ** i for i in range(norder))

        # Accept both DISP_ORDER / DLDP (old) and DISPL_ORDER / DISPL (new) conventions
        if f"DISPL_ORDER_{beam}" in self._conf:
            ndldp = int(self._conf[f"DISPL_ORDER_{beam}"]) + 1
            dldp = [_axe_field_dependent(xi, yi, self._conf.get(f"DISPL_{beam}_{i}", 0.0))
                    for i in range(ndldp)]
        else:
            ndldp = int(self._conf.get(f"DISP_ORDER_{beam}", 1)) + 1
            dldp = [_axe_field_dependent(xi, yi, self._conf.get(f"DLDP_{beam}_{i}", 0.0))
                    for i in range(ndldp)]

        dp = _axe_arc_length(dx - xoff_beam, dydx)
        lam = sum(dldp[i] * dp ** i for i in range(ndldp))

        return dy, lam


# ---------------------------------------------------------------------------
# Reader 2: GRISMCONF text .conf files
# ---------------------------------------------------------------------------

class GRISMCONFReader:
    """
    Read a GRISMCONF-format text configuration file.

    Provides DISPX / DISPY / DISPL and their inverses without requiring
    the external ``grismconf`` package.

    Parameters
    ----------
    conf_file : str
        Path to a GRISMCONF ``.conf`` text file.
    """

    def __init__(self, conf_file):
        self.conf_file = conf_file
        self.wx = 0.0
        self.wy = 0.0

        with open(conf_file) as f:
            self._lines = f.readlines()

        self.fwcpos_ref = self._read_scalar("FWCPOS_REF")
        self.orders = self._read_orders()

        self._dispx = {o: self._read_poly("DISPX", o) for o in self.orders}
        self._dispy = {o: self._read_poly("DISPY", o) for o in self.orders}
        self._displ = {o: self._read_poly("DISPL", o) for o in self.orders}

        self._invdispx = {o: self._read_poly("INVDISPX", o) for o in self.orders}
        self._invdispy = {o: self._read_poly("INVDISPY", o) for o in self.orders}
        self._invdispl = {o: self._read_poly("INVDISPL", o) for o in self.orders}

    def _read_scalar(self, key):
        """Return the float value of a bare ``key value`` line, or None if absent."""
        for line in self._lines:
            parts = line.split("#")[0].split()
            if parts and parts[0] == key and len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    return None
        return None

    def _read_orders(self):
        orders = []
        for line in self._lines:
            parts = line.split("#")[0].split()
            if parts and parts[0].startswith("BEAM_"):
                orders.append(parts[0][len("BEAM_"):])
        return orders

    def _read_poly(self, name, order):
        """Read coefficient array for keyword ``name`` and spectral order."""
        prefix = f"{name}_{order}_"
        rows = {}
        ncols = None
        for line in self._lines:
            if line.startswith("#"):
                continue
            parts = line.split()
            if parts and parts[0].startswith(prefix):
                idx = int(parts[0][len(prefix):])
                vals = np.array(parts[1:], dtype=float)
                rows[idx] = vals
                ncols = len(vals)
        if not rows:
            return np.zeros((0, 0))
        arr = np.zeros((max(rows) + 1, ncols))
        for i, v in rows.items():
            arr[i] = v
        return arr

    def DISPX(self, order, x0, y0, t):
        return -self.wx + _eval_poly(self._dispx[order], x0, y0, t)

    def DISPY(self, order, x0, y0, t):
        return -self.wy + _eval_poly(self._dispy[order], x0, y0, t)

    def DISPL(self, order, x0, y0, t):
        return _eval_poly(self._displ[order], x0, y0, t)

    def INVDISPX(self, order, x0, y0, dx, t0=np.linspace(-1, 2, 128)):
        c = self._invdispx[order]
        if c.size:
            return _eval_poly(c, x0, y0, dx)
        return _inv_poly(self._dispx[order], x0, y0, dx + self.wx, t0)

    def INVDISPY(self, order, x0, y0, dy, t0=np.linspace(-1, 2, 128)):
        c = self._invdispy[order]
        if c.size:
            return _eval_poly(c, x0, y0, dy)
        return _inv_poly(self._dispy[order], x0, y0, dy + self.wy, t0)

    def INVDISPL(self, order, x0, y0, lam, t0=np.linspace(-1, 2, 128)):
        c = self._invdispl[order]
        if c.size:
            return _eval_poly(c, x0, y0, lam)
        return _inv_poly(self._displ[order], x0, y0, lam, t0)


# ---------------------------------------------------------------------------
# Reader 2: JWST CRDS specwcs ASDF files
# ---------------------------------------------------------------------------

class CRDSReader:
    """
    Read a JWST CRDS ``specwcs`` ASDF file.

    Provides DISPX / DISPY / DISPL and their inverses without requiring
    ``jwst.datamodels`` — only ``asdf`` and ``astropy`` are needed.

    Parameters
    ----------
    file : str
        Path to a CRDS specwcs ``.asdf`` file.
    """

    def __init__(self, file):
        import asdf

        self.file = file
        self.full_path = self._resolve_path(file)

        with asdf.open(self.full_path) as af:
            tree = af.tree
            self._meta = copy.deepcopy(dict(tree.get("meta", {})))
            self._dm_orders = list(tree["orders"])
            self._dispx = copy.deepcopy(list(tree["dispx"]))
            self._dispy = copy.deepcopy(list(tree["dispy"]))
            self._displ = copy.deepcopy(list(tree["displ"]))
            raw_invdispl = tree.get("invdispl")
            self._invdispl = copy.deepcopy(list(raw_invdispl)) if raw_invdispl else None
            fwcpos = tree.get("fwcpos_ref")
            self.fwcpos_ref = float(fwcpos) if fwcpos is not None else None

    @staticmethod
    def _resolve_path(file):
        if os.path.exists(file):
            return file
        crds_path = os.environ.get("CRDS_PATH", "")
        if file.startswith("references"):
            return os.path.join(crds_path, file)
        if "/references" in file:
            return os.path.join(crds_path, "references", file.split("references/")[1])
        raise FileNotFoundError(f"Cannot resolve specwcs path: {file}")

    @property
    def orders(self):
        return [f"+{o}" if o > 0 else str(o) for o in self._dm_orders]

    def _oi(self, order):
        return self.orders.index(order)

    def _eval(self, model, x0, y0, t):
        """Evaluate an astropy modeling object (or list thereof) from the ASDF tree."""
        if hasattr(model, "n_inputs"):
            return model(t) if model.n_inputs == 1 else model(x0, y0)
        if len(model) == 1:
            m = model[0]
            return m(t) if m.n_inputs == 1 else m(x0, y0)
        coeffs = [m(t) if m.n_inputs == 1 else m(x0, y0) for m in model]
        return np.polynomial.Polynomial(coeffs)(t)

    def _inv_eval(self, model, x0, y0, val, t0=np.linspace(0, 1, 128)):
        forward = self._eval(model, x0, y0, t0)
        so = np.argsort(forward)
        return np.interp(val, forward[so], t0[so])

    def DISPX(self, order, x0, y0, t):
        return self._eval(self._dispx[self._oi(order)], x0, y0, t)

    def DISPY(self, order, x0, y0, t):
        return self._eval(self._dispy[self._oi(order)], x0, y0, t)

    def DISPL(self, order, x0, y0, t):
        return self._eval(self._displ[self._oi(order)], x0, y0, t)

    def INVDISPX(self, order, x0, y0, dx, t0=np.linspace(-1, 2, 128)):
        return self._inv_eval(self._dispx[self._oi(order)], x0, y0, dx, t0)

    def INVDISPY(self, order, x0, y0, dy, t0=np.linspace(-1, 2, 128)):
        return self._inv_eval(self._dispy[self._oi(order)], x0, y0, dy, t0)

    def INVDISPL(self, order, x0, y0, lam, t0=np.linspace(0, 1, 128)):
        if self._invdispl is not None:
            return self._eval(self._invdispl[self._oi(order)], x0, y0, lam)
        return self._inv_eval(self._displ[self._oi(order)], x0, y0, lam, t0)


# ---------------------------------------------------------------------------
# Reader 4: Roman WFI grism YAML optical model
# ---------------------------------------------------------------------------

def _eval_poly2d(coeffs, x, y):
    """
    Evaluate a full-power 2D polynomial: sum_i sum_j coeffs[i,j] * x^i * y^j.
    Used for xmap/ymap in the Roman model.
    """
    coeffs = np.asarray(coeffs)
    ni, nj = coeffs.shape
    result = 0.0
    for i in range(ni):
        for j in range(nj):
            result = result + coeffs[i, j] * x ** i * y ** j
    return result


def _eval_poly3d(coeffs, v0, v1, v2):
    """
    Evaluate a full-power 3D polynomial:
    sum_i sum_j sum_k coeffs[i,j,k] * v0^i * v1^j * v2^k.
    Used for crv/ids in the Roman model.
    """
    coeffs = np.asarray(coeffs)
    ni, nj, nk = coeffs.shape
    result = 0.0
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                result = result + coeffs[i, j, k] * v0 ** i * v1 ** j * v2 ** k
    return result


class RomanConfReader:
    """
    Read a Roman WFI grism optical model YAML file.

    The model is parameterized in FPA coordinates (mm).  Polynomials use a
    full-power matrix convention (coeff[i,j] * x^i * y^j) rather than the
    triangular basis of aXe or GRISMCONF.

    Parameters
    ----------
    yaml_file : str
        Path to the Roman grism optical model YAML file.

    Attributes
    ----------
    orders : list of str
        Order names with defined (non-null) coefficients.
    plate_scale : float
        Detector plate scale in pixel/mm.
    wl_reference : float
        Reference wavelength in microns.
    """

    def __init__(self, yaml_file):
        import yaml

        self.yaml_file = yaml_file
        with open(yaml_file) as f:
            doc = yaml.safe_load(f)

        roman = doc["roman"]
        self.meta = roman["meta"]
        det = roman["detector_model"]
        opt = roman["optical_model"]

        self.plate_scale = det["plate_scale"]   # pixel/mm
        self.naxis = (det["naxis1"], det["naxis2"])
        self.wl_min = opt["wl_min"]             # micron
        self.wl_max = opt["wl_max"]             # micron
        self.wl_reference = opt["wl_reference"] # micron

        raw = opt["orders"]
        self.orders = [
            o for o in opt["orders_defined"]
            if raw[o]["xmap_ij_coeff"] is not None
        ]

        self._xmap = {o: np.array(raw[o]["xmap_ij_coeff"]) for o in self.orders}
        self._ymap = {o: np.array(raw[o]["ymap_ij_coeff"]) for o in self.orders}
        self._crv  = {o: np.array(raw[o]["crv_ijk_coeff"]) for o in self.orders}
        self._ids  = {o: np.array(raw[o]["ids_ijk_coeff"]) for o in self.orders}

    def xmap(self, order, fpa_x, fpa_y):
        """X offset of trace origin from direct image position, in mm."""
        return _eval_poly2d(self._xmap[order], fpa_x, fpa_y)

    def ymap(self, order, fpa_x, fpa_y):
        """Y offset of trace origin from direct image position, in mm."""
        return _eval_poly2d(self._ymap[order], fpa_x, fpa_y)

    def crv(self, order, dy_mm, fpa_x, fpa_y):
        """
        X offset along the trace as a function of y offset (trace curvature).

        Parameters
        ----------
        dy_mm : float or array-like
            Y distance along the trace in mm.
        fpa_x, fpa_y : float
            Source FPA position in mm.

        Returns
        -------
        float or array-like
            X curvature offset in mm.
        """
        return _eval_poly3d(self._crv[order], dy_mm, fpa_x, fpa_y)

    def ids(self, order, wl_micron, fpa_x, fpa_y):
        """
        Y offset along the trace as a function of wavelength (inverse
        dispersion solution).

        Parameters
        ----------
        wl_micron : float or array-like
            Wavelength in microns.
        fpa_x, fpa_y : float
            Source FPA position in mm.

        Returns
        -------
        float or array-like
            Y offset in mm.
        """
        return _eval_poly3d(self._ids[order], wl_micron, fpa_x, fpa_y)

    def get_trace(self, order, fpa_x, fpa_y, wl_micron):
        """
        Compute dispersed trace pixel offsets from the source position.

        Parameters
        ----------
        order : str
        fpa_x, fpa_y : float
            Source FPA coordinates in mm.
        wl_micron : array-like
            Wavelengths in microns.

        Returns
        -------
        dx_pix, dy_pix : array-like
            Pixel offsets from the source position along x and y.
        """
        wl = np.asarray(wl_micron)
        dy_mm = self.ids(order, wl, fpa_x, fpa_y)
        dx_mm = self.xmap(order, fpa_x, fpa_y) + self.crv(order, dy_mm, fpa_x, fpa_y)
        dy_mm_total = self.ymap(order, fpa_x, fpa_y) + dy_mm
        return dx_mm * self.plate_scale, dy_mm_total * self.plate_scale
