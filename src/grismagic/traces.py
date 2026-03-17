"""
Unified grism trace computation wrapping all grismagic readers.

Explictly not ported from grizli/grismconf.py:

- Sensitivity curve loading (get_beams, SENS) — tied to grizli's file layout and astropy.table
- JwstDispersionTransform — JWST coordinate rotation to align dispersion with +x; only needed if you're working in grizli's rotated convention
- NIRISS fwcpos filter wheel rotation correction — niche
- Instrument-specific empirical polynomial corrections (the V4/V8 NIRCam offsets in get_beam_trace) — grizli calibrations
- load_grism_config / load_nircam_sensitivity_curve — grizli infrastructure

"""

import numpy as np
from .readers import aXeConfReader, GRISMCONFReader, CRDSReader, RomanConfReader


class GrismTrace:
    """
    Unified grism trace calculator.

    Wraps any of the four grismagic readers and exposes a single
    ``get_trace`` method that returns detector-frame trace positions and
    wavelengths for a given source location and set of pixel offsets.

    Parameters
    ----------
    reader : aXeConfReader | GRISMCONFReader | CRDSReader | RomanConfReader
        An already-initialised reader instance.

    Examples
    --------
    >>> tr = GrismTrace.from_axe("WFC3.G141.conf")
    >>> dx = np.arange(-100, 200)
    >>> x_tr, y_tr, lam = tr.get_trace(507, 507, order="A", dx=dx)

    >>> tr = GrismTrace.from_grismconf("NIRCAM_F444W_modA_R.conf")
    >>> x_tr, y_tr, lam = tr.get_trace(1024, 1024, order="+1", dx=dx)

    GrismTrace.from_axe("file.conf")
    GrismTrace.from_grismconf("file.conf")
    GrismTrace.from_crds("file.asdf")
    GrismTrace.from_roman("file.yaml")
    GrismTrace.from_file("file.*")          # auto-detects by extension/content

    x_trace, y_trace, lam = tr.get_trace(x, y, order, dx)

    """

    def __init__(self, reader):
        if isinstance(reader, aXeConfReader):
            self._kind = "axe"
        elif isinstance(reader, GRISMCONFReader):
            self._kind = "grismconf"
        elif isinstance(reader, CRDSReader):
            self._kind = "crds"
        elif isinstance(reader, RomanConfReader):
            self._kind = "roman"
        else:
            raise TypeError(f"Unsupported reader type: {type(reader).__name__}")
        self.reader = reader

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_axe(cls, conf_file):
        """Create from an aXe text .conf file."""
        return cls(aXeConfReader(conf_file))

    @classmethod
    def from_grismconf(cls, conf_file):
        """Create from a GRISMCONF text .conf file."""
        return cls(GRISMCONFReader(conf_file))

    @classmethod
    def from_crds(cls, asdf_file):
        """Create from a JWST CRDS specwcs .asdf file."""
        return cls(CRDSReader(asdf_file))

    @classmethod
    def from_roman(cls, yaml_file):
        """Create from a Roman WFI grism YAML file."""
        return cls(RomanConfReader(yaml_file))

    @classmethod
    def from_file(cls, path):
        """
        Auto-detect reader type from file extension and content.

        * ``.asdf``            → CRDS
        * ``.yaml`` / ``.yml`` → Roman
        * ``.conf`` with ``DISPX_`` keywords → GRISMCONF
        * ``.conf`` otherwise  → aXe
        """
        path_lower = str(path).lower()
        if path_lower.endswith(".asdf"):
            return cls.from_crds(path)
        if path_lower.endswith((".yaml", ".yml")):
            return cls.from_roman(path)
        with open(path) as fh:
            content = fh.read()
        if any(k in content for k in ("DISPX_", "DISPY_", "DISPL_")):
            return cls.from_grismconf(path)
        return cls.from_axe(path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def orders(self):
        """
        Available spectral orders.

        For aXe: beam names like ``'A'``, ``'B'``.
        For GRISMCONF / CRDS: signed strings like ``'+1'``, ``'0'``, ``'-1'``.
        For Roman: order strings from the YAML file.
        """
        if self._kind == "axe":
            return self.reader.beams
        return self.reader.orders

    def remove_beam(self, order):
        """
        Remove a spectral order from ``self.orders``.

        Parameters
        ----------
        order : str
            Order identifier as listed in ``self.orders``.
        """
        lst = self.reader.beams if self._kind == "axe" else self.reader.orders
        if order in lst:
            lst.remove(order)

    def dx_range(self, order, x=None, y=None, nt=512):
        """
        Pixel extent of the trace along x for a given order.

        For aXe the range comes directly from the ``BEAM{order}`` entry and is
        position-independent; ``x`` and ``y`` are ignored.  For GRISMCONF /
        CRDS the t parameter is swept from 0 to 1 at ``(x, y)``.  For Roman
        the full wavelength grid is evaluated at ``(x, y)``.  ``x`` and ``y``
        must be supplied for non-aXe formats.

        Parameters
        ----------
        order : str
            Spectral order identifier.
        x, y : float, optional
            Source position (pixels for GRISMCONF / CRDS; mm for Roman).
            Required for non-aXe formats.
        nt : int
            Number of points in the parameter sweep.

        Returns
        -------
        dx_min, dx_max : float
            Minimum and maximum x pixel offset from the source.
        """
        if self._kind == "axe":
            lo, hi = self.reader.beam_range[order]
            return float(lo), float(hi)
        if x is None or y is None:
            raise ValueError("x and y are required for GRISMCONF / CRDS / Roman")
        if self._kind in ("grismconf", "crds"):
            t = np.linspace(0, 1, nt)
            r = self.reader
            if self._primary_axis(order, x, y) == 'y':
                vals = r.DISPY(order, x, y, t)
            else:
                vals = r.DISPX(order, x, y, t)
            return float(vals.min()), float(vals.max())
        # Roman
        r = self.reader
        wl_grid = np.linspace(r.wl_min, r.wl_max, nt)
        dx_grid, _ = r.get_trace(order, x, y, wl_grid)
        return float(dx_grid.min()), float(dx_grid.max())

    def get_trace_at_wavelength(self, x, y, order, lam, n_interp=512):
        """
        Compute trace positions at specified wavelengths.

        Parameters
        ----------
        x, y : float
            Source position (pixels for aXe / GRISMCONF / CRDS; mm for Roman).
        order : str
            Spectral order identifier.
        lam : array-like
            Target wavelengths.  Units: Angstrom for aXe and GRISMCONF;
            micron for CRDS and Roman.
        n_interp : int
            Grid size for the numerical dx → wavelength inversion used by the
            aXe format.

        Returns
        -------
        x_trace : np.ndarray
            Absolute x pixel positions.
        y_trace : np.ndarray
            Absolute y pixel positions.
        """
        lam = np.asarray(lam, dtype=float)
        if self._kind == "axe":
            return self._axe_at_wavelength(x, y, order, lam, n_interp)
        if self._kind in ("grismconf", "crds"):
            r = self.reader
            t = r.INVDISPL(order, x, y, lam)
            return x + r.DISPX(order, x, y, t), y + r.DISPY(order, x, y, t)
        # Roman: ids() maps wavelength (micron) directly to trace offsets
        r = self.reader
        dy_mm = r.ids(order, lam, x, y)
        dx_mm = r.xmap(order, x, y) + r.crv(order, dy_mm, x, y)
        return (
            x + dx_mm * r.plate_scale,
            y + (r.ymap(order, x, y) + dy_mm) * r.plate_scale,
        )

    def get_traces(self, xs, ys, order, dx, n_lam_roman=512):
        """
        Compute traces for multiple source positions.

        Loops over ``(xs, ys)`` pairs and stacks the results.  See
        ``get_trace`` for parameter and return-value details.

        Parameters
        ----------
        xs, ys : array-like of float
            Source positions, one per source.
        order : str
            Spectral order identifier.
        dx : array-like
            Pixel offsets from each source along x (same grid for all sources).
        n_lam_roman : int
            Wavelength grid size for Roman inversion.

        Returns
        -------
        x_traces : np.ndarray, shape (n_sources, n_dx)
        y_traces : np.ndarray, shape (n_sources, n_dx)
        lams : np.ndarray, shape (n_sources, n_dx)
        """
        results = [self.get_trace(x, y, order, dx, n_lam_roman) for x, y in zip(xs, ys)]
        return tuple(np.array([r[i] for r in results]) for i in range(3))

    def get_traces_at_wavelength(self, xs, ys, order, lam, n_interp=512):
        """
        Compute trace positions at specified wavelengths for multiple sources.

        Loops over ``(xs, ys)`` pairs and stacks the results.  See
        ``get_trace_at_wavelength`` for parameter and return-value details.

        Parameters
        ----------
        xs, ys : array-like of float
            Source positions, one per source.
        order : str
            Spectral order identifier.
        lam : array-like
            Target wavelengths (same grid for all sources).
        n_interp : int
            Grid size for the aXe numerical inversion.

        Returns
        -------
        x_traces : np.ndarray, shape (n_sources, n_lam)
        y_traces : np.ndarray, shape (n_sources, n_lam)
        """
        results = [self.get_trace_at_wavelength(x, y, order, lam, n_interp) for x, y in zip(xs, ys)]
        return tuple(np.array([r[i] for r in results]) for i in range(2))

    def get_trace(self, x, y, order, dx, n_lam_roman=512):
        """
        Compute the grism trace for a source at detector position ``(x, y)``.

        Parameters
        ----------
        x, y : float
            Source position.  Detector pixel coordinates for aXe / GRISMCONF /
            CRDS; FPA coordinates in mm for Roman.
        order : str
            Spectral order identifier as listed in ``self.orders``.
        dx : array-like
            Pixel offsets from the source along the x-axis.
        n_lam_roman : int
            Wavelength grid size used when inverting the Roman dispersion model.

        Returns
        -------
        x_trace : np.ndarray
            Absolute x pixel positions along the trace.
        y_trace : np.ndarray
            Absolute y pixel positions along the trace.
        lam : np.ndarray
            Wavelength along the trace.  Units: Angstrom for aXe and
            GRISMCONF; micron for CRDS and Roman.
        """
        dx = np.asarray(dx, dtype=float)
        if self._kind == "axe":
            return self._trace_axe(x, y, order, dx)
        if self._kind in ("grismconf", "crds"):
            return self._trace_grismconf(x, y, order, dx)
        return self._trace_roman(x, y, order, dx, n_lam=n_lam_roman)

    # ------------------------------------------------------------------
    # Per-format implementations
    # ------------------------------------------------------------------

    def _trace_axe(self, x, y, beam, dx):
        dy, lam = self.reader.get_beam_trace(x, y, dx, beam=beam)
        return x + dx, y + dy, lam

    def _primary_axis(self, order, x, y, nt=64):
        """Return 'x' (row grism) or 'y' (column grism) based on which DISP has larger range."""
        t = np.linspace(0, 1, nt)
        r = self.reader
        if np.ptp(r.DISPY(order, x, y, t)) > np.ptp(r.DISPX(order, x, y, t)):
            return 'y'
        return 'x'

    def _trace_grismconf(self, x, y, order, dx):
        r = self.reader
        if self._primary_axis(order, x, y) == 'y':
            # Column grism: the input 'dx' is really a dy offset
            t = r.INVDISPY(order, x, y, dx)
            x_trace = x + r.DISPX(order, x, y, t)
            y_trace = y + dx
        else:
            t = r.INVDISPX(order, x, y, dx)
            x_trace = x + dx
            y_trace = y + r.DISPY(order, x, y, t)
        lam = r.DISPL(order, x, y, t)
        return x_trace, y_trace, lam

    def _axe_at_wavelength(self, x, y, beam, lam, n_interp):
        lo, hi = self.dx_range(beam)
        dx_grid = np.linspace(lo, hi, n_interp)
        _, _, lam_grid = self._trace_axe(x, y, beam, dx_grid)
        so = np.argsort(lam_grid)
        dx = np.interp(lam, lam_grid[so], dx_grid[so])
        x_trace, y_trace, _ = self._trace_axe(x, y, beam, dx)
        return x_trace, y_trace

    def _trace_roman(self, x, y, order, dx, n_lam=512):
        r = self.reader
        wl_grid = np.linspace(r.wl_min, r.wl_max, n_lam)
        dx_grid, dy_grid = r.get_trace(order, x, y, wl_grid)
        so = np.argsort(dx_grid)
        y_trace = y + np.interp(dx, dx_grid[so], dy_grid[so])
        lam = np.interp(dx, dx_grid[so], wl_grid[so])
        return x + dx, y_trace, lam
