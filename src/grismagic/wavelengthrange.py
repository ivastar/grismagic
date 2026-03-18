"""
Cache management for JWST CRDS wavelengthrange reference files.

Resolution order for the reference file:

1. Explicit ``wavelengthrange_file`` argument
2. ``$GRISMAGIC_WAVELENGTHRANGE_FILE`` environment variable
3. ``~/.cache/grismagic/wavelengthrange/`` — downloaded on first use and
   re-fetched when ``check_update=True`` and the CRDS operational context
   has changed since the last download.

The only network calls made by this module are:

* A lightweight GET to ``https://jwst-crds.stsci.edu/get_default_context``
  (returns a short string like ``"jwst_1413.pmap"``) — only when
  ``check_update=True``.
* A full download of the reference file — only on first use **or** when the
  CRDS context has changed.

If the ``crds`` Python package is available it is used to resolve the best
reference filename for the current context.  Otherwise the module falls back
to ``crds.client.api`` (a lightweight sub-package that does not require
``$CRDS_PATH`` to be configured).

"""

import json
import os
import urllib.request

_CACHE_DIR = os.path.expanduser("~/.cache/grismagic/wavelengthrange")
_META_FILE = os.path.join(_CACHE_DIR, "meta.json")
_CRDS_CONTEXT_URL = "https://jwst-crds.stsci.edu/get_default_context"
_CRDS_DOWNLOAD_URL = (
    "https://jwst-crds.stsci.edu/unchecked_get/references/jwst/{instr}/{filename}"
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_wavelength_range(filter_name, order=None, instrument="niriss",
                         wavelengthrange_file=None, check_update=False):
    """
    Return ``(lam_min, lam_max)`` in microns for *filter_name*.

    Parameters
    ----------
    filter_name : str
        Filter name, e.g. ``'F200W'``.
    order : int or str, optional
        Spectral order.  ``None`` (default) returns the range for the first
        entry that matches *filter_name* (all orders share the same range in
        the CRDS reference).
    instrument : str
        JWST instrument, lower-case, e.g. ``'niriss'``.
    wavelengthrange_file : str or path-like, optional
        Explicit path to a wavelengthrange ASDF file.  Skips the cache.
    check_update : bool
        If ``True``, query CRDS for the current operational context and
        re-download the reference file if the context has changed since the
        last download.

    Returns
    -------
    lam_min, lam_max : float
        Wavelength limits in microns.

    Raises
    ------
    ValueError
        If no entry matching *filter_name* / *order* is found.
    RuntimeError
        If the reference file cannot be resolved and no cache exists.
    """
    path = _resolve(instrument, wavelengthrange_file, check_update)
    return _read_range(path, filter_name, order)


# ---------------------------------------------------------------------------
# Resolution chain
# ---------------------------------------------------------------------------


def _resolve(instrument, wavelengthrange_file, check_update):
    """Return a local path to the wavelengthrange reference file."""
    # 1. Explicit override
    if wavelengthrange_file is not None:
        return str(wavelengthrange_file)
    # 2. Environment variable
    env = os.environ.get("GRISMAGIC_WAVELENGTHRANGE_FILE")
    if env:
        return env
    # 3. Cache (download if needed)
    return _ensure_cached(instrument, check_update)


def _ensure_cached(instrument, check_update):
    """Return path to the cached file, downloading or refreshing as needed."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    meta = _load_meta()
    entry = meta.get(instrument, {})
    cached_path = entry.get("path")
    cached_context = entry.get("context")

    if cached_path and os.path.exists(cached_path):
        if not check_update:
            return cached_path
        current_ctx = _fetch_crds_context()
        if current_ctx is None or current_ctx == cached_context:
            return cached_path
        # Context changed — fall through to re-download

    return _download_and_cache(instrument, meta)


# ---------------------------------------------------------------------------
# CRDS helpers
# ---------------------------------------------------------------------------


def _fetch_crds_context():
    """Return current JWST CRDS operational context string, or None on failure."""
    try:
        with urllib.request.urlopen(_CRDS_CONTEXT_URL, timeout=5) as r:
            return r.read().decode().strip().strip('"')
    except Exception:
        return None


def _find_best_filename(instrument, context):
    """
    Return the best wavelengthrange reference filename from CRDS.

    Uses ``crds.client.api`` which does not require ``$CRDS_PATH``.
    Returns ``None`` if the lookup fails.
    """
    try:
        from crds.client import api as capi
        best = capi.get_best_references(
            context,
            {"INSTRUME": instrument.upper(), "EXP_TYPE": "NIS_WFSS"},
            reftypes=["wavelengthrange"],
        )
        return best.get("wavelengthrange")
    except Exception:
        return None


def _download_and_cache(instrument, meta):
    """Download the best-reference file from CRDS and store it in the cache."""
    context = _fetch_crds_context()
    filename = _find_best_filename(instrument, context) if context else None

    if filename is None:
        raise RuntimeError(
            f"Cannot resolve a wavelengthrange reference for instrument={instrument!r}. "
            "Pass wavelengthrange_file= explicitly, or set the "
            "$GRISMAGIC_WAVELENGTHRANGE_FILE environment variable."
        )

    url = _CRDS_DOWNLOAD_URL.format(instr=instrument.lower(), filename=filename)
    dest = os.path.join(_CACHE_DIR, filename)
    with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as fh:
        fh.write(r.read())

    meta.setdefault(instrument, {}).update(
        path=dest, context=context, filename=filename
    )
    _save_meta(meta)
    return dest


# ---------------------------------------------------------------------------
# Meta-file helpers
# ---------------------------------------------------------------------------


def _load_meta():
    if os.path.exists(_META_FILE):
        try:
            with open(_META_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_meta(meta):
    with open(_META_FILE, "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# ASDF reader
# ---------------------------------------------------------------------------


def _read_range(path, filter_name, order):
    """Parse (lam_min, lam_max) from a wavelengthrange ASDF file."""
    import asdf

    filter_name = filter_name.upper()
    # Normalise order for comparison: strip '+', convert to int string
    order_str = None
    if order is not None:
        try:
            order_str = str(int(str(order).lstrip("+")))
        except ValueError:
            order_str = str(order)

    with asdf.open(path) as af:
        for entry in af.tree["wavelengthrange"]:
            # entry: [order_int, filter_str, lam_min, lam_max]
            entry_order, entry_filter, lmin, lmax = entry
            if entry_filter.upper() != filter_name:
                continue
            if order_str is None or str(int(entry_order)) == order_str:
                return float(lmin), float(lmax)

    raise ValueError(
        f"No wavelength range found for filter={filter_name!r}, order={order!r} "
        f"in {path}"
    )
