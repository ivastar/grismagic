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

_TABLE_CACHE: dict = {}   # {resolved_path: {(filter, order): (lmin, lmax)}}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_all_ranges(instrument="niriss", wavelengthrange_file=None,
                    check_update=False):
    """
    Load the entire wavelengthrange reference into a dict and return it.

    The dict is keyed by ``(filter_name_upper, order_str)`` where
    *order_str* is the integer order as a plain string (e.g. ``'-1'``,
    ``'0'``, ``'1'``).  Values are ``(lam_min, lam_max)`` in microns.

    This is the preferred entry point when computing traces for many
    sources: call it once, then look up ranges with plain dict access
    instead of re-reading the file on every call.

    Parameters
    ----------
    instrument : str
        JWST instrument, lower-case, e.g. ``'niriss'``.
    wavelengthrange_file : str or path-like, optional
        Explicit path to a wavelengthrange ASDF file.  Skips the cache.
    check_update : bool
        If ``True``, query CRDS for the current operational context and
        re-download the reference file if the context has changed.

    Returns
    -------
    dict
        ``{(filter_str, order_str): (lam_min, lam_max)}``
    """
    path = _resolve(instrument, wavelengthrange_file, check_update)
    if path not in _TABLE_CACHE:
        _TABLE_CACHE[path] = _read_all_ranges(path)
    return _TABLE_CACHE[path]


def get_wavelength_range(filter_name, order=None, instrument="niriss",
                         wavelengthrange_file=None, check_update=False):
    """
    Return ``(lam_min, lam_max)`` in microns for *filter_name* and *order*.

    Parameters
    ----------
    filter_name : str
        Filter name, e.g. ``'F200W'``.
    order : int or str, optional
        Spectral order.  ``None`` returns the range for the first entry
        matching *filter_name* (all orders share the same range in the
        current CRDS reference).
    instrument : str
        JWST instrument, lower-case, e.g. ``'niriss'``.
    wavelengthrange_file : str or path-like, optional
        Explicit path to a wavelengthrange ASDF file.  Skips the cache.
    check_update : bool
        If ``True``, query CRDS for the current operational context and
        re-download the reference file if the context has changed.

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
    # 3. grismagic's own cache (download if needed)
    try:
        return _ensure_cached(instrument, check_update)
    except Exception:
        pass
    # 4. Fall back to any wavelengthrange file already present in $CRDS_PATH
    crds_file = _find_in_crds_path(instrument)
    if crds_file:
        return crds_file
    raise RuntimeError(
        f"Cannot resolve a wavelengthrange reference for instrument={instrument!r}. "
        "Options:\n"
        "  • Pass wavelengthrange_file= explicitly\n"
        "  • Set $GRISMAGIC_WAVELENGTHRANGE_FILE\n"
        "  • Set $CRDS_PATH to a directory containing the JWST reference files\n"
        "  • Call grismagic.wavelengthrange._download_and_cache() once with network access"
    )


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


def _find_in_crds_path(instrument):
    """
    Look for an existing wavelengthrange ASDF file in ``$CRDS_PATH``.

    Checks the standard CRDS directory layout
    ``$CRDS_PATH/references/jwst/<instrument>/``.  Returns the path of the
    highest-versioned file found, or ``None`` if nothing is found.
    """
    import glob

    crds_root = os.environ.get("CRDS_PATH")
    if not crds_root:
        return None
    pattern = os.path.join(
        crds_root, "references", "jwst", instrument.lower(),
        f"*{instrument.lower()}*wavelengthrange*.asdf",
    )
    matches = glob.glob(pattern)
    if not matches:
        return None
    return sorted(matches)[-1]  # highest version by filename sort


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


def _read_all_ranges(path):
    """Return a dict of all (filter, order) -> (lam_min, lam_max) from *path*."""
    import asdf

    result = {}
    with asdf.open(path) as af:
        for entry in af.tree["wavelengthrange"]:
            # entry: [order_int, filter_str, lam_min, lam_max]
            entry_order, entry_filter, lmin, lmax = entry
            key = (entry_filter.upper(), str(int(entry_order)))
            result[key] = (float(lmin), float(lmax))
    return result


def _read_range(path, filter_name, order):
    """Parse (lam_min, lam_max) from a wavelengthrange ASDF file."""
    table = _read_all_ranges(path)
    filter_name = filter_name.upper()
    if order is None:
        # Return the first entry for this filter (order-independent in practice)
        for (f, _), v in table.items():
            if f == filter_name:
                return v
    else:
        try:
            order_str = str(int(str(order).lstrip("+")))
        except ValueError:
            order_str = str(order)
        key = (filter_name, order_str)
        if key in table:
            return table[key]

    raise ValueError(
        f"No wavelength range found for filter={filter_name!r}, order={order!r} "
        f"in {path}"
    )
