"""
JAX grism disperser.

Core operation: for each spatial pixel in a 2D image, deposit its flux
(weighted by a 1D spectrum sampled along the trace) onto the output detector
using bilinear scatter-add.

Design
------
Follows the shift-invariant trace approximation used by grizli:

    trace position for source pixel at offset (di, dj) from source center
        = (x_trace + di,  y_trace + dj)

where ``x_trace, y_trace`` are pre-computed by ``GrismTrace.get_trace`` for
the source center.  This is exact for point sources and a good approximation
for compact sources; for extended objects where the trace field-dependence
across the source extent matters, re-compute traces per pixel with
``GrismTrace.get_traces``.

Memory is managed by chunking over the wavelength axis (same strategy as
roman_disperser), keeping the working set to
``N_spatial × chunk_size`` floats per chunk.

Coordinate convention
---------------------
All pixel coordinates are **0-indexed** with pixel centers at integers.
``GrismTrace.get_trace`` returns coordinates in the same convention.

Typical usage
-------------
>>> import jax.numpy as jnp
>>> from grismagic.traces import GrismTrace
>>> from grismagic.disperse import disperse_obj
>>>
>>> tr = GrismTrace.from_axe("WFC3.G141.conf")
>>> dx = jnp.arange(*tr.dx_range("A"))
>>> x_tr, y_tr, lam = tr.get_trace(507., 507., "A", dx)
>>> sens = jnp.interp(lam, spec_wave, spec_flux)
>>>
>>> output = jnp.zeros((1014, 1014))
>>> output = disperse_obj(galaxy_image, 507., 507., x_tr, y_tr, sens, output)
"""

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Bilinear scatter-add
# ---------------------------------------------------------------------------

def bilinear_scatter_add(output, x, y, values):
    """
    Accumulate ``values`` onto ``output`` using bilinear interpolation.

    Each (x, y) coordinate distributes its value to the four surrounding
    integer grid points weighted by proximity.  Out-of-bounds points are
    silently discarded.

    Parameters
    ----------
    output : jnp.ndarray, shape (H, W)
        Accumulator array (0-indexed, pixel centers at integers).
    x : jnp.ndarray, shape (N,)
        x (column) coordinates, 0-indexed.
    y : jnp.ndarray, shape (N,)
        y (row) coordinates, 0-indexed.
    values : jnp.ndarray, shape (N,)
        Flux values to scatter.

    Returns
    -------
    output : jnp.ndarray, shape (H, W)
    """
    x_floor = jnp.floor(x).astype(jnp.int32)
    y_floor = jnp.floor(y).astype(jnp.int32)
    fx = x - x_floor
    fy = y - y_floor

    kw = dict(mode="drop", wrap_negative_indices=False)
    output = output.at[y_floor,     x_floor    ].add(values * (1 - fx) * (1 - fy), **kw)
    output = output.at[y_floor,     x_floor + 1].add(values *      fx  * (1 - fy), **kw)
    output = output.at[y_floor + 1, x_floor    ].add(values * (1 - fx) *      fy,  **kw)
    output = output.at[y_floor + 1, x_floor + 1].add(values *      fx  *      fy,  **kw)
    return output


# ---------------------------------------------------------------------------
# Core disperser
# ---------------------------------------------------------------------------

def disperse_obj(
    image,
    x_src,
    y_src,
    x_trace,
    y_trace,
    sens,
    output,
    mask=None,
    chunk_size=128,
):
    """
    Disperse a 2D spatial image × 1D spectrum onto the output detector.

    For each spatial pixel (row, col) in ``image``, the flux is deposited
    along the grism trace at positions::

        x_out = x_trace + (col - cx)
        y_out = y_trace + (row - cy)

    where ``cx = (Nx-1)/2``, ``cy = (Ny-1)/2`` place the source center at
    (``x_src``, ``y_src``).  The spectrum weight at each trace position is
    ``image[row, col] * sens[k]``.

    Parameters
    ----------
    image : jnp.ndarray, shape (Ny, Nx)
        Spatial flux image.  The source center is assumed to be at the image
        center pixel ``(cy, cx) = ((Ny-1)/2, (Nx-1)/2)``.
    x_src, y_src : float
        Source center position on the output detector (0-indexed).
    x_trace : jnp.ndarray, shape (N_lam,)
        Absolute x pixel positions of the trace, from
        ``GrismTrace.get_trace(x_src, y_src, ...)``.
    y_trace : jnp.ndarray, shape (N_lam,)
        Absolute y pixel positions of the trace.
    sens : jnp.ndarray, shape (N_lam,)
        Spectrum (or sensitivity × spectrum) sampled at each trace position.
        Units: flux per pixel.
    output : jnp.ndarray, shape (H, W)
        Accumulator array to add into.
    mask : jnp.ndarray, shape (Ny, Nx), optional
        Boolean mask; only pixels where ``mask != 0`` contribute.  Pass
        ``None`` to use all pixels.
    chunk_size : int
        Number of trace positions (wavelengths) processed per chunk.
        Larger values use more memory but fewer loop iterations.

    Returns
    -------
    output : jnp.ndarray, shape (H, W)
        Updated accumulator.

    Notes
    -----
    Uses ``jax.lax.scan`` for the wavelength loop so the function is
    JIT-compilable.  For JIT, ``chunk_size`` must be static::

        disperse_jit = jax.jit(disperse_obj, static_argnames=("chunk_size",))
    """
    Ny, Nx = image.shape
    N_lam = x_trace.shape[0]

    # ----- spatial setup ------------------------------------------------
    # Pixel offsets from source center
    cy = (Ny - 1) / 2.0
    cx = (Nx - 1) / 2.0
    rows, cols = jnp.meshgrid(jnp.arange(Ny), jnp.arange(Nx), indexing="ij")
    di = (cols - cx).ravel()   # x offset from source center [N_spatial]
    dj = (rows - cy).ravel()   # y offset from source center [N_spatial]
    fl = image.ravel()         # [N_spatial]

    if mask is not None:
        fl = jnp.where(mask.ravel() != 0, fl, 0.0)

    # ----- pad trace arrays to a multiple of chunk_size -----------------
    n_chunks = (N_lam + chunk_size - 1) // chunk_size
    pad_len = n_chunks * chunk_size - N_lam

    def _pad(arr):
        return jnp.pad(arr, (0, pad_len))

    x_trace_p = _pad(x_trace)
    y_trace_p = _pad(y_trace)
    sens_p    = _pad(sens)

    # ----- wavelength-chunked scatter-add --------------------------------
    def process_chunk(output, chunk_idx):
        start = chunk_idx * chunk_size
        xt = jax.lax.dynamic_slice(x_trace_p, (start,), (chunk_size,))  # [C]
        yt = jax.lax.dynamic_slice(y_trace_p, (start,), (chunk_size,))
        s  = jax.lax.dynamic_slice(sens_p,    (start,), (chunk_size,))

        # Broadcast: [N_spatial, 1] + [1, C] → [N_spatial, C]
        x_out = di[:, None] + xt[None, :]
        y_out = dj[:, None] + yt[None, :]
        vals  = fl[:, None] * s[None, :]

        output = bilinear_scatter_add(
            output, x_out.ravel(), y_out.ravel(), vals.ravel()
        )
        return output, None

    output, _ = jax.lax.scan(process_chunk, output, jnp.arange(n_chunks))
    return output


# ---------------------------------------------------------------------------
# Multi-source disperser
# ---------------------------------------------------------------------------

def disperse_galaxies(
    images,
    x_srcs,
    y_srcs,
    x_traces,
    y_traces,
    sensitivities,
    output_shape,
    masks=None,
    chunk_size=128,
):
    """
    Disperse multiple sources sequentially onto a single output detector.

    Parameters
    ----------
    images : jnp.ndarray, shape (N_gal, Ny, Nx)
        Spatial images, one per source.  All must share the same (Ny, Nx).
    x_srcs, y_srcs : jnp.ndarray, shape (N_gal,)
        Source center positions on the output detector (0-indexed).
    x_traces, y_traces : jnp.ndarray, shape (N_gal, N_lam)
        Pre-computed trace positions for each source.  All traces must have
        the same length ``N_lam``; pad shorter traces with the last valid
        position and zero the corresponding ``sensitivities`` entries.
    sensitivities : jnp.ndarray, shape (N_gal, N_lam)
        Spectrum weights at each trace position.
    output_shape : (int, int)
        ``(H, W)`` of the output detector array.
    masks : jnp.ndarray, shape (N_gal, Ny, Nx), optional
        Per-galaxy spatial masks.  ``None`` includes all pixels.
    chunk_size : int
        Wavelength chunk size passed to ``disperse_obj``.

    Returns
    -------
    output : jnp.ndarray, shape (H, W)
        Accumulated dispersed flux from all sources.
    """
    n_gal = images.shape[0]
    output = jnp.zeros(output_shape, dtype=jnp.float32)

    def body(i, out):
        msk = None if masks is None else masks[i]
        return disperse_obj(
            images[i],
            x_srcs[i],
            y_srcs[i],
            x_traces[i],
            y_traces[i],
            sensitivities[i],
            out,
            mask=msk,
            chunk_size=chunk_size,
        )

    return jax.lax.fori_loop(0, n_gal, body, output)
