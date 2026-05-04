# grismagic

Grism configuration readers and trace/dispersion utilities for slitless spectroscopy simulations.

Provides a unified Python interface for reading grism configuration files across multiple formats (aXe, GRISMCONF, JWST CRDS, Roman WFI), computing grism traces, and dispersing 2D spatial images onto a detector using JAX.

## Modules

### `Ex`
In the directory "Ex", my new simultaneous dispersion and recovery of actual images is available.


### `grismagic.readers`

Low-level readers for four grism configuration formats:

| Class | Format | Instruments |
|---|---|---|
| `aXeConfReader` | aXe `.conf` text files | HST WFC3, ACS, NIRISS |
| `GRISMCONFReader` | GRISMCONF `.conf` text files | JWST NIRCam, NIRISS |
| `CRDSReader` | JWST CRDS `specwcs` `.asdf` files | JWST NIRCam, NIRISS |
| `RomanConfReader` | Roman WFI optical model `.yaml` files | Roman WFI |

Each reader exposes `DISPX`, `DISPY`, `DISPL` and their inverses for GRISMCONF/CRDS, and `get_beam_trace` for aXe.

### `grismagic.traces`

`GrismTrace` wraps any reader with a consistent API:

```python
from grismagic.traces import GrismTrace
import numpy as np

# Load from any supported format
tr = GrismTrace.from_axe("WFC3.G141.conf")
tr = GrismTrace.from_grismconf("NIRCAM_F444W_modA_R.conf")
tr = GrismTrace.from_crds("jwst_nircam_specwcs_0136.asdf")
tr = GrismTrace.from_roman("roman_grism_model.yaml")
tr = GrismTrace.from_file("any_supported_file")  # auto-detects format

# Available spectral orders
print(tr.orders)  # e.g. ['A', 'B'] for aXe, ['+1', '0', '-1'] for GRISMCONF

# Trace at pixel offsets dx from source at (x, y)
dx = np.arange(*tr.dx_range("A"))  # valid pixel range for this order
x_trace, y_trace, lam = tr.get_trace(507., 507., order="A", dx=dx)

# Trace at specific wavelengths
x_trace, y_trace = tr.get_trace_at_wavelength(507., 507., order="A",
                                               lam=np.linspace(8000, 17000, 200))

# Multiple sources at once
x_traces, y_traces, lams = tr.get_traces(xs, ys, order="A", dx=dx)

# Remove an order
tr.remove_beam("B")
```

Wavelength units follow the native format convention: **Angstrom** for aXe and GRISMCONF, **micron** for CRDS and Roman.

### `grismagic.disperse`

JAX-based disperser. Convolves a 2D spatial image with a 1D spectrum and scatters the flux onto an output detector using bilinear interpolation.

```python
import jax
import jax.numpy as jnp
from grismagic.traces import GrismTrace
from grismagic.disperse import disperse_obj, disperse_galaxies

tr = GrismTrace.from_axe("WFC3.G141.conf")
dx = jnp.arange(*tr.dx_range("A"))
x_trace, y_trace, lam = tr.get_trace(507., 507., "A", dx)

# Sample spectrum at trace wavelengths
sens = jnp.interp(lam, spec_wave, spec_flux)

# Disperse a single galaxy
output = jnp.zeros((1014, 1014))
output = disperse_obj(galaxy_image, 507., 507., x_trace, y_trace, sens, output)

# JIT compile for repeated calls
disperse_jit = jax.jit(disperse_obj, static_argnames=("chunk_size",))

# Disperse multiple galaxies onto a single detector
output = disperse_galaxies(images, x_srcs, y_srcs, x_traces, y_traces,
                           sensitivities, output_shape=(1014, 1014))
```

The disperser uses the shift-invariant trace approximation (same as grizli): the trace computed at the source center is translated by each spatial pixel's offset. For extended objects where field-dependent trace variations across the source extent matter, compute per-pixel traces with `GrismTrace.get_traces`.

## Installation

```bash
pip install -e .
```

For Roman WFI support (requires `pyyaml`):

```bash
pip install -e ".[roman]"
```

For development:

```bash
pip install -e ".[test]"
pytest --cov=src/grismagic tests/
```

## Dependencies

- `numpy`
- `jax` — required for `grismagic.disperse`
- `asdf` — required for CRDS `.asdf` readers
- `pyyaml` — required for Roman `.yaml` readers (optional extra)

## Acknowledgements

This package was built using the [MPIA Python Package Template](https://github.com/mpi-astronomy/mpia-python-template) [![DOI](https://zenodo.org/badge/472725375.svg)](https://zenodo.org/badge/latestdoi/472725375).
