# `regrid` — ne120pg3 → lat-lon on pressure levels

Preprocessing for the `finite_amplitude_wave_diag` POD. Raw CAM output is
`VAR(time, lev, ncol)` on an unstructured cubed-sphere mesh with hybrid-sigma
levels; the MDTF preprocessor needs X/Y axes and a pressure Z axis. This package
does both conversions.

It is a Python port of `../regrid_ne120_to_latlon.sh`, issuing the same NCO
commands in the same order. The shell script stays in place as the reference
implementation.

## Why this is a separate step, not an MDTF hook

The framework's `user_pp_scripts` hook runs *after* `query_catalog()` and
`parse_ds()` (`src/preprocessor.py:1637`), by which point the data has already
been opened and validated against the POD's declared `lat`/`lon`/`plev`
dimensions. Native `ncol`/hybrid data does not survive that far. There is no
pre-catalog hook, so regridding has to happen before the framework runs.

(Two further reasons, if you are considering the hook anyway: its signature is
Dataset-in/Dataset-out, not file-based; and `execute_pp_functions` nests the
user-script loop *inside* the loop over built-in preprocessor functions, so a
user script runs six times per variable.)

## Install

Needs NCO, ESMF and tempest-remap on `PATH`, plus any Python ≥ 3.9. The package
itself imports nothing outside the standard library.

```bash
conda create -n mdtf_regrid -c conda-forge nco esmf tempest-remap
conda activate mdtf_regrid
```

If that environment has no Python, put its `bin` on `PATH` and use any other
interpreter:

```bash
export PATH="$CONDA_PREFIX/envs/mdtf_regrid/bin:$PATH"
```

## Use

Run from the POD directory (`diagnostics/finite_amplitude_wave_diag`).

```bash
python -m regrid config                  # show resolved settings, run nothing
python -m regrid levels                  # print the vertical grid, run nothing
python -m regrid setup                   # mesh, target grid, weights, vertical grid
python -m regrid regrid T U V            # the actual work
python -m regrid validate out/*.nc       # check the results
python -m regrid catalog -o esm_catalog_regridded.json
```

`-n/--dry-run` prints every command without running it, and still reads file
metadata, so the chunk boundaries and the model-top warning it shows are the
ones a real run would produce.

Settings come from defaults, then environment variables, then flags. The shell
script's invocations carry over unchanged:

```bash
DATA_DIR=/data OUT_DIR=/data/regridded WORK_DIR=/data/work \
  RANGE=TEST1D python -m regrid regrid T
```

## Input

| | |
|---|---|
| 3-D field | `{case}.cam.{stream}.{VAR}.{range}.nc`, `VAR(time, lev, ncol)` |
| Surface pressure | `{case}.cam.h7i.PS.{range}.nc` — **only** published in `h7i` |

`PS` is required and cannot be derived. `PSL` is not a substitute: sea-level
pressure is a hydrostatic reduction to z = 0 and is wrong in the hybrid formula
wherever there is terrain.

Use `stream=h7i` (instantaneous) for science. Local wave activity is a nonlinear
functional of the instantaneous PV field, so the `h8a` 6-hour means bias the
diagnostic.

## Output

`{out_dir}/{case}.{VAR}.{i0:06d}-{i1:06d}.nc`, one data variable each, on
`(time, plev, lat, lon)`. 42 levels evenly spaced in pseudoheight
(z = 0…41 km, p = 1000.0000…2.8594 hPa), 181×360 cell-centred lat-lon.

Each file carries `regrid_source_stream`, `regrid_source_file`,
`regrid_vrt_xtr` and `regrid_algo` global attributes. The filename has no stream
token, so these attributes are the only record of provenance — and the restart
logic checks `regrid_source_stream` before accepting an existing chunk, so
switching streams rebuilds rather than silently reusing.

## Settings that are not free choices

Three defaults look like tuning knobs and are not:

- **`rnr_thr = 0.0`** (the `-r` flag). Without renormalization the conservative
  remap counts below-ground missing values as zero. On the one-day test that
  produced 2599 cells at 1000 hPa holding 0 < T < 150 K — wrong, but not
  obviously wrong. Never remove it.
- **`vrt_ntp = "log"`**. `ncremap` does *no* vertical interpolation unless this
  is set. Naming a target grid is not enough.
- **`vrt_xtr = "mss_val"`**. Leaves below-ground cells missing. The POD's
  `DataPreprocessor` detects NaN, Poisson-fills it, and saves masks so the
  figures can mark the filled regions. `nrs_ngh` would arrive NaN-free, bypass
  gridfill, and plot fabricated values as real data.

Also: `algo = "traave"` because `ESMF_RegridWeightGen` segfaults on this grid
pair in the conda-forge osx-arm64 build; `ncoaave` works but is slower. And
never request levels above the model top (2.838 hPa / 41.05 km) — `setup` and
`regrid` both warn, but the check is advisory.

## Tests

```bash
python -m unittest regrid.test_regrid -v
```

27 tests covering the level table, chunk boundaries, the noleap calendar
arithmetic and the `ncdump` parsers. Everything that needs neither data nor NCO.
The pipeline itself is verified by running it on a one-day slice and then
`validate`.

## Layout

| Module | Contents |
|---|---|
| `config.py` | `RegridConfig`: settings, environment overrides, derived paths |
| `levels.py` | pseudoheight ↔ pressure, level tables |
| `nco.py` | subprocess wrappers, `ncdump`/`ncks` metadata readers |
| `mesh.py` | `setup`: SCRIP mesh, target grid, weights, vertical grid |
| `pipeline.py` | `regrid_variable`: the per-chunk pipeline, restart logic |
| `validate.py` | post-hoc checks on finished output |
| `catalog.py` | intake-ESM catalog emission, noleap calendar |
| `cli.py` | argument parsing |
