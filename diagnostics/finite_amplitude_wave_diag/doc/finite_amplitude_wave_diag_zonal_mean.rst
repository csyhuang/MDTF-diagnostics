.. This is a comment in RestructuredText format (two periods and a space).

.. Note that all "statements" and "paragraphs" need to be separated by a blank 
   line. This means the source code can be hard-wrapped to 80 columns for ease 
   of reading. Multi-line comments or commands like this need to be indented by
   exactly three spaces.

.. Underline with '='s to set top-level heading: 
   https://docutils.sourceforge.io/docs/user/rst/quickref.html#section-structure

Finite Amplitude Rossby Wave Diagnostics Documentation
======================================================

.. rst-class:: center

Clare S. Y. Huang\ |^1|, Christopher Polster |^2| and Noboru Nakamura\ |^1|

.. rst-class:: center

|^1|\ The University of Chicago, Chicago, Illinois

|^2|\ Johannes Gutenberg-Universität Mainz, Germany

.. rst-class:: center

Last update: 03/12/2024

Description
-----------
For a comprehensive review of the finite-amplitude Rossby wave activity (FAWA) theory, please refer to the review article Nakamura (2024).

This POD computes the seasonal climatologies of various finite-amplitude wave diagnostics. Each of the diagnostics captures different aspects of eddy-mean interactions.





Physical assumptions made in FAWA framework
--------------------------------------------





Preprocessing of Climate Model Output
-------------------------------------

The POD requires ``U``, ``V`` and ``T`` on a **regular latitude-longitude grid**
and on **pressure levels**. Nothing in the MDTF framework converts either a
horizontal grid or a vertical coordinate, so native model output that is on an
unstructured mesh, on hybrid-sigma levels, or both, has to be converted before
the framework runs.

For CAM/CESM spectral-element output (``VAR(time, lev, ncol)`` on a cubed
sphere with hybrid-sigma levels), a converter ships with this POD in
``regrid/``. It wraps NCO and TempestRemap and does the horizontal remap and
the vertical interpolation in one pass. See ``regrid/README.md`` for the full
manual; the short version is::

    conda activate mdtf_regrid           # nco, esmf, tempest-remap
    cd diagnostics/finite_amplitude_wave_diag
    python -m regrid setup               # mesh, target grid, weights (once)
    python -m regrid regrid T U V
    python -m regrid validate <out_dir>/*.nc
    python -m regrid catalog -o esm_catalog_<case>.json

Three of its defaults are not free choices and should not be changed without
reading the notes in ``regrid/README.md``: conservative remapping must be
renormalised (``-r``), the vertical interpolation must be requested explicitly
(``--vrt_ntp``), and below-ground cells should be left missing (``mss_val``)
rather than extrapolated, because this POD fills them itself with a Poisson
solver and records a mask of what it filled.

Vertical coordinate
^^^^^^^^^^^^^^^^^^^

The analysis is carried out in pseudoheight, :math:`z = -H \ln(p/p_0)` with
:math:`H` = 7000 m and :math:`p_0` = 1000 hPa. The POD inspects the input
pressure levels and adapts:

- If they are already evenly spaced in pseudoheight and start at the ground,
  ``falwa`` is told so (``data_on_evenly_spaced_pseudoheight_grid=True``); it
  then takes :math:`\Delta z` and ``kmax`` from the data and performs **no**
  vertical interpolation of its own.
- Otherwise the fields are interpolated onto a uniform pseudoheight grid of
  spacing ``TARGET_DZ`` (1000 m by default) reaching as high as the data does.

Supplying levels evenly spaced in pseudoheight therefore avoids one resampling
step. Levels above the model top must not be requested: they return entirely
missing and propagate into every column diagnostic.

inline :math:`\frac{ \sum_{t=0}^{N}f(t,k) }{N}`

.. Underline with '-'s to make a second-level heading.

Running this POD
----------------

1. **Create the environment.** ``src/conda/env_finite_amplitude_wave_diag.yml``.
   Note that ``falwa`` is distributed as source only -- its F2PY extensions are
   compiled by Meson for the host machine -- so a Fortran compiler is required,
   which is why ``fortran-compiler`` is among the dependencies.

   ``gridfill`` has no ``osx-arm64`` build on conda-forge, so on Apple Silicon
   the environment cannot be solved as written. It can be installed from source
   there (``pip install git+https://github.com/ajdawson/gridfill.git``) if you
   need to run the POD on that platform.

2. **Prepare the data** as described under *Preprocessing* above, and build an
   intake-ESM catalog for it.

3. **Edit the runtime configuration**,
   ``templates/runtime_config_finite_amplitude_wave_diag.yml``: point
   ``DATA_CATALOG`` at your catalog, and set the case name and the
   ``startdate``/``enddate`` to match the catalog's ``time_range``.

   Use a frequency string the framework can parse. ``6hr`` works even for
   instantaneous data; ``6hrPt`` appears in the catalog documentation but is
   rejected by ``DateFrequency`` in ``src/util/datelabel.py``.

4. **Run**::

       ./mdtf -f templates/runtime_config_finite_amplitude_wave_diag.yml

The framework queries the catalog on ``standard_name`` + ``frequency`` +
``realm`` + a regex of the case name against ``path``, preprocesses the
matches, writes ``case_info.yml`` into the POD's working directory, and passes
its location to the driver in the ``case_env_file`` environment variable. The
driver reads that file to locate the postprocessed catalog and the model's own
variable and coordinate names.

You do **not** need to create ``case_info.yml`` yourself. An annotated example
of what the framework produces is in ``case_info_example.yml``, which also
explains how to use a hand-written copy to run the driver standalone -- useful
when iterating on the diagnostic without re-running the whole framework.

Input data requirements
^^^^^^^^^^^^^^^^^^^^^^^

- ``U``, ``V``, ``T`` on ``(time, plev, lat, lon)``
- Regular lat-lon grid; the POD interpolates onto its own 1-degree analysis
  grid internally, so the input grid need not match it
- Pressure levels, stored in Pa or hPa (the driver converts)
- At least two timesteps per season; seasons with fewer are skipped with a
  warning, since the LWA/U covariance is undefined otherwise
- Missing values below ground are expected and are filled horizontally with a
  Poisson solver, with the filled region recorded in a mask that the figures
  mark

Cautions and known limitations
------------------------------

Read this before interpreting a model--observation comparison.

Vertical extent differs between model and reanalysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``kmax``, the number of levels in the pseudoheight analysis grid, is set to the
largest value each dataset can support rather than to a common value. That
keeps the maximum information from each, but it is a deliberate compromise
rather than a like-for-like comparison:

.. list-table::
   :header-rows: 1

   * - Source
     - Top level
     - ``kmax``
     - Analysis grid
   * - CAM ne120L58 (this case)
     - 2.859 hPa
     - 42
     - z = 0--41 km
   * - ERA5 (37 pressure levels)
     - 1 hPa
     - 49
     - z = 0--48 km

**The consequence is not confined to the top of the column.** The reference
state ``uref`` is obtained by inverting the quasi-geostrophic potential
vorticity over the whole column, so changing ``kmax`` changes ``uref`` at
*every* height, not only above 41 km. Differences in ``uref`` and in
``zonal_mean_u - uref`` therefore carry a methodological component throughout
the profile. ``lwa_baro`` and ``u_baro`` are density-weighted column integrals
and are affected more weakly, but they are affected.

Truncating the reanalysis to the model's top does not fix this cleanly: after
dropping the 1 and 2 hPa levels the next level is 3 hPa, giving ``kmax`` = 41
rather than 42.

Each output file records ``kmax``, ``dz`` and the top pressure in its global
attributes, and the POD warns when the model and observational values differ,
so the asymmetry is visible at run time rather than only here.

Below-ground values are constructed differently in model and reanalysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Where the surface lies above a pressure level -- 1000 hPa over the Tibetan
Plateau, for instance -- there is no atmosphere to report, and the two sources
fill that space by different means:

- **Model.** Regridding leaves those cells missing (``VRT_XTR=mss_val``), and
  the POD fills them by solving Poisson's equation horizontally on each level,
  recording a mask of what was filled.
- **ERA5.** Pressure-level fields are already extrapolated beneath the surface
  by ECMWF. The POD sees no missing values, so gridfill is bypassed and the
  mask is empty.

Nothing errors, and both look equally plausible. At 1000 hPa roughly 47% of the
grid is below ground in the model case, and 13% at 925 hPa, so **the lowest one
or two analysis levels (z = 0 and 1 km) should not be read as a clean
model--observation difference.** The barotropic quantities are density-weighted
and comparatively insensitive; ``zonal_mean_u`` near the ground is the most
exposed.

This is documented rather than corrected, on the grounds that published work
with ERA5 has not masked below-ground points either. Masking both sources
against a surface-pressure criterion is the obvious refinement if the lowest
levels turn out to matter.

Axis orientation is enforced, not assumed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``QGField`` requires **latitude ascending** and **pressure descending**.
Reanalysis is commonly stored the other way round: ERA5 from the CDS is
latitude-descending with pressure levels ascending from 1 hPa. The POD
reorients the input once, at load time, on the xarray Dataset -- which flips
the coordinate and the data together.

The reason for doing it on the Dataset rather than on arrays is that flipping a
coordinate and its data separately is easy to get half-right, and the result is
an inverted column or a hemisphere-flipped field: output that looks entirely
reasonable and is wrong.

Calendars are equalised by discarding 29 February
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The model uses a ``noleap`` calendar; reanalysis does not. The POD drops
29 February so that both calendars agree and every year contributes the same
number of timesteps -- which also makes a per-year mean and a pooled mean
identical, removing the weighting question rather than answering it.

Over 1991--2020 this discards 32 of 43832 six-hourly timesteps, 0.073%.

Note also that DJF is *climatological*: months 12, 1 and 2 are pooled, so
December of a given year is grouped with January and February of that same
year, not of the following one. Every winter then contains exactly 90 days.

Settings that are not free choices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Three regridding options look like tuning knobs and are not; see
``regrid/README.md`` before altering them.

- Conservative remapping must be renormalised (``-r``). Without it, below-ground
  missing values are counted as zero, which on a one-day test produced 2599
  cells at 1000 hPa holding temperatures between 0 and 150 K -- wrong, but not
  obviously wrong.
- ``ncremap`` performs no vertical interpolation unless ``--vrt_ntp`` is given.
  Naming a target grid is not sufficient.
- ``VRT_XTR=mss_val`` leaves below-ground cells missing so that the POD's own
  gridfill can act on them and record what it filled. The alternative hands the
  POD fabricated values it cannot distinguish from real ones.

Version & Contact info
----------------------

Here you should describe who contributed to the diagnostic, and who should be
contacted for further information:

- Version/revision information: version 1 (03/12/2024)
- PI (name, affiliation, email): Clare S. Y. Huang (The University of Chicago, csyhuang@uchicago.edu)
- Developer/point of contact: Clare S. Y. Huang (The University of Chicago, csyhuang@uchicago.edu)
- Other contributors: Christopher Polster, Noboru Nakamura

.. Underline with '^'s to make a third-level heading.

Open source copyright agreement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The MDTF framework is distributed under the LGPLv3 license (see LICENSE.txt). 
Unless you've distributed your script elsewhere, you don't need to change this.

Functionality
-------------

For each of the four seasons (DJF, MAM, JJA, SON) the POD:

1. selects every timestep in the season -- at 6-hourly input, all four samples
   a day. Local wave activity is a nonlinear functional of the instantaneous
   field, so the diagnostics are computed per timestep and the *results* are
   averaged, rather than averaging the input fields first;
2. fills any missing values horizontally at each level with a Poisson solver
   (``gridfill``), saving masks of the filled regions;
3. interpolates onto the analysis grid, ``xlon`` = 0(1)359, ``ylat`` =
   -90(1)90;
4. computes the reference state, local wave activity and barotropic components
   via ``falwa`` (``QGDataset`` with ``QGFieldNH18``);
5. interpolates the results back onto the input grid, averages over the season,
   and computes the covariance of barotropic LWA and U;
6. plots.

Step 4 is run in blocks of ``TIME_BATCH_SIZE`` timesteps. ``QGDataset`` holds
one ``QGField`` per timestep and each retains several full three-dimensional
fields, so a whole 6-hourly season at once would need over 100 GB. The
quantities kept afterwards are two-dimensional and cheap, so batching bounds
peak memory without changing any result -- the timesteps are independent.

Outputs, per season: height-latitude sections of zonal-mean U, LWA, Uref and
:math:`U - U_{ref}`; latitude-longitude maps of barotropic U, barotropic LWA
and their covariance. Twenty-eight figures in total, plus the preprocessed
intermediate fields as netCDF.

Required programming language and libraries
-------------------------------------------

Python 3. Dependencies are pinned in
``src/conda/env_finite_amplitude_wave_diag.yml``:

- ``falwa`` -- the finite-amplitude wave activity calculations. Source-only
  distribution; needs a Fortran compiler at install time.
- ``gridfill`` -- Poisson filling of missing values.
- ``intake-esm`` and ``pyyaml`` -- reading the framework's ``case_info.yml``
  hand-off and the data catalog it names.
- ``xarray``, ``numpy``, ``scipy``, ``netCDF4``, ``dask``, ``bottleneck`` --
  data handling.
- ``matplotlib`` and ``cartopy`` -- plotting.

The regridding helper in ``regrid/`` is separate and deliberately depends on
nothing outside the Python standard library, since it only orchestrates NCO and
TempestRemap command-line tools. Those must be on ``PATH``; a suitable
environment is ``conda create -n mdtf_regrid -c conda-forge nco esmf
tempest-remap``.

Required model output variables
-------------------------------

Three, all four-dimensional and at the same frequency:

.. list-table::
   :header-rows: 1

   * - POD name
     - standard_name
     - Units
     - Dimensions
   * - ``ua``
     - ``eastward_wind``
     - m s-1
     - time, plev, lat, lon
   * - ``va``
     - ``northward_wind``
     - m s-1
     - time, plev, lat, lon
   * - ``ta``
     - ``air_temperature``
     - K
     - time, plev, lat, lon

The names above are the CMIP conventions declared in ``settings.jsonc``; the
framework translates from the model's own convention, so CESM output supplies
them as ``U``, ``V`` and ``T``. No surface field is required by the POD itself,
though surface pressure is needed by the regridding step for any model on
hybrid-sigma levels.

The POD was developed against 6-hourly data. Any frequency the framework can
parse will run, but the wave activity budget is most meaningful at sub-daily
sampling.

References
----------

.. _ref-Nakamura-annual-review:

10241. Nakamura, N. (2024). Large-Scale Eddy-Mean Flow Interaction in the Earth's Extratropical Atmosphere. *Annual Review of Fluid Mechanics*, **56**, 349-377,
`doi:10.1146/annurev-fluid-121021-035602 <https://doi.org/10.1146/annurev-fluid-121021-035602>`__.

.. _ref-Neal-et-al-GRL:

10242. Neal, E., Huang, C. S., & Nakamura, N. (2022). The 2021 Pacific Northwest heat wave and associated blocking: meteorology and the role of an upstream cyclone as a diabatic source of wave activity. *Geophysical Research Letters*, **49(8)**, e2021GL097699. `doi:10.1029/2021GL097699 <https://doi.org/10.1029/2021GL097699>`__.

.. _ref-Nakamura-Science:

10243. Nakamura, N., & Huang, C. S. (2018). Atmospheric blocking as a traffic jam in the jet stream. *Science*, **361(6397)**, 42-47, `doi:10.1126/science.aat0721 <https://doi.org/10.1126/science.aat0721>`__.

.. _ref-Nakamura-Solomon-JAS-2010:

10244. Nakamura, N., & Solomon, A. (2010). Finite-amplitude wave activity and mean flow adjustments in the atmospheric general circulation. Part I: Quasigeostrophic theory and analysis. *Journal of the atmospheric sciences*, **67(12)**, 3967-3983, `doi:10.1175/2010JAS3503.1 <https://doi.org/10.1175/2010JAS3503.1>`__.

.. _ref-Nakamura-Solomon-JAS-2011:

10245. Nakamura, N., & Solomon, A. (2011). Finite-amplitude wave activity and mean flow adjustments in the atmospheric general circulation. Part II: Analysis in the isentropic coordinate. Journal of the atmospheric sciences, 68(11), 2783-2799, `doi:10.1175/2011JAS3685.1 <https://doi.org/10.1175/2011JAS3685.1>`__.

.. _ref-Huang-Nakamura-JAS-2016:

10246. Huang, C. S., & Nakamura, N. (2016). Local finite-amplitude wave activity as a diagnostic of anomalous weather events. Journal of the Atmospheric Sciences, 73(1), 211-229, `doi:10.1175/JAS-D-15-0194.1 <https://doi.org/10.1175/JAS-D-15-0194.1>`__.

.. _ref-Huang-Nakamura-GRL-2017:

10247. Huang, C. S., & Nakamura, N. (2017). Local wave activity budgets of the wintertime Northern Hemisphere: Implication for the Pacific and Atlantic storm tracks. Geophysical Research Letters, 44(11), 5673-5682, `doi:10.1002/2017GL073760 <https://doi.org/10.1002/2017GL073760>`__.

More about this diagnostic
--------------------------

(to be filled in)

Links to external sites
^^^^^^^^^^^^^^^^^^^^^^^

(to be filled in)

More references and citations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

(to be filled in)

Figures
^^^^^^^

Images **must** be provided in either .png or .jpeg formats in order to be 
displayed properly in both the html and pdf output.

Here's the syntax for including a figure in the document:

.. code-block:: restructuredtext

   .. _my-figure-tag: [only needed for linking to figures]

   .. figure:: [path to image file, relative to the source.rst file]
      :align: left
      :width: 75 % [these both need to be indented by three spaces]

      Paragraphs or other text following the figure that are indented by three
      spaces are treated as a caption/legend, eg:

      - red line: a Gaussian
      - blue line: another Gaussian

which produces

.. _my-figure-tag:

.. figure:: gaussians.jpg
   :align: left
   :width: 75 %

   Paragraphs or other text following the figure that are indented by three
   spaces are treated as a caption/legend, eg:

   - blue line: a Gaussian
   - orange line: another Gaussian

The tag lets you refer to figures in the text, e.g. 
``:ref:`Figure 1 <my-figure-tag>``` → :ref:`Figure 1 <my-figure-tag>`.

Equations
^^^^^^^^^

Accented and Greek letters can be written directly using Unicode: é, Ω. 
(Make sure your text editor is saving the file in UTF-8 encoding).

Use the following syntax for superscripts and subscripts in in-line text:

.. code-block:: restructuredtext

   W m\ :sup:`-2`\ ; CO\ :sub:`2`\ .

which produces: W m\ :sup:`-2`\ ; CO\ :sub:`2`\ .
Note one space is needed after both forward slashes in the input; these spaces 
are not included in the output.

Equations can be written using standard 
`latex <https://www.reed.edu/academic_support/pdfs/qskills/latexcheatsheet.pdf>`__ 
(PDF link) syntax. Short equations in-line with the text can be written as 
``:math:`f = 2 \Omega \sin \phi``` → :math:`f = 2 \Omega \sin \phi`.

Longer display equations can be written as follows. Note that a blank line is 
needed after the ``.. math::`` heading and after each equation, with the 
exception of aligned equations.

.. code-block:: restructuredtext

   .. math::

      \frac{D \mathbf{u}_g}{Dt} + f_0 \hat{\mathbf{k}} \times \mathbf{u}_a &= 0; \\
      \frac{Dh}{Dt} + f \nabla_z \cdot \mathbf{u}_a &= 0,

      \text{where } \mathbf{u}_g = \frac{g}{f_0} \hat{\mathbf{k}} \times \nabla_z h.

which produces:

.. math::

   \frac{D \mathbf{u}_g}{Dt} + f_0 \hat{\mathbf{k}} \times \mathbf{u}_a &= 0; \\
   \frac{Dh}{Dt} + f \nabla_z \cdot \mathbf{u}_a &= 0,

   \text{where } \mathbf{u}_g = \frac{g}{f_0} \hat{\mathbf{k}} \times \nabla_z h.

The editor at `https://livesphinx.herokuapp.com/ 
<https://livesphinx.herokuapp.com/>`__ can have issues formatting complicated 
equations, so you may want to check its output with a latex-specific editor, 
such as `overleaf <https://www.overleaf.com/>`__ or other `equation editors 
<https://www.codecogs.com/latex/eqneditor.php>`__.
