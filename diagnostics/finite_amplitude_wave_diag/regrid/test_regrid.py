"""Unit tests for the parts that need no data and no NCO.

Run with::

    python -m unittest regrid.test_regrid -v

These cover the pure logic: the vertical level table, chunk boundaries, the
noleap calendar arithmetic and the ncdump parsers. The pipeline itself is
verified by running ``regrid`` on a one-day slice and then ``validate``; see
clare_notes/regrid_guide.md.
"""

from __future__ import annotations

import unittest

from .catalog import _format_stamp, _noleap_from_day_number, _noleap_to_day_number
from .levels import (
    format_levels_cdl,
    levels_above_model_top,
    parse_level_list,
    pressure_to_pseudoheight,
    pseudoheight_levels,
    pseudoheight_to_pressure,
)
from .nco import data_variable_names, dimension_names, n_time
from .pipeline import chunk_bounds


class TestLevels(unittest.TestCase):

    def test_round_trip(self):
        for z in (0.0, 1000.0, 20000.0, 41000.0):
            p = pseudoheight_to_pressure(z)
            self.assertAlmostEqual(pressure_to_pseudoheight(p), z, places=6)

    def test_ground_level_is_p_ground(self):
        self.assertAlmostEqual(pseudoheight_to_pressure(0.0), 100000.0, places=6)

    def test_default_grid_matches_validated_values(self):
        """The 42-level grid signed off on 2026-08-07."""
        levels = pseudoheight_levels(41, 1)
        self.assertEqual(len(levels), 42)
        self.assertAlmostEqual(levels[0] / 100.0, 1000.0, places=4)
        self.assertAlmostEqual(levels[-1] / 100.0, 2.8594, places=4)

    def test_spacing_is_uniform_in_pseudoheight(self):
        levels = pseudoheight_levels(41, 1)
        z = [pressure_to_pseudoheight(p) for p in levels]
        spacing = [z[i + 1] - z[i] for i in range(len(z) - 1)]
        for s in spacing:
            self.assertAlmostEqual(s, 1000.0, places=3)

    def test_monotonically_decreasing_pressure(self):
        levels = pseudoheight_levels(41, 1)
        for a, b in zip(levels, levels[1:]):
            self.assertGreater(a, b)

    def test_fractional_spacing(self):
        levels = pseudoheight_levels(2, 0.5)
        self.assertEqual(len(levels), 5)   # 0, 0.5, 1.0, 1.5, 2.0

    def test_rejects_bad_spacing(self):
        with self.assertRaises(ValueError):
            pseudoheight_levels(41, 0)
        with self.assertRaises(ValueError):
            pseudoheight_levels(-1, 1)

    def test_parse_level_list(self):
        self.assertEqual(parse_level_list("100000, 92500"), [100000.0, 92500.0])
        self.assertEqual(parse_level_list("100000 92500"), [100000.0, 92500.0])
        with self.assertRaises(ValueError):
            parse_level_list("")
        with self.assertRaises(ValueError):
            parse_level_list("100000, -5")

    def test_levels_above_model_top(self):
        # Model top 2.838 hPa: ERA-Interim's 1 and 2 hPa levels have no data.
        levels = [100.0, 200.0, 500.0]          # Pa == 1, 2, 5 hPa
        above = levels_above_model_top(levels, 2.838)
        self.assertEqual(above, [100.0, 200.0])

    def test_format_levels_cdl(self):
        self.assertEqual(format_levels_cdl([1.0, 2.5]), "1.000000, 2.500000")


class TestChunkBounds(unittest.TestCase):

    def test_exact_multiple(self):
        self.assertEqual(chunk_bounds(8, 4), [(0, 3), (4, 7)])

    def test_ragged_last_chunk(self):
        self.assertEqual(chunk_bounds(10, 4), [(0, 3), (4, 7), (8, 9)])

    def test_single_short_chunk(self):
        self.assertEqual(chunk_bounds(4, 124), [(0, 3)])

    def test_chunk_of_one(self):
        self.assertEqual(chunk_bounds(3, 1), [(0, 0), (1, 1), (2, 2)])

    def test_covers_every_step_exactly_once(self):
        for n in (1, 7, 124, 2920):
            for c in (1, 5, 124, 5000):
                covered = []
                for i0, i1 in chunk_bounds(n, c):
                    covered.extend(range(i0, i1 + 1))
                self.assertEqual(covered, list(range(n)), f"n={n} chunk={c}")

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            chunk_bounds(0, 4)


class TestNoleapCalendar(unittest.TestCase):

    def test_day_number_round_trip(self):
        for y, m, d in [(2000, 1, 1), (2001, 1, 1), (2002, 12, 31),
                        (2000, 2, 28), (2000, 3, 1), (1999, 7, 15)]:
            n = _noleap_to_day_number(y, m, d)
            self.assertEqual(_noleap_from_day_number(n), (y, m, d))

    def test_no_leap_day(self):
        """29 Feb does not exist, and 1 Mar is one day after 28 Feb."""
        with self.assertRaises(ValueError):
            _noleap_to_day_number(2000, 2, 29)
        self.assertEqual(
            _noleap_to_day_number(2000, 3, 1) - _noleap_to_day_number(2000, 2, 28), 1
        )

    def test_year_is_365_days(self):
        self.assertEqual(
            _noleap_to_day_number(2001, 1, 1) - _noleap_to_day_number(2000, 1, 1), 365
        )

    def test_format_stamp_matches_real_data(self):
        """The TEST1D slice: 365.125..365.875 days since 2000-01-01, noleap.

        Verified against the file on 2026-08-08. This is also the case that
        proved the filename token 21600 (=06:00) was the averaging interval
        *end*, not the time coordinate, which sits at the midpoint 03:00.
        """
        ref = _noleap_to_day_number(2000, 1, 1)
        self.assertEqual(_format_stamp(365.125, ref), "20010101:030000")
        self.assertEqual(_format_stamp(365.875, ref), "20010101:210000")

    def test_format_stamp_midnight(self):
        ref = _noleap_to_day_number(2000, 1, 1)
        self.assertEqual(_format_stamp(0.0, ref), "20000101:000000")

    def test_format_stamp_carries_rounding_to_next_day(self):
        """A value a hair under midnight must not render as 24:00:00."""
        ref = _noleap_to_day_number(2000, 1, 1)
        self.assertEqual(_format_stamp(0.9999999, ref), "20000102:000000")

    def test_full_dataset_range(self):
        """The committed catalog's range, recomputed from the time coordinate."""
        ref = _noleap_to_day_number(2000, 1, 1)
        self.assertEqual(_format_stamp(365.125, ref), "20010101:030000")
        self.assertEqual(_format_stamp(1094.875, ref), "20021231:210000")


_HEADER = """netcdf foo {
dimensions:
\ttime = UNLIMITED ; // (4 currently)
\tlat = 181 ;
\tlon = 360 ;
\tplev = 42 ;
\tnbnd = 2 ;
variables:
\tdouble time(time) ;
\t\ttime:units = "days since 2000-01-01" ;
\tdouble time_bnds(time, nbnd) ;
\tdouble lat(lat) ;
\tfloat T(time, plev, lat, lon) ;
\t\tT:units = "K" ;

// global attributes:
\t\t:regrid_source_stream = "h7i" ;
}
"""


class TestHeaderParsing(unittest.TestCase):

    def test_dimension_names(self):
        self.assertEqual(
            dimension_names("", _HEADER), ["time", "lat", "lon", "plev", "nbnd"]
        )

    def test_data_variable_names_excludes_coords_and_bounds(self):
        """`time(time)` is a coordinate, `time_bnds` is bounds; only T is data."""
        self.assertEqual(data_variable_names("", _HEADER), ["T"])

    def test_n_time_unlimited(self):
        self.assertEqual(n_time("", _HEADER), 4)

    def test_n_time_fixed_dimension(self):
        header = _HEADER.replace("time = UNLIMITED ; // (4 currently)", "time = 7 ;")
        self.assertEqual(n_time("", header), 7)


if __name__ == "__main__":
    unittest.main()
