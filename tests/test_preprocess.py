import math

import numpy as np
import pytest

from vimes.preprocess import (
    get_stellar_types,
    interp,
    detect_large_jump,
    detect_phases_indices,
    detect_mt_starts,
    make_event_string,
    sample_indices,
)


class TestGetStellarTypes:
    """
    Test the mapping of stellar types.
    """
    @classmethod
    def setup_class(cls):
        cls.expected_types = [
            "MS",
            "MS",
            "HG",
            "FGB",
            "CHeB",
            "EAGB",
            "TPAGB",
            "HeMS",
            "HeHG",
            "HeGB",
            "HeWD",
            "COWD",
            "ONeWD",
            "NS",
            "BH",
            "MR",
            "CHE",
        ]

    def test_known_indices(self):
        type_map = get_stellar_types()
        for i, expected in enumerate(self.expected_types):
            assert type_map(i) == expected

    def test_out_of_range_returns_unknown(self):
        type_map = get_stellar_types()
        assert type_map(len(self.expected_types) + 1) == "unknown"


class TestInterp:
    """
    Test the interp function.
    """
    @classmethod
    def setup_class(cls):
        cls.data = np.linspace(0, 100, 10)

    def test_exact_first_index(self):
        result = interp(self.data, 0)
        assert math.isclose(result, 0.0)

    def test_exact_last_index(self):
        result = interp(self.data, 9)
        assert math.isclose(result, 100.0)

    def test_midpoint_interpolation(self):
        result = interp(self.data, 0.5)
        expected = (self.data[0] + self.data[1]) / 2
        assert math.isclose(result, expected)

    def test_interpolation_is_linear(self):
        # Between two points the contributions from these points should scale linearly
        result = interp(self.data, 3.25)
        lower = float(self.data[3])
        upper = float(self.data[4])
        expected = lower * 0.75 + upper * 0.25
        assert math.isclose(result, expected)


class TestDetectLargeJump:
    """
    Test the detection of large jumps between two values.
    """
    def test_no_jump_zero_value(self):
        assert not detect_large_jump(0.0, 0.0)
        assert not detect_large_jump(0.0, 10.0)
        assert not detect_large_jump(10.0, 0.0)

    def test_no_jump_similar_values(self):
        assert not detect_large_jump(10.0, 10.0, threshold_ratio=1.2)
        assert not detect_large_jump(11.0, 10.0, threshold_ratio=1.2)
        assert not detect_large_jump(10.0, 11.0, threshold_ratio=1.2)

    def test_ratio_jump(self):
        assert detect_large_jump(10.0, 20.0, threshold_ratio=1.2)

    def test_abs_jump(self):
        assert detect_large_jump(100.0, 160.0, threshold_ratio=1.2, threshold_abs=50)


class TestDetectPhasesIndices:
    """
    Test the detection of changes in phase based on stellar types.
    """
    def test_single_phase_no_type_change(self):
        stellar_type_1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        stellar_type_2 = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        phases = detect_phases_indices(stellar_type_1, stellar_type_2)
        assert phases == [(0, 9)]

    def test_phase_changes_one_star(self):
        stellar_type_1 = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        stellar_type_2 = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        phases = detect_phases_indices(stellar_type_1, stellar_type_2)
        assert len(phases) == 2
        assert phases[0] == (0, 4)
        assert phases[1] == (5, 9)

    def test_phase_changes_both_stars(self):
        stellar_type_1 = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        stellar_type_2 = np.array([0, 0, 1, 1, 1, 1, 1, 1, 2, 2])
        phases = detect_phases_indices(stellar_type_1, stellar_type_2)
        assert len(phases) == 4
        assert phases[0] == (0, 1)
        assert phases[1] == (2, 4)
        assert phases[2] == (5, 7)
        assert phases[3] == (8, 9)


class TestDetectMtStarts:
    """
    Test the detection of mass transfer starts.
    """
    def test_no_mt(self):
        mt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert detect_mt_starts(mt) == set()

    def test_single_mt_start(self):
        mt = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        assert detect_mt_starts(mt) == {3}

    def test_multiple_mt_starts(self):
        mt = [0, 1, 0, 0, 1, 1, 0, 1, 0, 0]
        assert detect_mt_starts(mt) == {1, 4, 7}
