"""
QuantGuard — ML Anomaly Scorer Tests
======================================
Tests the 6-feature ML scoring logic from pathway_engine.py
by replicating the feature extraction independently.
"""

import math
import pytest


def compute_z_score(amount: float, history: list[float]) -> float:
    """Z-score: (amount - mean) / std."""
    if len(history) < 3:
        return 0.0
    mean = sum(history) / len(history)
    std = max((sum((x - mean) ** 2 for x in history) / len(history)) ** 0.5, 0.01)
    return round((amount - mean) / std, 3)


def compute_iqr_score(amount: float, history: list[float]) -> float:
    """IQR outlier score: (amount - Q3) / IQR."""
    sorted_h = sorted(history)
    if len(sorted_h) < 5:
        return 0.0
    q1 = sorted_h[len(sorted_h) // 4]
    q3 = sorted_h[3 * len(sorted_h) // 4]
    iqr_val = max(q3 - q1, 0.01)
    if amount > q3:
        return round((amount - q3) / iqr_val, 3)
    elif amount < q1:
        return round((q1 - amount) / iqr_val, 3)
    return 0.0


def compute_percentile(amount: float, history: list[float]) -> float:
    """Percentile rank of amount in history."""
    if len(history) < 3:
        return 0.5
    rank = sum(1 for x in history if x <= amount)
    return round(rank / len(history), 3)


def compute_geo_entropy(locations: list[str]) -> float:
    """Shannon entropy over location frequency."""
    if len(locations) < 2:
        return 0.0
    counts: dict = {}
    for loc in locations:
        counts[loc] = counts.get(loc, 0) + 1
    total = sum(counts.values())
    return round(-sum(
        (c / total) * math.log2(c / total)
        for c in counts.values() if c > 0
    ), 3)


def compute_composite_risk(
    z_score, iqr_score, percentile, geo_entropy,
    velocity, category_dev, pw_ratio
) -> float:
    """Weighted ensemble composite risk score."""
    return round(min(1.0,
        0.25 * min(abs(z_score) / 4.0, 1.0)
        + 0.20 * min(iqr_score / 3.0, 1.0)
        + 0.10 * min(geo_entropy / 3.5, 1.0)
        + 0.10 * min(velocity / 2.0, 1.0)
        + 0.10 * category_dev
        + 0.15 * min(pw_ratio / 5.0, 1.0)
        + 0.10 * (1.0 if percentile > 0.95 else 0.0)
    ), 4)


class TestZScore:
    def test_normal_amount(self):
        """Amount near mean → z-score near 0."""
        history = [100, 110, 90, 105, 95, 100, 98, 102]
        z = compute_z_score(100, history)
        assert abs(z) < 0.5

    def test_high_anomaly(self):
        """Amount far above mean → large positive z-score."""
        history = [50, 55, 60, 45, 50, 48, 52, 55]
        z = compute_z_score(500, history)
        assert z > 3.0

    def test_short_history(self):
        """History < 3 → returns 0."""
        assert compute_z_score(100, [50, 60]) == 0.0


class TestIQRScore:
    def test_within_iqr(self):
        """Amount inside IQR → score 0."""
        history = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        assert compute_iqr_score(55, history) == 0.0

    def test_above_q3(self):
        """Amount above Q3 → positive score."""
        history = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        score = compute_iqr_score(200, history)
        assert score > 1.0

    def test_short_history(self):
        """History < 5 → returns 0."""
        assert compute_iqr_score(100, [10, 20, 30]) == 0.0


class TestPercentile:
    def test_max_is_100th(self):
        """Maximum amount in history → percentile 1.0."""
        history = [10, 20, 30, 40, 50]
        assert compute_percentile(50, history) == 1.0

    def test_min_is_low(self):
        """Minimum amount → low percentile."""
        history = [10, 20, 30, 40, 50]
        p = compute_percentile(10, history)
        assert p <= 0.3


class TestGeoEntropy:
    def test_single_location(self):
        """All same location → entropy 0."""
        locs = ["Mumbai"] * 10
        assert compute_geo_entropy(locs) == 0.0

    def test_diverse_locations(self):
        """Many distinct locations → high entropy."""
        locs = ["Mumbai", "Delhi", "London", "Tokyo", "Paris",
                "Berlin", "Dubai", "NYC", "Sydney", "Toronto"]
        h = compute_geo_entropy(locs)
        assert h > 3.0  # log2(10) ≈ 3.32


class TestCompositeRisk:
    def test_low_risk(self):
        """All low features → low composite score."""
        score = compute_composite_risk(
            z_score=0.1, iqr_score=0.0, percentile=0.5,
            geo_entropy=0.5, velocity=0.1, category_dev=0.1,
            pw_ratio=1.0,
        )
        assert score < 0.15

    def test_high_risk(self):
        """All extreme features → high composite score."""
        score = compute_composite_risk(
            z_score=4.0, iqr_score=5.0, percentile=0.99,
            geo_entropy=3.5, velocity=3.0, category_dev=0.95,
            pw_ratio=8.0,
        )
        assert score > 0.8

    def test_bounded_zero_to_one(self):
        """Composite risk always in [0, 1]."""
        score = compute_composite_risk(
            z_score=100, iqr_score=100, percentile=1.0,
            geo_entropy=100, velocity=100, category_dev=1.0,
            pw_ratio=100,
        )
        assert 0.0 <= score <= 1.0
