import math

import pytest

from bot.world import space


def test_position_distance_is_euclidean() -> None:
    assert space.Position(x=0.0, y=0.0).distance_to(space.Position(x=3.0, y=4.0)) == 5.0
    assert space.Position(x=1.0, y=2.0, z=3.0).distance_to(space.Position(x=1.0, y=2.0, z=3.0)) == 0.0


def test_placement_defaults_to_no_threshold() -> None:
    """A placed participant with no threshold hears everything: opting into geometry is explicit."""

    placement = space.Placement(position=space.Position(x=0.0, y=0.0))

    assert placement.threshold_db is None


@pytest.mark.parametrize(
    ("amplitude_db", "distance_m", "expected_db"),
    [
        # At the reference distance the received level is the declared amplitude.
        (60.0, space.D_REF, 60.0),
        # Inverse-square law: -20*log10(2) = -6.0206 dB per doubling of distance.
        (60.0, 2.0, 53.9794),
        (60.0, 4.0, 47.9588),
        (60.0, 8.0, 41.9382),
        # Closer than the reference gains level.
        (60.0, 0.5, 66.0206),
        # Distance zero is the hot path, not an edge case: an earpiece sits at the ear.
        (25.0, 0.0, 25.0 + 40.0),
        # ... and anything below the clamp reads the same as the clamp.
        (25.0, space.D_MIN / 2.0, 25.0 + 40.0),
        (25.0, space.D_MIN, 25.0 + 40.0),
    ],
)
def test_received_level_db_follows_inverse_square_law(
    amplitude_db: float, distance_m: float, expected_db: float
) -> None:
    assert space.received_level_db(amplitude_db, distance_m) == pytest.approx(expected_db, abs=1e-3)


def test_received_level_db_loses_six_db_per_doubling() -> None:
    """Pinned as a relationship, so a rewrite of the formula cannot quietly change the curve."""

    for distance_m in (0.25, 1.0, 3.0, 17.5):
        near = space.received_level_db(60.0, distance_m)
        far = space.received_level_db(60.0, distance_m * 2.0)
        assert near - far == pytest.approx(6.0206, abs=1e-3)


def test_received_level_db_survives_zero_distance() -> None:
    """``log10(0)`` is a domain error; the clamp is what keeps the earpiece path from raising.

    Kept separate from the table because this is the assertion that fails if someone simplifies
    ``max(distance_m, D_MIN)`` away, and it should fail with an obvious name.
    """

    level = space.received_level_db(25.0, 0.0)

    assert math.isfinite(level)
    assert level == pytest.approx(space.received_level_db(25.0, space.D_MIN))
