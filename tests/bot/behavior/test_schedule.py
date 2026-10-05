"""Wall-clock ``at`` schedules: weekdays, time zones, and daylight saving."""

import datetime
import zoneinfo

import pytest

from mosfet.behavior import schedule

_NEW_YORK = zoneinfo.ZoneInfo("America/New_York")


def _weekday_eight() -> schedule.At:
    return schedule.At.from_element(
        {"time": "08:00", "days": ("mon", "tue", "wed", "thu", "fri"), "tz": "America/New_York"}
    )


def test_next_after_skips_weekend_days() -> None:
    saturday = datetime.datetime(2026, 10, 3, 7, 0, tzinfo=_NEW_YORK)
    assert _weekday_eight().next_after(saturday) == datetime.datetime(2026, 10, 5, 8, 0, tzinfo=_NEW_YORK)


def test_next_after_same_day_before_and_after_the_time() -> None:
    monday_early = datetime.datetime(2026, 10, 5, 7, 59, tzinfo=_NEW_YORK)
    assert _weekday_eight().next_after(monday_early) == datetime.datetime(2026, 10, 5, 8, 0, tzinfo=_NEW_YORK)
    monday_late = datetime.datetime(2026, 10, 5, 8, 0, 1, tzinfo=_NEW_YORK)
    assert _weekday_eight().next_after(monday_late) == datetime.datetime(2026, 10, 6, 8, 0, tzinfo=_NEW_YORK)


def test_next_after_never_returns_the_occurrence_that_is_firing() -> None:
    # A timer that fires a hair early must not schedule the same occurrence again.
    early = datetime.datetime(2026, 10, 5, 7, 59, 59, 999_000, tzinfo=_NEW_YORK)
    assert _weekday_eight().next_after(early) == datetime.datetime(2026, 10, 6, 8, 0, tzinfo=_NEW_YORK)
    assert _weekday_eight().latest_at_or_before(early) == datetime.datetime(2026, 10, 5, 8, 0, tzinfo=_NEW_YORK)


def test_delay_counts_elapsed_time_across_daylight_saving_end() -> None:
    friday = datetime.datetime(2026, 10, 30, 9, 0, tzinfo=_NEW_YORK)
    # Sun Nov 1 falls back an hour: Fri 09:00 EDT -> Mon 08:00 EST is 72 elapsed hours.
    assert _weekday_eight().delay_from(friday) == datetime.timedelta(hours=72)
    assert _weekday_eight().delay_from(friday.astimezone(datetime.UTC)) == datetime.timedelta(hours=72)


def test_delay_counts_elapsed_time_across_daylight_saving_start() -> None:
    daily = schedule.At.from_element({"time": "08:00", "days": schedule.WEEKDAYS, "tz": "America/New_York"})
    saturday = datetime.datetime(2026, 3, 7, 8, 0, 5, tzinfo=_NEW_YORK)
    # Sun Mar 8 springs forward an hour: Sat 08:00 EST -> Sun 08:00 EDT is 23 elapsed hours.
    assert daily.delay_from(saturday) == datetime.timedelta(hours=23) - datetime.timedelta(seconds=5)


def test_time_inside_the_spring_forward_gap_resolves_past_it() -> None:
    gap = schedule.At.from_element({"time": "02:30", "days": ("sun",), "tz": "America/New_York"})
    saturday = datetime.datetime(2026, 3, 7, 12, 0, tzinfo=_NEW_YORK)
    assert gap.next_after(saturday).astimezone(datetime.UTC) == datetime.datetime(
        2026, 3, 8, 7, 30, tzinfo=datetime.UTC
    )


def test_schedule_rejects_naive_times() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _ = _weekday_eight().next_after(datetime.datetime(2026, 10, 5, 8, 0))
