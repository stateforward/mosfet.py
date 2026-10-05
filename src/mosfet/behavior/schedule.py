"""Wall-clock schedules for persistent routines (``hsm.at``).

An ``At`` schedule is a time of day on chosen weekdays in one IANA time zone. Occurrences are
computed with stdlib ``zoneinfo`` from an injected wall clock, so daylight-saving shifts and
weekdays follow the routine's zone, not the host's. There is no catch-up: a schedule only ever
looks forward from "now" for its next occurrence.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import datetime
import typing
import zoneinfo

WallClock: typing.TypeAlias = collections.abc.Callable[[], datetime.datetime]
"""Injected source of the current timezone-aware wall-clock time."""

WEEKDAYS: tuple[str, ...] = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

# A timer can fire a hair before its deadline (event-loop resolution, wall-clock adjustment);
# occurrences closer than this to "now" belong to the tick that is firing, not the next one.
OCCURRENCE_TOLERANCE = datetime.timedelta(seconds=1)
# Seven days plus one covers every weekday combination from any starting day.
_SEARCH_DAYS = 8


@dataclasses.dataclass(frozen=True, kw_only=True)
class At:
    """A wall-clock time of day on chosen weekdays in one IANA time zone."""

    hour: int
    minute: int
    weekdays: frozenset[int]
    zone: zoneinfo.ZoneInfo

    @classmethod
    def from_element(cls, element: collections.abc.Mapping[str, object]) -> At:
        """Build from a validated source ``at`` element (``time``, ``days``, ``tz``)."""

        time_of_day = typing.cast(str, element["time"])
        hour, minute = (int(part) for part in time_of_day.split(":", maxsplit=1))
        days = typing.cast(tuple[str, ...], element["days"])
        return cls(
            hour=hour,
            minute=minute,
            weekdays=frozenset(WEEKDAYS.index(day) for day in days),
            zone=zoneinfo.ZoneInfo(typing.cast(str, element["tz"])),
        )

    def _occurrences(self, around: datetime.datetime, *, forward: bool) -> collections.abc.Iterator[datetime.datetime]:
        local_date = around.astimezone(self.zone).date()
        offsets = range(_SEARCH_DAYS) if forward else range(0, -_SEARCH_DAYS, -1)
        for offset in offsets:
            day = local_date + datetime.timedelta(days=offset)
            if day.weekday() not in self.weekdays:
                continue
            # A local wall time in a spring-forward gap resolves past the gap; in a fall-back
            # overlap fold=0 picks the first of the two instants.
            yield datetime.datetime(day.year, day.month, day.day, self.hour, self.minute, tzinfo=self.zone)

    def next_after(self, now: datetime.datetime) -> datetime.datetime:
        """The first occurrence later than ``now`` (beyond the firing tolerance)."""

        _require_aware(now)
        threshold = now.astimezone(datetime.UTC) + OCCURRENCE_TOLERANCE
        for occurrence in self._occurrences(now, forward=True):
            if occurrence.astimezone(datetime.UTC) > threshold:
                return occurrence
        raise ValueError("at() schedule has no weekday to occur on.")

    def latest_at_or_before(self, moment: datetime.datetime) -> datetime.datetime:
        """The occurrence a tick firing at ``moment`` was due for (within the firing tolerance)."""

        _require_aware(moment)
        threshold = moment.astimezone(datetime.UTC) + OCCURRENCE_TOLERANCE
        for occurrence in self._occurrences(moment, forward=False):
            if occurrence.astimezone(datetime.UTC) <= threshold:
                return occurrence
        raise ValueError("at() schedule has no weekday to occur on.")

    def delay_from(self, now: datetime.datetime) -> datetime.timedelta:
        """Elapsed (UTC) time from ``now`` until the next occurrence."""

        return self.next_after(now).astimezone(datetime.UTC) - now.astimezone(datetime.UTC)


def _require_aware(moment: datetime.datetime) -> None:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("schedule times must be timezone-aware.")


__all__ = ["At", "OCCURRENCE_TOLERANCE", "WEEKDAYS", "WallClock"]
