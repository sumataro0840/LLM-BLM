"""
Acceptance times become New York time with daylight saving; the sessions and the event dates follow 設計.md 5.
"""
import unittest

from events.timing import timing

# a Friday, a Monday holiday (2024-01-15, Martin Luther King Jr. Day) and ordinary days
DAYS = ["2024-01-11", "2024-01-12", "2024-01-16", "2024-01-17", "2024-07-15", "2024-07-16", "2024-07-17"]


class TestTiming(unittest.TestCase):
    def test_summer_time_pre_market(self):
        # 11:36 UTC in July = 7:36 EDT, before the open: the same day
        at, et, session, day = timing("2024-07-16T11:36:19.000Z", DAYS)
        self.assertEqual(at, "2024-07-16T11:36:19+00:00")
        self.assertEqual(et, "2024-07-16T07:36:19")
        self.assertEqual((session, day), ("pre", "2024-07-16"))

    def test_winter_time_after_close(self):
        # 21:05 UTC in January = 16:05 EST: after the close, the next trading day
        _, et, session, day = timing("2024-01-11T21:05:00.000Z", DAYS)
        self.assertEqual(et, "2024-01-11T16:05:00")
        self.assertEqual((session, day), ("post", "2024-01-12"))

    def test_boundaries(self):
        # January: ET = UTC - 5
        self.assertEqual(timing("2024-01-16T14:29:59Z", DAYS)[2], "pre")
        self.assertEqual(timing("2024-01-16T14:30:00Z", DAYS)[2:], ("open", "2024-01-16"))
        self.assertEqual(timing("2024-01-16T21:00:00Z", DAYS)[2:], ("post", "2024-01-17"))
        self.assertEqual(timing("2024-01-16T22:30:00Z", DAYS)[2:], ("late", "2024-01-17"))

    def test_weekend_and_holiday(self):
        # Friday after the close -> Tuesday (Monday is a holiday)
        self.assertEqual(timing("2024-01-12T21:30:00Z", DAYS)[3], "2024-01-16")
        # filed on the holiday morning -> the next trading day
        self.assertEqual(timing("2024-01-15T12:00:00Z", DAYS)[3], "2024-01-16")

    def test_after_the_last_known_day(self):
        self.assertIsNone(timing("2024-07-17T21:00:00Z", DAYS)[3])
        self.assertIsNone(timing("2024-08-01T12:00:00Z", DAYS)[3])

    def test_late_night_utc_is_previous_day_in_new_york(self):
        # 2:00 UTC on Jan 17 = 21:00 EST on Jan 16: late, reacts on Jan 17
        _, et, session, day = timing("2024-01-17T02:00:00Z", DAYS)
        self.assertEqual(et, "2024-01-16T21:00:00")
        self.assertEqual((session, day), ("late", "2024-01-17"))


if __name__ == "__main__":
    unittest.main()
