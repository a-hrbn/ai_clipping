import unittest

from ml.build_eval_manifest import dedup_moments


def _match_row(start, end, *, views=0.0, url="u", sim=0.7, kind="single_window",
               title="clip", ambiguous=False):
    row = {
        "window_start": start,
        "window_end": end,
        "views": views,
        "source_url": url,
        "transcript_similarity": sim,
        "transcript_match_kind": kind,
        "repost_title": title,
    }
    if ambiguous:
        row["transcript_position_ambiguous"] = True
    return row


class DedupMomentsTests(unittest.TestCase):
    def test_overlapping_windows_collapse_highest_views_wins(self) -> None:
        rows = [
            _match_row(100.0, 120.0, views=5, url="low"),
            _match_row(105.0, 124.0, views=50, url="high"),  # overlaps → wins the slot
        ]
        moments = dedup_moments(rows)

        self.assertEqual(len(moments), 1)
        self.assertEqual(moments[0]["source_url"], "high")
        self.assertEqual(moments[0]["views"], 50)

    def test_distant_windows_are_kept_and_sorted_by_start(self) -> None:
        rows = [
            _match_row(400.0, 420.0, views=7, url="late"),
            _match_row(100.0, 120.0, views=10, url="early"),
            _match_row(105.0, 124.0, views=5, url="dup"),  # overlaps early → dropped
        ]
        moments = dedup_moments(rows)

        self.assertEqual([m["start"] for m in moments], [100.0, 400.0])
        self.assertEqual([m["source_url"] for m in moments], ["early", "late"])

    def test_position_ambiguous_flag_passed_through(self) -> None:
        moments = dedup_moments([_match_row(10.0, 30.0, ambiguous=True)])

        self.assertTrue(moments[0]["position_ambiguous"])

    def test_non_ambiguous_moment_omits_the_flag(self) -> None:
        moments = dedup_moments([_match_row(10.0, 30.0)])

        self.assertNotIn("position_ambiguous", moments[0])

    def test_drops_zero_and_negative_length_windows(self) -> None:
        rows = [
            _match_row(10.0, 10.0, url="zero"),
            _match_row(30.0, 20.0, url="inverted"),
            _match_row(40.0, 60.0, url="valid"),
        ]
        moments = dedup_moments(rows)

        self.assertEqual([m["source_url"] for m in moments], ["valid"])


if __name__ == "__main__":
    unittest.main()
