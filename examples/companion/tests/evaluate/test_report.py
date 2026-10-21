"""``companion_example.evaluate.report``: scores, the one-line summary and the self-contained HTML report, from
fixtures."""

import json
import os
import pathlib
import re
import stat

from companion_example.evaluate import record, report, scenarios, turns


def _measures(latency: float | None, *, talk_over: bool = False, replied: bool = True) -> turns.Measures:
    return turns.Measures(
        replied=replied,
        latency_seconds=latency,
        reply_seconds=2.0 if replied else 0.0,
        talk_over=talk_over,
        cut_in_seconds=None,
    )


def _results() -> record.Results:
    held = {scenario.id: scenario for scenario in scenarios.builtin().scenarios}
    child = held["curious-child"]
    stranger = held["stranger-arrives"]
    graded = record.Verdict(
        grades=tuple(
            record.Grade(criterion=criterion, score=4, rationale=f"Good {criterion}.") for criterion in child.criteria
        ),
        overall=4,
        rationale="Kind and <simple>.",
    )
    return record.Results(
        judge=record.Judge(model="gemma-4-26b-a4b", shares_companions_model=True),
        scenarios=(
            record.ScenarioResult(
                id=child.id,
                title=child.title,
                criteria=child.criteria,
                people=child.people,
                ended="done",
                utterances=(
                    record.Utterance(
                        scenario=child.id,
                        exchange=0,
                        speaker="lily",
                        name="Lily",
                        text="Why is the sky blue? <script>",
                        started_s=1.0,
                        ended_s=2.5,
                    ),
                    record.Utterance(
                        scenario=child.id,
                        exchange=0,
                        speaker=record.COMPANION,
                        name="Companion",
                        text="Sunlight scatters in the air.",
                        started_s=3.5,
                        ended_s=6.0,
                    ),
                ),
                exchanges=(
                    record.Exchange(
                        scenario=child.id, exchange=0, speaker="lily", measures=_measures(1.0), audio="a-0.wav"
                    ),
                    record.Exchange(
                        scenario=child.id,
                        exchange=1,
                        speaker="lily",
                        measures=_measures(3.0, talk_over=True),
                        audio="a-1.wav",
                    ),
                ),
                graded=graded,
            ),
            record.ScenarioResult(
                id=stranger.id,
                title=stranger.title,
                criteria=stranger.criteria,
                people=stranger.people,
                ended="turns_used",
                utterances=(),
                exchanges=(
                    record.Exchange(
                        scenario=stranger.id,
                        exchange=0,
                        speaker="visitor",
                        measures=_measures(None, replied=False),
                        audio="b-0.wav",
                    ),
                    record.Exchange(
                        scenario=stranger.id, exchange=1, speaker="visitor", measures=_measures(2.0), audio="b-1.wav"
                    ),
                ),
                graded=record.Ungraded(reason="the judge's answer failed the schema twice"),
            ),
        ),
    )


def test_scores_are_per_scenario_and_overall_with_median_and_p90_latency() -> None:
    scored = report.scores(_results())

    child, stranger = scored.scenarios
    assert (child.overall, child.graded, child.talk_overs, child.no_replies) == (4, True, 1, 0)
    assert child.grades == {criterion: 4 for criterion in scenarios.builtin().scenarios[3].criteria}
    assert (stranger.overall, stranger.graded, stranger.no_replies) == (None, False, 1)
    assert stranger.ungraded == "the judge's answer failed the schema twice"
    overall = scored.overall
    assert (overall.scenarios, overall.graded, overall.exchanges, overall.no_replies, overall.talk_overs) == (
        2,
        1,
        4,
        1,
        1,
    )
    assert overall.mean_overall == 4.0
    # Latencies 1, 2 and 3 s: the no reply has none.
    assert overall.latency_median_s == 2.0
    assert overall.latency_p90_s == 3.0
    assert scored.judge.shares_companions_model


def test_the_one_line_summary_says_what_a_glance_needs() -> None:
    line = report.summary(report.scores(_results()))

    assert "\n" not in line
    assert line == (
        "companion-evaluate: 2 scenarios, 1 graded, overall 4.0/5; 4 exchanges, latency median 2.0 s, p90 3.0 s; "
        + "1 talk-over, 1 no reply"
    )


def test_the_report_is_self_contained_html_with_scores_latencies_and_transcripts() -> None:
    results = _results()
    page = report.html(results, report.scores(results))

    assert page.startswith("<!doctype html>")
    # Nothing is fetched: no script, stylesheet, image or font from anywhere.
    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", page)
    assert "<script" not in page
    assert "Why is the sky blue? &lt;script&gt;" in page
    assert "Kind and &lt;simple&gt;." in page
    assert "Sunlight scatters in the air." in page
    assert "gemma-4-26b-a4b" in page and "the companion's own model" in page
    assert "2.0 s" in page and "3.0 s" in page
    assert "ungraded" in page.lower()
    assert "a-0.wav" in page


def test_outputs_are_written_readable_by_their_owner_alone(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "scores.json"

    report.write_private(path, json.dumps({"a": 1}).encode())
    report.write_private(path, b"again")

    assert path.read_bytes() == b"again"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
