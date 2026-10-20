"""What an evaluation reports: scores (``scores.json``), a one-line summary, and a self-contained HTML report.

:func:`scores` reads the record: each scenario's grades, overall grade, no replies, talk-overs and median latency,
and overall the mean of the graded scenarios' overall grades and the median and 90th-percentile (nearest rank)
response latency over every exchange Willy answered. :func:`html` is one page with no script and nothing fetched:
the judge (and whether it is Willy's own model, a bias), the scores, the latencies and every transcript.
:func:`write_private` writes a file readable by its owner alone (``0600``): everything the evaluator keeps is what
was said in a household.
"""

import html as markup
import math
import os
import pathlib
import statistics
import typing

import pydantic

from willy_example.evaluate import record, scenarios


class ScenarioScore(pydantic.BaseModel):
    """One scenario's scores."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    id: str
    title: str
    ended: record.Ended
    graded: bool
    overall: int | None = pydantic.Field(description="The judge's overall grade, 1 to 5; none when ungraded.")
    grades: dict[scenarios.Criterion, int] = pydantic.Field(description="Each criterion's grade, 1 to 5.")
    rationale: str | None = pydantic.Field(description="The judge's reason for the overall grade.")
    ungraded: str | None = pydantic.Field(description="Why the judge could not grade it.")
    exchanges: int
    no_replies: int
    talk_overs: int
    latency_median_s: float | None


class Summary(pydantic.BaseModel):
    """Every scenario together."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    scenarios: int
    graded: int
    mean_overall: float | None = pydantic.Field(description="The mean overall grade of the graded scenarios.")
    exchanges: int
    no_replies: int
    talk_overs: int
    cut_ins: int
    latency_median_s: float | None
    latency_p90_s: float | None


class Scores(pydantic.BaseModel):
    """``scores.json``: the judge, every scenario's scores, and overall."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    judge: record.Judge
    scenarios: tuple[ScenarioScore, ...]
    overall: Summary


def _p90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(0.9 * len(ordered)) - 1]


def _latencies(exchanges: tuple[record.Exchange, ...]) -> list[float]:
    return [
        exchange.measures.latency_seconds for exchange in exchanges if exchange.measures.latency_seconds is not None
    ]


def scores(results: record.Results) -> Scores:
    """Score ``results``: a pure reading of the record."""

    scored: list[ScenarioScore] = []
    for result in results.scenarios:
        verdict = result.graded if isinstance(result.graded, record.Verdict) else None
        latencies = _latencies(result.exchanges)
        scored.append(
            ScenarioScore(
                id=result.id,
                title=result.title,
                ended=result.ended,
                graded=verdict is not None,
                overall=None if verdict is None else verdict.overall,
                grades={} if verdict is None else {grade.criterion: grade.score for grade in verdict.grades},
                rationale=None if verdict is None else verdict.rationale,
                ungraded=result.graded.reason if isinstance(result.graded, record.Ungraded) else None,
                exchanges=len(result.exchanges),
                no_replies=sum(not exchange.measures.replied for exchange in result.exchanges),
                talk_overs=sum(exchange.measures.talk_over for exchange in result.exchanges),
                latency_median_s=statistics.median(latencies) if latencies else None,
            )
        )
    every = tuple(exchange for result in results.scenarios for exchange in result.exchanges)
    latencies = _latencies(every)
    overalls = [score.overall for score in scored if score.overall is not None]
    return Scores(
        judge=results.judge,
        scenarios=tuple(scored),
        overall=Summary(
            scenarios=len(scored),
            graded=len(overalls),
            mean_overall=statistics.fmean(overalls) if overalls else None,
            exchanges=len(every),
            no_replies=sum(not exchange.measures.replied for exchange in every),
            talk_overs=sum(exchange.measures.talk_over for exchange in every),
            cut_ins=sum(exchange.measures.cut_in_seconds is not None for exchange in every),
            latency_median_s=statistics.median(latencies) if latencies else None,
            latency_p90_s=_p90(latencies),
        ),
    )


def _seconds(value: float | None) -> str:
    return "–" if value is None else f"{value:.1f} s"


def _plural(count: int, one: str, many: str) -> str:
    return f"{count} {one if count == 1 else many}"


def summary(scored: Scores) -> str:
    """The one line printed at the end of a run."""

    overall = scored.overall
    mean = "–" if overall.mean_overall is None else f"{overall.mean_overall:.1f}"
    return (
        f"willy-evaluate: {_plural(overall.scenarios, 'scenario', 'scenarios')}, {overall.graded} graded, "
        + f"overall {mean}/5; {_plural(overall.exchanges, 'exchange', 'exchanges')}, latency median "
        + f"{_seconds(overall.latency_median_s)}, p90 {_seconds(overall.latency_p90_s)}; "
        + f"{_plural(overall.talk_overs, 'talk-over', 'talk-overs')}, {_plural(overall.no_replies, 'no reply', 'no replies')}"
    )


_STYLE = """
:root { --bg: #fbfaf7; --fg: #1f2328; --muted: #656d76; --line: #d8dee4; --card: #ffffff; --accent: #0b6e4f;
  --warn: #9a6700; --bad: #b42318; }
@media (prefers-color-scheme: dark) { :root { --bg: #0d1117; --fg: #e6edf3; --muted: #8d96a0; --line: #30363d;
  --card: #161b22; --accent: #3fb950; --warn: #d29922; --bad: #f85149; } }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px 64px; background: var(--bg); color: var(--fg);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 960px; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 4px; } h2 { font-size: 1.2rem; margin: 32px 0 8px; }
.muted { color: var(--muted); }
.note { border-left: 3px solid var(--warn); padding: 8px 12px; background: var(--card); margin: 16px 0; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 16px 0; }
.tile { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 12px; }
.tile b { display: block; font-size: 1.4rem; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; background: var(--card); }
th, td { border: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align: top; }
th { font-weight: 600; }
.score { font-variant-numeric: tabular-nums; text-align: center; }
.transcript { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px; }
.line { padding: 4px 0; } .line .who { font-weight: 600; } .line.willy .who { color: var(--accent); }
.flag { color: var(--bad); }
"""


def html(results: record.Results, scored: Scores) -> str:
    """The report: one self-contained page with no script and nothing fetched."""

    escape = markup.escape
    overall = scored.overall
    judge = results.judge
    parts: list[str] = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Willy evaluation</title>",
        f"<style>{_STYLE}</style></head><body><main>",
        "<h1>Willy evaluation</h1>",
        f'<p class="muted">Spoken conversations with Willy, graded by {escape(judge.model)}.</p>',
    ]
    if judge.shares_willys_model:
        parts.append(
            '<p class="note">The judge is Willy\'s own model, so it may favour answers like its own. Run with '
            + "<code>--judge-url</code> and <code>--judge-model</code> to judge with a bigger model.</p>"
        )
    mean = "–" if overall.mean_overall is None else f"{overall.mean_overall:.1f} / 5"
    tiles = [
        ("Overall", mean),
        ("Graded", f"{overall.graded} of {overall.scenarios}"),
        ("Latency median", _seconds(overall.latency_median_s)),
        ("Latency p90", _seconds(overall.latency_p90_s)),
        ("Talk-overs", str(overall.talk_overs)),
        ("No replies", str(overall.no_replies)),
        ("Exchanges", str(overall.exchanges)),
    ]
    parts.append('<div class="tiles">')
    parts.extend(f'<div class="tile"><span class="muted">{name}</span><b>{value}</b></div>' for name, value in tiles)
    parts.append("</div>")

    criteria: list[scenarios.Criterion] = [
        criterion for criterion in scenarios.CRITERIA if any(criterion in s.grades for s in scored.scenarios)
    ]
    parts.append('<h2>Scores</h2><div class="scroll"><table><thead><tr><th>Scenario</th><th>Overall</th>')
    parts.extend(f"<th>{escape(criterion.replace('_', ' '))}</th>" for criterion in criteria)
    parts.append("<th>Latency median</th><th>Talk-overs</th><th>No replies</th></tr></thead><tbody>")
    for score in scored.scenarios:
        parts.append(f"<tr><td>{escape(score.title)}</td>")
        parts.append(f'<td class="score">{"ungraded" if score.overall is None else score.overall}</td>')
        parts.extend(f'<td class="score">{score.grades.get(criterion, "")}</td>' for criterion in criteria)
        parts.append(
            f"<td>{_seconds(score.latency_median_s)}</td><td>{score.talk_overs}</td><td>{score.no_replies}</td></tr>"
        )
    parts.append("</tbody></table></div>")

    for result, score in zip(results.scenarios, scored.scenarios, strict=True):
        parts.append(f"<h2>{escape(result.title)}</h2>")
        people = ", ".join(f"{escape(person.name)} ({escape(person.voice)})" for person in result.people)
        parts.append(f'<p class="muted">{people}; ended: {escape(result.ended.replace("_", " "))}.</p>')
        if isinstance(result.graded, record.Verdict):
            parts.append(f"<p><b>Overall {result.graded.overall}/5.</b> {escape(result.graded.rationale)}</p>")
            parts.append("<ul>")
            parts.extend(
                f"<li><b>{escape(grade.criterion.replace('_', ' '))} {grade.score}/5</b>: {escape(grade.rationale)}</li>"
                for grade in result.graded.grades
            )
            parts.append("</ul>")
        else:
            parts.append(f'<p class="flag">Ungraded: {escape(score.ungraded or "")}</p>')
        measured = {exchange.exchange: exchange for exchange in result.exchanges}
        parts.append('<div class="transcript">')
        for utterance in result.utterances:
            when = "" if utterance.started_s is None else f"{utterance.started_s:.1f} s "
            who = "willy" if utterance.speaker == record.WILLY else "person"
            text = escape(utterance.text) if utterance.text else '<span class="muted">(nothing)</span>'
            note = f' <span class="flag">({escape(utterance.note)})</span>' if utterance.note else ""
            extra = ""
            exchange = measured.get(utterance.exchange)
            if utterance.speaker == record.WILLY and exchange is not None:
                extra = (
                    f' <span class="muted">latency {_seconds(exchange.measures.latency_seconds)}, '
                    + f"reply {_seconds(exchange.measures.reply_seconds)}</span>"
                    + f'<br><audio controls preload="none" src="{escape(exchange.audio)}"></audio>'
                )
            parts.append(
                f'<div class="line {who}"><span class="muted">{when}</span><span class="who">'
                + f"{escape(utterance.name)}:</span> {text}{note}{extra}</div>"
            )
        parts.append("</div>")
    parts.append("</main></body></html>")
    return "\n".join(parts)


def write_private(path: pathlib.Path, content: bytes) -> None:
    """Write ``content`` to ``path``, readable and writable by its owner alone (``0600``), replacing what was there."""

    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "wb") as file:
        _ = file.write(content)


__all__ = ["ScenarioScore", "Scores", "Summary", "html", "scores", "summary", "write_private"]
