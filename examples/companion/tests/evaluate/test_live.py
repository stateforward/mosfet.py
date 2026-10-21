"""The evaluator, live: one short scenario with a running companion, through the lab's speech, speech to text and model.

    cd examples/companion && set -a && . ./.env.example && set +a && COMPANION_LIVE=1 \\
        .venv/bin/python -m pytest -m live tests/evaluate/test_live.py -q -s

Two gates, as the live check has: marked ``live``, so ``-m "not live"`` deselects it, and skipped unless
``COMPANION_LIVE=1``. It needs the companion already running (the bot file, with a room the evaluator can join) at
``COMPANION_EVALUATE_URL`` (default ``ws://127.0.0.1:8765``) in room ``COMPANION_EVALUATE_PEER`` (default ``eval-room``,
so ``COMPANION_ROOMS`` must list it), and the ``computer_token_secret`` the companion was started with (``secrets/``
here, or ``COMPUTER_TOKEN_SECRET_FILE``). A companion or a lab service that does not answer fails it, naming what
failed. It asserts that the evaluation finished and kept what it found; what the companion said, and how the judge
graded it, are printed and kept, never asserted.
"""

import json
import os
import pathlib

import pytest
from companion_example.evaluate import cli

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("COMPANION_LIVE") != "1",
        reason="the evaluator runs against a running companion and the lab only with COMPANION_LIVE=1",
    ),
]

_SECRET = pathlib.Path(__file__).resolve().parents[2] / "secrets" / "computer_token_secret"


def test_the_evaluator_holds_a_short_conversation_with_a_running_companion_and_grades_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = os.environ.get("COMPUTER_TOKEN_SECRET_FILE", str(_SECRET))
    out = tmp_path / "evaluation"
    # Telemetry's files are written under the working directory: the test's own.
    monkeypatch.chdir(tmp_path)

    code = cli.main(
        [
            "--url",
            os.environ.get("COMPANION_EVALUATE_URL", "ws://127.0.0.1:8765"),
            "--peer",
            os.environ.get("COMPANION_EVALUATE_PEER", "eval-room"),
            "--secret-file",
            secret,
            "--out",
            str(out),
            "--scenario",
            "stranger-arrives",
        ]
    )

    captured = capsys.readouterr()
    print(captured.out)
    assert code == 0, f"the evaluation did not finish: {captured.err}"
    assert captured.out.startswith("companion-evaluate: 1 scenario")
    lines = (out / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    for line in lines:
        print(line)
    assert lines, "nothing was said"
    scores = json.loads((out / "scores.json").read_text(encoding="utf-8"))
    print(json.dumps(scores["scenarios"], indent=2))
    assert scores["overall"]["exchanges"] >= 1
    assert (out / "report.html").exists()
    assert sorted((out / "audio").iterdir())
