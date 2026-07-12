import json
import pathlib
import subprocess
import sys
import typing

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BENCHMARK_PATH = _ROOT / "benchmarks" / "yamux_session.py"


def test_yamux_session_benchmark_cli_runs_small_workload() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(_BENCHMARK_PATH),
            "--iterations",
            "2",
            "--repeats",
            "1",
            "--warmup",
            "0",
            "--payload-size",
            "8",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = typing.cast(list[dict[str, object]], json.loads(completed.stdout))
    assert {result["name"] for result in payload} == {
        "frame_encode_decode",
        "session_receive_read",
        "session_send_data",
        "session_ping_round_trip",
    }
    assert all(result["iterations"] == 2 for result in payload)
    assert all(result["payload_size"] == 8 for result in payload)
    assert all(isinstance(result["best_ns_per_op"], float) for result in payload)


def test_yamux_session_benchmark_cli_rejects_invalid_configuration() -> None:
    completed = subprocess.run(
        [sys.executable, str(_BENCHMARK_PATH), "--iterations", "0"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "iterations must be positive" in completed.stderr
