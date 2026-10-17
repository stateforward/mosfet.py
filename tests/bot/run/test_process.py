"""``bot.process``: the process options, from ``MOSFET_*`` variables and then flags, and B0107 for a bad one.

DSL-21 of the wave 7 bot DSL design (W7-10; OQ-27, OQ-28, OQ-34). Every call passes ``argv`` and
``environ``, so nothing here reads the test process's own.
"""

import datetime
import pathlib
import sys
import types

import pytest

from mosfet import bot, runtime

_S = datetime.timedelta(seconds=1)


def _bad(capsys: pytest.CaptureFixture[str], argv: list[str], environ: dict[str, str] | None = None) -> str:
    with pytest.raises(SystemExit) as exited:
        _ = bot.process(argv=argv, environ={} if environ is None else environ)
    assert exited.value.code == 64
    captured = capsys.readouterr()
    assert captured.out == ""
    return captured.err


def test_the_defaults_are_a_private_local_process() -> None:
    process = bot.process(argv=[], environ={"HOSTNAME": "host-1"})

    assert process.instance_id == "host-1"
    assert process.data_dir == pathlib.Path(".mosfet").absolute()
    assert process.ready_file is None
    assert process.policy == runtime.Policy()
    assert not process.traceback
    assert process.name


def test_the_name_of_a_python_c_process_is_mosfet(monkeypatch: pytest.MonkeyPatch) -> None:
    """OQ-34: ``python -c`` runs a ``__main__`` with no spec and no ``__file__``, so the name falls back."""

    monkeypatch.setitem(sys.modules, "__main__", types.ModuleType("__main__"))

    assert bot.process(argv=[], environ={}).name == "mosfet"


def test_every_option_has_a_flag_and_a_variable(tmp_path: pathlib.Path) -> None:
    flags = bot.process(
        argv=[
            "--name",
            "greeter",
            "--instance-id",
            "box-2",
            "--data-dir",
            str(tmp_path / "data"),
            "--secrets-dir",
            str(tmp_path / "secrets"),
            "--ready-file",
            str(tmp_path / "ready"),
            "--stop-grace",
            "6",
            "--max-restarts",
            "2",
        ],
        environ={},
    )
    variables = bot.process(
        argv=[],
        environ={
            "MOSFET_NAME": "greeter",
            "MOSFET_INSTANCE_ID": "box-2",
            "MOSFET_DATA_DIR": str(tmp_path / "data"),
            "MOSFET_SECRETS_DIR": str(tmp_path / "secrets"),
            "MOSFET_READY_FILE": str(tmp_path / "ready"),
            "MOSFET_STOP_GRACE": "6",
            "MOSFET_MAX_RESTARTS": "2",
        },
    )

    for process in (flags, variables):
        assert (process.name, process.instance_id) == ("greeter", "box-2")
        assert process.data_dir == tmp_path / "data"
        assert process.ready_file == tmp_path / "ready"
        assert process.policy.stop_grace == 6 * _S
        assert process.policy.max_restarts == 2


def test_a_flag_wins_over_its_variable() -> None:
    process = bot.process(argv=["--stop-grace", "6"], environ={"MOSFET_STOP_GRACE": "7", "MOSFET_NAME": "env"})

    assert process.policy.stop_grace == 6 * _S
    assert process.name == "env"


def test_an_empty_variable_is_unset() -> None:
    assert bot.process(argv=[], environ={"MOSFET_READY_FILE": "", "MOSFET_STOP_GRACE": ""}).ready_file is None


def test_secrets_are_read_from_the_secrets_dir_first(tmp_path: pathlib.Path) -> None:
    _ = (tmp_path / "api_key").write_text("mounted\n", encoding="utf-8")

    process = bot.process(argv=["--secrets-dir", str(tmp_path)], environ={"API_KEY": "variable"})

    assert process.secrets.get("api_key").get_secret_value() == "mounted"
    assert "secrets" not in process.model_dump()


def test_mosfet_traceback_asks_for_tracebacks() -> None:
    assert bot.process(argv=[], environ={"MOSFET_TRACEBACK": "1"}).traceback


def test_the_process_is_frozen() -> None:
    process = bot.process(argv=[], environ={})

    with pytest.raises(ValueError):
        setattr(process, "name", "other")


@pytest.mark.parametrize(
    ("argv", "named"),
    [
        pytest.param(["--stop-grace", "soon"], "`--stop-grace` must be a number of seconds", id="not-a-number"),
        pytest.param(["--stop-grace", "nan"], "`--stop-grace` must be a number of seconds", id="not-finite"),
        pytest.param(["--max-restarts", "two"], "`--max-restarts` must be a whole number", id="not-a-count"),
        pytest.param(["--stop-grace"], "`--stop-grace` expected one argument", id="no-value"),
        pytest.param(["--bogus"], "unknown option `--bogus`", id="unknown"),
        pytest.param(["extra"], "an argument that is not an option", id="positional"),
        pytest.param(["--name", ""], "`--name` is empty", id="empty-name"),
        pytest.param(["--stop-grace", "0"], "the policy options do not fit together", id="not-positive"),
        pytest.param(["--max-restarts", "0"], "the policy options do not fit together", id="no-restarts"),
        pytest.param(
            ["--stop-grace", "4", "--bot-stop-timeout", "3", "--bot-release-timeout", "1"],
            "the policy options do not fit together",
            id="bot-stop-outside-grace",
        ),
        pytest.param(
            ["--backoff-initial", "5", "--backoff-max", "1"], "the policy options do not fit together", id="backoff"
        ),
    ],
)
def test_a_bad_flag_is_b0107_naming_it_and_exits_64(
    capsys: pytest.CaptureFixture[str], argv: list[str], named: str
) -> None:
    err = _bad(capsys, argv)

    assert err.startswith("error[B0107]: bad process option: ")
    assert named in err.splitlines()[0]
    assert err.rstrip().endswith("error: could not read the process options: 1 error")
    assert '= explain: bot.explain("B0107")' in err


def test_a_bad_variable_is_named_by_its_variable(capsys: pytest.CaptureFixture[str]) -> None:
    err = _bad(capsys, [], {"MOSFET_STOP_GRACE": "soon"})

    assert "`MOSFET_STOP_GRACE` must be a number of seconds" in err.splitlines()[0]
    assert "help: write it `--stop-grace SECONDS` or `MOSFET_STOP_GRACE=SECONDS`" in err


def test_b0107_points_at_the_bot_files_own_call(capsys: pytest.CaptureFixture[str]) -> None:
    err = _bad(capsys, ["--bogus"])

    assert f"--> {pathlib.Path(__file__).relative_to(pathlib.Path.cwd())}:" in err
    assert "help: the options are `--name`, `--instance-id`, `--data-dir`" in err


def test_a_bad_value_is_never_echoed(capsys: pytest.CaptureFixture[str]) -> None:
    err = _bad(capsys, [], {"MOSFET_STOP_GRACE": "hunter2"})

    assert "hunter2" not in err


def test_help_is_requested_output_and_exits_0(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        _ = bot.process(argv=["--help"], environ={})

    assert exited.value.code == 0
    assert "--stop-grace SECONDS" in capsys.readouterr().out


def test_b0107_is_explained() -> None:
    assert bot.explain("B0107").startswith("error[B0107]: ")
