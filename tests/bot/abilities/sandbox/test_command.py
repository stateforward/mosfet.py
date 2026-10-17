import datetime
import typing

import pydantic
import pytest

from mosfet.abilities import sandbox

from tests.type_helpers import object_dict


def _examples(schema: dict[str, object]) -> list[object]:
    examples = schema["examples"]
    assert isinstance(examples, list) and examples
    return typing.cast(list[object], examples)


def test_run_defaults_to_the_workspace_no_stdin_and_a_minute() -> None:
    run = sandbox.RunData(argv=("python", "-c", "print(1)"))

    assert run.cwd is None
    assert run.env == {}
    assert run.stdin is None
    assert run.timeout == datetime.timedelta(seconds=60)


def test_run_argv_is_one_to_sixty_four_bounded_arguments() -> None:
    assert len(sandbox.RunData(argv=("x",) * 64).argv) == 64
    for argv in ((), ("x",) * 65, ("",), ("x" * 4097,), ("a\x00b",)):
        with pytest.raises(pydantic.ValidationError):
            _ = sandbox.RunData(argv=argv)


def test_run_argv_is_kept_verbatim_never_split_as_a_shell_line() -> None:
    argv = ("sh", "-c", "echo \"$HOME\" > 'out file'; ls *")

    assert sandbox.RunData(argv=argv).argv == argv


def test_run_cwd_is_an_absolute_sandbox_path() -> None:
    assert sandbox.RunData(argv=("ls",), cwd="/workspace/src").cwd == "/workspace/src"
    for cwd in ("src", "/workspace/../etc", "/workspace//src"):
        with pytest.raises(pydantic.ValidationError):
            _ = sandbox.RunData(argv=("ls",), cwd=cwd)


def test_run_env_is_at_most_sixteen_plain_entries() -> None:
    assert len(sandbox.RunData(argv=("env",), env={f"V{index}": "x" for index in range(16)}).env) == 16
    with pytest.raises(pydantic.ValidationError):
        _ = sandbox.RunData(argv=("env",), env={f"V{index}": "x" for index in range(17)})
    with pytest.raises(pydantic.ValidationError):
        _ = sandbox.RunData.model_validate({"argv": ["env"], "env": {"TOKEN": pydantic.SecretStr("hunter2")}})


def test_run_stdin_is_at_most_sixty_four_kib_of_utf8() -> None:
    assert sandbox.RunData(argv=("cat",), stdin="x" * 65_536).stdin == "x" * 65_536
    with pytest.raises(pydantic.ValidationError):
        _ = sandbox.RunData(argv=("cat",), stdin="x" * 65_537)
    with pytest.raises(pydantic.ValidationError, match="64 KiB"):
        _ = sandbox.RunData(argv=("cat",), stdin="é" * 32_769)
    with pytest.raises(pydantic.ValidationError):
        _ = sandbox.RunData(argv=("cat",), stdin="\ud800")


def test_run_timeout_is_one_second_to_the_hard_ceiling() -> None:
    assert sandbox.RunData.model_validate({"argv": ["true"], "timeout": "PT1S"}).timeout == datetime.timedelta(
        seconds=1
    )
    assert sandbox.RunData(argv=("true",), timeout=sandbox.MAX_RUN_TIMEOUT).timeout == sandbox.MAX_RUN_TIMEOUT
    with pytest.raises(pydantic.ValidationError):
        _ = sandbox.RunData(argv=("true",), timeout=datetime.timedelta(milliseconds=999))
    with pytest.raises(pydantic.ValidationError):
        _ = sandbox.RunData(argv=("true",), timeout=sandbox.MAX_RUN_TIMEOUT + datetime.timedelta(seconds=1))


def test_run_json_schema_golden() -> None:
    schema = object_dict(sandbox.RunData.model_json_schema())
    properties = object_dict(schema["properties"])

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["argv"]
    assert schema["description"]
    assert schema["examples"]
    assert set(properties) == {"argv", "cwd", "env", "stdin", "timeout"}
    argv = object_dict(properties["argv"])
    assert (argv["type"], argv["minItems"], argv["maxItems"]) == ("array", 1, 64)
    assert object_dict(argv["items"]) == {"maxLength": 4096, "minLength": 1, "pattern": "^[^\\x00]*$", "type": "string"}
    env = object_dict(properties["env"])
    assert (env["type"], env["maxProperties"]) == ("object", 16)
    assert object_dict(env["patternProperties"]) == {
        "^[A-Za-z_][A-Za-z0-9_]{0,127}$": {"maxLength": 4096, "pattern": "^[^\\x00]*$", "type": "string"}
    }
    assert object_dict(properties["cwd"])["anyOf"] == [
        {"maxLength": 1024, "minLength": 2, "type": "string"},
        {"type": "null"},
    ]
    assert object_dict(properties["stdin"])["anyOf"] == [{"maxLength": 65536, "type": "string"}, {"type": "null"}]
    assert "``" not in str(schema["description"])
    assert object_dict(properties["timeout"])["format"] == "duration"
    assert object_dict(properties["timeout"])["default"] == "PT1M"
    for name, field in properties.items():
        assert object_dict(field)["description"], name
        assert object_dict(field)["examples"], name
    for example in _examples(schema):
        _ = sandbox.RunData.model_validate(example)


def test_ran_carries_an_exit_code_only_when_the_process_exited() -> None:
    ran = sandbox.RanData(outcome="exited", exit_code=2, stderr="boom\n", duration=datetime.timedelta(seconds=1))

    assert (ran.exit_code, ran.stdout, ran.stdout_truncated, ran.stderr_truncated) == (2, "", False, False)
    assert sandbox.RanData(outcome="timed_out", duration=datetime.timedelta(seconds=2)).exit_code is None
    with pytest.raises(pydantic.ValidationError):
        _ = sandbox.RanData(outcome="exited", duration=datetime.timedelta(0))
    for outcome in ("timed_out", "out_of_memory", "killed"):
        with pytest.raises(pydantic.ValidationError):
            _ = sandbox.RanData(outcome=outcome, exit_code=137, duration=datetime.timedelta(0))
    for exit_code in (-1, 256):
        with pytest.raises(pydantic.ValidationError):
            _ = sandbox.RanData(outcome="exited", exit_code=exit_code, duration=datetime.timedelta(0))
    with pytest.raises(pydantic.ValidationError):
        _ = sandbox.RanData.model_validate({"outcome": "crashed", "duration": "PT0S"})
    with pytest.raises(pydantic.ValidationError):
        _ = sandbox.RanData(outcome="exited", exit_code=0, duration=datetime.timedelta(seconds=-1))


def test_ran_output_is_bounded_by_the_hard_output_ceiling() -> None:
    whole = "x" * sandbox.MAX_OUTPUT_CHARS

    assert sandbox.RanData(outcome="killed", stdout=whole, stderr=whole, duration=datetime.timedelta(0)).stdout == whole
    with pytest.raises(pydantic.ValidationError):
        _ = sandbox.RanData(outcome="killed", stdout=whole + "x", duration=datetime.timedelta(0))
    with pytest.raises(pydantic.ValidationError):
        _ = sandbox.RanData(outcome="killed", stderr=whole + "x", duration=datetime.timedelta(0))


def test_write_is_a_whole_file_of_at_most_256_kib_of_utf8() -> None:
    write = sandbox.WriteData(path="/workspace/src/main.py", text="print(1)\n")

    assert (write.path, write.text) == ("/workspace/src/main.py", "print(1)\n")
    assert len(sandbox.WriteData(path="/tmp/big", text="x" * 262_144).text) == 262_144
    with pytest.raises(pydantic.ValidationError, match="256 KiB"):
        _ = sandbox.WriteData(path="/tmp/big", text="é" * 131_073)
    with pytest.raises(pydantic.ValidationError):
        _ = sandbox.WriteData(path="main.py", text="")
    written = sandbox.WrittenData(path="/workspace/src/main.py", bytes=9)
    assert written.bytes == 9
    with pytest.raises(pydantic.ValidationError):
        _ = sandbox.WrittenData(path="/workspace/src/main.py", bytes=-1)


def test_read_answers_bounded_contents_with_the_file_size() -> None:
    assert sandbox.ReadData(path="/workspace/out.txt").path == "/workspace/out.txt"
    with pytest.raises(pydantic.ValidationError):
        _ = sandbox.ReadData(path="/workspace/../etc/passwd")
    contents = sandbox.ContentsData(path="/workspace/out.txt", text="ok\n", truncated=False, bytes=3)
    assert (contents.text, contents.truncated, contents.bytes) == ("ok\n", False, 3)
    with pytest.raises(pydantic.ValidationError):
        _ = sandbox.ContentsData(
            path="/workspace/out.txt", text="x" * (sandbox.MAX_OUTPUT_CHARS + 1), truncated=True, bytes=1
        )


def test_reset_and_fresh_carry_nothing_and_reset_says_what_is_lost() -> None:
    assert sandbox.ResetData().model_dump() == {}
    assert sandbox.FreshData().model_dump() == {}
    description = object_dict(sandbox.ResetData.model_json_schema())["description"]
    assert isinstance(description, str) and "workspace is lost" in description
    with pytest.raises(pydantic.ValidationError):
        _ = sandbox.ResetData.model_validate({"keep": True})


@pytest.mark.parametrize(
    "model",
    [
        sandbox.RunData,
        sandbox.RanData,
        sandbox.WriteData,
        sandbox.WrittenData,
        sandbox.ReadData,
        sandbox.ContentsData,
        sandbox.ResetData,
        sandbox.FreshData,
    ],
)
def test_every_command_model_is_frozen_closed_described_and_exemplified(model: type[pydantic.BaseModel]) -> None:
    schema = object_dict(model.model_json_schema())

    assert model.model_config.get("frozen") is True
    assert model.model_config.get("extra") == "forbid"
    assert schema["description"]
    assert schema["examples"]
    for example in _examples(schema):
        _ = model.model_validate(example)
