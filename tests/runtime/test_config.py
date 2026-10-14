import pathlib
import typing

import pydantic
import pytest
from mosfet import runtime


def _secrets(directory: pathlib.Path, **environ: str) -> runtime.Secrets:
    return runtime.Secrets(directory=directory, environ=environ)


def test_mounted_secret_file_wins_over_file_pointer_and_variable(tmp_path: pathlib.Path) -> None:
    _ = (tmp_path / "api_key").write_text("mounted\n", encoding="utf-8")
    pointed = tmp_path / "pointed"
    _ = pointed.write_text("pointed", encoding="utf-8")

    secret = _secrets(tmp_path, API_KEY_FILE=str(pointed), API_KEY="variable").get("API_KEY")

    assert secret.get_secret_value() == "mounted"


def test_file_pointer_wins_over_variable(tmp_path: pathlib.Path) -> None:
    pointed = tmp_path / "pointed"
    _ = pointed.write_text("pointed\r\n", encoding="utf-8")

    secret = _secrets(tmp_path / "absent", API_KEY_FILE=str(pointed), API_KEY="variable").get("api_key")

    assert secret.get_secret_value() == "pointed"


def test_variable_is_the_last_source(tmp_path: pathlib.Path) -> None:
    assert _secrets(tmp_path, API_KEY="variable").get("api_key").get_secret_value() == "variable"


def test_missing_secret_names_the_secret_and_no_value(tmp_path: pathlib.Path) -> None:
    secrets = _secrets(tmp_path, OTHER="hunter2")

    with pytest.raises(runtime.MissingSecretError) as raised:
        _ = secrets.get("api_key")

    assert raised.value.name == "api_key"
    assert "hunter2" not in str(raised.value)


@pytest.mark.parametrize("name", ["../etc/passwd", "a/b", "", "1key", "key-name"])
def test_secret_names_must_be_identifiers(tmp_path: pathlib.Path, name: str) -> None:
    with pytest.raises(ValueError):
        _ = _secrets(tmp_path).get(name)


def test_secrets_repr_lists_no_values(tmp_path: pathlib.Path) -> None:
    secrets = _secrets(tmp_path, API_KEY="hunter2")

    assert "hunter2" not in repr(secrets)
    assert "API_KEY" not in repr(secrets)


def test_config_is_a_frozen_snapshot_that_hides_the_environment(tmp_path: pathlib.Path) -> None:
    environ = {"API_KEY": "hunter2"}
    config = runtime.Config(
        name="builder_bot",
        instance_id="host-1",
        data_dir=tmp_path,
        environ=environ,
        secrets=_secrets(tmp_path),
    )
    environ["API_KEY"] = "changed"

    assert config.environ == {"API_KEY": "hunter2"}
    with pytest.raises(TypeError):
        typing.cast(dict[str, str], config.environ)["API_KEY"] = "mutated"
    with pytest.raises(pydantic.ValidationError):
        setattr(config, "name", "other")
    assert "hunter2" not in repr(config)


@pytest.mark.parametrize("field", ["name", "instance_id"])
def test_config_requires_names(tmp_path: pathlib.Path, field: str) -> None:
    values: dict[str, object] = {
        "name": "builder_bot",
        "instance_id": "host-1",
        "data_dir": tmp_path,
        "environ": {},
        "secrets": _secrets(tmp_path),
    }
    values[field] = ""

    with pytest.raises(pydantic.ValidationError):
        _ = runtime.Config.model_validate(values)
