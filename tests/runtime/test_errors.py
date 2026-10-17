import typing

from mosfet import runtime


def test_boot_failure_kinds_are_the_closed_set_bot_run_reports() -> None:
    assert typing.get_args(runtime.BootFailureKind) == ("constructor_raised", "invalid_models")


def test_missing_secret_is_a_runtime_error_that_names_only_the_secret() -> None:
    error = runtime.MissingSecretError("livekit_api_key")

    assert isinstance(error, runtime.Error)
    assert error.name == "livekit_api_key"
    assert str(error) == "secret 'livekit_api_key' is not provided"


def test_usage_error_is_a_runtime_error() -> None:
    assert issubclass(runtime.UsageError, runtime.Error)
