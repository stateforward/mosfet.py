from mosfet import runtime


def test_boot_error_carries_its_kind_and_message() -> None:
    error = runtime.BootError("factory_raised", "builder_bot.app:build raised ValueError: bad")

    assert isinstance(error, runtime.Error)
    assert error.kind == "factory_raised"
    assert str(error) == "builder_bot.app:build raised ValueError: bad"


def test_missing_secret_is_a_boot_failure_that_names_only_the_secret() -> None:
    error = runtime.MissingSecretError("livekit_api_key")

    assert isinstance(error, runtime.BootError)
    assert error.kind == "missing_secret"
    assert error.name == "livekit_api_key"
    assert str(error) == "secret 'livekit_api_key' is not provided"


def test_usage_error_is_a_runtime_error() -> None:
    assert issubclass(runtime.UsageError, runtime.Error)
    assert not issubclass(runtime.UsageError, runtime.BootError)
