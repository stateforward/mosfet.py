"""Package metadata sanity: name, version, and public surface."""

import bot.providers.typesafe as package


def test_package_name_and_version() -> None:
    assert package.__name__.endswith("bot.providers.typesafe")
    assert package.__version__ == "0.1.0"


def test_public_surface_is_minimal_and_importable() -> None:
    for name in package.__all__:
        assert hasattr(package, name), name
    assert set(package.__all__) == {
        "AsyncSystemOneClient",
        "SystemOneError",
        "ProcessingError",
        "Processor",
        "__version__",
    }
