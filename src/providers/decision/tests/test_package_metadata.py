"""Package metadata sanity: name, version, and public surface."""

from mosfet.providers import decision


def test_package_name_and_version() -> None:
    assert decision.__name__ == "mosfet.providers.decision"
    assert decision.__version__ == "0.1.0"


def test_public_surface_is_minimal_and_importable() -> None:
    for name in decision.__all__:
        assert hasattr(decision, name), name
    assert set(decision.__all__) == {
        "Choice",
        "ChoiceAnswer",
        "Client",
        "Error",
        "PayloadGroundingError",
        "ProcessingError",
        "Processor",
        "Request",
        "Response",
        "__version__",
    }
