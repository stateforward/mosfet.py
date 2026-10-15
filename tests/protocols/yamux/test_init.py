import mosfet.protocols.yamux as yamux


def test_init_exports_namespace_scoped_public_api() -> None:
    for name in (
        "Client",
        "Server",
        "Session",
        "SessionSnapshot",
        "FrameData",
        "Stream",
        "StreamSnapshot",
        "StreamState",
        "OpenStreamEvent",
        "OpenStreamFailedEvent",
        "TransmitEvent",
        "StreamAcceptedEvent",
        "EndedEvent",
        "CloseEvent",
        "ResetEvent",
        "HandOverEvent",
        "FrameOutEvent",
        "StreamEndedEvent",
        "ViolationEvent",
    ):
        assert name in yamux.__all__
        assert hasattr(yamux, name)
    for removed in ("FrameStream", "ReadStream", "SendDataEvent", "SendData", "CloseStreamEvent", "ResetStreamEvent"):
        assert removed not in yamux.__all__
        assert not hasattr(yamux, removed)

    assert not any(name.startswith("Yamux") for name in yamux.__all__)
    assert not any(name.startswith("YAMUX_") for name in yamux.__all__)
    assert not any("Endpoint" in name for name in yamux.__all__)
    assert not any(name.startswith("STREAM_") for name in yamux.__all__)
