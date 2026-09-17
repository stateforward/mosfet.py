import mosfet.protocols.yamux as yamux


def test_init_exports_namespace_scoped_public_api() -> None:
    assert yamux.Client.__name__ == "Client"
    assert yamux.Server.__name__ == "Server"
    assert yamux.Session.__name__ == "Session"
    assert yamux.SessionSnapshot.__name__ == "SessionSnapshot"
    assert yamux.FrameData.__name__ == "FrameData"
    assert yamux.FrameStream.__name__ == "FrameStream"
    assert yamux.ReadStream.__name__ == "ReadStream"
    assert yamux.Stream.__name__ == "Stream"
    assert yamux.StreamSnapshot.__name__ == "StreamSnapshot"

    assert not any(name.startswith("Yamux") for name in yamux.__all__)
    assert not any(name.startswith("YAMUX_") for name in yamux.__all__)
    assert not any("Endpoint" in name for name in yamux.__all__)
    assert not any(name.startswith("STREAM_") for name in yamux.__all__)
