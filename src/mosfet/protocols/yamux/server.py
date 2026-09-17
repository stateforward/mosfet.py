import datetime

from mosfet.protocols.yamux.frame import INITIAL_STREAM_WINDOW
from mosfet.protocols.yamux.session import DEFAULT_PING_TIMEOUT, Session


class Server(Session):
    """Server-role Yamux session."""

    def __init__(
        self,
        *,
        initial_stream_window: int = INITIAL_STREAM_WINDOW,
        ping_timeout: datetime.timedelta = DEFAULT_PING_TIMEOUT,
    ) -> None:
        super().__init__(role="server", initial_stream_window=initial_stream_window, ping_timeout=ping_timeout)
