import datetime

import hsm

from mosfet.protocols.yamux.frame import INITIAL_STREAM_WINDOW
from mosfet.protocols.yamux.session import DEFAULT_PING_TIMEOUT, Session


class Client(Session):
    """Client-role Yamux session: it opens odd stream ids. ``connection`` owns the socket."""

    def __init__(
        self,
        *,
        connection: hsm.Instance,
        initial_stream_window: int = INITIAL_STREAM_WINDOW,
        ping_timeout: datetime.timedelta = DEFAULT_PING_TIMEOUT,
    ) -> None:
        super().__init__(
            role="client",
            connection=connection,
            initial_stream_window=initial_stream_window,
            ping_timeout=ping_timeout,
        )
