from typing import Iterable
from textual.widgets import Markdown

from botbot.widgets.menus import MenuItem


class MarkdownNote(Markdown):
    def get_block_menu(self) -> Iterable[MenuItem]:
        return
        yield

    def get_block_content(self, destination: str) -> str | None:
        return self.source
