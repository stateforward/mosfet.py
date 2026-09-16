from textual.widgets import Input

from playground.directory_suggester import DirectorySuggester


class DirectoryInput(Input):
    def on_mount(self) -> None:
        self.suggester = DirectorySuggester()
