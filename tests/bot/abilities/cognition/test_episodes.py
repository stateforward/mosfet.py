"""Cognitive episodes: prior turns stored in and recalled from bot memory."""

from mosfet.abilities import memory
from mosfet.abilities.cognition import episodes


def test_episode_recall_returns_the_most_recent_window_oldest_first() -> None:
    """Once more episodes exist than the window holds, recall still reaches the latest turns.

    Rows written within the same second tie on ``created_at`` and fall back to ``memory_id``, which
    these rows carry in insertion order.
    """

    store = memory.Memory()
    for index in range(60):
        _ = store.execute(
            episodes.episode_insert_input(
                episodes.CognitiveEpisode(focus=f"device-{index:02d}", stimulus_name="environment.sound", output=()),
                context_ref="phone",
                memory_id=f"episode-{index:02d}",
            )
        )

    recalled = episodes.episodes_from_output(store.execute(episodes.episode_select_input(context_ref="phone")))

    assert [episode.focus for episode in recalled] == [f"device-{index:02d}" for index in range(10, 60)]
