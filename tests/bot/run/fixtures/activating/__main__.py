"""A bot that never finishes activating; ``<data dir>/activating`` appears once it has begun."""

from mosfet import bot
from mosfet.bot import testing

from tests.runtime.fixtures import doubles

if __name__ == "__main__":
    process = bot.process()
    marker = process.data_dir / "activating"
    bot.run(
        bot.define(
            "slow",
            bot.cognition(reasoning=testing.Scripted(), reflex=testing.Decider()),
            bot.attach(doubles.MarkingDevice(marker)),
        ),
        process=process,
    )
