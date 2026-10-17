"""A quiet bot and a shared actor whose stop ignores cancellation and is never released."""

from mosfet import bot
from mosfet.bot import testing

from tests.runtime.fixtures import doubles

if __name__ == "__main__":
    bot.run(
        bot.define("quiet", bot.cognition(testing.Scripted()), bot.allow("no_devices")),
        actors=(doubles.StubbornActor("stubborn", []),),
    )
