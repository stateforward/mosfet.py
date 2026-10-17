"""A bot that activates promptly and perceives nothing, so it does nothing until it is stopped."""

from mosfet import bot
from mosfet.bot import testing

if __name__ == "__main__":
    bot.run(bot.define("quiet", bot.cognition(testing.Scripted()), bot.allow("no_devices")))
