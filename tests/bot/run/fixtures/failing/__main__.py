"""A bot whose device's bring-up always fails, so its activation fails until its budget is spent."""

from mosfet import bot
from mosfet.bot import testing

from tests.runtime.fixtures import doubles

if __name__ == "__main__":
    bot.run(bot.define("failing", bot.cognition(testing.Scripted()), bot.attach(doubles.FailingDevice())))
