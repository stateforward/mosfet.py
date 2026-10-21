"""A quiet bot run with a boundary observer that writes each body request it sees to ``<data dir>/observed``."""

import typing

import hsm

import mosfet
from mosfet import bot
from mosfet.bot import testing

if __name__ == "__main__":
    process = bot.process()
    record = process.data_dir / "observed"

    def watch(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> None:
        del ctx, instance
        with record.open("a", encoding="utf-8") as written:
            _ = written.write(typing.cast(dict[str, hsm.Event[typing.Any]], event.data)["event"].name + "\n")

    bot.run(
        bot.define(
            "quiet", bot.cognition(reasoning=testing.Scripted(), reflex=testing.Decider()), bot.allow("no_devices")
        ),
        process=process,
        observe=(bot.observe(mosfet.ActivateEvent, mosfet.DeactivateEvent, watch),),
    )
