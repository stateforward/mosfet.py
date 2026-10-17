# Bot diagnostics

<!-- Generated from mosfet.bot.diagnostics.EXPLANATIONS. Do not edit by hand; regenerate with
python -c 'from mosfet.bot import diagnostics; print(diagnostics.document(), end="")' > docs/bot_diagnostics.md -->

`bot.define` reports every mistake in a bot file at once, the way rustc does: each with a stable
code, the source span, notes on why, and a `help:` that shows the rewrite. Errors (`B00xx` from
`bot.define`, `B01xx` from `bot.start` and `bot.run`) stop the bot; lints (snake_case names) are
warnings. `bot.explain("<code>")` prints any entry below.

