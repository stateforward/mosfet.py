"""Bot files ``tests/bot/run`` runs as ``python -m tests.bot.run.fixtures.<name>`` child processes.

Each package's ``__main__`` defines one bot and runs it with ``bot.run``, as a bot file does; their parts
are the doubles in ``tests.runtime.fixtures.doubles``, each with one behavior a test needs.
"""
