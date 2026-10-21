"""A voice agent that evaluates the companion: it joins one of the companion's rooms, talks with the companion aloud,
and grades it.

    python -m companion_example.evaluate --url ws://127.0.0.1:8765 --peer eval-room \\
        --secret-file examples/companion/secrets/computer_token_secret --out /tmp/companion-eval [--scenario ID ...]

The evaluator is a room's far side, exactly as ``companion_example.room`` is, so the companion hears it through its ear
and answers through its mouth like anyone in the house. Everything it uses is local: the people speak through the
lab's speech engine in its catalog voices (never the companion's own), the companion's replies are heard by the lab's
batch speech to text, and the persona and the judge run on the companion's language model unless another is named.

* ``scenarios``: what each conversation is for, as data (``scenarios.json``).
* ``turns``: the person's turn-taking (speak once the companion's voice has died away and the echo window passed; wait
  for the companion's reply until it pauses) and what each exchange measures.
* ``sound``: the evaluator's audio at the room's rate, mixed and written as WAV.
* ``minds``: the persona that plays the people and the judge that grades the companion.
* ``record`` and ``report``: what was said and found; ``scores.json``, the summary line and ``report.html``.
* ``conversation``: one whole evaluation.
* ``cli``: the command line.
"""
