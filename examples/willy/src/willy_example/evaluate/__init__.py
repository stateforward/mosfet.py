"""A voice agent that evaluates Willy: it joins a room, holds spoken conversations with Willy, and grades them.

    python -m willy_example.evaluate --url ws://127.0.0.1:8765 --peer eval-room \\
        --secret-file examples/willy/secrets/computer_token_secret --out /tmp/willy-eval [--scenario ID ...]

``scenarios`` holds what each conversation is for, as data.
"""
