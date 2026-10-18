# Willy's secrets

Each secret is one file here, named for it, holding only its value. Nothing in this directory but this README and
`.gitignore` is ever committed.

- `computer_token_secret`: the key every room's far side signs its token with
  (`websocket.HmacTokens(secret).issue(<room>)`). Willy's server refuses a room whose token it did not sign.
  Make one with `python -c "import secrets; print(secrets.token_urlsafe(32))" > secrets/computer_token_secret`.
- `openai_api_key` (optional): the key for the lab's OpenAI-compatible servers, which take none; without it,
  `OPENAI_API_KEY` in `.env` is the placeholder.

Run Willy with `--secrets-dir examples/willy/secrets`, or point `COMPUTER_TOKEN_SECRET_FILE` at the file.
