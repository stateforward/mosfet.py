# Willy's secrets

Each secret is one file here, named for it, holding only its value. Nothing in this directory but this README and
`.gitignore` is ever committed.

- `computer_token_secret`: the key every room's far side signs its token with
  (`websocket.HmacTokens(secret).issue(<room>)`). Willy's server refuses a room whose token it did not sign.
  Make one with `python -c "import secrets; print(secrets.token_urlsafe(32))" > secrets/computer_token_secret`.
- `openai_api_key` (optional): the key for the lab's OpenAI-compatible servers, which take none; without it,
  `OPENAI_API_KEY` in `.env` is the placeholder.
- `decision_api_key` (optional): the key of the decision server Willy's reflex decides on, sent as a bearer
  only when this file (or `DECISION_API_KEY`) is given. The local decide `DECISION_BASE_URL` names in
  `.env.example` takes no key, so leave it out for the lab.

Run Willy with `--secrets-dir examples/willy/secrets`, or point `COMPUTER_TOKEN_SECRET_FILE` at the file.
