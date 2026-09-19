# Smart phone acoustic assets

The call sounds (ring, busy, reorder) are the phone's own; see `mosfet/devices/phone/assets/SOURCES.md`.

## notification.wav

- **Content:** Handset notification ding, 0.5 s, mono PCM 16 kHz 16-bit: two struck sine notes,
  E6 (1318.5 Hz) for 0.15 s then A6 (1760 Hz) for 0.35 s, each decaying exponentially with an
  80 ms time constant.
- **Source:** Generated, not sampled. Unlike the call-progress tones there is no published
  specification: every handset maker designs its own alert. What they share, and what this file
  keeps, is the shape — a short, bright, rising chime that fetches attention and then stops. A
  maker's own alert sound is a copyrighted recording, so a generic synthesized chime is the
  honest artifact rather than a stand-in for somebody's product.
- **Generation:** run from the repository root with the interpreter's `wave` and `math` modules
  only. Each note starts at 0.5 of full scale (−6 dBFS peak) and decays, so nothing clips.

  ```
  python3 - <<'PY'
  import math, wave

  def notification_ding(path, notes, rate=16000, amp=0.5, decay_s=0.08):
      """Two-note handset notification ding: struck sine notes, each decaying exponentially."""
      frames = bytearray()
      for freq, dur_s in notes:
          for n in range(round(rate * dur_s)):
              t = n / rate
              level = amp * math.exp(-t / decay_s) * math.sin(2 * math.pi * freq * t)
              frames += round(level * 32767).to_bytes(2, "little", signed=True)
      with wave.open(path, "wb") as out:
          out.setnchannels(1)
          out.setsampwidth(2)
          out.setframerate(rate)
          out.writeframes(bytes(frames))

  notification_ding("src/mosfet/devices/smart_phone/assets/notification.wav", ((1318.5, 0.15), (1760.0, 0.35)))
  PY
  ```

- **Use:** Elevated as `environment.sound` (`kind` `phone.notification`) whenever the smart
  phone notifies. One generic ding for every notification, as on a real handset; what caused it travels
  on the sound as the originating phone event (today only a new text message, `phone.sms.text`).
