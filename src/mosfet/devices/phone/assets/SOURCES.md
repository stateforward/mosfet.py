# Phone acoustic assets

## ring.wav

- **Content:** Short landline telephone ring clip (~1.5 s, mono PCM 16 kHz).
- **Source:** Derived from a publicly available UK telephone ringtone sample at
  https://telephonesuk.org.uk/sounds/ (`ringtone.wav`), trimmed and resampled for
  tests and ring elevation.
- **Use:** Elevated as `environment.sound` when the phone commits ringing so input
  listening can classify real acoustic energy.

## busy.wav, reorder.wav

- **Content:** US call-progress tones, 2.0 s each, mono PCM 16 kHz 16-bit.
  - `busy.wav` — busy tone (slow busy): 480 Hz + 620 Hz, 0.5 s on / 0.5 s off, two cycles.
  - `reorder.wav` — reorder tone (fast busy): 480 Hz + 620 Hz, 0.25 s on / 0.25 s off, four cycles.
- **Source:** Generated from the published specification, not sampled from a recording. These
  tones genuinely *are* generated signals in the real network — a switch gates a pair of
  continuously running oscillators — so a generated file is the honest artifact here rather than
  a stand-in for one. Frequencies and cadences are the North American precise-tone plan
  (ITU-T E.180 / Bellcore call-progress tones): both tones use the same 480+620 Hz pair and
  differ only in cadence, which is exactly why a caller can tell them apart by ear.
- **Generation:** run from the repository root with the interpreter's `wave` and `math` modules
  only. Each component sits at 0.25 of full scale, so the summed pair peaks at −6 dBFS with no
  clipping. Time runs continuously across the whole file and the interrupter chops it, so the
  oscillator phase is continuous across bursts the way a real tone plant's is, and the gate edges
  are abrupt because a real interrupter's are.

  ```
  python3 - <<'PY'
  import math, wave

  def call_progress_tone(path, on_s, off_s, cycles, rate=16000, freqs=(480.0, 620.0), amp=0.25):
      """Precise-spec US call-progress tone: continuous oscillators chopped by an interrupter."""
      period, on = round(rate * (on_s + off_s)), round(rate * on_s)
      frames = bytearray()
      for n in range(period * cycles):
          level = sum(amp * math.sin(2 * math.pi * f * n / rate) for f in freqs) if n % period < on else 0.0
          frames += round(level * 32767).to_bytes(2, "little", signed=True)
      with wave.open(path, "wb") as out:
          out.setnchannels(1)
          out.setsampwidth(2)
          out.setframerate(rate)
          out.writeframes(bytes(frames))

  call_progress_tone("src/mosfet/devices/phone/assets/busy.wav", 0.5, 0.5, 2)
  call_progress_tone("src/mosfet/devices/phone/assets/reorder.wav", 0.25, 0.25, 4)
  PY
  ```

- **Use:** Elevated as `environment.sound` when a dial attempt produces no call and the exchange
  would have put a tone in the caller's ear. Which tone is the network's verdict, not the
  handset's: a refused or engaged line gives busy, and every other way the network fails to
  complete a call gives reorder.

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

  notification_ding("src/mosfet/devices/phone/assets/notification.wav", ((1318.5, 0.15), (1760.0, 0.35)))
  PY
  ```

- **Use:** Elevated as `environment.sound` (`kind` `phone.notification`) whenever the phone
  notifies. One generic ding for every notification, as on a real handset; what caused it travels
  on the sound as the originating phone event (today only a new text message, `phone.sms.text`).
