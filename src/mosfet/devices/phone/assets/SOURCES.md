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
