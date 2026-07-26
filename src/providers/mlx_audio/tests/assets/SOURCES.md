# MLX Audio voice detection test assets

## speech.wav

- **Content:** Real human speech (~1.5 s, mono PCM s16, 16 kHz) — Neil Armstrong's 1969 Apollo 11
  lunar-surface transmission.
- **Source:** Trimmed from `Neil Armstrong small step.wav` on Wikimedia Commons
  (https://commons.wikimedia.org/wiki/File:Neil_Armstrong_small_step.wav), a NASA recording in the
  public domain as a work of the U.S. federal government. The 24.1 s original (mono, 8-bit,
  11.025 kHz) was cut at 3.75–5.25 s and converted with
  `ffmpeg -ss 3.75 -t 1.5 -ac 1 -ar 16000 -sample_fmt s16 -c:a pcm_s16le`, giving the same encoding
  as `src/bot/devices/phone/assets/ring.wav` so the two clips differ only in content.
- **Use:** Positive control for the live voice-detection test. The claim that a real Silero VAD
  hears no voice in the phone ring is only worth anything when the same detector is shown to answer
  `True` on real speech, so this clip is what separates "correctly heard no voice" from "the model
  never ran."
