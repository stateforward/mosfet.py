# bot-provider-pyannote

The pyannote provider adapts `pyannote.audio` speaker diarization pipelines
and speaker-embedding inference to the provider-neutral voice contracts in
`stateforward.bot`.

The package declares the approved runtime dependency:

```text
pyannote.audio>=4.0.7,<5.0.0
```

The package imports pyannote lazily, so its deterministic tests do not need
model weights, credentials, network access, or `ffmpeg`.

The adapters accept injected pipeline/inference instances or loaders. They
stage typed raw signed 16-bit PCM input as temporary WAV files for pyannote.
The diarizer converts pyannote-like turns into metadata-bearing clipped core
diarization segments; provider labels are used only inside the adapter and are
omitted from the provider-neutral output. The identification `Classifier`
accepts metadata-bearing `VoiceSegment` values and invokes the
`pyannote.audio.Inference(model)` contract once per segment. It converts each
inference result by unwrapping pyannote `SlidingWindowFeature.data` (and
accepting direct `tolist()` test doubles) into a validated core
`VoiceEmbedding` in input order, rejecting non-finite values. Participant
identity resolution is owned downstream by Conversation; this provider only
produces embeddings.

Run the focused tests from the workspace root with:

```sh
PYTHONPATH=src/providers/pyannote/src:src \
  .venv/bin/python -m pytest src/providers/pyannote/tests
```

The default loaders expect the approved runtime and an authenticated
pyannote/Hugging Face model environment. The identification default model is
`pyannote/wespeaker-voxceleb-resnet34-LM`. For tests and custom runtimes,
inject `VoiceDiarizationPipeline` / `VoiceDiarizationPipelineLoader` or
`SpeakerEmbeddingInference` / `SpeakerEmbeddingInferenceLoader` into
`Classifier`.
