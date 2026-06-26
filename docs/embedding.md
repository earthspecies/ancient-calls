# Embeddings (bring your own)

The pipeline starts from **embeddings**, not audio (see
[`data-format.md`](data-format.md)). You produce an `(N, D)` float32 array,
L2-normalized per row, with a row-aligned metadata table. Any model works —
the rest of the pipeline never sees audio.

## Today: produce embeddings yourself

Embed your audio with whatever model you use (BirdNET, a BEATS/EfficientNet
variant, a generic audio encoder, …), then normalize and save:

```python
import numpy as np
emb = your_model.embed(clips)            # (N, D)
emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
np.save("embeddings.npy", emb.astype("float32"))
```

Write a metadata table with one row per clip (same order), with at least
`example_id` and `species`.

## Planned: a config-driven embedding producer (not yet implemented)

A future optional component (`ancientcalls[embedding]` extra) will turn audio
into conforming embeddings, with the backend chosen in config, e.g.:

```yaml
embedding:
  backend: birdnet        # birdnet | <other backends> | byo
  audio_root: data/audio
  sample_rate: 48000
```

The seam is intentionally clean: a producer reads an audio manifest, runs a
chosen backend, L2-normalizes, and writes exactly the `embeddings.npy` +
metadata that the pipeline consumes. Because the contract is the only coupling
point, backends can be added (or a dataset we used linked) without touching the
analysis pipeline. This is scoped but deliberately deferred.
