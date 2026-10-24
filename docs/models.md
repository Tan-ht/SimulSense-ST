# Models

The Hub repository `tan-ht/SimulSense` is intended to contain four models:

| Directory | Role | Required base models |
| --- | --- | --- |
| `SimulSense/` | Unit detector used by all three directions | Whisper large-v3 encoder |
| `translation/en-zh/` | English speech → Chinese text | Whisper large-v3 + Qwen3-8B |
| `translation/en-de/` | English speech → German text | Whisper large-v3 + Qwen3-8B |
| `translation/en-ja/` | English speech → Japanese text | Whisper large-v3 + Qwen3-8B |

Each translation directory contains `model.pt` (speech normalization/projection parameters and portable configuration), `adapter_model.safetensors` (LoRA and saved embedding parameters), and `adapter_config.json`. These are custom Fairseq + PEFT checkpoints, not standalone `transformers.pipeline` models. Use this release's loaders.

The detector contains `sense_units_detector.pt`, `other_states.pt`, and its Whisper decoder adapter/config. ST inference uses the detector tensors and configuration; decoder adapter weights are retained for the original detector model. The original expert tensor dimensions are unchanged; this release exposes English source speech only.

Inference exports remove optimizer state, metrics, and private paths. They preserve all saved inference tensors and adapter bytes. Checksums and export validation are included in the Hub bundle. Full frozen backbones are not redistributed. Resume training from a private training checkpoint, not an inference export.

Checkpoint files are PyTorch serialized objects and should only be loaded from a trusted source. Known limitations include dependence on source-domain/unit annotations, translation errors, and possible hallucinations inherited from the speech and language backbones. No deployment fitness or benchmark replication claim is made by this source release.

Project-authored additions retain MIT licensing. The base models and adapted model implementations have their own Apache-2.0 notices; see `THIRD_PARTY_NOTICES.md` and the upstream model cards.
