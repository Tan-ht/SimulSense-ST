# SimulSense-ST: Learning When to Speak

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Xplore-brightgreen)](https://ieeexplore.ieee.org/document/11732033)
[![Models](https://img.shields.io/badge/Models-Hugging%20Face-blue)](https://huggingface.co/tan-ht/SimulSense)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

Speech-to-text release of **Learning When to Speak: Human-Inspired Unit Sensing
for Machine Simultaneous Interpretation**.
SimulSense detects interpretation units in incoming speech to decide when a
speech translation model should produce text. This repository provides the
shared unit detector, Whisper + Qwen translation models, training code, and
simultaneous and continuous-stream evaluation for **English → Chinese, German,
and Japanese**.

**[Quick Start](#-quick-start)** · **[Models](#models)** ·
**[Data](#data-layout)** · **[Inference](#inference)** ·
**[Training](#model-training)** · **[Citation](#citation)**

---

## 🚀 Quick Start

### 1. Set up the environment

Use **Linux, Python 3.10, and an NVIDIA GPU** for model execution. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/), a C/C++ compiler,
Python development headers, `ffmpeg`, and `libsndfile1`, then run:

```bash
git clone https://github.com/Tan-ht/SimulSense-ST.git
cd SimulSense-ST
uv sync --frozen
source .venv/bin/activate
```

The environment pins PyTorch 2.7.0, Transformers 4.51.3, and PEFT 0.17.0.
Use the vendored Fairseq, SimulEval, and SimulStream versions: the system relies
on local modifications to these toolkits. FlashAttention is optional; attention
uses SDPA when FlashAttention is unavailable.

### 2. Download the models

The model release is prepared for
[**tan-ht/SimulSense**](https://huggingface.co/tan-ht/SimulSense).
**Public weight availability has not yet been verified.** Once the files are
public, download the detector and all three translation models with:

```bash
python scripts/download_models.py --target all
```

For just English → German and the shared detector:

```bash
python scripts/download_models.py --target de
```

The default destination is `checkpoints/`. Use `--output /path/to/checkpoints`
to change it, or `--revision <Hub-commit>` to pin a specific model revision.

> The release contains the detector, adapters, and custom model components.
> Qwen3-8B and Whisper large-v3 are downloaded separately on first use.
> Allow additional disk space and GPU memory for these base models.

### 3. Prepare evaluation data and run

Prepare `source.txt` and `target.txt` as described in [Data layout](#data-layout).
For a short first run, use a small set of paired examples, then evaluate
English → German on GPU `0`:

```bash
python scripts/run.py evaluate \
  --target de \
  --checkpoint checkpoints/translation/en-de \
  --detector checkpoints/SimulSense \
  --source data/en-de/source.txt \
  --reference data/en-de/target.txt \
  --output outputs/en-de-simulsense \
  --device 0 --chunk-ms 2000 --threshold 1.0
```

Replace `de` with `zh` or `ja` and use the matching checkpoint and evaluation
files. Results are written to the directory specified by `--output`; each run
requires a new or empty directory.

## Models

The release contains one shared interpretation-unit detector and three
speech-to-text translation models. The following paths are relative to the
model repository; download sizes refer to the locally prepared release files.

| Model | Directory | Approximate size |
| --- | --- | ---: |
| Shared English unit detector | `SimulSense/` | 1.21 GB |
| English → Chinese | `translation/en-zh/` | 2.59 GB |
| English → German | `translation/en-de/` | 2.59 GB |
| English → Japanese | `translation/en-ja/` | 2.59 GB |

The four groups total approximately **9.0 GB**, excluding the base models.
Translation uses **[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)** and
**[Whisper large-v3](https://huggingface.co/openai/whisper-large-v3)**.
The download layout is:

```text
checkpoints/
├── SimulSense/
│   ├── sense_units_detector.pt
│   ├── other_states.pt
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── translation/
    ├── en-zh/
    ├── en-de/
    │   ├── model.pt
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    └── en-ja/
```

Pass the **entire translation directory** to `--checkpoint` so the loader can
read both the custom checkpoint and the PEFT adapter. These are custom
Fairseq + PEFT exports; use this repository's loaders.

The exports omit optimizer state, training logs, and machine-specific paths.
They are intended for inference and are not resumable training snapshots.
See [model details](docs/models.md) for checkpoint contents and compatibility.

## Configuration

The public entry point is `scripts/run.py`. Dataset, checkpoint, and output
paths are passed explicitly:

| Option | Default | Purpose |
| --- | --- | --- |
| `--target` | Required for translation | Target language: `zh`, `de`, or `ja` |
| `--checkpoint` | Required for evaluation | Translation checkpoint and adapter directory |
| `--detector` | Required for SimulSense inference | Shared unit detector directory |
| `--output` | Required | Training checkpoints or evaluation results |
| `--device` | `0` | Evaluation GPU index |
| `--chunk-ms` | `2000` | Incoming audio chunk duration in milliseconds |
| `--threshold` | `1.0` | SimulSense unit detection threshold |
| `--decode` | `greedy` | Decoding method: `greedy`, `beam`, or `sample` |

To use locally stored base models, set:

```bash
export SIMULSENSE_LLM_PATH=/models/Qwen3-8B
export SIMULSENSE_WHISPER_PATH=/models/whisper-large-v3
```

Add `--dry-run` to a training or evaluation command to inspect its generated
commands before loading models. Input paths and checkpoint layout must still
be valid. Run `python scripts/run.py <command> --help` for all options.

## Data layout

Datasets and evaluation audio are obtained separately. The repository includes
the vocabulary and prompt configuration used by the public launchers.
Detailed manifest formats are documented in [docs/data.md](docs/data.md).

Simultaneous evaluation expects:

```text
data/en-<target-language>/
├── source.txt              # One absolute WAV path per line
└── target.txt              # One reference translation per line
```

Use **mono, 16 kHz audio**, with each example at most **30 seconds** long.
Source and reference lists must contain matching nonempty line counts and
preserve pairing order. Audio paths must resolve on the machine running
inference.

Continuous-stream evaluation expects:

```text
data/stream/en-<target-language>/
├── src.txt                 # One absolute long-recording path per line
├── tgt_segments.txt        # One reference translation per annotated segment
├── segments.yaml           # Recording names, offsets, and durations
└── src_segments.txt        # Optional source transcripts for COMET
```

Segment annotations and references must have identical ordering, grouped by
recording in the order used by `src.txt`. The streaming processor maintains a
rolling context, so long recordings do not need to be cut into 30-second files.
See [streaming data conventions](docs/data.md#continuous-stream-evaluation).

## Inference

Run these commands from the repository root with `.venv` activated and the
models and evaluation data prepared.

### Simultaneous speech-to-text translation

Use the `evaluate` command in Quick Start for SimulSense inference. The launcher
chooses the evaluation settings for each translation direction:

| Target | Latency unit | BLEU tokenizer |
| --- | --- | --- |
| Chinese (`zh`) | Character | `zh` |
| German (`de`) | Word | `13a` |
| Japanese (`ja`) | Character | `ja-mecab` |

Sweep `--threshold` across separate runs to explore the quality–latency
tradeoff. The example chunk size and threshold are usage settings, not claims
to reproduce a paper table.

The same evaluation entry point supports these baseline policies:

| Policy | Options |
| --- | --- |
| Fixed chunk | `--policy fixed` |
| Wait-k | `--policy waitk --waitk 3` |
| Voice activity detection | `--policy vad --vad-threshold 0.5` |

Add the selected options to the `evaluate` command. These baselines do not
require `--detector`.

### Continuous-stream evaluation

Install the additional scoring dependencies:

```bash
uv sync --frozen --extra streaming-metrics
```

Then run English → German streaming evaluation on GPU `0`:

```bash
python scripts/run.py stream \
  --target de \
  --checkpoint checkpoints/translation/en-de \
  --detector checkpoints/SimulSense \
  --source data/stream/en-de/src.txt \
  --reference data/stream/en-de/tgt_segments.txt \
  --audio-definition data/stream/en-de/segments.yaml \
  --output outputs/en-de-stream \
  --device 0 --chunk-ms 2000 --threshold 1.0
```

This runs SimulStream inference, StreamLAAL, resegmented BLEU, and timing
statistics. Add `--transcript data/stream/en-de/src_segments.txt` to compute
COMET. Predictions, runtime configuration, and metric logs are saved under
`--output`. The `mweralign` dependency supplies reference resegmentation and
may require platform-specific build dependencies.

## Model training

Prepare Fairseq-style TSV manifests as described in
[data formats](docs/data.md). Both trainers operate on raw speech and use the
provided vocabulary and prompt configuration.

### Train the speech translation model

For English → Japanese on GPU `0`:

```bash
python scripts/run.py train-st \
  --target ja --data /datasets/en-ja \
  --train-subset train --valid-subset dev \
  --devices 0 --output checkpoints/train-en-ja
```

Use `--target zh` or `--target de` with the corresponding data for the other
directions. Set `--devices 0,1` for two GPUs. Experiment tracking is disabled
by default.

### Train the unit detector

```bash
python scripts/run.py train-detector \
  --data /datasets/english-units \
  --train-subset train --valid-subset dev \
  --devices 0 --output checkpoints/train-simulsense
```

Detector supervision requires English transcripts with `<seg>` unit boundaries.
The annotated training corpus and automatic generation of the paper's unit
annotations are not included. The released detector retains its original
expert tensor dimensions for checkpoint compatibility; public input and
training interfaces accept English source speech. This English-only training
example does not reproduce the multilingual training history of the released
weights.

## Repository layout

```text
SimulSense-ST/
├── main_scripts/           # Models, datasets, criteria, trainers, and agents
├── scripts/                # Training, evaluation, and model download launchers
├── fairseq/                # Vendored Fairseq with local modifications
├── SimulEval/              # Vendored simultaneous evaluation toolkit
├── simulstream/            # Vendored continuous-stream evaluation toolkit
├── docs/                   # Data, model, and validation documentation
└── tests/                  # Release and command validation
```

## Release checks

Run the lightweight release checks:

```bash
python -m unittest discover -s tests -v
uv lock --check --offline
```

The tests cover release scope and training/evaluation command construction;
the runtime integration check additionally uses the installed SimulEval parser.
See [validation scope](docs/validation.md) for the recorded checks and limits.
A fresh environment installation, full training, end-to-end GPU translation,
GPU-memory profiling, and paper metric reproduction have not been validated
for this release.

## Third-party components

Project code retains the existing [MIT license](LICENSE). Vendored and adapted
components retain their upstream licenses, including the bundled SimulEval
license. Source links and notices are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Dataset audio, transcripts, and complete base-model weights are not bundled.
External datasets and models remain subject to their respective terms.

## Citation

If you use this repository, please cite **Learning When to Speak:
Human-Inspired Unit Sensing for Machine Simultaneous Interpretation**.
Use the publisher-provided citation from the
[paper page on IEEE Xplore](https://ieeexplore.ieee.org/document/11732033).
