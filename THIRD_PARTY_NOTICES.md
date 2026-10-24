# Third-party notices

The root MIT license covers project-authored code. It does not replace licenses on vendored or derived files.

| Component | Origin | License retained in this distribution |
| --- | --- | --- |
| Fairseq (locally modified) | https://github.com/facebookresearch/fairseq | `fairseq/LICENSE` (MIT); nested components keep their own notices |
| SimulEval (locally modified) | https://github.com/facebookresearch/SimulEval | `SimulEval/LICENSE` (CC BY-SA 4.0; preserve per-file notices) |
| SimulStream (locally modified) | https://github.com/hlt-mt/simulstream | `simulstream/LICENSE` (Apache-2.0) |
| Qwen3 model implementation | Hugging Face Transformers / Qwen | `licenses/transformers-APACHE-2.0.txt` and headers in `main_scripts/models/modeling_qwen3.py` |
| Whisper model implementation | Hugging Face Transformers / OpenAI | `licenses/transformers-APACHE-2.0.txt` and headers in `main_scripts/models/modeling_whisper.py` |

The two adapted Transformers files include local changes and retain their upstream copyright headers. Fairseq-derived dataset/training files retain their original headers. The bundled toolkit code is a snapshot of the research checkout; an exact upstream commit is not asserted where provenance was not recorded.

Base models are downloaded separately: [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) and [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) are distributed by their respective publishers under Apache-2.0. Base-model terms and notices continue to apply to the derived adapters. No training or evaluation datasets are distributed here.
