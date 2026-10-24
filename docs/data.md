# Data formats

Training and evaluation datasets are supplied by the user. This release does not redistribute CoVoST, MuST-C, ACL audio, transcripts, or private annotations. Example text below is synthetic.

## Translation training

The manifest directory must contain `train.tsv` and `dev.tsv` (or names passed to the launcher). Files are UTF-8, tab-delimited, with a header:

```text
id	audio	n_frames	src_text	tgt_text	src_lang	tgt_lang
example-0001	/absolute/path/to/example.wav	32000	Hello everyone.	大家好。	en	zh
```

Use real tab characters. `n_frames` counts waveform samples for raw audio (32,000 samples = two seconds at 16 kHz). `audio` is an absolute path to mono audio. The loader also accepts Fairseq ZIP references when created according to Fairseq conventions. `src_lang` must be `en`; `tgt_lang` must match the selected training direction (`zh`, `de`, or `ja`). Optional `speaker` identifies the speaker. Target audio is not supported.

The training configuration reads raw audio at 16 kHz. Keep examples within the 30-second Whisper window; the default translation training launcher further caps source length at 200,000 samples, matching the research launcher. `main_scripts/config/speech_llm.yaml` selects the Qwen3 vocabulary. Do not substitute a different backbone/tokenizer without regenerating the vocabulary and adapting the model.

## Detector training

```text
id	audio	n_frames	src_text	lang
example-0001	/absolute/path/to/example.wav	32000	Hello <seg> everyone. <seg>	en
```

`<seg>` marks a human/interpreter unit boundary, not a fixed-duration chunk. The loader maps it to Whisper's `<|nospeech|>` token and derives unit group labels. Boundary annotation generation is not provided. All rows must use English (`en`).

## Simultaneous evaluation

`source.txt` contains one absolute path per line; `target.txt` contains one translation per matching line. No headers. Audio must be mono, 16 kHz, and at most 30 seconds per example. Files must have the same number of nonempty lines and preserve pairing order. A synthetic format example:

```text
# source.txt (omit this comment in the actual file)
/absolute/path/to/example.wav
```

```text
# target.txt (omit this comment in the actual file)
大家好。
```

## Continuous-stream evaluation

`src.txt` lists absolute paths to long recordings, one per line. `tgt_segments.txt` contains one reference translation per annotated segment, and optional `src_segments.txt` contains the corresponding source transcripts. `segments.yaml` is a YAML list whose entries describe the segment boundaries in seconds:

```yaml
- wav: recording.wav
  offset: 0.0
  duration: 2.0
- wav: recording.wav
  offset: 2.0
  duration: 3.0
```

Use the recording basename in `wav`, matching `src.txt`. Keep YAML and reference segments in identical order and group them by recording in the same order as `src.txt`. See the bundled `simulstream/simulstream/metrics/readers.py` for the exact reader contract. The streaming processor keeps a rolling context; it does not require cutting long recordings into 30-second files first.
