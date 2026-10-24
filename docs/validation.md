# Validation scope

Release preparation verified:

- CPU-only unit tests for language restrictions, deterministic checkpoint selection, ST-only project registrations, argument construction, and paths containing spaces.
- Real SimulEval argument parsing for all three languages and the SimulSense, fixed, wait-k, and VAD policies.
- Offline imports of the public models and evaluation agents, without downloading backbones.
- Loading all three exported translation checkpoint/task configurations and their 151,675-entry vocabulary.
- Loading the exported detector and running a finite CPU forward pass.
- Exact learned-tensor equality for translation projections, and byte equality for copied detector/adapter files.
- Source syntax checks, release dependency lock consistency, and a Fairseq wheel build from source.
- Pattern scans for credentials/private machine paths in source and checkpoint metadata; source archives exclude local binary extensions, data, logs, environments, and Git history.

Runtime checks and the wheel build used the existing Linux/Python 3.10.18 environment with PyTorch 2.7.0, Transformers 4.51.3, PEFT 0.17.0, NumPy 1.26.4, and Cython 3.2.3. The release pins those core package versions and build tools. A complete fresh-environment installation of that lock has not been executed; passing imports in the research environment does not prove all dependencies are sufficient in isolation.

Full training, GPU translation evaluation, GPU memory profiling, and reproduction of paper tables were not performed. CPU tests and checkpoint integrity do not establish translation quality or end-to-end CUDA correctness. For optional runtime integration checks, install the release dependencies before running `python -m unittest discover -s tests -v`.
