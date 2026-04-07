# fl-compression-study

A research codebase comparing quantization vs SZ lossy compression for gradient compression in federated learning. Experiments run on Tiny ImageNet (ResNet-18) and ETTh1 time series (2-layer LSTM), with a distributed training loop using PyTorch DDP and the gloo backend.

## Setup

```bash
git clone <repo-url>
cd fl-compression-study
pip install -r requirements.txt
python scripts/download_data.py
```

## Folder structure

```
configs/        Experiment YAML configs (base + per-experiment overrides)
scripts/        Entry points: train.py, run_sweep.sh, download_data.py
src/
  compressors/  NoCompression, QuantizationCompressor, SZCompressor, HybridCompressor
  data/         Tiny ImageNet and ETTh1 data loaders
  models/       ResNet-18 and TimeSeriesLSTM
  training/     Trainer, comm hook, metrics
  utils/        Logger, Timer, seed, distributed helpers
tests/          Unit tests for compressors, metrics, logger
results/        CSV output (gitignored)
```

## Run a single experiment

```bash
torchrun --nproc_per_node=4 scripts/train.py --config configs/exp_baseline.yaml
```

## Run the full sweep

```bash
bash scripts/run_sweep.sh
```

Results are written to `results/` as CSV files and timestamped in `sweep_log.txt`.
