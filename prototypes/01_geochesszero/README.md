
# GeoChessZero — Prototype 1

## Overview

GeoChessZero is a geometry-aware Transformer for chess move prediction.

The research objective is to investigate whether chess-specific geometric attention improves move prediction compared with standard self-attention.

## Models

### GeoChessZero

A Transformer with eight attention heads, including geometry-aware heads based on:

- Rook movement geometry
- Bishop movement geometry
- Knight movement geometry
- King movement geometry

The remaining attention heads use unrestricted attention.

### Standard Transformer

A baseline Transformer using unrestricted self-attention.

Both models share the same general architecture.

## Architecture

- Input: 64 chessboard tokens
- Features per token: 18
- Transformer dimension: 128
- Transformer layers: 4
- Attention heads: 8
- Output actions: 4,272
- Training objective: legal-move-masked cross-entropy

## Dataset

Source: Lichess rated chess games.

| Split | Games | Positions |
|---|---:|---:|
| Training | 80 | 4,877 |
| Validation | 10 | 697 |
| Test | 10 | 619 |
| Total | 100 | 6,193 |

The dataset is split by game to avoid sharing positions from the same game across splits.

## Training

- Framework: PyTorch
- Optimizer: AdamW
- Learning rate: 0.0003
- Weight decay: 0.01
- Batch size: 16
- Epochs: 10
- GPU: NVIDIA RTX 3060 Laptop GPU

## Results

| Metric | GeoChessZero | Standard Transformer |
|---|---:|---:|
| Test Loss | 2.6474 | 2.7794 |
| Top-1 Accuracy | 24.23% | 24.07% |
| Top-5 Accuracy | 61.23% | 52.50% |

GeoChessZero achieved lower test loss and higher move-prediction accuracy on this test set.

These results are preliminary and do not establish stronger chess playing ability.

## Best Checkpoints

- `checkpoints/geochesszero_lichess_best.pt` — epoch 6
- `checkpoints/standard_transformer_lichess_best.pt` — epoch 9

Checkpoint paths are relative to the ChessAIResearch project root.

## Run the Comparison

From the project root:

```powershell
python -m prototypes.01_geochesszero.compare_test
```

## Predict Moves

```powershell
python -m prototypes.01_geochesszero.predict_position
```

## Launch the Interactive Demo

```powershell
python -m streamlit run demo/app.py
```

The demo allows users to play against either model.

## Research Artifacts

Results are stored in:

`results/geochesszero/`

Including:

- `test_comparison.json`
- `test_comparison.csv`
- `plots/test_loss_comparison.png`
- `plots/top1_comparison.png`
- `plots/top5_comparison.png`
- `final_report.md`

## Limitations

- Small training dataset
- Single experimental configuration
- Limited training duration
- No statistical significance analysis
- No engine-strength evaluation
- No Monte Carlo Tree Search
- No self-play reinforcement learning

## Status

Prototype 1: Completed.

The architecture, dataset processing, training, testing, comparison, visualization and interactive demonstration have been implemented.
