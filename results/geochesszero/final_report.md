
# GeoChessZero — Prototype 1 Research Report

## 1. Project Overview

GeoChessZero is a research prototype designed to investigate whether chess-specific geometric attention masks can improve Transformer-based chess move prediction.

The project compares two models:

1. GeoChessZero: a geometry-aware Transformer.
2. Standard Transformer: a baseline without geometric attention restrictions.

Both models use the same overall architecture and are trained on the same chess dataset.

## 2. Research Question

Can chess-specific geometric attention improve the prediction of human chess moves compared with standard self-attention?

### Hypothesis

By incorporating geometric relationships between chessboard squares into the attention mechanism, GeoChessZero may learn useful spatial patterns more efficiently than a standard Transformer.

## 3. Dataset

Source: Lichess rated chess games.

Dataset preparation:

- Original games are stored in PGN format.
- PGN games are converted into board positions and target moves.
- Each position is encoded as a 64 × 18 feature matrix.
- Each target move is represented by an action index.
- The action vocabulary contains 4,272 possible actions.
- Legal move masks are generated for each position.
- The dataset is divided by game to avoid sharing positions from the same game between training and evaluation splits.

### Dataset Statistics

| Split | Games | Positions |
|---|---:|---:|
| Training | 80 | 4,877 |
| Validation | 10 | 697 |
| Test | 10 | 619 |
| Total | 100 | 6,193 |

## 4. Model Architecture

Both models use a Transformer architecture with:

- 64 chessboard tokens.
- 18 input features per square.
- Model dimension: 128.
- 4 Transformer layers.
- 8 attention heads.
- A policy head predicting chess moves.

### GeoChessZero

GeoChessZero uses geometry-aware attention.

Some attention heads are restricted using chess movement geometries:

- Rook geometry.
- Bishop geometry.
- Knight geometry.
- King geometry.

Other attention heads remain unrestricted.

These geometric masks encode possible spatial relationships between squares, rather than the complete legal moves of individual pieces.

### Standard Transformer

The baseline uses unrestricted self-attention.

The main experimental difference between the models is the geometric attention mechanism.

## 5. Training Methodology

Both models were trained using:

- PyTorch.
- NVIDIA GPU with CUDA.
- AdamW optimizer.
- Learning rate: 0.0003.
- Weight decay: 0.01.
- Batch size: 16.
- Training epochs: 10.
- Cross-entropy loss.
- Legal move masking.

The best checkpoint for each model was selected using validation loss.

### Best Checkpoints

GeoChessZero:

- Best epoch: 6.
- Validation loss: 2.6808.
- Validation accuracy: 26.69%.

Standard Transformer:

- Best epoch: 9.
- Validation loss: 2.8051.
- Validation accuracy: 23.24%.

## 6. Test Results

Both models were evaluated on the same held-out test split.

| Metric | GeoChessZero | Standard Transformer |
|---|---:|---:|
| Test Loss | 2.6474 | 2.7794 |
| Top-1 Accuracy | 24.23% | 24.07% |
| Top-5 Accuracy | 61.23% | 52.50% |

### Observed Differences

- Test loss difference: -0.1320.
- Top-1 accuracy improvement: +0.16 percentage points.
- Top-5 accuracy improvement: +8.73 percentage points (using rounded displayed values).

GeoChessZero achieved a lower test loss and higher Top-1 and Top-5 accuracy.

The largest observed improvement was in Top-5 accuracy.

## 7. Interactive Demonstration

A Streamlit interface was developed to demonstrate both models.

The application supports:

- Displaying a chessboard.
- Selecting GeoChessZero or the Standard Transformer.
- Playing legal chess moves.
- Generating AI responses.
- Displaying the selected model and checkpoint epoch.
- Displaying predicted move probabilities.
- Tracking the game state.

The application uses python-chess for chess rules and legal move generation.

## 8. Interpretation

The results suggest that geometry-aware attention may be beneficial for chess move prediction.

GeoChessZero performed better than the baseline on the evaluated test set.

However, the Top-1 improvement was small, while the Top-5 improvement was substantially larger.

This indicates that geometric attention may help rank plausible moves more effectively without necessarily producing a large improvement in the single highest-ranked move.

## 9. Limitations

Several limitations must be considered:

1. Only 100 chess games were used.
2. The experiment used a relatively small dataset.
3. Both models were trained for only 10 epochs.
4. The comparison was performed with a single training configuration.
5. Statistical significance was not established.
6. Human move prediction accuracy is not equivalent to chess playing strength.
7. The models were not evaluated through a large tournament or against a strong chess engine.
8. The models do not currently use Monte Carlo Tree Search or reinforcement learning.

Therefore, the experiment provides preliminary evidence rather than a definitive conclusion.

## 10. Future Improvements

Possible improvements include:

- Training on a larger Lichess dataset.
- Running experiments with multiple random seeds.
- Comparing learning curves and training efficiency.
- Testing different geometric attention configurations.
- Evaluating performance across chess opening, middlegame, and endgame positions.
- Evaluating actual playing strength.
- Exploring policy-value learning.
- Integrating search or self-play reinforcement learning.

## 11. Conclusion

Prototype 1 successfully implemented and evaluated a geometry-aware chess Transformer.

On the current test set, GeoChessZero achieved:

- 24.23% Top-1 accuracy.
- 61.23% Top-5 accuracy.
- 2.6474 test loss.

The Standard Transformer achieved:

- 24.07% Top-1 accuracy.
- 52.50% Top-5 accuracy.
- 2.7794 test loss.

The results are encouraging, particularly for Top-5 move prediction.

Further experiments are required to determine whether geometric attention consistently improves performance and whether these improvements translate into stronger chess play.
