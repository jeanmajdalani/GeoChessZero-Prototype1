# GeoChessZero — Prototype 1
### Geometry-Aware Transformer for Chess Move Prediction

## 1. Project Overview

GeoChessZero is a research prototype investigating whether incorporating chess-specific geometric relationships into Transformer attention improves move prediction compared with a standard Transformer.

The central research question is:

**Can geometry-aware attention help a Transformer learn chess move patterns more effectively than unrestricted self-attention?**

We implement two models with comparable architectures:

- **GeoChessZero:** A Transformer with attention heads constrained by chess-piece movement geometry.
- **Standard Transformer:** A baseline Transformer with unrestricted attention.

Both models are trained and evaluated on the same chess dataset, using the same action space and legal-move constraints.

This is a move-prediction research prototype, not a complete AlphaZero-style chess engine.

## 2. Main Research Idea

A standard Transformer learns relationships between board squares using unrestricted self-attention.

GeoChessZero introduces a chess-specific inductive bias by restricting selected attention heads according to piece movement patterns.

The geometric relationships include:

- **Rook geometry:** Horizontal and vertical connections.
- **Bishop geometry:** Diagonal connections.
- **Knight geometry:** L-shaped connections.
- **King geometry:** Adjacent squares.

The architecture uses eight attention heads:

- Four geometry-constrained heads.
- Four unrestricted attention heads.

The goal is to combine structured chess knowledge with the flexibility of learned attention.

These geometric masks describe possible square-to-square relationships. They are not equivalent to checking whether a move is currently legal.

## 3. Board Representation

Each chess position is represented as a tensor of shape:

`(64, 18)`

The 64 tokens correspond to the 64 chessboard squares.

Each token contains 18 features representing:

- 12 piece-type and color channels.
- Side to move.
- Four castling-right indicators.
- En passant information.

The model projects these features into a learned embedding space.

## 4. Model Architecture

Both models use:

| Component | Configuration |
|---|---|
| Input representation | 64 × 18 |
| Embedding dimension | 128 |
| Transformer blocks | 4 |
| Attention heads | 8 |
| Geometric heads | 4 for GeoChessZero |
| Baseline attention | Fully unrestricted |
| Action space | 4,272 actions |
| Encoder parameters | 803,968 |

The architecture contains:

1. Board feature projection.
2. Learned positional embeddings.
3. Transformer encoder blocks.
4. Move-prediction policy head.
5. Legal-action masking.

### Policy Head

The policy head predicts a score for each possible chess action.

The action space includes:

- 4,096 possible origin-destination combinations.
- 176 additional promotion actions.

**Total: 4,272 actions.**

Illegal moves are masked before the final prediction, ensuring that the selected action belongs to the legal-move set.

## 5. Dataset Preparation

The dataset is derived from Lichess chess games.

The preprocessing pipeline performs the following operations:

1. Read chess games from PGN files.
2. Replay each game move by move.
3. Encode every position into a `(64, 18)` tensor.
4. Convert the played move into its action index.
5. Assign game identifiers to positions.
6. Split the data by game.
7. Generate legal-action masks.

### Experimental Dataset

| Split | Games | Positions |
|---|---:|---:|
| Training | 80 | 4,877 |
| Validation | 10 | 697 |
| Testing | 10 | 619 |
| **Total** | **100** | **6,193** |

The split is performed at the game level to prevent positions from the same game appearing in multiple splits.

This initial dataset is relatively small and is intended for validating the architecture and experimental pipeline.

## 6. Training

Both models were trained using:

- PyTorch.
- NVIDIA CUDA acceleration.
- AdamW optimizer.
- Learning rate: `0.0003`.
- Weight decay: `0.01`.
- Batch size: `16`.
- Training duration: `10 epochs`.
- Cross-entropy loss over legal moves.

The best checkpoint is selected using validation loss.

### Hardware

The experiments were performed locally using an NVIDIA GeForce RTX 3060 Laptop GPU.

## 7. Experimental Results

The best checkpoints were evaluated on the held-out test dataset of 619 positions.

| Metric | GeoChessZero | Standard Transformer |
|---|---:|---:|
| Best epoch | 6 | 9 |
| Test loss | **2.6474** | 2.7794 |
| Top-1 accuracy | **24.23%** | 24.07% |
| Top-5 accuracy | **61.23%** | 52.50% |

### Observed Differences

GeoChessZero achieved:

- A test-loss reduction of **0.1320**.
- A Top-1 accuracy improvement of **0.16 percentage points**.
- A Top-5 accuracy improvement of **8.73 percentage points**, based on the displayed rounded accuracies.

The geometry-aware model performed better on these metrics in this experiment.

However, a single small-dataset experiment is insufficient to establish that geometry-aware attention consistently outperforms standard attention.

Further experiments with more games, multiple random seeds, and larger datasets are necessary.

## 8. Model Evaluation

The evaluation pipeline includes:

- Test cross-entropy loss.
- Top-1 move-prediction accuracy.
- Top-5 move-prediction accuracy.
- Comparison between geometric and standard attention.
- Prediction of legal moves from example chess positions.

Top-1 accuracy measures whether the highest-ranked predicted move matches the move recorded in the dataset.

Top-5 accuracy measures whether the recorded move appears among the five highest-ranked predictions.

These metrics evaluate imitation of recorded moves, not chess playing strength.

## 9. Interactive Demonstration

A Streamlit application allows users to play chess against either trained model.

The interface provides:

- An interactive chessboard display.
- Legal move selection.
- Model selection between GeoChessZero and the Standard Transformer.
- AI move prediction.
- Model prediction probabilities.
- Move history and game status.
- Game reset functionality.

The user plays White, and the selected AI model plays Black.

### Launch the Demo

From the project root:

```bash
streamlit run demo/app.py
```

The application normally opens at:

`http://localhost:8501`

## 10. Installation

Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd GeoChessZero-Prototype1
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install -r requirements.txt
```

PyTorch CUDA compatibility depends on the GPU, driver, and installed PyTorch build.

## 11. Repository Structure

```text
GeoChessZero-Prototype1/
├── data/
│   ├── pgn_converter.py
│   ├── split_dataset.py
│   ├── generate_legal_masks.py
│   └── attach_legal_masks.py
├── prototypes/
│   └── 01_geochesszero/
│       ├── geometry_masks.py
│       ├── attention.py
│       ├── transformer.py
│       ├── policy_value.py
│       ├── masked_dataset.py
│       ├── train_lichess.py
│       ├── train_standard.py
│       ├── compare_test.py
│       └── predict_position.py
├── shared/
├── tests/
├── demo/
│   └── app.py
├── results/
│   └── geochesszero/
├── requirements.txt
├── .gitignore
└── README.md
```

Some generated datasets and model checkpoints are excluded from version control because of their size.

The interactive demonstration requires trained checkpoints to be available locally.

## 12. Limitations

The current prototype has several limitations:

1. The training dataset contains only 100 games.
2. The model learns to imitate moves rather than directly maximize winning probability.
3. The architecture does not include Monte Carlo Tree Search.
4. The model has not been evaluated using an Elo-based chess tournament.
5. The experiment uses a limited number of training runs.
6. Prediction accuracy does not necessarily indicate strategic chess strength.
7. Geometric attention masks do not explicitly model all chess rules or piece blocking.

## 13. Future Improvements

Potential research extensions include:

- Training on a larger Lichess dataset.
- Repeating experiments with multiple random seeds.
- Performing attention-head ablation studies.
- Comparing computational efficiency and inference latency.
- Adding a position-value prediction head.
- Evaluating chess playing strength against established engines.
- Exploring self-play and search-based training.
- Comparing the architecture with other chess language-model approaches.

## 14. Research Conclusion

The initial experiment suggests that incorporating chess movement geometry into Transformer attention can improve certain move-prediction metrics.

On the current test dataset, GeoChessZero achieves lower loss and higher Top-5 accuracy than the standard Transformer baseline.

The Top-1 accuracy difference is small.

These findings are preliminary and should be validated with larger datasets and repeated experiments.

**Prototype 1 status: Implemented, trained, evaluated, and demonstrated.**