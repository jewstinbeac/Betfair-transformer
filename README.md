# Betfair Transformer

A transformer-based model for predicting expected value (EV) in Betfair horse racing markets using order book data and market microstructure.

## Overview

This project implements a deep learning model that processes Betfair exchange betting data to predict the expected value of back-to-lay arbitrage opportunities in horse racing markets. The model uses transformer architecture to capture temporal dependencies in market data and predict profitable trading opportunities.

## Key Features

- **Transformer Architecture**: Custom transformer model for sequential market data processing
- **EV-Focused Predictions**: Specifically targets expected value > 1.0 trading opportunities  
- **Comprehensive Data Pipeline**: Complete pipeline from raw .bz2 stream files to training-ready datasets
- **Advanced Evaluation**: Detailed analysis of model performance on positive EV predictions
- **Market Microstructure**: Incorporates order book depth, market timing, and liquidity factors

## Project Structure

```
betfair_transformer_project/
├── bf_transformer.py           # Main transformer model and training code
├── process_data_pipeline.py    # Complete data processing pipeline
├── data_processing/            # Data processing utilities
│   ├── build_race_json.py     # Convert .bz2 streams to race JSON
│   ├── split_races_streaming.py # Deterministic train/test/validation split
│   ├── split_scratchings.py   # Handle scratched horses
│   └── add_bin_counts.py      # Add sequence length metadata
├── evaluation/                 # Model evaluation tools
│   └── evaluate_ev.py         # Comprehensive EV-focused evaluation
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/jewstinbeac/Betfair-transformer.git
cd Betfair-transformer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Processing

Process raw Betfair .bz2 stream files into training-ready datasets:

```bash
# Complete pipeline: .bz2 files → final training datasets
python3 process_data_pipeline.py \
  --bz2-root /path/to/betfair/data \
  --output-dir datasets \
  --topk 5  # Keep top 5 price levels
```

This creates:
- `datasets/races_train_split_with_bins.jsonl.gz` (training set)
- `datasets/races_test_split.jsonl.gz` (test set)
- `datasets/races_validation_split.jsonl.gz` (validation set)

### 3. Training

Train the transformer model:

```bash
python3 bf_transformer.py
```

The model will:
- Load training data automatically
- Train with attention mechanism on market sequences
- Save checkpoints to `checkpoints/`
- Target log-EV prediction for back-to-lay opportunities

### 4. Evaluation

Evaluate model performance on positive EV predictions:

```bash
# Comprehensive EV evaluation
python3 evaluation/evaluate_ev.py \
  --checkpoint checkpoints/checkpoint_latest.pt \
  --save-dir eval_results

# Quick test with limited data
python3 evaluation/evaluate_ev.py \
  --checkpoint checkpoints/checkpoint_latest.pt \
  --max-races 100 \
  --save-dir eval_test
```

## Model Architecture

- **Input**: Market microstructure tokens (127 dimensions per time step)
  - Subject horse: back/lay prices, sizes, LTP, time-to-off
  - Other horses: top price levels for market context
  - Market state: total matched, active runners, suspensions

- **Architecture**: 6-layer transformer with 6 attention heads
  - 384 embedding dimensions
  - 256 sequence length (≈2+ minutes of market data)
  - Dropout regularization

- **Output**: Log expected value predictions for 32-step horizon
  - Target: `log(back_price / future_lay_price)`
  - Focus: Identifying EV > 1.0 opportunities

## Key Results

From evaluation on test data:
- **Correlation**: ~0.55 Pearson correlation between predicted and realized EV
- **Precision**: Model identifies positive EV opportunities with meaningful signal
- **Coverage**: ~1.5% of predictions exceed EV = 1.0 threshold
- **Calibration**: Higher confidence predictions show improved accuracy

## Data Pipeline Features

- **Deterministic Splitting**: Hash-based train/test splits ensure reproducibility
- **Scratching Handling**: Automatically segments races with scratched horses
- **Streaming Processing**: Memory-efficient processing of large datasets
- **Parallel Processing**: Optional multiprocessing for faster data builds

## Requirements

- Python 3.8+
- PyTorch
- NumPy, Pandas
- scikit-learn (for evaluation metrics)
- tqdm (progress bars)

See `requirements.txt` for complete dependencies.

## License

[Add your preferred license here]

## Contributing

[Add contribution guidelines if desired]
