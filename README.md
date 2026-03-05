# FluSight Forecasting - South Carolina

Probabilistic 1–4 week ahead influenza forecasting using a state-of-the-art Hybrid Parallel CNN-LSTM model with Multi-Head Attention.

## Description
This project implements an advanced deep learning pipeline to predict influenza-related trends in South Carolina. It supports multiple targets (ED visit proportions and hospital admissions) and generates submission-compliant quantile-based forecasts. The model is optimized for temporal alignment and magnitude accuracy through:
*   **Hybrid Architecture**: Parallel CNN kernels (sizes 3 & 5) combined with Bidirectional LSTMs.
*   **Adaptive Lag Compensation**: Multi-head temporal attention and residual-gated decoding.
*   **Robust Training**: AdamW optimization, Gaussian noise augmentation, and directional phase-alignment loss.

## Installation
Ensure you have Python 3.8+ installed, then set up the environment:
```bash
git clone <repository-url>
cd FluSight
pip install -r requirements.txt
```

## Usage
### 1. Training & Forecasting
Generate forecasts for all targets (ED visits and Hospital Admissions):
```bash
python train_forecast.py
```
Outputs are saved to the `results/` folder, including:
*   `forecast_results_{target}.csv`: Full historical backtest and future forecasts.
*   `{date}-team-model-{target}.csv`: Filtered file containing only the latest reference date.
*   `best_model_{target}.pth`: Trained model weights.

### 2. Visualization
Generate comparison plots against ground truth and lab trends:
```bash
python visualize_results.py
```
View the generated `.png` figures in the `results/` directory.

## Contributing
1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/improvement`).
3.  Commit your changes with clear messages.
4.  Push to the branch and open a Pull Request.

## License
Distributed under the MIT License. See `LICENSE` for more information.
