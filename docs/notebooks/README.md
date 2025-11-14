# Notebooks: Multivariate Time Series Forecasting

This folder contains a comprehensive example notebook demonstrating multivariate time series forecasting using multiple state-of-the-art techniques.

## Contents

- **`01_time_series_forecasting.ipynb`** — Main notebook with complete workflow:
  - Data loading and exploration
  - Preprocessing and normalization
  - **GARCH** model for volatility forecasting
  - **LSTM** (RNN-based) model
  - **Transformer** (attention-based) model
  - Comprehensive model comparison (RMSE, MAE, MAPE, R²)
  - Residual analysis
  - Next steps for production

- **`data/sample_multivar_timeseries.csv`** — Sample multivariate time series dataset
  - Target column: `NIFTY` (close prices)
  - Multiple feature columns for multivariate forecasting

## Prerequisites

Install the required packages:

```zsh
pip install -r requirements-notebook.txt
```

## Quick Start

From the repository root:

```zsh
pip install -e .
pip install -r docs/requirements-notebook.txt
cd docs/notebooks
jupyter lab 01_time_series_forecasting.ipynb
```

## Dataset Format

The notebook expects a CSV file with:
- A `NIFTY` column (or similar close price column)
- Multiple feature columns (e.g., OPEN, HIGH, LOW, VOLUME, RSI, MACD, etc.)
- Optional `date` column for time indexing

Example:
```
date,NIFTY,OPEN,HIGH,LOW,VOLUME,...
2020-01-01,10000.5,9999,10050,9980,1000000,...
2020-01-02,10050.2,10001,10100,10020,1100000,...
```

## Models Explained

### 1. GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
- Models volatility of returns
- Useful for risk assessment and uncertainty quantification
- Univariate approach (operates on target column alone)

### 2. LSTM (Long Short-Term Memory)
- Recurrent neural network with memory cells
- Captures temporal dependencies in sequences
- Multivariate: uses all features to predict next value
- Good for medium-term patterns

### 3. Transformer (Attention-Based)
- Uses multi-head self-attention mechanism
- Captures long-range dependencies efficiently
- Parallelizable training (faster than LSTM)
- State-of-the-art for many sequence tasks

## Performance Metrics

The notebook computes:
- **RMSE**: Root Mean Squared Error (focus on large errors)
- **MAE**: Mean Absolute Error (robust average error)
- **MAPE**: Mean Absolute Percentage Error (scale-independent)
- **R²**: Coefficient of determination (variance explained)

## Customization

### Change forecast horizon:
```python
FORECAST_HORIZON = 5  # Predict 5 steps ahead instead of 1
```

### Adjust window size:
```python
WINDOW_SIZE = 60  # Use 60 timesteps instead of 30
```

### Modify train/test split:
```python
TRAIN_SIZE = 0.7  # 70% training, 30% testing
```

### Tune model hyperparameters:
- LSTM units, dropout rates, learning rates
- Transformer head size, number of heads, feed-forward dimension
- GARCH p, q parameters

## Next Steps (For Production)

1. **Hyperparameter tuning**: Use Optuna or GridSearchCV
2. **Cross-validation**: Implement walk-forward validation
3. **Ensemble models**: Combine predictions from multiple architectures
4. **Feature engineering**: Add domain-specific technical indicators
5. **External data**: Incorporate macroeconomic variables
6. **Backtesting**: Test strategies with real trading scenarios
7. **Deployment**: Package as REST API or microservice

## Troubleshooting

### Issue: Out of memory
- Reduce batch size
- Use smaller WINDOW_SIZE
- Reduce number of LSTM units

### Issue: Poor predictions
- Increase training epochs
- Add more features
- Check data quality and stationarity

### Issue: TensorFlow not found
```zsh
pip install tensorflow --upgrade
```

## References

- [Keras Time Series Documentation](https://keras.io/examples/timeseries/)
- [ARCH Package](https://arch.readthedocs.io/)
- [Transformer Architecture](https://arxiv.org/abs/1706.03762)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

