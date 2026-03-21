# Quick Reference: Running the Brain-AI Notebook

## TL;DR - Get Started in 2 Minutes

```zsh
# 1. Run setup script
chmod +x setup_environment.sh
./setup_environment.sh

# 2. Activate environment
source .venv/bin/activate

# 3. Launch notebook
cd docs/notebooks
jupyter lab 01_time_series_forecasting.ipynb
```

---

## Environment Status

✅ **Virtual Environment Created**: `.venv/` (Python 3.13.7)
✅ **Kernel Registered**: brain-ai-env
✅ **Key Packages Installed**: 
- arch (GARCH models)
- tensorflow (LSTM & Transformer)
- numpy, pandas (data processing)
- scikit-learn (metrics)
- matplotlib, seaborn (visualization)

---

## Running the Notebook

### Option 1: Jupyter Lab (Recommended)
```zsh
source .venv/bin/activate
jupyter lab docs/notebooks/01_time_series_forecasting.ipynb
```

### Option 2: Jupyter Notebook
```zsh
source .venv/bin/activate
jupyter notebook docs/notebooks/01_time_series_forecasting.ipynb
```

### Option 3: From Anywhere
```zsh
jupyter lab --notebook-dir=/Volumes/MacSSD/Areas/Github_Repositories/brain-ai/docs/notebooks
```

---

## What the Notebook Does

The notebook (`01_time_series_forecasting.ipynb`) demonstrates:

1. **Data Loading** - Read multivariate time series (CSV)
2. **Preprocessing** - Normalize, split, create sequences
3. **GARCH Model** - Volatility forecasting
4. **LSTM Model** - RNN-based predictions
5. **Transformer Model** - Attention-based predictions
6. **Comparison** - RMSE, MAE, MAPE, R² metrics
7. **Visualization** - Predictions vs actual, residuals

---

## Project Structure

```
brain-ai/
├── .venv/                          ← Your virtual environment
├── src/brain_automl/               ← Project source code
├── docs/
│   ├── notebooks/
│   │   ├── 01_time_series_forecasting.ipynb  ← Example notebook
│   │   ├── data/
│   │   │   └── sample_multivar_timeseries.csv
│   │   └── README.md
│   └── requirements-notebook.txt
├── requirements.txt                ← All project dependencies
├── setup.py                        ← Project configuration
├── setup_environment.sh            ← Setup script
├── ENVIRONMENT.md                  ← Detailed setup guide
└── QUICKSTART.md                   ← This file
```

---

## Common Commands

### Activate/Deactivate Environment
```zsh
# Activate
source .venv/bin/activate

# Deactivate
deactivate
```

### Verify Installation
```zsh
# Check Python
python --version

# Check packages
pip list | grep -E "jupyter|tensorflow|arch"

# Test imports
python -c "import arch, tensorflow; print('OK')"
```

### Install Additional Packages
```zsh
# Make sure environment is activated first
pip install <package-name>
```

### Check Environment Info
```zsh
which python
python -m site
pip show tensorflow
```

---

## Notebook Features

| Feature | Details |
|---------|---------|
| **Models** | GARCH, LSTM, Transformer |
| **Target** | NIFTY (close prices) |
| **Features** | Multivariate (all columns) |
| **Metrics** | RMSE, MAE, MAPE, R² |
| **Train/Test** | 80/20 split |
| **Window Size** | 30 timesteps |
| **Epochs** | 30 (with early stopping) |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `venv not activated` | Run `source .venv/bin/activate` |
| `jupyter not found` | Reactivate environment or run setup script again |
| `ModuleNotFoundError` | Make sure environment is activated |
| `Permission denied` | Run `chmod +x setup_environment.sh` |
| `TensorFlow slow on M1 Mac` | Install `tensorflow-macos` instead |

---

## Next Steps

1. ✅ **Run the setup script** to create environment
2. 🔬 **Open the notebook** and explore the examples
3. 📊 **Load your own data** (CSV file) and test
4. 🔧 **Modify parameters** (window size, epochs, etc.)
5. 🚀 **Deploy** the trained models

---

## Getting Help

1. Check `ENVIRONMENT.md` for detailed setup guide
2. See `docs/notebooks/README.md` for notebook documentation
3. Review the notebook cells for inline comments
4. Check package documentation:
   - TensorFlow: https://www.tensorflow.org/
   - ARCH: https://arch.readthedocs.io/
   - Scikit-learn: https://scikit-learn.org/

---

**Environment Ready!** 🎉

Your Python environment is now configured with all dependencies for running the time series forecasting notebook.

