# Python Environment Setup Guide

This guide will help you set up the Python environment for the **brain-ai** project and run the example Jupyter notebooks.

## Quick Start (Automated Setup)

### macOS / Linux

Run the setup script to automatically create and configure the environment:

```zsh
chmod +x setup_environment.sh
./setup_environment.sh
```

This script will:
1. ✅ Create a Python virtual environment at `.venv/`
2. ✅ Activate the environment
3. ✅ Upgrade pip, setuptools, and wheel
4. ✅ Install all core dependencies from `requirements.txt`
5. ✅ Install notebook dependencies from `docs/requirements-notebook.txt`
6. ✅ Install the project in editable mode (`pip install -e .`)
7. ✅ Register the Jupyter kernel

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r docs/requirements-notebook.txt
pip install -e .
python -m ipykernel install --user --name brain-ai-env --display-name "Brain-AI Environment"
```

## Manual Setup

If you prefer manual setup, follow these steps:

### 1. Create Virtual Environment

```zsh
python3 -m venv .venv
```

### 2. Activate Virtual Environment

**macOS / Linux:**
```zsh
source .venv/bin/activate
```

**Windows:**
```powershell
.\.venv\Scripts\Activate
```

### 3. Upgrade Package Manager

```zsh
pip install --upgrade pip setuptools wheel
```

### 4. Install Dependencies

**Core project dependencies:**
```zsh
pip install -r requirements.txt
```

**Notebook-specific dependencies:**
```zsh
pip install -r docs/requirements-notebook.txt
```

Or install individual key packages:
```zsh
pip install jupyter ipykernel numpy pandas matplotlib seaborn scikit-learn scipy arch tensorflow
```

### 5. Install Project in Development Mode

```zsh
pip install -e .
```

### 6. Register Jupyter Kernel (Optional but Recommended)

```zsh
python -m ipykernel install --user --name brain-ai-env --display-name "Brain-AI Environment"
```

## Verify Installation

Check that everything is installed correctly:

```zsh
# Check Python version
python --version

# Check pip packages
pip list | grep -E "jupyter|numpy|pandas|tensorflow|arch|scikit-learn"

# Test imports
python -c "import numpy, pandas, sklearn, tensorflow, arch; print('All imports successful!')"
```

## Running the Jupyter Notebook

### From Project Root

```zsh
# Activate environment (if not already active)
source .venv/bin/activate

# Launch Jupyter Lab
jupyter lab docs/notebooks/01_time_series_forecasting.ipynb
```

### Or Navigate to Notebook Directory

```zsh
source .venv/bin/activate
cd docs/notebooks
jupyter lab 01_time_series_forecasting.ipynb
```

### Alternative: Jupyter Notebook

If you prefer the classic Jupyter interface:

```zsh
jupyter notebook docs/notebooks/01_time_series_forecasting.ipynb
```

## Environment Structure

After setup, your project structure will look like:

```
brain-ai/
├── .venv/                          # Virtual environment (created)
│   ├── bin/
│   │   ├── python                  # Python executable
│   │   ├── jupyter                 # Jupyter command
│   │   └── ...
│   └── lib/
│       └── python3.x/
│           └── site-packages/      # Installed packages
├── src/
│   └── brain_automl/               # Project source code
├── docs/
│   ├── notebooks/
│   │   ├── 01_time_series_forecasting.ipynb  # Example notebook
│   │   └── README.md
│   └── requirements-notebook.txt
├── requirements.txt                # Core dependencies
├── setup.py                        # Project setup file
├── setup_environment.sh            # Setup script
└── ENVIRONMENT.md                  # This file
```

## Deactivate Environment

When you're done working, deactivate the environment:

```zsh
deactivate
```

## Troubleshooting

### Issue: `ModuleNotFoundError` when importing packages

**Solution**: Make sure the virtual environment is activated:
```zsh
source .venv/bin/activate
```

### Issue: Jupyter kernel not found

**Solution**: Register the kernel again:
```zsh
python -m ipykernel install --user --name brain-ai-env --display-name "Brain-AI Environment"
```

### Issue: Permission denied when running setup script

**Solution**: Make the script executable:
```zsh
chmod +x setup_environment.sh
```

### Issue: TensorFlow installation fails on M1/M2 Mac

**Solution**: Install pre-built wheel:
```zsh
pip install tensorflow-macos
pip install tensorflow-metal
```

### Issue: Out of disk space during installation

**Solution**: Some ML packages are large. Clean pip cache:
```zsh
pip cache purge
```

### Issue: Old Python version

**Solution**: Ensure you have Python 3.6 or higher:
```zsh
python3 --version
```

If needed, install Python via Homebrew (macOS):
```zsh
brew install python@3.11
```

## Using Conda Instead (Alternative)

If you prefer Conda:

```zsh
# Create environment
conda create -n brain-ai python=3.11

# Activate
conda activate brain-ai

# Install dependencies
pip install -r requirements.txt
pip install -r docs/requirements-notebook.txt
pip install -e .

# Register kernel
python -m ipykernel install --user --name brain-ai --display-name "Brain-AI (Conda)"
```

## Project Dependencies Overview

### Core Packages (from requirements.txt)
- **Data Processing**: pandas, numpy
- **ML/AutoML**: scikit-learn, TensorFlow, torch, autogluon, pycaret
- **Time Series**: arch, statsmodels, prophet, pmdarima
- **Visualization**: matplotlib, seaborn, plotly, bokeh
- **Utilities**: scipy, requests, pyyaml

### Notebook-Specific (from docs/requirements-notebook.txt)
- **Jupyter**: jupyter, ipykernel, notebook
- **Scientific**: numpy, pandas, scipy
- **ML**: scikit-learn, tensorflow
- **Visualization**: matplotlib, seaborn
- **Time Series**: arch (GARCH models)

## Next Steps

1. ✅ **Verify environment**: Run `jupyter lab` and open the example notebook
2. 📊 **Explore the notebook**: Follow the multivariate time series forecasting example
3. 🔧 **Customize**: Modify the notebook with your own data and models
4. 🚀 **Integrate**: Incorporate the workflow into your projects

## Additional Resources

- [Virtual Environments Documentation](https://docs.python.org/3/tutorial/venv.html)
- [Jupyter Installation Guide](https://jupyter.org/install)
- [TensorFlow Installation](https://www.tensorflow.org/install)
- [Conda Documentation](https://docs.conda.io/)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify Python version: `python --version`
3. List installed packages: `pip list`
4. Check environment status: `pip show ipykernel jupyter`

---

**Last Updated**: November 14, 2025
