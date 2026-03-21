#!/bin/bash

# Setup script for brain-ai project
# This script creates and configures a Python virtual environment with all dependencies

set -e  # Exit on error

echo "================================"
echo "Brain-AI Environment Setup"
echo "================================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Python is installed
echo -e "${BLUE}Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.6 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✓ Found $PYTHON_VERSION${NC}"

# Check if venv already exists
if [ -d ".venv" ]; then
    echo -e "${BLUE}Virtual environment already exists at .venv${NC}"
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Removing existing virtual environment...${NC}"
        rm -rf .venv
    else
        echo -e "${BLUE}Using existing virtual environment...${NC}"
        source .venv/bin/activate
        echo -e "${GREEN}✓ Virtual environment activated${NC}"
        echo ""
        echo "To activate the environment manually, run:"
        echo "  source .venv/bin/activate"
        exit 0
    fi
fi

# Create virtual environment
echo -e "${BLUE}Creating virtual environment...${NC}"
python3 -m venv .venv
echo -e "${GREEN}✓ Virtual environment created${NC}"

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source .venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip, setuptools, and wheel
echo -e "${BLUE}Upgrading pip, setuptools, and wheel...${NC}"
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ Pip tools upgraded${NC}"

# Install core dependencies
echo -e "${BLUE}Installing core project dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Core dependencies installed${NC}"
else
    echo -e "${BLUE}Note: requirements.txt not found, skipping core dependencies${NC}"
fi

# Install notebook dependencies
echo -e "${BLUE}Installing notebook dependencies...${NC}"
if [ -f "docs/requirements-notebook.txt" ]; then
    pip install -r docs/requirements-notebook.txt
    echo -e "${GREEN}✓ Notebook dependencies installed${NC}"
else
    echo -e "${BLUE}Installing essential notebook packages...${NC}"
    pip install jupyter ipykernel numpy pandas matplotlib seaborn scikit-learn scipy arch tensorflow
    echo -e "${GREEN}✓ Essential packages installed${NC}"
fi

# Install the project in development mode
echo -e "${BLUE}Installing project in development mode...${NC}"
pip install -e .
echo -e "${GREEN}✓ Project installed in editable mode${NC}"

# Register the kernel for Jupyter
echo -e "${BLUE}Registering Python kernel for Jupyter...${NC}"
python -m ipykernel install --user --name brain-ai-env --display-name "Brain-AI Environment"
echo -e "${GREEN}✓ Kernel registered${NC}"

echo ""
echo "================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "================================"
echo ""
echo "Virtual environment location: .venv/"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo ""
echo "Next steps:"
echo "1. Activate the environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Run Jupyter with the notebook:"
echo "   cd docs/notebooks"
echo "   jupyter lab 01_time_series_forecasting.ipynb"
echo ""
echo "3. Or run Jupyter from the project root:"
echo "   jupyter lab docs/notebooks/"
echo ""
echo "To deactivate the environment, run:"
echo "   deactivate"
echo ""
