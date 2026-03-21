
<div align="center">
  <img src="https://raw.githubusercontent.com/chandraveshchaudhari/personal-information/initial_setup/logos/my%20github%20logo%20template-Brain-AutoML.drawio.png" width="640" height="320">
</div>

# Brain-Multiple-Modalities-AutoML (BMMA)

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Contribution](#contribution)
- [Future Improvements](#future-improvements)

## Introduction
BMMA framework is capable of scaling to multiple modalities such as tabular, sentiment data, time series, and computer vision data. The architecture of BMMA is centred around the main component, Brain (Facade Design), which manages all internal parts.

### Authors
<img align="left" width="231.95" height="75" src="https://raw.githubusercontent.com/chandraveshchaudhari/personal-information/initial_setup/images/christ.png">

The package [Brain-AutoML](https://github.com/chandraveshchaudhari/brain-ai/) is part of thesis created by [Chandravesh chaudhari][chandravesh linkedin], Doctoral candidate at [CHRIST (Deemed to be University), Bangalore, India][christ university website] under the supervision of [Dr. Geetanjali purswani][geetanjali linkedin].

<br/>

[chandravesh linkedin]: https://www.linkedin.com/in/chandravesh-chaudhari "chandravesh linkedin profile"
[geetanjali linkedin]: https://www.linkedin.com/in/dr-geetanjali-purswani-546336b8 "geetanjali linkedin profile"
[christ university website]: https://christuniversity.in/ "website"

## Features
- Highly customisable
- provide separate Json for customisation

#### Significance
- Saves time
- Automate monotonous tasks
- Provides replicable results

## Installation 

### From PyPI
This project is available at [PyPI](https://pypi.org/project/brain-automl/). For help in installation check 
[instructions](https://packaging.python.org/tutorials/installing-packages/#installing-from-pypi)
```bash
python3 -m pip install brain−multiple−modalities−automl 
```

### Development Setup (Local Installation)

For development or running examples locally, follow these steps:

#### Quick Setup (Automated)
```zsh
chmod +x setup_environment.sh
./setup_environment.sh
```

#### Manual Setup
```zsh
# Create virtual environment
python3 -m venv .venv

# Activate environment
source .venv/bin/activate  # macOS/Linux
# OR
.\.venv\Scripts\Activate   # Windows

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r docs/requirements-notebook.txt
pip install -e .
```

### Documentation
- 📖 **Setup Guide**: See [ENVIRONMENT.md](ENVIRONMENT.md) for detailed instructions
- ⚡ **Quick Start**: See [QUICKSTART.md](QUICKSTART.md) for common commands
- 📊 **Example Notebook**: See [docs/notebooks/README.md](docs/notebooks/README.md)

### Running Example Notebooks

After setup, run the time series forecasting example:

```zsh
# Activate environment
source .venv/bin/activate

# Launch Jupyter
jupyter lab docs/notebooks/01_time_series_forecasting.ipynb
```

See [docs/notebooks/README.md](docs/notebooks/README.md) for more details on available examples.

### Implemented AutoML libraries
- [AutoGluon](https://auto.gluon.ai/stable/tutorials/tabular/index.html)
- AutoKeras
- AutoSklearn
- TPOT (Tree-based Pipeline Optimization Tool)
- H2O.ai
- ML Jar
- PyCaret
  Sentiment Analysis models
- BERT
- RoBERTa

## Important links
- [Documentation](https://chandraveshchaudhari.github.io/brain-ai/)
- [Quick tour](https://chandraveshchaudhari.github.io/brain-ai/brain-ai%20tutorial.html)
- [Project maintainer (feel free to contact)](mailto:chandraveshchaudhari@gmail.com?subject=[GitHub]%20Source%20brain-ai) 
- [Future Improvements](https://github.com/chandraveshchaudhari/brain-ai/projects)
- [License](https://github.com/chandraveshchaudhari/brain-ai/blob/master/LICENSE.txt)

## Contribution
all kinds of contributions are appreciated.
- [Improving readability of documentation](https://chandraveshchaudhari.github.io/brain-ai/)
- [Feature Request](https://github.com/chandraveshchaudhari/brain-ai/issues/new/choose)
- [Reporting bugs](https://github.com/chandraveshchaudhari/brain-ai/issues/new/choose)
- [Contribute code](https://github.com/chandraveshchaudhari/brain-ai/compare)
- [Asking questions in discussions](https://github.com/chandraveshchaudhari/brain-ai/discussions)

## Future Improvements
- [ ] Web based GUI


