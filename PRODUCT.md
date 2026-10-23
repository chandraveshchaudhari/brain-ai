# Brain-AI (brain-automl): product requirements

Owner: Chandravesh Chaudhari (SolarpunkWorks Pvt Ltd). Status: **draft 2026-10-23, written
by Claude from the repository's own docs; items marked _(proposed)_ need the owner's
confirmation.** This file states *what* the product must do; the Supervisor plans *how*.
Release order: LitSynth → Root → the rest; development may run in parallel.

## Product in one line

An open-source engine that decides and executes multimodal ML pipelines
(`PipelineSpec = modalities, granularity, fusion, model backend, hyperparameters`).

## Who it is for

ML researchers and practitioners with multimodal data (tabular, time series, text).
It also serves as a portfolio piece for the owner's job search.

## Must-have outcomes _(proposed)_

1. `pip install brain-automl` installs cleanly on Python 3.10–3.12 and the README quick-start runs as written.
2. The example notebooks in `examples/` run end to end on bundled or downloadable data.
3. The test suite passes and covers the fusion strategies and adapters.
4. Docs explain `PipelineSpec` and show one complete multimodal example.

## Open decisions for the owner

1. Keep as a portfolio/research project, or grow it into a product?
2. Next PyPI release: when, and with which changes?
