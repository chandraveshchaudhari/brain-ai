"""Tests for modality notebook planning and hybrid roadmap helpers."""

import json

from brain_automl import (
    build_modality_notebook_template,
    create_modality_notebooks,
    get_decomposition_algorithms,
    get_hybrid_modeling_roadmap,
    get_modality_notebook_plan,
    get_supported_modalities,
)
from brain_automl.config import get_default_config


def test_supported_modalities_match_config():
    """Notebook planner stays aligned with config defaults."""
    config = get_default_config()
    assert get_supported_modalities() == config["planning"]["modality_notebooks"]["enabled_modalities"]


def test_build_modality_notebook_template_is_json_notebook_shape():
    """Generated template uses the notebook JSON structure and cell language metadata."""
    notebook = build_modality_notebook_template("time_series")

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    assert len(notebook["cells"]) == 2
    assert notebook["cells"][0]["metadata"]["language"] == "markdown"
    assert notebook["cells"][1]["metadata"]["language"] == "python"


def test_create_modality_notebooks_writes_expected_files(tmp_path):
    """Scaffolder writes one notebook per supported modality."""
    created_paths = create_modality_notebooks(tmp_path)

    assert len(created_paths) == len(get_supported_modalities())
    assert created_paths[0].exists()

    parsed = json.loads(created_paths[0].read_text(encoding="utf-8"))
    assert parsed["cells"][0]["metadata"]["language"] == "markdown"


def test_hybrid_roadmap_covers_decomposition_and_backtesting():
    """Hybrid roadmap includes decomposition and evaluation milestones."""
    roadmap = get_hybrid_modeling_roadmap()
    summaries = [item["summary"] for item in roadmap]

    assert any("decomposition-based hybrids" in summary for summary in summaries)
    assert any("backtesting" in summary for summary in summaries)
    assert get_decomposition_algorithms() == ["stl", "wavelet", "emd", "ceemdan"]


def test_modality_notebook_plan_metadata():
    """Planner exposes notebook creation plan metadata."""
    plan = get_modality_notebook_plan()

    assert plan["format"] == "json_notebook"
    assert plan["template_naming"] == "<modality>_template.ipynb"
    assert "multimodal" in plan["modalities"]