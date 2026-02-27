"""
utils package initializer

This package provides core utility functions and pipeline components for:
- JSON parsing and cleaning
- Model interface (OpenRouter / API wrappers)
- Complaint processing, model analysis, and deduplication pipeline
"""

from .complaint_Stage1extract import process_complaints_batch
from .complaint_Stage1modelchoosing import analyze_model_similarity
from .complaint_Stage2dedup import export_results, run_pipeline_deduplicate
from .complaint_Stage3Construction import construction
from .model_interface import ModelInterface, call_openrouter_model
from .prepare_json_string import prepare_json_string

__all__ = [
    "prepare_json_string",
    "call_openrouter_model",
    "ModelInterface",
    "process_complaints_batch",
    "analyze_model_similarity",
    "run_pipeline_deduplicate",
    "export_results",
    "construction",
]
