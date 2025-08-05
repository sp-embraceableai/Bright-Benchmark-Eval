"""
BRIGHT Benchmark Evaluation Package
"""

from .bright_evaluator import BrightEvaluator
from .config import (
    MODEL_CONFIG, 
    AVAILABLE_DOMAINS, 
    AVAILABLE_CONFIGS, 
    DEFAULT_SETTINGS,
    METRICS
)

__version__ = "1.0.0"
__author__ = "BRIGHT Benchmark Team"

__all__ = [
    "BrightEvaluator",
    "MODEL_CONFIG",
    "AVAILABLE_DOMAINS", 
    "AVAILABLE_CONFIGS",
    "DEFAULT_SETTINGS",
    "METRICS"
] 