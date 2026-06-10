"""
BrainLat: A comprehensive library for neurodegeneration and aging analysis.

This library provides tools for regression modeling, statistical analysis, 
and diagnostic visualizations for neurodegenerative disease research.

Modules:
    - age_gap_models: Gap-corrected regression models
    - regresion_model: Standard regression models
    - stats: Statistical measures and tests
    - tools: Utility functions (VIF, directions, scaling)
    - graphics: Diagnostic plots and visualizations
    - diagnostics: Data quality checks and logging
    - clasification_model: Classification framework (future)
"""

__version__ = "0.1.0"
__author__ = "BrainLat Team"

from . import age_gap_models
from . import regresion_model
from . import stats
from . import tools
from . import graphics
from . import diagnostics
from . import clasification_model
from . import gam_model

__all__ = [
    'age_gap_models',
    'regresion_model',
    'stats',
    'tools',
    'graphics',
    'diagnostics',
    'clasification_model',
    'gam_model',
]

