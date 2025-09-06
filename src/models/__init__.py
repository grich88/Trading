"""
Models package.

This package provides model classes for the application.
"""

from src.models.base_model import BaseModel
from src.models.macro_events_analyzer import MacroEventsAnalyzer, analyze_cpi_impact, analyze_macro_events

__all__ = [
    'BaseModel',
    'MacroEventsAnalyzer',
    'analyze_cpi_impact',
    'analyze_macro_events'
]
