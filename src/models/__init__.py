"""
Models package.

This package provides model classes for the application.
"""

from src.models.base_model import BaseModel
from src.models.funding_rate_analyzer import FundingRateAnalyzer, detect_funding_anomalies, analyze_funding_rates

__all__ = [
    'BaseModel',
    'FundingRateAnalyzer',
    'detect_funding_anomalies',
    'analyze_funding_rates'
]
