"""
Models package.

This package provides model classes for the application.
"""

from src.models.base_model import BaseModel
from src.models.onchain_flows_analyzer import OnChainFlowsAnalyzer, analyze_onchain_flows, track_whale_movements

__all__ = [
    'BaseModel',
    'OnChainFlowsAnalyzer',
    'analyze_onchain_flows',
    'track_whale_movements'
]
