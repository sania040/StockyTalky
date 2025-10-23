# src/ml/__init__.py
"""Machine learning module for StockyTalky."""

from .forecasting import (
    ProphetModel,
    XGBoostModel,
    get_available_models
)

__all__ = [
    'ProphetModel',
    'XGBoostModel',
    'get_available_models'
]
