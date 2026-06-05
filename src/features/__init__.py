"""Feature engineering package: tsfresh pipeline, domain features, physics model."""

from src.features.feature_engineering import FeatureEngineer, extract_window_features
from src.features.physics_model import PhysicsValidation, PolynomialDegradationModel
from src.features.sensor_features import add_domain_features

__all__ = [
    "FeatureEngineer",
    "extract_window_features",
    "PhysicsValidation",
    "PolynomialDegradationModel",
    "add_domain_features",
]
