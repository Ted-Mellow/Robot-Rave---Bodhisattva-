#!/usr/bin/env python3
"""
Apply Trained Keypoint Mapping
Helper function to apply the trained mapping model to normalize keypoints.
"""

import pickle
import numpy as np
from pathlib import Path


def load_mapping_model(model_path="keypoint_mapping_model.pkl"):
    """Load the trained mapping model."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data


def apply_mapping(x_raw, y_raw, model_data):
    """
    Apply trained mapping to normalize raw keypoint coordinates.
    
    Args:
        x_raw: Raw x coordinate (wrist relative to shoulder, normalized by arm length)
        y_raw: Raw y coordinate (wrist relative to shoulder, normalized by arm length)
        model_data: Loaded model data from load_mapping_model()
    
    Returns:
        (x_norm, y_norm): Normalized coordinates in [0, 1] range
    """
    models = model_data['models']
    model_type = models.get('type', 'statistical')
    
    if model_type == 'statistical':
        # Statistical mapping
        x_params = models['x']
        y_params = models['y']
        
        x_norm = (x_raw + x_params['offset']) * x_params['scale']
        y_norm = (y_raw + y_params['offset']) * y_params['scale']
        
        # Clamp to [0, 1]
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        
    else:
        # sklearn model
        X_raw = np.array([[x_raw, y_raw]])
        
        x_norm = models['x'].predict(X_raw)[0]
        y_norm = models['y'].predict(X_raw)[0]
        
        # Clamp to [0, 1]
        x_norm = max(0.0, min(1.0, float(x_norm)))
        y_norm = max(0.0, min(1.0, float(y_norm)))
    
    return x_norm, y_norm


# For backward compatibility - default mapping (before training)
def default_mapping(x_raw, y_raw):
    """Default mapping: simple linear transformation."""
    x = (x_raw + 1.0) / 2.0  # Maps [-1, 1] to [0, 1]
    y = (y_raw + 1.0) / 2.0  # Maps [-1, 1] to [0, 1]
    return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))
