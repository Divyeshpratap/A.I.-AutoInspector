# utils/data_helpers.py

import numpy as np

def convert_to_serializable(obj):
    """
    Convert various data types (e.g., numpy types) into serializable Python data types.
    Args:
    - obj: Object to be converted.
    
    Returns:
    - Serializable Python object.
    """
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    return obj
