"""
Custom JSON encoder to handle numpy types and other non-serializable objects.
"""
import json
import numpy as np
from datetime import datetime, date
from decimal import Decimal
from typing import Any


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles numpy types and other common non-serializable objects.
    """
    
    def default(self, obj: Any) -> Any:
        """
        Convert non-serializable objects to serializable types.
        
        Args:
            obj: Object to encode
            
        Returns:
            Serializable representation of the object
        """
        # Handle numpy integer types
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        
        # Handle numpy float types
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        
        # Handle numpy bool types
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        
        # Handle numpy arrays
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle numpy scalars
        elif isinstance(obj, np.generic):
            return obj.item()
        
        # Handle datetime objects
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        
        # Handle Decimal objects
        elif isinstance(obj, Decimal):
            return float(obj)
        
        # Handle sets
        elif isinstance(obj, set):
            return list(obj)
        
        # Handle bytes
        elif isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')
        
        # Let the base class handle other types
        return super().default(obj)


def numpy_to_python(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types.
    
    Args:
        obj: Object to convert
        
    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(numpy_to_python(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                         np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='ignore')
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    else:
        return obj


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safely serialize an object to JSON, handling numpy types.
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments to pass to json.dumps
        
    Returns:
        JSON string
    """
    # Use our custom encoder by default
    kwargs.setdefault('cls', NumpyEncoder)
    return json.dumps(obj, **kwargs)


def safe_json_loads(json_str: str, **kwargs) -> Any:
    """
    Safely deserialize JSON string to Python object.
    
    Args:
        json_str: JSON string to deserialize
        **kwargs: Additional arguments to pass to json.loads
        
    Returns:
        Deserialized object
    """
    return json.loads(json_str, **kwargs)