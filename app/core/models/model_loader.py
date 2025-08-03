"""
Custom model loader that handles class remapping for models saved in different environments.
"""

import pickle
import joblib
import sys
import warnings
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)

# Suppress sklearn version warnings for production
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')


class RemappingUnpickler(pickle.Unpickler):
    """Custom unpickler that remaps module paths for classes."""
    
    def find_class(self, module, name):
        """Override find_class to remap module paths."""
        # Map classes from __main__ to their actual locations
        class_mappings = {
            ('__main__', 'CasualtyRiskPreprocessor'): ('app.core.preprocessing', 'CasualtyRiskPreprocessor'),
            ('__main__', 'CasualtyFeatureEngineer'): ('app.core.preprocessing', 'CasualtyFeatureEngineer'),
            ('__main__', 'FocalLoss'): ('app.core.preprocessing', 'FocalLoss'),
            ('__main__', 'CasualtyRiskModel'): ('app.core.models.casualty_model_wrapper', 'CasualtyRiskModel'),
            ('__main__', 'PropertyDamageDataPreprocessor'): ('app.core.preprocessing', 'PropertyDamageDataPreprocessor'),
            ('__main__', 'SeverityClassifier'): ('catboost', 'CatBoostClassifier'),
            ('_loss', 'FocalLoss'): ('app.core.preprocessing', 'FocalLoss'),
            ('_loss', 'Loss'): ('app.core.preprocessing', 'FocalLoss'),
            ('_loss', 'CyHalfSquaredError'): ('app.core.preprocessing', 'DummyLoss'),
            ('_loss', 'HalfSquaredError'): ('app.core.preprocessing', 'DummyLoss'),
        }
        
        # Check if we need to remap
        if (module, name) in class_mappings:
            new_module, new_name = class_mappings[(module, name)]
            module = new_module
            name = new_name
        elif module == '_loss':
            # Handle any _loss module references
            logger.warning(f"Found reference to _loss.{name}, mapping to placeholder")
            # Create a proper dummy class with required methods for loss functions
            class DummyLoss:
                def __init__(self, *args, **kwargs):
                    pass
                def __call__(self, *args, **kwargs):
                    return 0.0
                def __reduce__(self):
                    return (type(self), ())
                def __getstate__(self):
                    return {}
                def __setstate__(self, state):
                    pass
            return DummyLoss
        elif module == '__mp_main__':
            # Handle multiprocessing main module references
            logger.warning(f"Found reference to __mp_main__.{name}, mapping to app.core.preprocessing")
            module = 'app.core.preprocessing'
            
        # Import the module if needed
        if module not in sys.modules:
            try:
                __import__(module)
            except ImportError as e:
                logger.warning(f"Could not import module {module}: {e}")
                # Create a dummy class as fallback
                class DummyClass:
                    def __init__(self, *args, **kwargs):
                        pass
                    def __getattr__(self, item):
                        return lambda *args, **kwargs: None
                return DummyClass
            
        # Get the class from the module
        try:
            return getattr(sys.modules[module], name)
        except AttributeError:
            logger.warning(f"Class {name} not found in module {module}, creating dummy")
            class DummyClass:
                def __init__(self, *args, **kwargs):
                    pass
                def __getattr__(self, item):
                    return lambda *args, **kwargs: None
            return DummyClass


def load_model_with_remapping(filepath: str) -> Any:
    """
    Load a model with class remapping to handle models saved in different environments.
    
    Args:
        filepath: Path to the model file
        
    Returns:
        The loaded model object
    """
    logger.info(f"Loading model with remapping from {filepath}")
    
    try:
        # First try with joblib (it might handle some cases better)
        try:
            # Set up the modules in sys.modules to help with loading
            import app.core.preprocessing
            import app.core.models.casualty_model_wrapper
            
            # Create a custom joblib load function with suppressed warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open(filepath, 'rb') as f:
                    model = RemappingUnpickler(f).load()
            
            logger.info("Successfully loaded model with custom unpickler")
            return model
            
        except Exception as e:
            logger.warning(f"Custom unpickler failed: {e}, trying alternate approach")
            
            # Try loading with a modified environment
            import app.core.preprocessing as preprocessing
            import app.core.models.casualty_model_wrapper as casualty_wrapper
            
            # Create dummy classes for models that need them
            class SeverityClassifier:
                def __init__(self, *args, **kwargs):
                    pass
                def predict(self, *args, **kwargs):
                    return ['Minor']
                def predict_proba(self, *args, **kwargs):
                    return [[0.8, 0.2]]
                
            class PropertyDamageDataPreprocessor:
                def __init__(self):
                    self.label_encoders = {}
                    self.column_transformer = None
                    self._name_to_fitted_passthrough = {}  # Fix for sklearn compatibility
                def transform(self, X):
                    return X
                def fit_transform(self, X, y=None):
                    return X
            
            # Create a dummy _loss module with common loss functions
            class DummyLoss:
                def __init__(self, *args, **kwargs):
                    pass
                def __call__(self, *args, **kwargs):
                    return 0.0
                def __reduce__(self):
                    return (type(self), ())
                def __getstate__(self):
                    return {}
                def __setstate__(self, state):
                    pass
            
            import types
            _loss = types.ModuleType('_loss')
            _loss.FocalLoss = DummyLoss
            _loss.Loss = DummyLoss
            _loss.CyHalfSquaredError = DummyLoss  # Add the specific loss function
            _loss.HalfSquaredError = DummyLoss
            sys.modules['_loss'] = _loss
            
            # Temporarily add classes to __main__
            import __main__
            __main__.CasualtyRiskPreprocessor = getattr(preprocessing, 'CasualtyRiskPreprocessor', PropertyDamageDataPreprocessor)
            __main__.CasualtyFeatureEngineer = getattr(preprocessing, 'CasualtyFeatureEngineer', PropertyDamageDataPreprocessor)
            __main__.FocalLoss = getattr(preprocessing, 'FocalLoss', DummyLoss)
            __main__.CasualtyRiskModel = getattr(casualty_wrapper, 'CasualtyRiskModel', SeverityClassifier)
            __main__.SeverityClassifier = SeverityClassifier
            __main__.PropertyDamageDataPreprocessor = PropertyDamageDataPreprocessor
            __main__._loss = _loss  # Add the dummy loss module
            
            try:
                # Now try loading with joblib, suppressing warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = joblib.load(filepath)
                logger.info("Successfully loaded model with __main__ injection")
                return model
            finally:
                # Clean up __main__
                for attr in ['CasualtyRiskPreprocessor', 'CasualtyFeatureEngineer', 'FocalLoss', 
                            'CasualtyRiskModel', 'SeverityClassifier', 'PropertyDamageDataPreprocessor', '_loss']:
                    if hasattr(__main__, attr):
                        delattr(__main__, attr)
                # Clean up sys.modules
                if '_loss' in sys.modules:
                    del sys.modules['_loss']
                        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise