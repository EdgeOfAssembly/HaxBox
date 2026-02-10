#!/usr/bin/env python3
"""OCR engine registry and factory functions.

This module provides the public API for working with OCR engines:
- get_engine(): Factory function to instantiate engines by name
- available_engines(): List installed/usable engines
- Engine auto-discovery through imports

All engine modules in this package should register themselves using the
@register_engine decorator from base.py. This module then imports all
engines to trigger registration.

Usage:
    # Get an engine instance
    engine = get_engine("tesseract", lang="eng", gpu=False)
    
    # Check what's available
    engines = available_engines()
    print(f"Available engines: {engines}")
    
    # Use the engine
    result = engine.ocr(image, return_boxes=False)
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

from pdfocr.engines.base import OCREngine, get_registry

# ============================================================================
# IMPORT ALL ENGINES TO TRIGGER REGISTRATION
# ============================================================================
# Each engine module uses @register_engine decorator, which adds the engine
# to the registry when the module is imported. By importing all engines here,
# we ensure they're all registered and discoverable.

from pdfocr.engines import tesseract  # noqa: F401
from pdfocr.engines import easyocr  # noqa: F401
from pdfocr.engines import trocr  # noqa: F401
from pdfocr.engines import paddleocr  # noqa: F401
from pdfocr.engines import doctr  # noqa: F401

# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "get_engine",
    "available_engines",
    "get_all_engines",
    "OCREngine",
]


def get_engine(name: str, **kwargs: Any) -> OCREngine:
    """Factory function to instantiate an OCR engine by name.
    
    This is the main entry point for getting engine instances. It looks up
    the engine class in the registry and instantiates it with the provided
    parameters.
    
    Args:
        name: Engine name (e.g., "tesseract", "easyocr", "trocr").
        **kwargs: Engine-specific initialization parameters.
                 Common parameters:
                   - lang: Language code (default: "eng")
                   - gpu: Use GPU acceleration (default: False)
                 Engine-specific:
                   - batch_size: For PaddleOCR
                   - model_variant: For TrOCR ("printed" or "handwritten")
                   
    Returns:
        Initialized OCREngine instance.
        
    Raises:
        KeyError: If the engine name is not recognized.
        ImportError: If the engine's dependencies are not installed.
        
    Example:
        ```python
        # Get tesseract with German language
        engine = get_engine("tesseract", lang="deu")
        
        # Get EasyOCR with GPU
        engine = get_engine("easyocr", lang="eng", gpu=True)
        
        # Get PaddleOCR with custom batch size
        engine = get_engine("paddleocr", lang="eng", batch_size=4)
        ```
    """
    registry = get_registry()
    
    if name not in registry:
        available = list(registry.keys())
        raise KeyError(
            f"Unknown OCR engine: '{name}'. "
            f"Available engines: {', '.join(available)}"
        )
    
    engine_class = registry[name]
    return engine_class(**kwargs)


def available_engines() -> List[str]:
    """Get list of engines whose dependencies are installed.
    
    This function checks each registered engine's is_available() method
    to determine if it can actually be used (i.e., its dependencies are
    installed and functional).
    
    Returns:
        List of engine names that can be instantiated.
        
    Example:
        ```python
        engines = available_engines()
        # ['tesseract', 'easyocr']  (if only these two are installed)
        
        if 'paddleocr' in engines:
            engine = get_engine('paddleocr')
        ```
    """
    registry = get_registry()
    available: List[str] = []
    
    for name, engine_class in registry.items():
        if engine_class.is_available():
            available.append(name)
    
    return available


def get_all_engines() -> Dict[str, Type[OCREngine]]:
    """Get dictionary of all registered engines (installed or not).
    
    This returns the complete registry, including engines whose dependencies
    may not be installed. Use available_engines() to filter to only usable
    engines.
    
    Returns:
        Dict mapping engine names to engine classes.
        
    Example:
        ```python
        all_engines = get_all_engines()
        for name, engine_class in all_engines.items():
            print(f"{name}: {engine_class.display_name}")
            print(f"  Available: {engine_class.is_available()}")
            print(f"  Install: {engine_class.install_hint}")
        ```
    """
    return get_registry()
