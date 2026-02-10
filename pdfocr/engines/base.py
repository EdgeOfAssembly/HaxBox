#!/usr/bin/env python3
"""Abstract base class for OCR engines.

This module defines the OCREngine abstract base class that all OCR engine
implementations must inherit from. It establishes a consistent interface
for engine discovery, initialization, and OCR operations.

Adding a New OCR Engine:
    1. Create a new file in pdfocr/engines/ (e.g., my_engine.py)
    2. Import OCREngine from this module
    3. Subclass OCREngine and implement all abstract methods
    4. Use @register_engine decorator on your class
    5. Add the engine name to CLI choices in cli.py
    
Example:
    ```python
    from pdfocr.engines.base import OCREngine, register_engine
    from PIL import Image
    
    @register_engine
    class MyEngine(OCREngine):
        name = "myengine"
        display_name = "My Custom OCR Engine"
        supports_gpu = True
        supports_boxes = True
        install_hint = "pip install my-ocr-library"
        
        def __init__(self, lang: str = "eng", gpu: bool = False, **kwargs):
            self.lang = lang
            self.gpu = gpu
            # Initialize your engine here
            
        def ocr(self, image: Image.Image, return_boxes: bool = False):
            # Perform OCR
            if return_boxes:
                return [{"bbox": [...], "text": "...", "confidence": 0.95}]
            return "extracted text"
            
        @classmethod
        def is_available(cls) -> bool:
            try:
                import my_ocr_library
                return True
            except ImportError:
                return False
    ```

Threading:
    All engines use lazy initialization with thread-safe locking in their
    __init__ methods. The engine registry instantiates engines only when
    requested, so expensive imports and model loading happen just-in-time.

GPU Support:
    Engines that support GPU acceleration should check the `gpu` parameter
    in __init__ and configure their backend accordingly. Some engines may
    automatically fall back to CPU if GPU memory is exhausted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Type, Dict

try:
    from PIL import Image
except ImportError:
    # Fallback type hint if PIL not installed (shouldn't happen in practice)
    Image = Any  # type: ignore[misc, assignment]

# ============================================================================
# ENGINE REGISTRY
# ============================================================================

# Global registry mapping engine name → engine class (not instance).
# Engines register themselves using the @register_engine decorator.
_REGISTRY: Dict[str, Type["OCREngine"]] = {}


def register_engine(cls: Type["OCREngine"]) -> Type["OCREngine"]:
    """Decorator to register an OCR engine class in the global registry.
    
    This decorator should be applied to all OCREngine subclasses. It adds
    the engine to the _REGISTRY dict, making it discoverable by get_engine()
    and available_engines().
    
    Args:
        cls: The OCREngine subclass to register.
        
    Returns:
        The same class, unmodified (allows decorator use).
        
    Example:
        ```python
        @register_engine
        class TesseractEngine(OCREngine):
            name = "tesseract"
            # ...
        ```
    """
    _REGISTRY[cls.name] = cls
    return cls


def get_registry() -> Dict[str, Type["OCREngine"]]:
    """Get the complete engine registry.
    
    Returns:
        Dict mapping engine names to engine classes.
    """
    return _REGISTRY.copy()


# ============================================================================
# ABSTRACT BASE CLASS
# ============================================================================


class OCREngine(ABC):
    """Abstract base class for OCR engines.
    
    All OCR engines must inherit from this class and implement its abstract
    methods. The class defines the contract that all engines must follow,
    ensuring a consistent interface for the rest of the pdfocr package.
    
    Class Attributes:
        name: Unique identifier for this engine (lowercase, no spaces).
              Used in CLI with -e/--engine flag. Example: "tesseract"
        display_name: Human-readable name for display. Example: "Tesseract OCR"
        supports_gpu: Whether this engine can use GPU acceleration.
                      If False, --gpu flag has no effect.
        supports_boxes: Whether return_boxes=True produces real bounding boxes.
                       If False, return_boxes may be ignored or return empty boxes.
        install_hint: Installation command hint for users.
                     Example: "pip install pytesseract"
    
    Instance Lifecycle:
        1. User selects engine via CLI (-e tesseract) or API (get_engine("tesseract"))
        2. Engine registry looks up the class in _REGISTRY
        3. __init__() is called with user-specified parameters (lang, gpu, etc.)
        4. Engine initializes lazily (imports libraries, loads models)
        5. ocr() method is called for each image to process
        
    Thread Safety:
        Engines should implement thread-safe lazy initialization if they use
        shared resources (models, readers, etc.). Use threading.Lock() for
        synchronization if needed.
    """

    # ========================================================================
    # CLASS ATTRIBUTES (must be defined by each engine)
    # ========================================================================

    name: str = ""  # e.g., "tesseract"
    display_name: str = ""  # e.g., "Tesseract OCR"
    supports_gpu: bool = False  # Does this engine support GPU?
    supports_boxes: bool = False  # Does return_boxes work?
    install_hint: str = ""  # e.g., "pip install pytesseract"

    # ========================================================================
    # ABSTRACT METHODS (must be implemented by each engine)
    # ========================================================================

    @abstractmethod
    def __init__(self, lang: str = "eng", gpu: bool = False, **kwargs: Any) -> None:
        """Initialize the OCR engine.
        
        This method is called lazily—only when the user selects this engine.
        Expensive operations (imports, model loading) should happen here.
        
        Args:
            lang: Language code in tesseract format (e.g., "eng", "deu", "chi_sim").
                  Engines should convert to their own format using convert_language().
            gpu: Whether to use GPU acceleration (if supported).
            **kwargs: Engine-specific options (e.g., batch_size for PaddleOCR).
                     Document accepted kwargs in subclass docstring.
        
        Raises:
            ImportError: If required dependencies are not installed.
            RuntimeError: If engine initialization fails.
        """
        ...

    @abstractmethod
    def ocr(self, image: Image.Image, return_boxes: bool = False) -> Any:
        """Perform OCR on a single image.
        
        The image is already preprocessed (CLAHE enhancement) if requested
        by the user. This method should focus only on the OCR operation.
        
        Args:
            image: PIL Image to perform OCR on. Already preprocessed.
            return_boxes: If True, return structured data with bounding boxes.
                         If False, return plain text string.
                         
        Returns:
            If return_boxes is False: str with extracted text.
            If return_boxes is True: Engine-specific structured data.
                - EasyOCR: List of [bbox, text, confidence]
                - PaddleOCR: List of [bbox, (text, confidence)]
                - docTR: Dict with nested word/line/block structure
                - Tesseract: May return empty list or approximate boxes
                
        Raises:
            RuntimeError: If OCR operation fails.
            
        Notes:
            - The image is not modified by this method
            - Text should be UTF-8 encoded
            - Bounding box formats vary by engine (coordinate systems differ)
        """
        ...

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check if this engine's dependencies are installed.
        
        This is a class method so it can be called without instantiating
        the engine. Used for engine discovery and fallback logic.
        
        Returns:
            True if the engine can be used, False if dependencies are missing.
            
        Example:
            ```python
            @classmethod
            def is_available(cls) -> bool:
                try:
                    import pytesseract
                    return True
                except ImportError:
                    return False
            ```
        """
        ...

    # ========================================================================
    # OPTIONAL METHODS (can be overridden if needed)
    # ========================================================================

    @classmethod
    def convert_language(cls, tesseract_lang: str) -> str:
        """Convert tesseract language code to this engine's format.
        
        Many engines use different language code conventions. This method
        provides a hook for engines to convert from the standardized
        tesseract codes (which users provide) to engine-specific codes.
        
        Args:
            tesseract_lang: Tesseract language code (e.g., "eng", "deu", "chi_sim").
            
        Returns:
            Language code in this engine's format.
            
        Notes:
            - Default implementation returns input unchanged
            - Override in subclass if engine uses different codes
            - Should handle unknown codes gracefully (return default or original)
            
        Example:
            ```python
            @classmethod
            def convert_language(cls, tesseract_lang: str) -> str:
                mapping = {"eng": "en", "deu": "de", "fra": "fr"}
                return mapping.get(tesseract_lang, tesseract_lang[:2])
            ```
        """
        return tesseract_lang
