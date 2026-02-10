#!/usr/bin/env python3
"""Tesseract OCR engine implementation.

This module provides a Tesseract OCR engine implementation that wraps
the pytesseract library. Tesseract is a mature, open-source OCR engine
originally developed by HP and now maintained by Google. It offers fast
OCR with good accuracy on clear text.

Features:
    - Fast OCR processing (CPU-only)
    - Support for 100+ languages
    - Reliable text extraction
    - No GPU acceleration (CPU-only)
    - Limited bounding box support

Installation:
    pip install pytesseract
    
    Also requires tesseract binary:
        Ubuntu/Debian: apt install tesseract-ocr
        macOS: brew install tesseract
        Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

Usage Example:
    ```python
    from pdfocr.engines.tesseract import TesseractEngine
    from PIL import Image
    
    # Initialize engine
    engine = TesseractEngine(lang="eng")
    
    # Perform OCR
    image = Image.open("document.jpg")
    text = engine.ocr(image)
    print(text)
    ```

Thread Safety:
    This engine uses thread-safe lazy loading with double-checked locking
    to ensure pytesseract is imported only once across all threads.

Performance:
    Tesseract is CPU-only and generally faster than GPU-based engines for
    small batches of images. For large-scale processing, consider PaddleOCR
    or EasyOCR with GPU acceleration.

Limitations:
    - No GPU support
    - No bounding box support (supports_boxes=False)
    - Requires external tesseract binary installation
    - Performance degrades on low-quality images
"""

from __future__ import annotations

import threading
from typing import Any

from pdfocr.engines.base import OCREngine, register_engine

try:
    from PIL import Image
except ImportError:
    # Fallback type hint if PIL not installed (shouldn't happen in practice)
    Image = Any  # type: ignore[misc, assignment]

# ============================================================================
# GLOBAL STATE FOR LAZY LOADING
# ============================================================================

# Module-level cache for pytesseract library.
# Using None as sentinel to indicate "not yet loaded".
# This enables lazy loading: we only import pytesseract when TesseractEngine
# is first instantiated, avoiding unnecessary imports if user chooses a
# different OCR engine.
_pytesseract: Any = None

# Thread lock for synchronizing pytesseract import.
# This ensures thread-safe lazy loading with double-checked locking pattern:
# 1. Check if _pytesseract is None (fast path, no lock)
# 2. Acquire lock if None
# 3. Check again inside lock (second check)
# 4. Import if still None
# This pattern minimizes lock contention while ensuring single initialization.
_pytesseract_lock = threading.Lock()


# ============================================================================
# LAZY LOADING HELPER
# ============================================================================


def _get_pytesseract() -> Any:
    """Lazy import pytesseract with thread-safe double-checked locking.
    
    This function implements the double-checked locking pattern to ensure
    pytesseract is imported exactly once, even in multi-threaded scenarios.
    
    Pattern explanation:
        1. First check (outside lock): Fast path for already-loaded module.
           Most calls after initialization take this path, avoiding lock overhead.
           
        2. Lock acquisition: Only acquired if module not yet loaded.
        
        3. Second check (inside lock): Prevents race condition where multiple
           threads pass the first check simultaneously. Only the first thread
           through will actually import; others will see _pytesseract is set.
           
        4. Import: Performed only by the first thread to reach this point.
    
    Returns:
        The pytesseract module if available, otherwise None.
        Returning None allows graceful handling of missing dependency:
        TesseractEngine.__init__() can raise ImportError with helpful message.
        
    Thread Safety:
        Fully thread-safe. Multiple threads can call this concurrently;
        pytesseract will be imported exactly once.
        
    Performance:
        After first import, this is a simple None check (no lock overhead).
    """
    # First check: Fast path for already-loaded module
    # Uses global keyword to read/write module-level _pytesseract variable
    global _pytesseract
    if _pytesseract is None:
        # Module not loaded yet, acquire lock for import
        with _pytesseract_lock:
            # Second check: Verify another thread didn't import while we waited
            # This is the key to double-checked locking pattern
            if _pytesseract is None:
                try:
                    # Import pytesseract library
                    # This is an optional dependency; if not installed,
                    # we catch ImportError and leave _pytesseract as None
                    import pytesseract

                    # Cache the imported module for future calls
                    _pytesseract = pytesseract
                except ImportError:
                    # pytesseract not installed; leave _pytesseract as None
                    # TesseractEngine.__init__() will raise informative error
                    pass
    
    # Return cached module (or None if import failed)
    return _pytesseract


# ============================================================================
# TESSERACT ENGINE IMPLEMENTATION
# ============================================================================


@register_engine
class TesseractEngine(OCREngine):
    """Tesseract OCR engine using pytesseract wrapper.
    
    This engine wraps the pytesseract library, which in turn wraps the
    tesseract command-line tool. Tesseract is a mature, widely-used OCR
    engine with support for 100+ languages.
    
    Class Attributes:
        name: Engine identifier used in CLI (-e tesseract)
        display_name: Human-readable name for display/logging
        supports_gpu: False - Tesseract is CPU-only
        supports_boxes: False - This implementation doesn't provide bounding boxes
        install_hint: Installation instructions shown on ImportError
        
    Instance Attributes:
        lang: Tesseract language code (e.g., "eng", "deu", "chi_sim")
        _pytesseract: Cached pytesseract module reference
        
    Initialization:
        The __init__ method performs lazy loading of pytesseract using
        _get_pytesseract(). If pytesseract is not installed, raises
        ImportError with installation instructions.
        
    OCR Operation:
        The ocr() method calls pytesseract.image_to_string() to extract
        text from a PIL Image. Returns plain text string; does not
        support bounding boxes (supports_boxes=False).
        
    Thread Safety:
        Lazy loading is thread-safe via double-checked locking in
        _get_pytesseract(). Multiple TesseractEngine instances can
        safely coexist across threads.
        
    Example:
        ```python
        # Check if available before use
        if TesseractEngine.is_available():
            engine = TesseractEngine(lang="eng")
            text = engine.ocr(my_image)
        else:
            print("Please install: pip install pytesseract")
        ```
    """

    # ========================================================================
    # CLASS ATTRIBUTES
    # ========================================================================
    
    # Engine identifier for CLI (-e tesseract) and registry lookup
    name: str = "tesseract"
    
    # Human-readable name for display in help text and error messages
    display_name: str = "Tesseract OCR"
    
    # Tesseract does not support GPU acceleration
    # GPU parameter in __init__() is ignored
    supports_gpu: bool = False
    
    # This implementation does not provide bounding boxes
    # pytesseract can extract boxes via image_to_data(), but we keep
    # the interface simple by only supporting text extraction
    supports_boxes: bool = False
    
    # Installation hint shown when pytesseract is not available
    # User needs both the Python wrapper and the binary
    install_hint: str = "pip install pytesseract"

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def __init__(self, lang: str = "eng", gpu: bool = False, **kwargs: Any) -> None:
        """Initialize Tesseract OCR engine.
        
        Performs lazy loading of pytesseract library and validates that
        it's available. Stores the language parameter for use during OCR.
        
        Args:
            lang: Tesseract language code (default: "eng" for English).
                  Common codes: "eng", "deu", "fra", "spa", "chi_sim", "chi_tra".
                  Multiple languages can be combined with "+": "eng+fra".
                  See tesseract --list-langs for available languages.
                  
            gpu: Ignored for Tesseract (CPU-only engine).
                 Included for API consistency across engines.
                 
            **kwargs: Additional keyword arguments (ignored).
                     Included for forward compatibility if we add
                     tesseract-specific options in the future.
                     
        Raises:
            ImportError: If pytesseract is not installed. Error message
                        includes installation instructions.
                        
        Notes:
            - Language data must be installed separately via tesseract-ocr-<lang>
            - If language not installed, tesseract will fail at OCR time
            - No validation is performed in __init__ for performance
            
        Thread Safety:
            Multiple threads can call __init__ concurrently. The _get_pytesseract()
            helper ensures thread-safe lazy loading of the pytesseract module.
        """
        # Lazy load pytesseract using thread-safe helper
        # Returns None if pytesseract is not installed
        pytesseract = _get_pytesseract()
        
        # Validate that pytesseract is available
        # If None, raise ImportError with helpful installation hint
        if pytesseract is None:
            raise ImportError(
                "pytesseract not installed. "
                "Install with: pip install pytesseract\n"
                "Also requires tesseract binary:\n"
                "  Ubuntu/Debian: apt install tesseract-ocr\n"
                "  macOS: brew install tesseract\n"
                "  Windows: https://github.com/UB-Mannheim/tesseract/wiki"
            )
        
        # Cache pytesseract module reference for use in ocr() method
        # Avoids repeated _get_pytesseract() calls during OCR operations
        self._pytesseract = pytesseract
        
        # Store language parameter for OCR operations
        # Passed to pytesseract.image_to_string(lang=...) in ocr() method
        self.lang: str = lang
        
        # Note: gpu parameter is ignored (Tesseract is CPU-only)
        # Note: **kwargs are ignored (no additional options currently supported)

    # ========================================================================
    # OCR OPERATION
    # ========================================================================

    def ocr(self, image: Image.Image, return_boxes: bool = False) -> str:
        """Perform OCR on an image using Tesseract.
        
        Extracts text from the provided PIL Image using pytesseract.
        The image should be preprocessed (if requested) before calling this
        method; this method only performs the OCR operation itself.
        
        Args:
            image: PIL Image object to perform OCR on.
                   Can be any mode (RGB, L, RGBA, etc.); pytesseract handles conversion.
                   
            return_boxes: Ignored (Tesseract engine doesn't support bounding boxes).
                         Included for API consistency across engines.
                         supports_boxes=False indicates this parameter has no effect.
                         
        Returns:
            Extracted text as a string.
            Text is UTF-8 encoded and preserves line breaks from the document.
            Returns empty string if no text detected.
            
        Raises:
            RuntimeError: If tesseract binary is not found or OCR fails.
                         This can happen if tesseract is not installed or if
                         the requested language is not available.
                         
        Notes:
            - Line breaks in the original document are preserved in output
            - Confidence scores are not available in this simple interface
            - For bounding boxes, use an engine with supports_boxes=True
            
        Performance:
            Tesseract processes images on CPU. Performance scales with image
            size and complexity. Typical processing time: 0.5-2 seconds per
            page on modern hardware.
            
        Thread Safety:
            This method is thread-safe. Multiple threads can call ocr()
            on the same or different TesseractEngine instances concurrently.
        """
        # Call pytesseract.image_to_string() to perform OCR
        # lang parameter specifies the language(s) to use for recognition
        # Returns extracted text as a string with preserved line breaks
        result: str = self._pytesseract.image_to_string(image, lang=self.lang)
        
        # Return extracted text
        # Note: return_boxes parameter is ignored (supports_boxes=False)
        return result

    # ========================================================================
    # AVAILABILITY CHECK
    # ========================================================================

    @classmethod
    def is_available(cls) -> bool:
        """Check if Tesseract engine is available.
        
        Attempts to import pytesseract to determine if this engine can be used.
        This is a class method so it can be called without instantiating the
        engine, useful for engine discovery and CLI fallback logic.
        
        Returns:
            True if pytesseract is installed and can be imported.
            False if pytesseract is not available.
            
        Notes:
            - Only checks Python package availability, not tesseract binary
            - Tesseract binary check would require subprocess call (expensive)
            - If pytesseract imports but binary missing, error occurs at OCR time
            
        Performance:
            This method is lightweight after first call, as Python caches
            import failures. First call may take ~10-50ms for import attempt.
            
        Example:
            ```python
            if TesseractEngine.is_available():
                engine = TesseractEngine()
            else:
                print("Tesseract not available")
                print(f"Install with: {TesseractEngine.install_hint}")
            ```
        """
        try:
            # Try importing pytesseract
            # If successful, engine is available
            import pytesseract  # noqa: F401
            
            return True
        except ImportError:
            # pytesseract not installed or import failed
            return False
