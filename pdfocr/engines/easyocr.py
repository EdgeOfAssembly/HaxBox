#!/usr/bin/env python3
"""EasyOCR engine implementation.

This module provides an EasyOCR engine implementation that wraps the
easyocr library. EasyOCR is a deep learning-based OCR engine that supports
80+ languages and provides good accuracy with built-in text detection.

Features:
    - GPU acceleration support for faster processing
    - 80+ languages supported
    - Built-in text detection (no preprocessing needed)
    - Bounding box extraction with confidence scores
    - Thread-safe lazy initialization

Installation:
    pip install easyocr
    
    For GPU support:
        pip install torch torchvision  # CUDA-enabled PyTorch
        
Usage Example:
    ```python
    from pdfocr.engines.easyocr import EasyOCREngine
    from PIL import Image
    
    # Initialize with GPU
    engine = EasyOCREngine(lang="eng", gpu=True)
    
    # Perform OCR with bounding boxes
    image = Image.open("document.jpg")
    results = engine.ocr(image, return_boxes=True)
    for item in results:
        print(f"Text: {item['text']}, Confidence: {item['confidence']}")
    ```

Thread Safety:
    This engine uses thread-safe lazy loading with double-checked locking
    to ensure easyocr is imported and initialized only once. The reader
    is reinitialized if the language set changes.

Performance:
    EasyOCR can utilize GPU acceleration for significantly faster processing.
    GPU mode typically processes images 10-50x faster than CPU mode. However,
    initial model loading takes 5-30 seconds depending on hardware.

Limitations:
    - Requires ~2GB disk space for language models
    - Initial startup is slow (model download/loading)
    - GPU mode requires CUDA-compatible GPU with sufficient VRAM
"""

from __future__ import annotations

import threading
from typing import Any, Dict, FrozenSet, List, Optional, Union

from pdfocr.engines.base import OCREngine, register_engine
from pdfocr.types import TESSERACT_TO_EASYOCR_LANG

try:
    from PIL import Image
except ImportError:
    # Fallback type hint if PIL not installed (shouldn't happen in practice)
    Image = Any  # type: ignore[misc, assignment]

# ============================================================================
# GLOBAL STATE FOR LAZY LOADING
# ============================================================================

# Module-level cache for easyocr reader.
# Using None as sentinel to indicate "not yet loaded".
# The reader is expensive to initialize (loads neural network models),
# so we cache it to avoid repeated initialization overhead.
_easyocr_reader: Any = None

# Tracks the set of languages used to initialize the cached reader.
# If requested languages differ from cached languages, we reinitialize
# the reader. Uses FrozenSet for efficient comparison and immutability.
_easyocr_reader_langs: Optional[FrozenSet[str]] = None

# Thread lock for synchronizing easyocr reader initialization.
# Ensures thread-safe lazy loading with double-checked locking pattern.
_easyocr_lock = threading.Lock()

# Module-level cache for numpy (required by easyocr for image conversion).
_np: Any = None

# Thread lock for numpy import synchronization.
_numpy_lock = threading.Lock()


# ============================================================================
# LAZY LOADING HELPERS
# ============================================================================


def _get_numpy() -> Any:
    """Lazy import numpy with thread-safe double-checked locking.
    
    EasyOCR requires numpy to convert PIL Images to numpy arrays before
    processing. This function ensures numpy is imported exactly once.
    
    Pattern:
        1. First check (outside lock): Fast path for already-loaded module
        2. Lock acquisition: Only if not yet loaded
        3. Second check (inside lock): Prevents race condition
        4. Import: Performed only once
    
    Returns:
        The numpy module if available, otherwise None.
        
    Thread Safety:
        Fully thread-safe. Multiple threads can call this concurrently;
        numpy will be imported exactly once.
    """
    # First check: Fast path for already-loaded module
    global _np
    if _np is None:
        # Module not loaded yet, acquire lock for import
        with _numpy_lock:
            # Second check: Verify another thread didn't import while we waited
            if _np is None:
                try:
                    # Import numpy library
                    import numpy as np

                    # Cache the imported module for future calls
                    _np = np
                except ImportError:
                    # numpy not installed; leave _np as None
                    pass
    
    # Return cached module (or None if import failed)
    return _np


def _get_easyocr_reader(
    langs: List[str], gpu: bool = False
) -> Any:
    """Lazy import and initialize easyocr reader with thread-safe locking.
    
    This function implements double-checked locking to ensure the easyocr
    reader is initialized exactly once, or reinitialized if the language
    set changes. The reader initialization is expensive (loads neural
    network models), so caching is critical for performance.
    
    Reinitialization Logic:
        The reader is recreated if the requested language set differs from
        the cached language set. This allows switching languages at runtime
        without restarting the application, while still benefiting from
        caching when languages remain constant.
        
    Args:
        langs: List of EasyOCR language codes (e.g., ["en", "de", "fr"]).
               Should already be converted from tesseract codes using
               convert_language() before calling this function.
               
        gpu: Whether to use GPU acceleration. If True and GPU not available,
             easyocr automatically falls back to CPU.
             
    Returns:
        EasyOCR Reader instance if available, otherwise None.
        Returning None allows graceful handling of missing dependency.
        
    Thread Safety:
        Fully thread-safe. Multiple threads can call this concurrently;
        the reader will be initialized exactly once (or reinitialized if
        languages change), and all threads will receive the same instance.
        
    Performance:
        After first initialization, this is a fast frozenset comparison
        (if languages unchanged) or a full reinitialization (if changed).
        First initialization takes 5-30 seconds depending on hardware.
    """
    # Convert list to frozenset for efficient comparison
    # FrozenSet is hashable and immutable, perfect for caching keys
    requested_langs: FrozenSet[str] = frozenset(langs)
    
    # First check: Fast path if reader exists with same languages
    # Most calls after initialization take this path, avoiding lock overhead
    global _easyocr_reader, _easyocr_reader_langs
    
    # Use lock for all reader access to ensure consistency
    with _easyocr_lock:
        # Check if we need to reinitialize with different languages
        # Reader is cached only if languages match exactly
        if _easyocr_reader is not None and _easyocr_reader_langs == requested_langs:
            return _easyocr_reader
        
        # Need to initialize or reinitialize reader
        try:
            # Import easyocr library
            # This is an optional dependency; if not installed,
            # we catch ImportError and return None
            import easyocr
            
            # Initialize reader with requested languages and GPU setting
            # This is expensive: loads neural network models from disk
            # Models are cached by easyocr in ~/.EasyOCR/
            _easyocr_reader = easyocr.Reader(langs, gpu=gpu)
            
            # Cache the language set for future comparisons
            _easyocr_reader_langs = requested_langs
        except ImportError:
            # easyocr is an optional dependency; if not installed, return None
            # Calling code will raise ImportError with installation instructions
            pass
    
    # Return cached reader (or None if import failed)
    return _easyocr_reader


# ============================================================================
# EASYOCR ENGINE IMPLEMENTATION
# ============================================================================


@register_engine
class EasyOCREngine(OCREngine):
    """EasyOCR engine using deep learning for accurate text recognition.
    
    This engine wraps the easyocr library, which uses neural networks for
    both text detection and recognition. EasyOCR provides high accuracy,
    especially on challenging documents with complex layouts or low quality.
    
    Class Attributes:
        name: Engine identifier used in CLI (-e easyocr)
        display_name: Human-readable name for display/logging
        supports_gpu: True - EasyOCR supports GPU acceleration via PyTorch
        supports_boxes: True - Returns detailed bounding boxes with confidence
        install_hint: Installation instructions shown on ImportError
        
    Instance Attributes:
        lang: Tesseract language code (e.g., "eng", "deu")
        easyocr_langs: List of EasyOCR language codes (converted from tesseract)
        gpu: Whether to use GPU acceleration
        _reader: Cached EasyOCR Reader instance
        _np: Cached numpy module reference
        
    Initialization:
        The __init__ method performs lazy loading of easyocr and numpy using
        thread-safe helpers. Languages are converted from tesseract format to
        EasyOCR format. The reader is initialized on first use or when
        languages change.
        
    OCR Operation:
        The ocr() method converts PIL Image to numpy array and calls
        reader.readtext(). Returns either plain text or structured data
        with bounding boxes and confidence scores, depending on return_boxes.
        
    Thread Safety:
        Lazy loading is thread-safe via double-checked locking in helper
        functions. Multiple EasyOCREngine instances share the same cached
        reader if they use the same language set, ensuring memory efficiency.
        
    Example:
        ```python
        # Check availability before use
        if EasyOCREngine.is_available():
            # Initialize with multiple languages and GPU
            engine = EasyOCREngine(lang="eng+fra", gpu=True)
            
            # Perform OCR with bounding boxes
            text_with_boxes = engine.ocr(image, return_boxes=True)
            
            # Or just extract text
            text = engine.ocr(image, return_boxes=False)
        else:
            print(f"Install with: {EasyOCREngine.install_hint}")
        ```
    """

    # ========================================================================
    # CLASS ATTRIBUTES
    # ========================================================================
    
    # Engine identifier for CLI (-e easyocr) and registry lookup
    name: str = "easyocr"
    
    # Human-readable name for display in help text and error messages
    display_name: str = "EasyOCR"
    
    # EasyOCR supports GPU acceleration via PyTorch CUDA
    # If GPU requested but not available, automatically falls back to CPU
    supports_gpu: bool = True
    
    # EasyOCR provides detailed bounding boxes with confidence scores
    # Each result includes 4-point polygon bbox, text, and confidence
    supports_boxes: bool = True
    
    # Installation hint shown when easyocr is not available
    # User needs easyocr plus PyTorch for GPU support
    install_hint: str = "pip install easyocr"

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def __init__(self, lang: str = "eng", gpu: bool = False, **kwargs: Any) -> None:
        """Initialize EasyOCR engine.
        
        Performs lazy loading of easyocr and numpy libraries, converts
        language codes from tesseract to EasyOCR format, and stores
        configuration for reader initialization.
        
        Args:
            lang: Tesseract language code (default: "eng" for English).
                  Common codes: "eng", "deu", "fra", "spa", "chi_sim", "chi_tra".
                  Multiple languages can be combined with "+": "eng+fra".
                  Will be converted to EasyOCR format using TESSERACT_TO_EASYOCR_LANG.
                  
            gpu: Whether to use GPU acceleration (default: False).
                 If True, EasyOCR will attempt to use CUDA-enabled PyTorch.
                 If GPU not available, automatically falls back to CPU.
                 GPU mode is typically 10-50x faster than CPU.
                 
            **kwargs: Additional keyword arguments (currently unused).
                     Included for forward compatibility.
                     
        Raises:
            ImportError: If easyocr or numpy is not installed.
                        Error message includes installation instructions.
                        
        Notes:
            - The reader is initialized lazily on first ocr() call
            - Language models are downloaded automatically to ~/.EasyOCR/
            - First run may take time to download models (varies by language)
            - GPU mode requires CUDA-compatible GPU and CUDA toolkit
            
        Thread Safety:
            Multiple threads can call __init__ concurrently. The lazy loading
            helpers ensure thread-safe initialization of shared resources.
        """
        # Lazy load numpy using thread-safe helper
        # Returns None if numpy is not installed
        np_module = _get_numpy()
        
        # Validate that numpy is available
        # EasyOCR requires numpy for image array conversion
        if np_module is None:
            raise ImportError(
                "numpy not installed. "
                "Install with: pip install numpy"
            )
        
        # Cache numpy module reference for use in ocr() method
        self._np = np_module
        
        # Store original tesseract language code
        self.lang: str = lang
        
        # Store GPU preference for reader initialization
        self.gpu: bool = gpu
        
        # Convert tesseract language codes to EasyOCR format
        # Handle multiple languages separated by "+" (e.g., "eng+fra")
        tesseract_langs: List[str] = lang.split("+")
        self.easyocr_langs: List[str] = [
            self.convert_language(tess_lang) for tess_lang in tesseract_langs
        ]
        
        # Initialize reader lazily - get or create cached reader
        # This may take 5-30 seconds on first call (loads neural network models)
        # If reader with different languages exists, it will be reinitialized
        reader = _get_easyocr_reader(self.easyocr_langs, gpu=gpu)
        
        # Validate that easyocr is available
        # If None, reader initialization failed (easyocr not installed)
        if reader is None:
            raise ImportError(
                "easyocr not installed. "
                "Install with: pip install easyocr\n"
                "For GPU support, also install: pip install torch torchvision"
            )
        
        # Cache reader reference for use in ocr() method
        # Note: Reader may be shared across multiple engine instances
        # if they use the same language set (memory efficiency)
        self._reader = reader

    # ========================================================================
    # OCR OPERATION
    # ========================================================================

    def ocr(
        self, image: Image.Image, return_boxes: bool = False
    ) -> Union[str, List[Dict[str, Any]]]:
        """Perform OCR on an image using EasyOCR.
        
        Converts the PIL Image to a numpy array and performs OCR using the
        cached EasyOCR reader. Can return either plain text or structured
        data with bounding boxes and confidence scores.
        
        Args:
            image: PIL Image object to perform OCR on.
                   Can be any mode (RGB, L, RGBA, etc.); numpy handles conversion.
                   
            return_boxes: If True, return structured data with bounding boxes.
                         If False, return plain text string (default).
                         
        Returns:
            If return_boxes is False:
                str: Extracted text with line breaks between text regions.
                     Returns empty string if no text detected.
                     
            If return_boxes is True:
                List[Dict[str, Any]]: List of dictionaries, one per detected text region.
                Each dict contains:
                    - "text" (str): Recognized text
                    - "confidence" (float): Recognition confidence (0.0-1.0)
                    - "bbox" (List[List[int]]): 4-point polygon bounding box
                      Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                      Points are ordered clockwise from top-left
                      
        Raises:
            RuntimeError: If OCR operation fails (rare, usually GPU memory issues).
                         
        Notes:
            - EasyOCR performs both text detection and recognition
            - Results are sorted roughly top-to-bottom, left-to-right
            - Confidence scores are generally high (>0.9) for clear text
            - Bounding boxes are polygons (4 points) not axis-aligned rectangles
            
        Performance:
            GPU mode: 0.1-1 second per page (depending on GPU, image size)
            CPU mode: 1-10 seconds per page (depending on CPU, image size)
            
        Thread Safety:
            This method is thread-safe. The cached reader can be used
            concurrently by multiple threads.
        """
        # Convert PIL Image to numpy array
        # EasyOCR's readtext() expects numpy array, not PIL Image
        # Array is RGB or grayscale depending on image mode
        img_array = self._np.array(image)
        
        # Perform OCR using cached reader
        # readtext() returns list of (bbox, text, confidence) tuples
        # bbox: 4-point polygon [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        # text: recognized string
        # confidence: float in range [0.0, 1.0]
        results = self._reader.readtext(img_array)
        
        # Process results based on return_boxes parameter
        if return_boxes:
            # Return structured data with bounding boxes and confidence
            # Convert to list of dicts for consistent API across engines
            structured_results: List[Dict[str, Any]] = []
            for bbox, text, confidence in results:
                # bbox is a list of 4 points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                # Convert to integers for consistent output format
                structured_results.append(
                    {
                        "text": text,
                        "confidence": float(confidence),
                        "bbox": [[int(p[0]), int(p[1])] for p in bbox],
                    }
                )
            return structured_results
        
        # Return plain text: extract text from results and join with newlines
        # Each detected region becomes a separate line in the output
        text_parts: List[str] = [result[1] for result in results]
        return "\n".join(text_parts)

    # ========================================================================
    # AVAILABILITY CHECK
    # ========================================================================

    @classmethod
    def is_available(cls) -> bool:
        """Check if EasyOCR engine is available.
        
        Attempts to import easyocr to determine if this engine can be used.
        This is a class method so it can be called without instantiating the
        engine, useful for engine discovery and CLI fallback logic.
        
        Returns:
            True if easyocr is installed and can be imported.
            False if easyocr is not available.
            
        Notes:
            - Only checks Python package availability, not GPU support
            - GPU availability check would require PyTorch import (expensive)
            - If easyocr imports but GPU not available, CPU fallback is automatic
            - Does not check for language model availability (downloaded on first use)
            
        Performance:
            This method is lightweight after first call, as Python caches
            import failures. First call may take ~100-500ms for import attempt.
            
        Example:
            ```python
            if EasyOCREngine.is_available():
                engine = EasyOCREngine(lang="eng", gpu=True)
            else:
                print("EasyOCR not available")
                print(f"Install with: {EasyOCREngine.install_hint}")
            ```
        """
        try:
            # Try importing easyocr
            # If successful, engine is available
            import easyocr  # noqa: F401
            
            return True
        except ImportError:
            # easyocr not installed or import failed
            return False

    # ========================================================================
    # LANGUAGE CONVERSION
    # ========================================================================

    @classmethod
    def convert_language(cls, tesseract_lang: str) -> str:
        """Convert tesseract language code to EasyOCR format.
        
        EasyOCR uses different language codes than Tesseract (mostly ISO 639-1
        two-letter codes). This method maps from tesseract's three-letter codes
        to EasyOCR's format using the TESSERACT_TO_EASYOCR_LANG mapping.
        
        Args:
            tesseract_lang: Tesseract language code (e.g., "eng", "deu", "chi_sim").
            
        Returns:
            EasyOCR language code (e.g., "en", "de", "ch_sim").
            If mapping not found, returns first two characters as fallback.
            
        Notes:
            - Mapping is defined in pdfocr/types.py
            - Not all tesseract languages are supported by EasyOCR
            - Fallback may not work for all unmapped languages
            - Chinese variants use special codes: "ch_sim", "ch_tra"
            
        Examples:
            >>> EasyOCREngine.convert_language("eng")
            'en'
            >>> EasyOCREngine.convert_language("deu")
            'de'
            >>> EasyOCREngine.convert_language("chi_sim")
            'ch_sim'
            >>> EasyOCREngine.convert_language("unknown")
            'un'
        """
        # Look up language code in mapping from pdfocr/types.py
        # If not found, fallback to first 2 characters (ISO 639-1 approximation)
        return TESSERACT_TO_EASYOCR_LANG.get(tesseract_lang, tesseract_lang[:2])
