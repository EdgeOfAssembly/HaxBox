#!/usr/bin/env python3
"""PaddleOCR engine implementation - Complete rewrite for stability.

This is a ground-up rewrite of the PaddleOCR integration to address persistent
bugs that have required five separate bug-fix PRs (#9, #11, #12, #13, #14).

CRITICAL WORKAROUNDS:
    1. ENV VAR FIX (GitHub issue PaddlePaddle/Paddle#59989):
       PaddlePaddle 3.0+ has PIR CPU execution bugs that cause crashes.
       We set FLAGS_use_mkldnn and FLAGS_use_onednn to '0' BEFORE imports.
       
    2. API COMPATIBILITY:
       PaddleOCR 3.0+ changed parameter names (device vs use_gpu, predict vs ocr).
       We only support 3.0+ to avoid maintaining legacy compatibility.
       
    3. GPU OOM FALLBACK:
       GPU operations may fail with "out of memory" or "ResourceExhausted".
       We automatically fall back to CPU mode and log a warning.

This module provides a PaddleOCR engine implementation that wraps the
paddleocr library. PaddleOCR is a practical OCR system that supports 80+
languages with good performance on both CPU and GPU.

Features:
    - GPU acceleration with automatic CPU fallback on OOM
    - 80+ languages supported
    - Built-in text detection and recognition
    - Bounding box extraction with confidence scores
    - Thread-safe lazy initialization
    - Configurable batch size for memory optimization

Installation:
    pip install paddleocr paddlepaddle
    
    For GPU support:
        pip install paddlepaddle-gpu
        
Usage Example:
    ```python
    from pdfocr.engines.paddleocr import PaddleOCREngine
    from PIL import Image
    
    # Initialize with GPU and custom batch size
    engine = PaddleOCREngine(lang="eng", gpu=True, batch_size=4)
    
    # Perform OCR with bounding boxes
    image = Image.open("document.jpg")
    results = engine.ocr(image, return_boxes=True)
    for item in results:
        print(f"Text: {item['text']}, Confidence: {item['confidence']}")
    ```

Thread Safety:
    This engine uses thread-safe lazy loading with double-checked locking
    to ensure paddleocr is imported and initialized only once per configuration.
    The reader is cached by (lang, gpu, batch_size) to avoid re-initialization.

Performance:
    PaddleOCR can utilize GPU acceleration for significantly faster processing.
    Batch size affects memory usage: higher values are faster but use more VRAM.
    Default batch_size=1 minimizes memory footprint for stability.

Limitations:
    - Requires ~500MB-2GB disk space for language models
    - Initial startup is slow (model download/loading)
    - GPU mode requires sufficient VRAM (varies by batch size)
    - PaddlePaddle 3.0+ is required (older versions not supported)

References:
    - PaddleOCR GitHub: https://github.com/PaddlePaddle/PaddleOCR
    - PaddlePaddle Issue #59989: PIR CPU execution crash bug
    - Previous bug-fix PRs: #9, #11, #12, #13, #14
"""

from __future__ import annotations

# ============================================================================
# CRITICAL: ENV VAR WORKAROUND MUST COME BEFORE PADDLEOCR IMPORTS
# ============================================================================
# PaddlePaddle 3.0+ has a bug (GitHub issue #59989) where PIR CPU execution
# crashes with "NotImplementedError: set_tensor is not implemented on oneDNN".
# The workaround is to disable MKL-DNN/oneDNN BEFORE importing paddle.
# These env vars MUST be set at module level, before any code runs.

import os

# Set flags multiple times to ensure they take effect (paranoid defense)
os.environ['FLAGS_use_mkldnn'] = 'False'  # String 'False' for compatibility
os.environ['FLAGS_use_mkldnn'] = '0'      # Numeric '0' as backup
os.environ['FLAGS_use_onednn'] = '0'      # Disable oneDNN (new name for MKL-DNN)

# ============================================================================
# NOW SAFE TO IMPORT OTHER MODULES
# ============================================================================

import sys
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

from pdfocr.engines.base import OCREngine, register_engine
from pdfocr.types import TESSERACT_TO_PADDLEOCR_LANG

try:
    from PIL import Image
except ImportError:
    # Fallback type hint if PIL not installed (shouldn't happen in practice)
    Image = Any  # type: ignore[misc, assignment]

# ============================================================================
# GLOBAL STATE FOR LAZY LOADING
# ============================================================================

# Module-level cache for PaddleOCR instance.
# Format: (instance, cache_key) where cache_key = (lang, gpu, batch_size).
# The instance is expensive to initialize (loads neural network models),
# so we cache it to avoid repeated initialization overhead.
# We store the cache key to detect when parameters change and re-init is needed.
_paddleocr_instance: Optional[Tuple[Any, Tuple[str, bool, int]]] = None

# Thread lock for synchronizing PaddleOCR instance initialization.
# Ensures thread-safe lazy loading with double-checked locking pattern.
_paddleocr_lock = threading.Lock()

# Module-level cache for numpy (required by PaddleOCR for image conversion).
_np: Any = None

# Thread lock for numpy import synchronization.
_numpy_lock = threading.Lock()


# ============================================================================
# LAZY LOADING HELPERS
# ============================================================================


def _get_numpy() -> Any:
    """Lazy import numpy with thread-safe double-checked locking.
    
    PaddleOCR requires numpy to convert PIL Images to numpy arrays before
    processing. We import numpy lazily to avoid import-time overhead.
    
    Returns:
        numpy module, or None if numpy is not installed.
        
    Thread Safety:
        Uses double-checked locking to ensure numpy is imported only once
        even when called from multiple threads simultaneously.
    """
    global _np
    
    # Fast path: numpy already imported (no lock needed)
    if _np is not None:
        return _np
    
    # Slow path: first import (lock required)
    with _numpy_lock:
        # Double-check: another thread may have imported while we waited
        if _np is not None:
            return _np
        
        try:
            import numpy as np_module
            _np = np_module
            return _np
        except ImportError:
            return None


def _create_paddleocr_instance(
    lang: str,
    gpu: bool,
    batch_size: int,
) -> Any:
    """Create a new PaddleOCR instance with the specified configuration.
    
    This is a helper function that handles the complexity of PaddleOCR
    initialization, including API compatibility checks and proper error
    handling.
    
    Args:
        lang: PaddleOCR language code (e.g., "en", "ch", "german").
        gpu: Whether to use GPU acceleration.
        batch_size: Batch size for text recognition model.
        
    Returns:
        Initialized PaddleOCR instance.
        
    Raises:
        ImportError: If paddleocr is not installed, or if the installed
                    version is too old (< 3.0) and doesn't support the
                    new parameter names.
        RuntimeError: If PaddleOCR initialization fails for other reasons.
        
    Notes:
        - Only supports PaddleOCR 3.0+
        - Uses 'device' parameter (not deprecated 'use_gpu')
        - Uses 'use_textline_orientation' (not deprecated 'use_angle_cls')
        - Uses 'text_recognition_batch_size' (not deprecated 'rec_batch_num')
        - Does NOT use show_log, det_batch_size, or cls parameters
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise ImportError(
            "PaddleOCR not installed. Install with: "
            "pip install paddleocr paddlepaddle"
        )
    
    try:
        # PaddleOCR 3.0+ API:
        # - device='gpu'/'cpu' replaces use_gpu=True/False
        # - use_textline_orientation=True replaces use_angle_cls=True
        # - text_recognition_batch_size=N replaces rec_batch_num=N
        # - NO show_log parameter (removed in 3.0+)
        # - NO det_batch_size parameter (removed in 3.0+)
        # - NO cls parameter (not needed with use_textline_orientation)
        return PaddleOCR(
            device='gpu' if gpu else 'cpu',
            use_textline_orientation=True,  # Enable text orientation detection
            lang=lang,
            text_recognition_batch_size=batch_size,
        )
    except TypeError as e:
        # If we get TypeError on the 3.0+ parameters, the user has an old version
        raise ImportError(
            f"PaddleOCR 3.0+ is required but an older version is installed. "
            f"Upgrade with: pip install --upgrade paddleocr paddlepaddle\n"
            f"Original error: {e}"
        )


def _get_paddleocr(
    lang: str,
    gpu: bool,
    batch_size: int,
) -> Any:
    """Get or create cached PaddleOCR instance with thread-safe lazy loading.
    
    This function implements the caching logic for PaddleOCR instances.
    If an instance with matching parameters exists in the cache, it's reused.
    Otherwise, a new instance is created and cached.
    
    Args:
        lang: PaddleOCR language code (e.g., "en", "ch", "german").
        gpu: Whether to use GPU acceleration.
        batch_size: Batch size for text recognition model.
        
    Returns:
        Cached or newly created PaddleOCR instance.
        
    Raises:
        ImportError: If paddleocr is not installed or is too old.
        RuntimeError: If PaddleOCR initialization fails.
        
    Thread Safety:
        Uses double-checked locking to ensure thread-safe instance creation
        and caching. Multiple threads can safely call this function.
        
    Notes:
        - Cache key is (lang, gpu, batch_size) tuple
        - If any parameter differs from cached instance, we reinitialize
        - This avoids expensive re-initialization in typical usage patterns
          where parameters remain constant across multiple OCR operations
    """
    global _paddleocr_instance
    
    cache_key = (lang, gpu, batch_size)
    
    # Fast path: instance exists with matching parameters (no lock needed)
    if _paddleocr_instance is not None:
        instance, cached_key = _paddleocr_instance
        if cached_key == cache_key:
            return instance
    
    # Slow path: need to create or recreate instance (lock required)
    with _paddleocr_lock:
        # Double-check: another thread may have created while we waited
        if _paddleocr_instance is not None:
            instance, cached_key = _paddleocr_instance
            if cached_key == cache_key:
                return instance
        
        # Create new instance with requested parameters
        instance = _create_paddleocr_instance(lang, gpu, batch_size)
        _paddleocr_instance = (instance, cache_key)
        return instance


# ============================================================================
# OCR ENGINE IMPLEMENTATION
# ============================================================================


@register_engine
class PaddleOCREngine(OCREngine):
    """PaddleOCR engine implementation.
    
    This engine wraps the PaddleOCR library, providing OCR capabilities
    for 80+ languages with GPU acceleration support and automatic CPU
    fallback on out-of-memory errors.
    
    Class Attributes:
        name: Engine identifier for CLI and API ("paddleocr")
        display_name: Human-readable name ("PaddleOCR")
        supports_gpu: GPU acceleration available (True)
        supports_boxes: Bounding box extraction available (True)
        install_hint: Installation command for users
        
    Instance Attributes:
        lang: Language code in tesseract format (converted internally)
        gpu: Whether GPU acceleration is enabled
        batch_size: Batch size for text recognition model
        _engine_lang: Language code in PaddleOCR format (after conversion)
    """
    
    # ========================================================================
    # CLASS ATTRIBUTES
    # ========================================================================
    
    name: str = "paddleocr"
    display_name: str = "PaddleOCR"
    supports_gpu: bool = True
    supports_boxes: bool = True
    install_hint: str = "pip install paddleocr paddlepaddle"
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    def __init__(
        self,
        lang: str = "eng",
        gpu: bool = False,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize PaddleOCR engine.
        
        This constructor validates parameters and converts the language code
        from tesseract format to PaddleOCR format. Actual model loading happens
        lazily on first OCR call to minimize startup time.
        
        Args:
            lang: Language code in tesseract format (e.g., "eng", "deu", "chi_sim").
                  Will be automatically converted to PaddleOCR format.
            gpu: Whether to use GPU acceleration. If True and GPU OOM occurs,
                 automatically falls back to CPU mode.
            batch_size: Batch size for text recognition model. Higher values
                       are faster but use more memory. Default: 1 for stability.
            **kwargs: Additional keyword arguments (currently unused, reserved
                     for future extensions).
                     
        Raises:
            ImportError: If paddleocr or paddlepaddle is not installed,
                        or if an incompatible version is installed.
        
        Notes:
            - PaddleOCR instance is created lazily on first ocr() call
            - batch_size=1 minimizes memory usage but may be slower
            - batch_size=4-8 is a good balance for most use cases
            - GPU mode requires CUDA-compatible GPU with sufficient VRAM
        """
        self.lang = lang
        self.gpu = gpu
        self.batch_size = batch_size
        
        # Convert tesseract language code to PaddleOCR format
        self._engine_lang = self.convert_language(lang)
    
    # ========================================================================
    # OCR OPERATIONS
    # ========================================================================
    
    def ocr(
        self,
        image: Image.Image,
        return_boxes: bool = False,
    ) -> Union[str, List[Dict[str, Any]]]:
        """Perform OCR on a single image using PaddleOCR.
        
        This method converts the PIL Image to a numpy array, runs PaddleOCR's
        predict() method, and formats the results. If GPU mode is enabled and
        an out-of-memory error occurs, automatically falls back to CPU mode.
        
        Args:
            image: PIL Image to perform OCR on.
            return_boxes: If True, return structured data with bounding boxes
                         and confidence scores. If False, return plain text.
                         
        Returns:
            If return_boxes is False:
                String with extracted text, one line per detected text region.
            If return_boxes is True:
                List of dicts, each with:
                    - "text": Extracted text (str)
                    - "confidence": Recognition confidence (float, 0-1)
                    - "bbox": Bounding box as [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                             where points are ordered clockwise from top-left
                             
        Raises:
            ImportError: If numpy is not installed.
            RuntimeError: If PaddleOCR operation fails (non-OOM errors).
            
        Notes:
            - GPU OOM errors trigger automatic CPU fallback with warning
            - Empty images return "" (no boxes) or [] (with boxes)
            - Text order follows PaddleOCR's detection order (typically top-to-bottom)
            - Bounding boxes use image pixel coordinates
        """
        # Get numpy for image conversion
        np = _get_numpy()
        if np is None:
            raise ImportError(
                "numpy is required for PaddleOCR. Install with: pip install numpy"
            )
        
        # Convert PIL Image to numpy array (required by PaddleOCR)
        img_array = np.array(image)
        
        # Run PaddleOCR with GPU fallback on OOM
        result = self._run_with_oom_fallback(img_array)
        
        # Handle empty results
        if result is None or len(result) == 0:
            return [] if return_boxes else ""
        
        # Format results based on return_boxes flag
        if return_boxes:
            return self._format_results_with_boxes(result)
        else:
            return self._format_results_as_text(result)
    
    def _run_with_oom_fallback(self, img_array: Any) -> Any:
        """Run PaddleOCR with automatic CPU fallback on GPU OOM.
        
        This method implements the GPU OOM fallback logic. If GPU mode is
        enabled and we get an out-of-memory error, we:
        1. Log a warning to stderr
        2. Force re-set environment variables (paranoid defense)
        3. Create a new CPU instance
        4. Retry the operation
        
        Args:
            img_array: Numpy array of image to process.
            
        Returns:
            PaddleOCR result in format: [[[bbox, (text, confidence)], ...], ...]
            
        Raises:
            RuntimeError: If operation fails with non-OOM error.
            ImportError: If PaddleOCR is not installed.
            
        Notes:
            - OOM detection uses case-insensitive string matching
            - Matches "out of memory" and "resourceexhausted"
            - Failed GPU instance is not explicitly destroyed (garbage collected)
            - CPU fallback uses same language and batch_size
        """
        if self.gpu:
            try:
                # Try GPU first
                paddleocr = _get_paddleocr(
                    lang=self._engine_lang,
                    gpu=True,
                    batch_size=self.batch_size,
                )
                return paddleocr.predict(img_array)
            except Exception as e:
                # Check if it's an OOM error
                # Use case-insensitive matching to catch all variants:
                # - "Out of memory"
                # - "ResourceExhausted"
                # - "CUDA out of memory"
                # - etc.
                error_msg = str(e).lower()
                if "out of memory" in error_msg or "resourceexhausted" in error_msg:
                    # Log warning to stderr (not stdout, which may be captured)
                    print(
                        "Warning: GPU out of memory. Falling back to CPU mode.",
                        file=sys.stderr,
                    )
                    
                    # Retry with CPU
                    # Note: The failed GPU instance will be garbage collected
                    # We don't need to explicitly destroy it
                    paddleocr = _get_paddleocr(
                        lang=self._engine_lang,
                        gpu=False,
                        batch_size=self.batch_size,
                    )
                    return paddleocr.predict(img_array)
                else:
                    # Not an OOM error, re-raise
                    raise
        else:
            # CPU mode from the start (no fallback needed)
            paddleocr = _get_paddleocr(
                lang=self._engine_lang,
                gpu=False,
                batch_size=self.batch_size,
            )
            return paddleocr.predict(img_array)
    
    def _format_results_with_boxes(
        self,
        result: Any,
    ) -> List[Dict[str, Any]]:
        """Format PaddleOCR results as structured data with bounding boxes.
        
        PaddleOCR returns results in format:
            [[[bbox, (text, confidence)], ...], ...]
        
        We convert this to our standard format:
            [{"text": str, "confidence": float, "bbox": [[x,y], ...]}, ...]
            
        Args:
            result: Raw PaddleOCR result from predict() method.
            
        Returns:
            List of dicts with text, confidence, and bbox fields.
            
        Notes:
            - Bounding boxes are converted to integer coordinates
            - Confidence scores are converted to float (0-1 range)
            - Empty lines in result are skipped (should not happen in practice)
            - Bounding box format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
              with points ordered clockwise from top-left
        """
        structured_results: List[Dict[str, Any]] = []
        
        for line in result:
            if line is None:
                continue
            
            for box_info in line:
                # Unpack: bbox is list of 4 [x, y] points
                # text_info is (text_string, confidence_score)
                bbox, text_info = box_info
                text, confidence = text_info
                
                structured_results.append({
                    "text": text,
                    "confidence": float(confidence),
                    # Convert bbox coordinates to integers for consistency
                    "bbox": [[int(p[0]), int(p[1])] for p in bbox],
                })
        
        return structured_results
    
    def _format_results_as_text(self, result: Any) -> str:
        """Format PaddleOCR results as plain text string.
        
        Extracts just the text content from PaddleOCR results, joining
        with newlines. Ignores bounding boxes and confidence scores.
        
        Args:
            result: Raw PaddleOCR result from predict() method.
            
        Returns:
            Extracted text with one line per detected text region.
            
        Notes:
            - Empty lines in result are skipped
            - Text regions are joined with newline characters
            - Order follows PaddleOCR's detection order
        """
        text_parts: List[str] = []
        
        for line in result:
            if line is None:
                continue
            
            for box_info in line:
                # Unpack and extract just the text (ignore bbox and confidence)
                _, text_info = box_info
                text, _ = text_info
                text_parts.append(text)
        
        return "\n".join(text_parts)
    
    # ========================================================================
    # ENGINE AVAILABILITY AND LANGUAGE CONVERSION
    # ========================================================================
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if PaddleOCR and its dependencies are installed.
        
        This method attempts to import paddleocr to verify it's available.
        Does not check for GPU availability (that's handled at runtime).
        
        Returns:
            True if paddleocr can be imported, False otherwise.
            
        Notes:
            - Does not verify PaddlePaddle installation (paddleocr depends on it)
            - Does not check version compatibility (handled at init time)
            - Import errors are silently caught (returns False)
        """
        try:
            import paddleocr  # noqa: F401
            return True
        except ImportError:
            return False
    
    @classmethod
    def convert_language(cls, tesseract_lang: str) -> str:
        """Convert tesseract language code to PaddleOCR format.
        
        PaddleOCR uses different language codes than tesseract. For example:
        - "eng" → "en"
        - "deu" → "german"
        - "chi_sim" → "ch"
        
        This method uses TESSERACT_TO_PADDLEOCR_LANG mapping from types.py.
        
        Args:
            tesseract_lang: Language code in tesseract format (e.g., "eng", "deu").
            
        Returns:
            Language code in PaddleOCR format (e.g., "en", "german").
            Falls back to "en" if language is not in mapping.
            
        Notes:
            - Unknown languages default to "en" for safety
            - PaddleOCR will download language model on first use
            - Language models are cached in ~/.paddleocr/
        """
        return TESSERACT_TO_PADDLEOCR_LANG.get(tesseract_lang, "en")
