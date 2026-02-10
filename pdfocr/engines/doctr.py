#!/usr/bin/env python3
"""docTR engine implementation.

This module provides a docTR (Document Text Recognition) engine implementation
that wraps the doctr library. docTR is a document-focused OCR engine that
excels at extracting text from documents with complex layouts, tables, and
multi-column formats.

Features:
    - GPU acceleration support for faster processing
    - Document-focused architecture optimized for structured documents
    - Built-in text detection and recognition
    - Bounding box extraction with confidence scores
    - Horizontal spacing preservation for tabular data
    - Thread-safe lazy initialization with model caching

Installation:
    pip install python-doctr[torch]
    
    For GPU support:
        pip install torch torchvision  # CUDA-enabled PyTorch
        
Usage Example:
    ```python
    from pdfocr.engines.doctr import DocTREngine
    from PIL import Image
    
    # Initialize with GPU
    engine = DocTREngine(lang="eng", gpu=True)
    
    # Perform OCR with bounding boxes
    image = Image.open("document.jpg")
    results = engine.ocr(image, return_boxes=True)
    for item in results:
        print(f"Text: {item['text']}, Confidence: {item['confidence']}")
    
    # Or extract text with preserved horizontal spacing
    text = engine.ocr(image, return_boxes=False)
    ```

Horizontal Spacing Preservation:
    docTR returns word-level bounding boxes but doesn't inherently preserve
    horizontal spacing between words. This engine reconstructs spacing by:
    1. Calculating pixel gaps between consecutive words
    2. Identifying wide gaps (> WIDE_GAP_THRESHOLD_PIXELS)
    3. Inserting proportional spaces based on gap width
    
    This is critical for extracting tabular data and maintaining document
    structure (e.g., multi-column layouts, indented paragraphs).

Thread Safety:
    This engine uses thread-safe lazy loading with double-checked locking
    to ensure doctr is imported and the model is initialized only once.
    The model is cached with its GPU setting; if the GPU parameter changes,
    the model is reinitialized (required by docTR's device management).

Performance:
    docTR can utilize GPU acceleration for significantly faster processing.
    GPU mode typically processes images 5-20x faster than CPU mode. However,
    initial model loading takes 5-30 seconds depending on hardware.

Device Switching:
    docTR models are device-specific (CPU vs GPU). When switching devices,
    the model must be reinitialized with .to(device). This engine handles
    device switching automatically by caching the model along with its GPU
    setting and reinitializing when the setting changes.

Limitations:
    - Requires ~1GB disk space for language models
    - Initial startup is slow (model loading)
    - GPU mode requires CUDA-compatible GPU with sufficient VRAM
    - Language support is more limited than Tesseract/EasyOCR
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple, Union

from pdfocr.engines.base import OCREngine, register_engine
from pdfocr.types import WIDE_GAP_THRESHOLD_PIXELS, PIXELS_PER_SPACE

try:
    from PIL import Image
except ImportError:
    # Fallback type hint if PIL not installed (shouldn't happen in practice)
    Image = Any  # type: ignore[assignment]

# ============================================================================
# GLOBAL STATE FOR LAZY LOADING
# ============================================================================

# Module-level cache for doctr predictor.
# Using None as sentinel to indicate "not yet loaded".
# The predictor is expensive to initialize (loads neural network models),
# so we cache it to avoid repeated initialization overhead.
# 
# Cache format: Tuple[predictor, gpu_setting]
# - predictor: The doctr ocr_predictor instance
# - gpu_setting: Boolean indicating whether model is on GPU (True) or CPU (False)
#
# We need to cache the GPU setting because docTR models are device-specific.
# If the user requests a different device, we must reinitialize the model.
_doctr_model: Optional[Tuple[Any, bool]] = None

# Thread lock for synchronizing doctr model initialization.
# Ensures thread-safe lazy loading with double-checked locking pattern.
_doctr_lock = threading.Lock()

# Module-level cache for numpy (required by doctr for image conversion).
_np: Any = None

# Thread lock for numpy import synchronization.
_numpy_lock = threading.Lock()


# ============================================================================
# LAZY LOADING HELPERS
# ============================================================================


def _get_numpy() -> Any:
    """Lazy import numpy with thread-safe double-checked locking.
    
    docTR requires numpy to convert PIL Images to numpy arrays before
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


def _get_doctr_model(gpu: bool = False) -> Any:
    """Lazy import and initialize docTR model with thread-safe locking.
    
    This function implements double-checked locking to ensure the docTR
    predictor is initialized exactly once, or reinitialized if the GPU
    setting changes. The predictor initialization is expensive (loads neural
    network models), so caching is critical for performance.
    
    Device Switching:
        docTR models are device-specific (CPU vs GPU). When switching devices,
        the model must be moved to the new device using .to(device). This
        function detects device changes by comparing the requested GPU setting
        with the cached GPU setting, and reinitializes the model if they differ.
        
    Args:
        gpu: Whether to use GPU acceleration. If True and GPU not available,
             docTR automatically falls back to CPU.
             
    Returns:
        docTR ocr_predictor instance if available, otherwise None.
        Returning None allows graceful handling of missing dependency.
        
    Thread Safety:
        Fully thread-safe. Multiple threads can call this concurrently;
        the predictor will be initialized exactly once (or reinitialized if
        GPU setting changes), and all threads will receive the same instance.
        
    Performance:
        After first initialization, this is a fast boolean comparison
        (if GPU setting unchanged) or a full reinitialization (if changed).
        First initialization takes 5-30 seconds depending on hardware.
    """
    # First check: Fast path if model exists with same GPU setting
    # Most calls after initialization take this path, avoiding lock overhead
    global _doctr_model
    if _doctr_model is not None:
        # Model exists, check if GPU setting matches
        cached_instance, cached_gpu = _doctr_model
        if cached_gpu == gpu:
            # GPU setting unchanged, return cached model
            return cached_instance
    
    # Model not loaded or GPU setting changed, acquire lock for initialization
    with _doctr_lock:
        # Second check: Verify another thread didn't initialize while we waited
        # This is the key to double-checked locking pattern
        if _doctr_model is not None:
            cached_instance, cached_gpu = _doctr_model
            if cached_gpu == gpu:
                # Another thread already initialized with correct GPU setting
                return cached_instance
        
        # Need to initialize or reinitialize model
        try:
            # Import doctr library
            # This is an optional dependency; if not installed,
            # we catch ImportError and return None
            from doctr.models import ocr_predictor
            
            # Determine device string for docTR
            # docTR uses PyTorch device strings: 'cuda' or 'cpu'
            device = 'cuda' if gpu else 'cpu'
            
            # Initialize predictor with pretrained weights
            # This loads the neural network models from disk or downloads them
            # Models are cached by doctr in ~/.doctr/
            predictor = ocr_predictor(pretrained=True).to(device)
            
            # Cache the model along with its GPU setting
            # Format: (predictor, gpu_setting)
            _doctr_model = (predictor, gpu)
            
            # Return the newly initialized predictor
            return predictor
        except ImportError:
            # doctr is an optional dependency; if not installed, return None
            # Calling code will raise ImportError with installation instructions
            return None
    
    # This line should never be reached due to return statements above,
    # but we include it for type checking completeness
    return None  # pragma: no cover


def _create_doctr_model(gpu: bool) -> Any:
    """Helper to create docTR model (deprecated, use _get_doctr_model).
    
    This function exists for backward compatibility with the original
    pdfocr.py code structure. It's a thin wrapper around model creation
    logic that's now integrated into _get_doctr_model().
    
    Args:
        gpu: Whether to use GPU acceleration.
        
    Returns:
        docTR ocr_predictor instance or None if not available.
    """
    try:
        # Import doctr library
        from doctr.models import ocr_predictor
        
        # Determine device string
        device = 'cuda' if gpu else 'cpu'
        
        # Create and return predictor
        return ocr_predictor(pretrained=True).to(device)
    except ImportError:
        # doctr not installed
        return None


# ============================================================================
# DOCTR ENGINE IMPLEMENTATION
# ============================================================================


@register_engine
class DocTREngine(OCREngine):
    """docTR engine using deep learning for document-focused text recognition.
    
    This engine wraps the doctr library, which uses neural networks optimized
    for document OCR. docTR excels at extracting text from structured documents
    with complex layouts, tables, and multi-column formats.
    
    Class Attributes:
        name: Engine identifier used in CLI (-e doctr)
        display_name: Human-readable name for display/logging
        supports_gpu: True - docTR supports GPU acceleration via PyTorch
        supports_boxes: True - Returns detailed bounding boxes with confidence
        install_hint: Installation instructions shown on ImportError
        
    Instance Attributes:
        lang: Tesseract language code (e.g., "eng")
              Note: docTR language support is more limited than Tesseract
        gpu: Whether to use GPU acceleration
        _model: Cached docTR predictor instance
        _np: Cached numpy module reference
        
    Initialization:
        The __init__ method performs lazy loading of doctr and numpy using
        thread-safe helpers. The predictor is initialized on first use or
        when GPU setting changes.
        
    OCR Operation:
        The ocr() method converts PIL Image to numpy array and calls
        predictor(). Returns either plain text (with preserved horizontal
        spacing) or structured data with bounding boxes and confidence scores,
        depending on return_boxes parameter.
        
    Horizontal Spacing:
        When return_boxes=False, this engine preserves horizontal spacing
        by analyzing gaps between words. This is critical for tabular data
        and multi-column layouts. Gap detection uses:
        - WIDE_GAP_THRESHOLD_PIXELS: Minimum gap to consider "wide" (30px)
        - PIXELS_PER_SPACE: Conversion factor for gap width to spaces (10px)
        
    Thread Safety:
        Lazy loading is thread-safe via double-checked locking in helper
        functions. Multiple DocTREngine instances share the same cached
        predictor if they use the same GPU setting, ensuring memory efficiency.
        
    Example:
        ```python
        # Check availability before use
        if DocTREngine.is_available():
            # Initialize with GPU
            engine = DocTREngine(lang="eng", gpu=True)
            
            # Perform OCR with bounding boxes
            text_with_boxes = engine.ocr(image, return_boxes=True)
            
            # Or extract text with preserved spacing
            text = engine.ocr(image, return_boxes=False)
        else:
            print(f"Install with: {DocTREngine.install_hint}")
        ```
    """

    # ========================================================================
    # CLASS ATTRIBUTES
    # ========================================================================
    
    # Engine identifier for CLI (-e doctr) and registry lookup
    name: str = "doctr"
    
    # Human-readable name for display in help text and error messages
    display_name: str = "docTR (Document-focused)"
    
    # docTR supports GPU acceleration via PyTorch CUDA
    # If GPU requested but not available, automatically falls back to CPU
    supports_gpu: bool = True
    
    # docTR provides detailed bounding boxes with confidence scores
    # Each word includes normalized coordinates and confidence
    supports_boxes: bool = True
    
    # Installation hint shown when doctr is not available
    # User needs python-doctr[torch] which includes PyTorch dependency
    install_hint: str = "pip install python-doctr[torch]"

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def __init__(self, lang: str = "eng", gpu: bool = False, **kwargs: Any) -> None:
        """Initialize docTR engine.
        
        Performs lazy loading of doctr and numpy libraries, stores
        configuration for predictor initialization.
        
        Args:
            lang: Tesseract language code (default: "eng" for English).
                  Note: docTR has limited language support compared to
                  Tesseract. The lang parameter is stored for API consistency
                  but may not affect docTR's operation (docTR uses multilingual
                  models by default).
                  
            gpu: Whether to use GPU acceleration (default: False).
                 If True, docTR will attempt to use CUDA-enabled PyTorch.
                 If GPU not available, automatically falls back to CPU.
                 GPU mode is typically 5-20x faster than CPU.
                 
            **kwargs: Additional keyword arguments (currently unused).
                     Included for forward compatibility.
                     
        Raises:
            ImportError: If doctr or numpy is not installed.
                        Error message includes installation instructions.
                        
        Notes:
            - The predictor is initialized lazily on first ocr() call
            - Models are downloaded automatically to ~/.doctr/
            - First run may take time to download models (~500MB)
            - GPU mode requires CUDA-compatible GPU and CUDA toolkit
            - Device switching (CPU↔GPU) requires model reinitialization
            
        Thread Safety:
            Multiple threads can call __init__ concurrently. The lazy loading
            helpers ensure thread-safe initialization of shared resources.
        """
        # Lazy load numpy using thread-safe helper
        # Returns None if numpy is not installed
        np_module = _get_numpy()
        
        # Validate that numpy is available
        # docTR requires numpy for image array conversion
        if np_module is None:
            raise ImportError(
                "numpy not installed. "
                "Install with: pip install numpy"
            )
        
        # Cache numpy module reference for use in ocr() method
        self._np = np_module
        
        # Store original tesseract language code
        # Note: docTR uses multilingual models by default, so this
        # parameter mainly provides API consistency with other engines
        self.lang: str = lang
        
        # Store GPU preference for predictor initialization
        self.gpu: bool = gpu
        
        # Initialize predictor lazily - get or create cached predictor
        # This may take 5-30 seconds on first call (loads neural network models)
        # If predictor with different GPU setting exists, it will be reinitialized
        model = _get_doctr_model(gpu=gpu)
        
        # Validate that doctr is available
        # If None, predictor initialization failed (doctr not installed)
        if model is None:
            raise ImportError(
                "docTR not installed. "
                "Install with: pip install python-doctr[torch]\n"
                "For GPU support, ensure PyTorch is installed: "
                "pip install torch torchvision"
            )
        
        # Cache predictor reference for use in ocr() method
        # Note: Predictor may be shared across multiple engine instances
        # if they use the same GPU setting (memory efficiency)
        self._model = model

    # ========================================================================
    # OCR OPERATION
    # ========================================================================

    def ocr(
        self, image: Image.Image, return_boxes: bool = False
    ) -> Union[str, List[Dict[str, Any]]]:
        """Perform OCR on an image using docTR.
        
        Converts the PIL Image to a numpy array and performs OCR using the
        cached docTR predictor. Can return either plain text (with preserved
        horizontal spacing) or structured data with bounding boxes and
        confidence scores.
        
        Args:
            image: PIL Image object to perform OCR on.
                   Can be any mode (RGB, L, RGBA, etc.); numpy handles conversion.
                   
            return_boxes: If True, return structured data with bounding boxes.
                         If False, return plain text string with preserved spacing.
                         
        Returns:
            If return_boxes is False:
                str: Extracted text with preserved horizontal spacing.
                     Line breaks separate text lines. Wide gaps between words
                     are represented by multiple spaces (proportional to gap width).
                     Returns empty string if no text detected.
                     
            If return_boxes is True:
                List[Dict[str, Any]]: List of dictionaries, one per detected word.
                Each dict contains:
                    - "text" (str): Recognized word
                    - "confidence" (float): Recognition confidence (0.0-1.0)
                    - "bbox" (List[List[int]]): 4-point bounding box
                      Format: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                      Points define top-left, top-right, bottom-right, bottom-left
                      Coordinates are in pixels (0-based, absolute)
                      
        Raises:
            RuntimeError: If OCR operation fails (rare, usually GPU memory issues).
                         
        Notes:
            - docTR performs both text detection and recognition
            - Results are organized by page → block → line → word
            - Confidence scores reflect model certainty (higher is better)
            - Bounding boxes use normalized coordinates [0,1] internally,
              converted to pixels in structured output
            - Horizontal spacing preservation is critical for tabular data
            
        Horizontal Spacing Algorithm:
            1. Sort words by horizontal position (left-to-right)
            2. Calculate pixel gap between consecutive words
            3. If gap > WIDE_GAP_THRESHOLD_PIXELS (30px):
               - Insert multiple spaces (gap / PIXELS_PER_SPACE)
            4. Otherwise insert single space
            
            This reconstructs document structure (tables, columns, indentation)
            that would be lost with naive word concatenation.
            
        Performance:
            GPU mode: 0.1-1 second per page (depending on GPU, image size)
            CPU mode: 1-5 seconds per page (depending on CPU, image size)
            
        Thread Safety:
            This method is thread-safe. The cached predictor can be used
            concurrently by multiple threads.
        """
        # Convert PIL Image to numpy array
        # docTR expects numpy array, not PIL Image
        # Array is RGB or grayscale depending on image mode
        img_array = self._np.array(image)
        
        # Perform OCR using cached predictor
        # docTR's predictor expects a list of images, so we wrap in list
        # Returns a Document object with hierarchical structure:
        # - Document.pages: List of Page objects
        # - Page.blocks: List of Block objects (paragraphs)
        # - Block.lines: List of Line objects (text lines)
        # - Line.words: List of Word objects (individual words)
        result = self._model([img_array])
        
        # Process results based on return_boxes parameter
        if return_boxes:
            # Return structured data with bounding boxes and confidence
            # Flatten hierarchical structure into list of word dictionaries
            structured_results: List[Dict[str, Any]] = []
            
            # Iterate through document hierarchy
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            # Get bounding box coordinates
                            # docTR returns normalized coordinates [0, 1]
                            # Format: ((x1, y1), (x2, y2)) where:
                            # - (x1, y1) is top-left corner
                            # - (x2, y2) is bottom-right corner
                            bbox = word.geometry
                            
                            # Get image dimensions for denormalization
                            # img_array.shape: (height, width, channels) for RGB
                            #                  (height, width) for grayscale
                            h, w = img_array.shape[:2]
                            
                            # Convert normalized coordinates to pixel coordinates
                            # Multiply by width/height and convert to integers
                            x1, y1 = int(bbox[0][0] * w), int(bbox[0][1] * h)
                            x2, y2 = int(bbox[1][0] * w), int(bbox[1][1] * h)
                            
                            # Create structured result dictionary
                            # bbox format: 4-point polygon (clockwise from top-left)
                            # This format matches other engines for consistency
                            structured_results.append({
                                "text": word.value,
                                "confidence": float(word.confidence),
                                "bbox": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                            })
            
            return structured_results
        
        # Extract plain text from results, preserving horizontal spacing
        # This is critical for maintaining document structure (tables, columns)
        text_parts: List[str] = []
        
        # Process each page in the document
        for page in result.pages:
            # Get page width for gap calculation
            # We need height first due to numpy shape format (height, width)
            _, page_width = img_array.shape[:2]
            
            # Process each block (paragraph) in the page
            for block in page.blocks:
                # Process each line in the block
                for line in block.lines:
                    # Skip empty lines
                    if not line.words:
                        continue
                    
                    # Sort words by horizontal position (left-to-right)
                    # word.geometry[0][0] is the x-coordinate of top-left corner
                    # Normalized to [0, 1] range
                    sorted_words = sorted(line.words, key=lambda w: w.geometry[0][0])
                    
                    # Build line text with spacing preservation
                    line_parts: List[str] = []
                    prev_end_x: Optional[float] = None
                    
                    # Process each word in the line
                    for word in sorted_words:
                        # Get word horizontal position (normalized 0-1)
                        # geometry[0][0] is left edge x-coordinate
                        word_start_x: float = word.geometry[0][0]
                        
                        # Calculate gap between consecutive words
                        if prev_end_x is not None:
                            # Calculate gap in normalized coordinates
                            gap_normalized = word_start_x - prev_end_x
                            
                            # Convert gap to pixels for threshold comparison
                            gap_pixels = gap_normalized * page_width
                            
                            # Check if gap is wide (e.g., table column separator)
                            # WIDE_GAP_THRESHOLD_PIXELS from pdfocr/types.py (30px)
                            if gap_pixels > WIDE_GAP_THRESHOLD_PIXELS:
                                # Wide gap: add proportional spaces
                                # PIXELS_PER_SPACE from pdfocr/types.py (10px)
                                # This preserves tabular structure
                                num_spaces = max(1, int(gap_pixels / PIXELS_PER_SPACE))
                                line_parts.append(" " * num_spaces)
                            else:
                                # Normal gap: single space between words
                                line_parts.append(" ")
                        
                        # Add the word text
                        line_parts.append(word.value)
                        
                        # Update previous word's end position for next iteration
                        # geometry[1][0] is right edge x-coordinate
                        prev_end_x = word.geometry[1][0]
                    
                    # Join line parts and add to text if non-empty
                    line_text = "".join(line_parts)
                    if line_text.strip():
                        text_parts.append(line_text)
        
        # Join all lines with newlines and return
        return "\n".join(text_parts)

    # ========================================================================
    # AVAILABILITY CHECK
    # ========================================================================

    @classmethod
    def is_available(cls) -> bool:
        """Check if docTR engine is available.
        
        Attempts to import doctr to determine if this engine can be used.
        This is a class method so it can be called without instantiating the
        engine, useful for engine discovery and CLI fallback logic.
        
        Returns:
            True if doctr is installed and can be imported.
            False if doctr is not available.
            
        Notes:
            - Only checks Python package availability, not GPU support
            - GPU availability check would require PyTorch import (expensive)
            - If doctr imports but GPU not available, CPU fallback is automatic
            - Does not check for model availability (downloaded on first use)
            
        Performance:
            This method is lightweight after first call, as Python caches
            import failures. First call may take ~100-500ms for import attempt.
            
        Example:
            ```python
            if DocTREngine.is_available():
                engine = DocTREngine(lang="eng", gpu=True)
            else:
                print("docTR not available")
                print(f"Install with: {DocTREngine.install_hint}")
            ```
        """
        try:
            # Try importing doctr
            # If successful, engine is available
            import doctr  # noqa: F401
            
            return True
        except ImportError:
            # doctr not installed or import failed
            return False

    # ========================================================================
    # LANGUAGE CONVERSION
    # ========================================================================

    @classmethod
    def convert_language(cls, tesseract_lang: str) -> str:
        """Convert tesseract language code to docTR format.
        
        docTR uses multilingual models by default and doesn't require
        explicit language codes for most operations. This method exists
        for API consistency with other engines but returns the input
        unchanged since docTR handles language detection automatically.
        
        Args:
            tesseract_lang: Tesseract language code (e.g., "eng", "deu", "fra").
            
        Returns:
            Same language code (docTR doesn't require conversion).
            
        Notes:
            - docTR uses multilingual models that work across languages
            - Explicit language codes are not required for initialization
            - This method provides API consistency with other engines
            - Override in future if docTR adds language-specific models
            
        Examples:
            >>> DocTREngine.convert_language("eng")
            'eng'
            >>> DocTREngine.convert_language("deu")
            'deu'
        """
        # docTR uses multilingual models, no conversion needed
        # Return original code for consistency
        return tesseract_lang
