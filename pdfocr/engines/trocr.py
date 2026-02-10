#!/usr/bin/env python3
"""TrOCR engine implementation.

This module provides a TrOCR (Transformer-based Optical Character Recognition)
engine implementation using Microsoft's TrOCR models from Hugging Face. TrOCR
is a state-of-the-art deep learning model designed specifically for text line
recognition, achieving excellent results on printed and handwritten text.

Important Notes:
    TrOCR is designed for LINE-LEVEL OCR, not full-page document processing.
    The model expects cropped text line images as input and will resize images
    to 384x384 pixels internally. For full-page documents, use tesseract or
    easyocr engines instead. Large images will trigger a warning message.

Features:
    - GPU acceleration support for faster processing
    - Specialized models for printed and handwritten text
    - Transformer-based architecture (SOTA accuracy on text lines)
    - Thread-safe lazy initialization with device-aware caching
    - Supports both CPU and CUDA devices

Installation:
    pip install transformers torch
    
    For GPU support:
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        
Usage Example:
    ```python
    from pdfocr.engines.trocr import TrOCREngine
    from PIL import Image
    
    # Initialize with GPU and handwritten text support
    engine = TrOCREngine(lang="eng", gpu=True, model_variant="handwritten")
    
    # Perform OCR on a text line image (best results)
    text_line = Image.open("text_line.jpg")
    text = engine.ocr(text_line, return_boxes=False)
    
    # Or get structured output (though TrOCR doesn't provide real boxes)
    result = engine.ocr(text_line, return_boxes=True)
    ```

Thread Safety:
    This engine uses thread-safe lazy loading with double-checked locking
    to ensure transformers models are imported and initialized only once per
    (model_name, device) tuple. The cache key includes device to prevent
    device mismatch errors when using GPU.

Performance:
    TrOCR can utilize GPU acceleration for significantly faster processing.
    GPU mode typically processes text lines 5-20x faster than CPU mode. 
    However, initial model loading takes 10-60 seconds depending on hardware
    and network speed (models are ~300-500MB).

Limitations:
    - Does NOT provide bounding boxes (designed for pre-segmented text lines)
    - Not suitable for full-page documents (will resize and distort)
    - Requires ~500MB-1GB disk space for model weights
    - Initial startup is slow (model download/loading)
    - GPU mode requires CUDA-compatible GPU with sufficient VRAM (~2GB+)
    - Best results on text line images, poor on full pages
"""

from __future__ import annotations

import sys
import threading
from typing import Any, Dict, Optional, Tuple, Union

from pdfocr.engines.base import OCREngine, register_engine

try:
    from PIL import Image
except ImportError:
    # Fallback type hint if PIL not installed (shouldn't happen in practice)
    Image = Any  # type: ignore[misc, assignment]

# ============================================================================
# GLOBAL STATE FOR LAZY LOADING
# ============================================================================

# Module-level cache for TrOCR processor and model tuples.
# Key format: (model_name, device) where device is "cuda" or "cpu"
# 
# Why cache by (model_name, device)?
#   - Multiple model variants exist (printed vs handwritten)
#   - Device matters: moving model between CPU/GPU is expensive and error-prone
#   - Keying by both prevents device mismatch errors and avoids reloading
#
# Value format: (processor, model) tuple
#   - processor: TrOCRProcessor for image preprocessing and text decoding
#   - model: VisionEncoderDecoderModel for actual OCR inference
#
# Both processor and model are expensive to initialize (download + load weights),
# so caching is critical for performance. Models are 300-500MB each.
_trocr_cache: Dict[Tuple[str, str], Tuple[Any, Any]] = {}

# Thread lock for synchronizing TrOCR processor/model initialization.
# Ensures thread-safe lazy loading with double-checked locking pattern.
# Without this lock, multiple threads could simultaneously initialize the same
# model, wasting memory and causing race conditions.
_trocr_lock = threading.Lock()


# ============================================================================
# LAZY LOADING HELPERS
# ============================================================================


def _get_trocr(
    model_name: str, device: str
) -> Tuple[Optional[Any], Optional[Any]]:
    """Lazy import and initialize TrOCR processor and model (thread-safe).
    
    This function implements double-checked locking to ensure TrOCR models
    are initialized exactly once per (model_name, device) combination. The
    initialization is expensive (downloads and loads 300-500MB of weights),
    so caching is essential for performance.
    
    Cache Key Design:
        The cache uses (model_name, device) as the key to prevent device
        mismatch errors. If a model is loaded on CPU and later requested
        on GPU, we create a separate cached instance. This prevents runtime
        errors from moving tensors between devices.
        
    Args:
        model_name: Hugging Face model identifier (e.g., "microsoft/trocr-base-printed").
                   Full list: https://huggingface.co/models?search=trocr
                   
        device: PyTorch device string ("cuda" or "cpu").
               Determines where model tensors are allocated.
               GPU (cuda) provides 5-20x speedup but requires CUDA-compatible GPU.
               
    Returns:
        Tuple of (processor, model) if successful, or (None, None) if import fails.
        
        Returning (None, None) allows graceful handling of missing dependencies
        without raising exceptions in the lazy loader. The __init__ method
        will raise ImportError with installation instructions if None returned.
        
    Thread Safety:
        Fully thread-safe. Multiple threads can call this concurrently;
        the model will be initialized exactly once per cache key, and all
        threads will receive the same (processor, model) tuple.
        
    Performance:
        After first initialization for a given (model_name, device):
            - Fast path: dict lookup (~1 microsecond)
            - No lock contention in common case
            
        First initialization:
            - Downloads model weights if not cached (~300-500MB)
            - Loads weights into memory (~5-30 seconds)
            - Moves model to device and sets to eval mode
            
    Raises:
        Does not raise exceptions directly. Returns (None, None) on import
        failure, allowing caller to handle error with context.
    """
    # Cache key combines model name and device to prevent device mismatch errors
    # Example: ("microsoft/trocr-base-printed", "cuda")
    cache_key: Tuple[str, str] = (model_name, device)
    
    # First check: Fast path for already-initialized models
    # Most calls after initialization take this path, avoiding lock overhead
    # This is safe because dict reads are atomic in CPython (GIL protected)
    global _trocr_cache
    if cache_key in _trocr_cache:
        return _trocr_cache[cache_key]
    
    # Model not loaded yet for this (model_name, device), acquire lock
    with _trocr_lock:
        # Second check: Verify another thread didn't initialize while we waited
        # This is the key to double-checked locking pattern - prevents duplicate
        # initialization when multiple threads race to the lock
        if cache_key in _trocr_cache:
            return _trocr_cache[cache_key]
        
        # Need to initialize processor and model for this cache key
        try:
            # Import transformers library (optional dependency)
            # TrOCRProcessor: handles image preprocessing and output decoding
            # VisionEncoderDecoderModel: the actual transformer model
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            # Download and load processor from Hugging Face Hub
            # Processor handles:
            #   - Image resizing to 384x384
            #   - Normalization for model input
            #   - Tokenization of output text
            # Cached in ~/.cache/huggingface/transformers/ after first download
            processor = TrOCRProcessor.from_pretrained(model_name)
            
            # Download and load model weights from Hugging Face Hub
            # Model architecture:
            #   - Vision encoder: Processes 384x384 image patches
            #   - Decoder: Generates text tokens autoregressively
            # Weights are 300-500MB depending on model variant
            model = VisionEncoderDecoderModel.from_pretrained(model_name)
            
            # Move model to target device (CPU or CUDA GPU)
            # This allocates model parameters on the specified device
            # GPU allocation fails if insufficient VRAM available
            model = model.to(device)
            
            # Set model to evaluation mode (disables dropout, etc.)
            # Required for inference - training mode would give non-deterministic results
            model.eval()
            
            # Cache the initialized (processor, model) tuple for future calls
            # Other threads/calls with same cache key will reuse this instance
            _trocr_cache[cache_key] = (processor, model)
            
            # Return the newly initialized processor and model
            return (processor, model)
            
        except ImportError:
            # transformers or one of its dependencies (torch, etc.) not installed
            # Return (None, None) to signal import failure
            # Caller (__init__) will raise ImportError with installation instructions
            return (None, None)
    
    # This line is technically unreachable (return inside lock), but included
    # for type checking and clarity. All paths return from inside the lock.
    return _trocr_cache.get(cache_key, (None, None))


# ============================================================================
# TROCR ENGINE IMPLEMENTATION
# ============================================================================


@register_engine
class TrOCREngine(OCREngine):
    """TrOCR engine using transformer models for text line recognition.
    
    This engine wraps Microsoft's TrOCR models from Hugging Face Transformers.
    TrOCR achieves state-of-the-art accuracy on text line recognition tasks
    using a vision transformer encoder and autoregressive text decoder.
    
    IMPORTANT: TrOCR is designed for LINE-LEVEL OCR, not full-page documents.
    Images are resized to 384x384 internally, so full-page documents will be
    distorted and produce poor results. For page-level OCR, use tesseract
    or easyocr engines instead.
    
    Class Attributes:
        name: Engine identifier used in CLI (-e trocr)
        display_name: Human-readable name for display/logging
        supports_gpu: True - TrOCR supports GPU acceleration via PyTorch
        supports_boxes: False - TrOCR doesn't provide bounding boxes (line-level only)
        install_hint: Installation instructions shown on ImportError
        
    Instance Attributes:
        lang: Tesseract language code (stored but not used by TrOCR)
        gpu: Whether to use GPU acceleration
        device: PyTorch device string ("cuda" or "cpu")
        model_name: Hugging Face model identifier
        _processor: Cached TrOCRProcessor instance
        _model: Cached VisionEncoderDecoderModel instance
        
    Model Variants:
        - "printed": microsoft/trocr-base-printed (default)
          Best for printed text (documents, books, signage)
          
        - "handwritten": microsoft/trocr-base-handwritten
          Best for handwritten text (notes, forms, manuscripts)
          
    Initialization:
        The __init__ method performs lazy loading of transformers models using
        thread-safe helpers. Models are cached by (model_name, device) to
        prevent device mismatch errors and avoid redundant loading.
        
    OCR Operation:
        The ocr() method converts PIL Image to tensor, passes through model,
        and decodes generated token IDs to text. Returns plain text or a
        simple dict (TrOCR doesn't provide real bounding boxes).
        
    Thread Safety:
        Lazy loading is thread-safe via double-checked locking in _get_trocr().
        Multiple TrOCREngine instances share cached models if they use the
        same (model_name, device), ensuring memory efficiency.
        
    Example:
        ```python
        # Check availability before use
        if TrOCREngine.is_available():
            # Initialize with GPU and handwritten model
            engine = TrOCREngine(
                lang="eng",
                gpu=True, 
                model_variant="handwritten"
            )
            
            # Best use: text line images
            text_line = Image.open("handwritten_line.jpg")
            text = engine.ocr(text_line)
            
            # Avoid: full-page documents (will be distorted)
            # page = Image.open("full_page.jpg")  # Don't do this!
        else:
            print(f"Install with: {TrOCREngine.install_hint}")
        ```
    """

    # ========================================================================
    # CLASS ATTRIBUTES
    # ========================================================================
    
    # Engine identifier for CLI (-e trocr) and registry lookup
    name: str = "trocr"
    
    # Human-readable name for display in help text and error messages
    # Emphasizes transformer-based architecture for differentiation
    display_name: str = "TrOCR (Transformer-based)"
    
    # TrOCR supports GPU acceleration via PyTorch CUDA
    # GPU provides significant speedup (5-20x) for inference
    # Requires CUDA-compatible GPU with ~2GB+ VRAM
    supports_gpu: bool = True
    
    # TrOCR does NOT provide bounding boxes
    # It's designed for line-level OCR (pre-segmented text lines)
    # If return_boxes=True, we return a simple dict with text only
    supports_boxes: bool = False
    
    # Installation hint shown when transformers/torch not available
    # Users need both transformers (model code) and torch (backend)
    install_hint: str = "pip install transformers torch"

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def __init__(
        self, 
        lang: str = "eng", 
        gpu: bool = False, 
        model_variant: str = "printed",
        **kwargs: Any
    ) -> None:
        """Initialize TrOCR engine.
        
        Performs lazy loading of transformers models, selects appropriate
        model variant (printed vs handwritten), and configures device.
        
        Args:
            lang: Tesseract language code (default: "eng" for English).
                 NOTE: This parameter is accepted for API consistency with other
                 engines, but TrOCR models are language-agnostic (work on any
                 language). The lang parameter is stored but not used.
                 
            gpu: Whether to use GPU acceleration (default: False).
                If True, TrOCR will use CUDA-enabled PyTorch for inference.
                GPU mode is typically 5-20x faster than CPU.
                If GPU not available, initialization will fail with error.
                Requires CUDA-compatible GPU with ~2GB+ VRAM.
                
            model_variant: Model variant to use (default: "printed").
                Options:
                    - "printed": microsoft/trocr-base-printed
                      Best for printed text (documents, books, signage)
                      
                    - "handwritten": microsoft/trocr-base-handwritten  
                      Best for handwritten text (notes, forms, manuscripts)
                      
                Invalid values default to "printed" with no warning.
                
            **kwargs: Additional keyword arguments (currently unused).
                     Included for forward compatibility with OCREngine interface.
                     
        Raises:
            ImportError: If transformers or torch is not installed.
                        Error message includes installation instructions.
                        
            RuntimeError: If GPU requested but not available, or if model
                         loading fails (network error, disk space, etc.).
                         
        Notes:
            - Models are downloaded automatically to ~/.cache/huggingface/
            - First run may take time to download models (~300-500MB)
            - Models are cached after first download (fast subsequent loads)
            - GPU mode requires CUDA toolkit and CUDA-enabled PyTorch
            - CPU mode works on any system but is significantly slower
            
        Thread Safety:
            Multiple threads can call __init__ concurrently. The lazy loading
            helper ensures thread-safe initialization of shared resources.
            
        Example:
            ```python
            # CPU mode with printed text model (default)
            engine = TrOCREngine()
            
            # GPU mode with handwritten text model
            engine = TrOCREngine(gpu=True, model_variant="handwritten")
            
            # Specify language for API compatibility (not actually used)
            engine = TrOCREngine(lang="eng", gpu=True)
            ```
        """
        # Store original tesseract language code
        # NOTE: TrOCR models are language-agnostic (work on any language)
        # This parameter is stored for API consistency but not used
        self.lang: str = lang
        
        # Store GPU preference
        self.gpu: bool = gpu
        
        # Determine PyTorch device based on GPU preference
        # "cuda" enables GPU acceleration (requires CUDA-compatible GPU)
        # "cpu" uses CPU inference (slower but works on any system)
        self.device: str = "cuda" if gpu else "cpu"
        
        # Map model variant to Hugging Face model identifier
        # Provides user-friendly names ("printed"/"handwritten") while
        # using full model paths internally
        model_map: Dict[str, str] = {
            "printed": "microsoft/trocr-base-printed",
            "handwritten": "microsoft/trocr-base-handwritten",
        }
        
        # Look up model name, defaulting to printed if variant invalid
        # Invalid variants fail silently (default to printed) to avoid
        # breaking existing code if new variants are requested
        self.model_name: str = model_map.get(model_variant, model_map["printed"])
        
        # Lazy load TrOCR processor and model using thread-safe helper
        # This may take 10-60 seconds on first call (downloads + loads weights)
        # Subsequent calls with same (model_name, device) are instant (cached)
        processor, model = _get_trocr(self.model_name, self.device)
        
        # Validate that transformers/torch are available
        # If None, lazy loader failed to import (dependencies not installed)
        if processor is None or model is None:
            raise ImportError(
                "TrOCR dependencies are missing or failed to import. "
                "Ensure that the 'transformers' library and its dependencies "
                "such as 'torch' are installed. "
                f"Install with: {self.install_hint}"
            )
        
        # Cache processor and model references for use in ocr() method
        # Note: These may be shared across multiple engine instances
        # if they use the same (model_name, device) tuple (memory efficiency)
        self._processor = processor
        self._model = model

    # ========================================================================
    # OCR OPERATION
    # ========================================================================

    def ocr(
        self, image: Image.Image, return_boxes: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """Perform OCR on an image using TrOCR.
        
        Converts the PIL Image to model input tensors, runs inference, and
        decodes the generated token IDs to text. Best results on text line
        images (cropped regions); poor results on full-page documents due
        to 384x384 resize.
        
        IMPORTANT: TrOCR is designed for text line recognition, not full-page
        documents. Images are resized to 384x384 internally, which will distort
        large images and produce poor results. For page-level OCR, use tesseract
        or easyocr instead.
        
        Args:
            image: PIL Image object to perform OCR on.
                  Best results: Text line images (<1000px width/height)
                  Poor results: Full-page documents (>1000px width/height)
                  Can be any mode (RGB, L, RGBA); converted to RGB internally.
                  
            return_boxes: If True, return structured data with text.
                         If False, return plain text string (default).
                         
                         NOTE: TrOCR doesn't provide real bounding boxes
                         (it's line-level only). If return_boxes=True, we
                         return a dict with text and None for bbox/confidence.
                         
        Returns:
            If return_boxes is False:
                str: Extracted text from the image.
                     Empty string if no text recognized.
                     
            If return_boxes is True:
                Dict[str, Any]: Dictionary containing:
                    - "text" (str): Recognized text
                    - "confidence" (None): TrOCR doesn't provide confidence
                    - "bbox" (None): TrOCR doesn't provide bounding boxes
                    
        Warnings:
            If image width or height exceeds 1000 pixels, prints a warning to
            stderr that TrOCR is designed for line-level OCR and the image will
            be resized to 384x384, causing distortion. This helps users understand
            why results may be poor on full-page documents.
                         
        Raises:
            RuntimeError: If OCR operation fails (rare, usually GPU memory issues).
                         
        Notes:
            - Image is converted to RGB if not already (model expects RGB)
            - Image is resized to 384x384 internally by TrOCRProcessor
            - Model generates text autoregressively (token by token)
            - Max output length is 256 tokens (configurable in generate())
            - No language specification needed (model works on any language)
            
        Performance:
            GPU mode: 0.05-0.2 seconds per text line
            CPU mode: 0.5-2 seconds per text line
            Full page (not recommended): 10x slower + poor accuracy
            
        Thread Safety:
            This method is thread-safe. The cached model can be used
            concurrently by multiple threads (PyTorch handles locking).
            
        Example:
            ```python
            # Best use case: text line image
            text_line = Image.open("cropped_line.jpg")  # e.g., 800x50px
            text = engine.ocr(text_line)
            
            # Structured output (though TrOCR doesn't provide real boxes)
            result = engine.ocr(text_line, return_boxes=True)
            # {"text": "extracted text", "confidence": None, "bbox": None}
            
            # Poor results: full page (triggers warning)
            page = Image.open("full_page.jpg")  # e.g., 2000x3000px
            text = engine.ocr(page)  # Will print warning to stderr
            ```
        """
        # Ensure image is in RGB mode (required by TrOCRProcessor)
        # Model expects 3-channel RGB input, not grayscale or RGBA
        # convert() is a no-op if already RGB (efficient)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Get image dimensions for size check
        width, height = image.size
        
        # Warn if image is too large (likely a full page document)
        # TrOCR works best on text lines (typically <1000px in both dimensions)
        # Full pages will be resized to 384x384, causing severe distortion
        if width > 1000 or height > 1000:
            print(
                f"Warning: Image size ({width}x{height}) is large for TrOCR. "
                f"TrOCR is designed for line-level OCR and will resize to 384x384, "
                f"causing distortion. For full-page OCR, use tesseract or easyocr instead.",
                file=sys.stderr
            )
        
        # Get device from model parameters
        # This ensures we move input tensors to the same device as model
        # Prevents "tensor on different device" runtime errors
        device_obj = next(self._model.parameters()).device
        
        # Process image through TrOCRProcessor
        # Returns dict with "pixel_values" tensor of shape (1, 3, 384, 384)
        # Processor handles:
        #   - Resize to 384x384 (can distort large images)
        #   - Normalize pixel values for model input
        #   - Convert to PyTorch tensor
        pixel_values = self._processor(
            images=image, 
            return_tensors="pt"  # Return PyTorch tensors
        ).pixel_values
        
        # Move pixel_values tensor to model device (CPU or CUDA)
        # Critical: input and model must be on same device
        pixel_values = pixel_values.to(device_obj)
        
        # Generate text autoregressively using transformer decoder
        # Model generates token IDs one at a time until EOS token or max length
        # Shape: (1, sequence_length) where sequence_length <= 256
        # max_new_tokens=256 limits output length (prevents infinite generation)
        generated_ids = self._model.generate(pixel_values, max_new_tokens=256)
        
        # Decode generated token IDs to text string
        # batch_decode returns list (batch size 1, so we take first element)
        # skip_special_tokens=True removes [BOS], [EOS], [PAD] tokens
        text: str = self._processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        # Return format depends on return_boxes parameter
        if return_boxes:
            # Return structured data for API consistency with other engines
            # TrOCR doesn't provide bounding boxes or confidence scores
            # (it's a line-level model, not a detection + recognition pipeline)
            return {
                "text": text,
                "confidence": None,  # TrOCR doesn't provide confidence
                "bbox": None,  # TrOCR doesn't provide bounding boxes
            }
        
        # Return plain text string (most common use case)
        return text

    # ========================================================================
    # AVAILABILITY CHECK
    # ========================================================================

    @classmethod
    def is_available(cls) -> bool:
        """Check if TrOCR engine is available.
        
        Attempts to import transformers to determine if this engine can be
        used. This is a class method so it can be called without instantiating
        the engine, useful for engine discovery and CLI fallback logic.
        
        Returns:
            True if transformers is installed and can be imported.
            False if transformers is not available.
            
        Notes:
            - Only checks Python package availability, not GPU support
            - GPU availability check would require torch import (expensive)
            - If transformers imports but torch not available, init will fail
            - Does not check for model weight availability (downloaded on first use)
            - Does not validate torch version or CUDA availability
            
        Performance:
            This method is lightweight after first call, as Python caches
            import failures. First call may take ~100-500ms for import attempt.
            Subsequent calls are instant (cached in sys.modules).
            
        Example:
            ```python
            if TrOCREngine.is_available():
                engine = TrOCREngine(lang="eng", gpu=True)
                text = engine.ocr(image)
            else:
                print("TrOCR not available")
                print(f"Install with: {TrOCREngine.install_hint}")
            ```
            
        Design Rationale:
            We check transformers import rather than torch because:
            1. transformers depends on torch (if transformers works, torch likely works)
            2. Checking torch is slower (larger import)
            3. Error handling in __init__ provides better context if torch missing
        """
        try:
            # Try importing transformers library (optional dependency)
            # If successful, engine is available (assuming torch also installed)
            import transformers  # noqa: F401
            
            return True
        except ImportError:
            # transformers not installed or import failed
            return False

    # ========================================================================
    # LANGUAGE CONVERSION
    # ========================================================================

    @classmethod
    def convert_language(cls, tesseract_lang: str) -> str:
        """Convert tesseract language code to TrOCR format.
        
        TrOCR models are language-agnostic and don't require language codes.
        This method is included for API consistency with OCREngine base class,
        but simply returns the input unchanged since TrOCR doesn't use language
        codes for inference.
        
        Args:
            tesseract_lang: Tesseract language code (e.g., "eng", "deu", "chi_sim").
            
        Returns:
            The same language code unchanged (TrOCR doesn't use language codes).
            
        Notes:
            - TrOCR models work on any language without configuration
            - Model selection (printed vs handwritten) is via model_variant kwarg
            - Language-specific TrOCR models exist on Hugging Face but are not
              currently supported by this engine (would require extending model_map)
            - This method exists for API compatibility with other engines
            
        Example:
            >>> TrOCREngine.convert_language("eng")
            'eng'
            >>> TrOCREngine.convert_language("deu")
            'deu'
            >>> TrOCREngine.convert_language("unknown")
            'unknown'
            
        Design Rationale:
            We could remove this method (default implementation in base class
            does the same), but explicitly overriding it with documentation
            clarifies that TrOCR's language handling is different from other
            engines. This helps future maintainers understand the design.
        """
        # TrOCR models are language-agnostic
        # Return input unchanged for API consistency
        return tesseract_lang
