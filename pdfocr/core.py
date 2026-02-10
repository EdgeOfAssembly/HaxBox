#!/usr/bin/env python3
"""Core OCR functions for the pdfocr package.

This module contains the main OCR functions that process images and PDFs:
- preprocess_image_for_ocr(): Enhance images for better OCR accuracy
- ocr_image(): OCR a single PIL Image using any engine
- ocr_pdf(): OCR all pages of a PDF and save results
- ocr_image_file(): OCR a single image file and save results

These functions form the core processing pipeline. They use the engine
architecture from pdfocr.engines to support multiple OCR backends
(Tesseract, EasyOCR, TrOCR, PaddleOCR, docTR).

All functions include comprehensive type hints for mypy --strict compatibility
and detailed Google-style docstrings.

Usage:
    # OCR a single image
    from PIL import Image
    from pdfocr.core import ocr_image
    
    image = Image.open("scan.png")
    text = ocr_image(image, engine="tesseract", lang="eng")
    
    # OCR a PDF file
    from pathlib import Path
    from pdfocr.core import ocr_pdf
    
    output = ocr_pdf(
        Path("document.pdf"),
        Path("output/"),
        engine="easyocr",
        lang="eng"
    )
    
    # OCR an image file  
    from pdfocr.core import ocr_image_file
    
    output = ocr_image_file(
        Path("scan.png"),
        Path("output/"),
        engine="tesseract"
    )
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar

from PIL import Image

from pdfocr.engines import get_engine
from pdfocr.types import (
    TESSERACT_TO_EASYOCR_LANG,
    TESSERACT_TO_PADDLEOCR_LANG,
    OCRResult,
)
from pdfocr.utils import pdf_to_images

# ============================================================================
# TQDM FALLBACK PATTERN
# ============================================================================
# tqdm is an optional dependency for progress bars. If not installed, we
# provide a passthrough fallback that returns the iterable unchanged.

_T = TypeVar("_T")

try:
    from tqdm import tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

    def tqdm(  # type: ignore[no-redef]
        iterable: Iterable[_T], *args: Any, **kwargs: Any
    ) -> Iterable[_T]:
        """Fallback tqdm that passes through the iterable unchanged.
        
        This no-op implementation is used when tqdm is not installed.
        It accepts the same signature as tqdm but ignores all arguments
        except the iterable, which is returned as-is.
        
        Args:
            iterable: The iterable to wrap (returned unchanged).
            *args: Ignored positional arguments (for tqdm compatibility).
            **kwargs: Ignored keyword arguments (for tqdm compatibility).
            
        Returns:
            The original iterable, unchanged.
        """
        return iterable


# ============================================================================
# LAZY CV2/NUMPY LOADING FOR IMAGE PREPROCESSING
# ============================================================================
# OpenCV (cv2) and NumPy are optional dependencies used for image enhancement.
# We lazy-load them with thread-safe initialization to avoid import overhead
# when preprocessing is disabled or dependencies are not installed.

_cv2: Any = None  # OpenCV module (cv2), loaded on first use
_np: Any = None  # NumPy module, loaded on first use
_cv2_lock = threading.Lock()  # Thread-safety for lazy initialization
_numpy_lock = threading.Lock()  # Thread-safety for lazy initialization


def _get_numpy() -> Any:
    """Lazy import numpy with thread-safe initialization.
    
    This function performs lazy loading of NumPy to avoid import overhead
    when image preprocessing is not needed. Thread-safety is guaranteed
    through a lock to prevent race conditions in multi-threaded environments.
    
    Returns:
        The numpy module if available and successfully imported.
        None if NumPy is not installed or import fails.
        
    Note:
        The first call to this function triggers the actual import and
        caches the module in a global variable. Subsequent calls return
        the cached module without re-importing.
    """
    global _np
    
    # Fast path: already loaded
    if _np is not None:
        return _np
    
    # Slow path: need to import (with lock for thread-safety)
    with _numpy_lock:
        # Double-check pattern: another thread may have loaded it
        # while we were waiting for the lock
        if _np is None:
            try:
                import numpy as np

                _np = np
            except ImportError:
                # NumPy is an optional dependency; if not installed, return None
                # Callers should gracefully handle None by skipping preprocessing
                pass
    
    return _np


def _get_cv2() -> Any:
    """Lazy import OpenCV (cv2) with thread-safe initialization.
    
    This function performs lazy loading of OpenCV to avoid import overhead
    when image preprocessing is not needed. Thread-safety is guaranteed
    through a lock to prevent race conditions in multi-threaded environments.
    
    As a side effect, this also loads NumPy (since OpenCV depends on it)
    and caches it in the global _np variable for use by other functions.
    
    Returns:
        The cv2 (OpenCV) module if available and successfully imported.
        None if OpenCV is not installed or import fails.
        
    Note:
        The first call to this function triggers the actual import and
        caches the module in a global variable. Subsequent calls return
        the cached module without re-importing.
        
        This function also sets the global _np variable as a side effect
        since OpenCV imports require NumPy.
    """
    global _cv2, _np
    
    # Fast path: already loaded
    if _cv2 is not None:
        return _cv2
    
    # Slow path: need to import (with lock for thread-safety)
    with _cv2_lock:
        # Double-check pattern: another thread may have loaded it
        # while we were waiting for the lock
        if _cv2 is None:
            try:
                import cv2
                import numpy as np

                _cv2 = cv2
                _np = np
            except ImportError:
                # OpenCV and NumPy are optional dependencies; if unavailable,
                # return None so callers can skip CV-based preprocessing gracefully
                pass
    
    return _cv2


# ============================================================================
# CORE OCR FUNCTIONS
# ============================================================================


def preprocess_image_for_ocr(image: Image.Image, enhance: bool = True) -> Image.Image:
    """Preprocess image to improve OCR accuracy using CLAHE enhancement.
    
    This function applies Contrast Limited Adaptive Histogram Equalization
    (CLAHE) to improve contrast and text readability in images before OCR.
    CLAHE works by equalizing histograms in small regions (tiles) of the image,
    which enhances local contrast without amplifying noise as much as global
    histogram equalization would.
    
    The enhancement is particularly effective for:
        - Low-contrast scanned documents
        - Faded or aged documents
        - Uneven lighting/shadows
        - Poor quality photocopies
    
    If OpenCV or NumPy are not installed, or if enhance=False, the original
    image is returned unchanged. This allows graceful degradation when optional
    dependencies are missing.
    
    Technical details:
        - For color images (RGB/RGBA): Converts to LAB color space, applies
          CLAHE to the lightness (L) channel only, then converts back to RGB.
          This preserves color information while enhancing contrast.
        - For grayscale images: Applies CLAHE directly to intensity values.
        - CLAHE parameters: clipLimit=2.0, tileGridSize=(8, 8)
            - clipLimit controls contrast enhancement strength (lower = less)
            - tileGridSize controls the size of local regions (smaller = more local)
    
    Args:
        image: PIL Image to preprocess. Can be RGB, RGBA, or grayscale.
            The image format is automatically detected and handled appropriately.
        enhance: Whether to apply enhancement. If False, returns the original
            image unchanged. Default is True.
    
    Returns:
        Preprocessed PIL Image in RGB or grayscale format (matching input).
        If enhancement is disabled or dependencies are unavailable, returns
        the original image unchanged.
    
    Note:
        This function requires opencv-python (cv2) and numpy to be installed.
        If they are not available, the image is returned unchanged without error.
        
    Examples:
        >>> from PIL import Image
        >>> from pdfocr.core import preprocess_image_for_ocr
        >>> 
        >>> # Enhance a scanned document
        >>> img = Image.open("scan.png")
        >>> enhanced = preprocess_image_for_ocr(img, enhance=True)
        >>> 
        >>> # Skip enhancement (faster, but may reduce accuracy)
        >>> raw = preprocess_image_for_ocr(img, enhance=False)
    """
    # Fast path: enhancement disabled, return original image
    if not enhance:
        return image

    # Try to load OpenCV (cv2) - returns None if not installed
    cv2 = _get_cv2()
    if cv2 is None:
        # No OpenCV available, return image as-is
        # This allows the OCR pipeline to continue without preprocessing
        return image

    # _get_cv2() also loads numpy into _np as a side effect
    # Check that NumPy was loaded successfully
    if _np is None:
        # No NumPy available (shouldn't happen if cv2 loaded, but check anyway)
        return image

    # Convert PIL Image to NumPy array for OpenCV processing
    img_array = _np.array(image)

    # Handle color images: convert from RGB(A) to BGR format for OpenCV
    # OpenCV uses BGR channel order, while PIL uses RGB
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:
            # RGBA format (with alpha channel)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif img_array.shape[2] == 3:
            # RGB format (no alpha channel)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if len(img_array.shape) == 3:
        # Color image: Apply CLAHE to lightness channel in LAB color space
        # This preserves color while enhancing contrast
        
        # Convert BGR to LAB color space
        # LAB separates lightness (L) from color (A, B)
        lab = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)
        
        # Split into L, A, B channels
        l, a, b = cv2.split(lab)
        
        # Create CLAHE object
        # clipLimit=2.0: Limits contrast enhancement to avoid over-amplifying noise
        # tileGridSize=(8,8): Divides image into 8x8 grid of tiles for local equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Apply CLAHE to lightness channel only
        cl = clahe.apply(l)
        
        # Merge enhanced lightness with original color channels
        enhanced_lab = cv2.merge((cl, a, b))
        
        # Convert back to BGR, then to RGB for PIL
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        result = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    else:
        # Grayscale image: Apply CLAHE directly to intensity values
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        result = clahe.apply(img_array)

    # Convert NumPy array back to PIL Image and return
    return Image.fromarray(result)


def ocr_image(
    image: Image.Image,
    engine: str = "tesseract",
    lang: str = "eng",
    enhance: bool = True,
    gpu: bool = False,
    return_boxes: bool = False,
    batch_size: int = 1,
) -> OCRResult:
    """OCR a single PIL Image using the specified engine.
    
    This is the primary function for performing OCR on a single image. It
    supports multiple OCR engines through a unified interface, handles image
    preprocessing, and returns either plain text or structured data with
    bounding boxes.
    
    The function automatically:
        1. Preprocesses the image if enhance=True (CLAHE contrast enhancement)
        2. Converts language codes between engine-specific formats
        3. Instantiates the appropriate engine with the new architecture
        4. Returns results in the requested format (text or structured)
    
    Supported engines:
        - "tesseract": Fast, good for printed text (default)
        - "easyocr": Higher accuracy, supports more languages
        - "trocr": Transformer-based, good for line-level OCR
        - "trocr-handwritten": Specialized for handwritten text
        - "paddleocr": State-of-the-art accuracy
        - "doctr": Document-focused with strong layout analysis
    
    Language codes:
        The function uses Tesseract language codes by default (e.g., "eng",
        "deu", "fra") and automatically converts them to engine-specific codes
        when needed:
            - EasyOCR uses ISO 639-1 codes ("en", "de", "fr")
            - PaddleOCR uses full language names ("english", "german", "french")
            - TrOCR and docTR are language-agnostic
    
    Args:
        image: PIL Image to OCR. Can be RGB, RGBA, or grayscale.
        engine: OCR engine name. One of: "tesseract", "easyocr", "trocr",
            "trocr-handwritten", "paddleocr", "doctr". Default is "tesseract".
        lang: Language code in Tesseract format (e.g., "eng", "deu", "fra").
            Default is "eng" (English). Automatically converted to engine-specific
            format when needed.
        enhance: Apply CLAHE preprocessing to improve OCR accuracy. Default is True.
            Set to False for faster processing if image quality is already good.
        gpu: Use GPU acceleration when supported by the engine. Default is False.
            Supported by: EasyOCR, TrOCR, PaddleOCR, docTR. Ignored by Tesseract.
        return_boxes: Return structured data with bounding boxes instead of plain text.
            Default is False (returns plain text string).
            - For EasyOCR: Returns list of [bbox, text, confidence] entries
            - For PaddleOCR: Returns list of [bbox, (text, confidence)] entries
            - For docTR: Returns dict with hierarchical document structure
            - For TrOCR: Returns dict with "text" key (no boxes, line-level OCR)
            - For Tesseract: Ignored, always returns plain text
        batch_size: Batch size for PaddleOCR text recognition model. Default is 1
            (minimal memory usage). Higher values may be faster but use more GPU memory.
            Ignored by other engines.
    
    Returns:
        OCR result in one of the following formats:
            - str: Plain text (if return_boxes=False or engine doesn't support boxes)
            - List[Any]: Structured data with bounding boxes (engine-specific format)
            - Dict[str, Any]: Document structure (docTR) or text dict (TrOCR)
    
    Raises:
        KeyError: If the engine name is not recognized.
        ImportError: If the engine's dependencies are not installed.
        
    Examples:
        >>> from PIL import Image
        >>> from pdfocr.core import ocr_image
        >>> 
        >>> # Simple OCR with Tesseract (default)
        >>> img = Image.open("document.png")
        >>> text = ocr_image(img)
        >>> print(text)
        
        >>> # Use EasyOCR with GPU acceleration
        >>> text = ocr_image(img, engine="easyocr", gpu=True)
        >>> 
        >>> # Get structured output with bounding boxes
        >>> results = ocr_image(img, engine="easyocr", return_boxes=True)
        >>> for bbox, text, confidence in results:
        ...     print(f"{text} ({confidence:.2f})")
        >>> 
        >>> # German language with PaddleOCR
        >>> text = ocr_image(img, engine="paddleocr", lang="deu")
    """
    # Step 1: Preprocess the image to enhance contrast and readability
    # This applies CLAHE if enhance=True and OpenCV is available
    processed = preprocess_image_for_ocr(image, enhance=enhance)

    # Step 2: Handle special cases for engines with variants or language conversion
    
    if engine == "trocr":
        # TrOCR printed text variant
        # Note: TrOCR works best on text lines, not full pages
        converted_lang = "eng"  # TrOCR is language-agnostic, but we pass a dummy value
        ocr_engine = get_engine(
            engine, lang=converted_lang, gpu=gpu, model_variant="printed"
        )
        return ocr_engine.ocr(processed, return_boxes=return_boxes)
        
    elif engine == "trocr-handwritten":
        # TrOCR handwritten text variant
        converted_lang = "eng"  # TrOCR is language-agnostic, but we pass a dummy value
        ocr_engine = get_engine(
            engine, lang=converted_lang, gpu=gpu, model_variant="handwritten"
        )
        return ocr_engine.ocr(processed, return_boxes=return_boxes)
        
    elif engine == "easyocr":
        # Convert Tesseract language code to EasyOCR format
        # EasyOCR uses ISO 639-1 two-letter codes (e.g., "en" instead of "eng")
        if lang in TESSERACT_TO_EASYOCR_LANG:
            # Use the mapping table for known languages
            converted_lang = TESSERACT_TO_EASYOCR_LANG[lang]
        else:
            # Fallback for unknown languages: try first 2 characters
            # This handles cases like "eng" -> "en" for languages not in the mapping
            converted_lang = lang[:2] if len(lang) > 2 else lang
        
        # Instantiate EasyOCR engine with converted language
        ocr_engine = get_engine(engine, lang=converted_lang, gpu=gpu)
        return ocr_engine.ocr(processed, return_boxes=return_boxes)
        
    elif engine == "paddleocr":
        # Convert Tesseract language code to PaddleOCR format
        # PaddleOCR uses full language names (e.g., "english" instead of "eng")
        converted_lang = TESSERACT_TO_PADDLEOCR_LANG.get(lang, "en")
        
        # Instantiate PaddleOCR engine with batch size parameter
        ocr_engine = get_engine(
            engine, lang=converted_lang, gpu=gpu, batch_size=batch_size
        )
        return ocr_engine.ocr(processed, return_boxes=return_boxes)
        
    elif engine == "doctr":
        # docTR doesn't require language specification (multilingual by default)
        ocr_engine = get_engine(engine, lang="multilingual", gpu=gpu)
        return ocr_engine.ocr(processed, return_boxes=return_boxes)
        
    else:
        # Default case: Tesseract or any other engine
        # Use language code as-is (Tesseract uses 3-letter codes like "eng", "deu")
        ocr_engine = get_engine(engine, lang=lang, gpu=gpu)
        return ocr_engine.ocr(processed, return_boxes=return_boxes)


def ocr_pdf(
    pdf_path: Path,
    output_dir: Path,
    engine: str = "tesseract",
    lang: str = "eng",
    dpi: int = 300,
    enhance: bool = True,
    save_images: bool = False,
    force: bool = False,
    quiet: bool = False,
    pages: Optional[List[int]] = None,
    output_format: str = "text",
    gpu: bool = False,
    batch_size: int = 1,
) -> Optional[Path]:
    """OCR a PDF file and save the extracted text to a file.
    
    This function provides a complete pipeline for OCR'ing PDF documents:
        1. Converts PDF pages to images at the specified DPI
        2. Optionally filters to specific pages
        3. OCRs each page using the specified engine
        4. Combines results and saves to output file
        5. Optionally saves page images for debugging
    
    The function handles errors gracefully, including per-page failures, and
    supports both plain text and JSON output formats. It also checks for
    existing output files and prompts for confirmation before overwriting
    (unless --force or --quiet is used).
    
    Output formats:
        - "text": Plain text file with page separators
          Format: "--- Page N ---\n<text>\n\n--- Page N+1 ---\n..."
        - "json": Structured JSON file with metadata
          Format: {"source": "...", "engine": "...", "pages": [...]}
          Each page entry contains: {"page": N, "text": "..."}
          Or with bounding boxes: {"page": N, "results": [...]}
    
    Page numbering:
        All page numbers are 1-indexed (first page is 1, not 0).
    
    Args:
        pdf_path: Path to the PDF file to process.
        output_dir: Directory where output file will be saved. Must exist.
        engine: OCR engine to use. One of: "tesseract", "easyocr", "trocr",
            "trocr-handwritten", "paddleocr", "doctr". Default is "tesseract".
        lang: Language code in Tesseract format (e.g., "eng", "deu", "fra").
            Default is "eng". Automatically converted to engine-specific format.
        dpi: DPI for PDF-to-image rendering. Default is 300. Higher values
            produce sharper images but use more memory. Range typically 72-600.
        enhance: Apply CLAHE preprocessing to improve OCR accuracy. Default is True.
        save_images: Also save each page as a PNG file in output_dir.
            Default is False. Files are named "{stem}_page_{N:03d}.png".
        force: Force overwrite of existing output files without prompting.
            Default is False.
        quiet: Suppress progress bars and prompts. If True, existing files
            are skipped without prompting. Default is False.
        pages: Optional list of 1-indexed page numbers to OCR. If None, all
            pages are processed. Example: [1, 2, 3] to OCR only first 3 pages.
        output_format: Output format. "text" for plain text (default) or "json"
            for structured JSON with metadata.
        gpu: Use GPU acceleration when supported by the engine. Default is False.
        batch_size: Batch size for PaddleOCR. Default is 1 (minimal memory).
    
    Returns:
        Path to the output file if successful, None if failed or skipped.
        Possible reasons for None return:
            - User declined overwrite prompt
            - PDF-to-image conversion failed
            - File write failed
    
    Raises:
        No exceptions are raised. All errors are caught and reported to stderr.
        
    Examples:
        >>> from pathlib import Path
        >>> from pdfocr.core import ocr_pdf
        >>> 
        >>> # Simple OCR of entire PDF
        >>> output = ocr_pdf(
        ...     Path("document.pdf"),
        ...     Path("output/"),
        ...     engine="tesseract"
        ... )
        >>> print(f"Saved to: {output}")
        
        >>> # OCR first 5 pages with EasyOCR and GPU
        >>> output = ocr_pdf(
        ...     Path("document.pdf"),
        ...     Path("output/"),
        ...     engine="easyocr",
        ...     pages=[1, 2, 3, 4, 5],
        ...     gpu=True
        ... )
        
        >>> # Save as JSON with bounding boxes
        >>> output = ocr_pdf(
        ...     Path("document.pdf"),
        ...     Path("output/"),
        ...     engine="easyocr",
        ...     output_format="json",
        ...     force=True
        ... )
        
        >>> # High DPI, save images, quiet mode
        >>> output = ocr_pdf(
        ...     Path("document.pdf"),
        ...     Path("output/"),
        ...     dpi=600,
        ...     save_images=True,
        ...     quiet=True
        ... )
    """
    # Determine output file name based on input PDF stem and format
    stem = pdf_path.stem
    ext = ".json" if output_format == "json" else ".txt"
    output_file = output_dir / f"{stem}{ext}"

    # Check if output file already exists and handle overwrite logic
    if output_file.exists() and not force:
        if quiet:
            # Quiet mode: skip without prompting
            print(
                f"Skipping existing file '{output_file}' (use --force to overwrite).",
                file=sys.stderr,
            )
            return None
        
        # Interactive mode: prompt user for confirmation
        print(f"File '{output_file}' exists. Overwrite? (Y/N): ", end="", flush=True)
        try:
            response = input().strip().lower()
            if response != "y":
                # User declined, skip this file
                return None
        except EOFError:
            # Input stream closed (e.g., in non-interactive environment)
            return None

    # Convert PDF to images
    # This uses pdf2image under the hood, which requires poppler
    try:
        page_images = pdf_to_images(pdf_path, dpi=dpi, pages=pages)
    except Exception as e:
        print(f"Error converting PDF to images: {e}", file=sys.stderr)
        return None

    # Initialize result accumulators based on output format
    if output_format == "json":
        # JSON format: list of page result dicts
        all_results: List[Dict[str, Any]] = []
    else:
        # Text format: list of text strings
        all_text: List[str] = []

    # Set up progress bar if tqdm is available and not in quiet mode
    pages_iter: Iterable[Tuple[int, Image.Image]]
    if not quiet and _HAS_TQDM:
        # Show progress bar with page count
        pages_iter = tqdm(page_images, desc=f"OCR {pdf_path.name}", unit="page")
    else:
        # No progress bar: just use the iterable as-is
        pages_iter = page_images

    # Process each page
    for page_num, image in pages_iter:
        try:
            # Determine if we need structured output (bounding boxes)
            # Currently only EasyOCR supports this in JSON format
            if output_format == "json" and engine == "easyocr":
                # Get structured results with bounding boxes
                results = ocr_image(
                    image,
                    engine=engine,
                    lang=lang,
                    enhance=enhance,
                    gpu=gpu,
                    return_boxes=True,
                    batch_size=batch_size,
                )
                # Store page number and results
                all_results.append({"page": page_num, "results": results})
            else:
                # Get plain text or convert structured to text
                text = ocr_image(
                    image,
                    engine=engine,
                    lang=lang,
                    enhance=enhance,
                    gpu=gpu,
                    batch_size=batch_size,
                )
                
                if output_format == "json":
                    # JSON format: store text with page number
                    all_results.append({"page": page_num, "text": text})
                else:
                    # Text format: add page separator
                    all_text.append(f"--- Page {page_num} ---\n{text}")

            # Optionally save page image for debugging
            if save_images:
                img_path = output_dir / f"{stem}_page_{page_num:03d}.png"
                image.save(str(img_path), "PNG")

        except Exception as e:
            # Per-page error handling: record error and continue with next page
            if output_format == "json":
                # JSON format: store error message
                all_results.append({"page": page_num, "error": str(e)})
            else:
                # Text format: insert error marker
                all_text.append(f"--- Page {page_num} ---\n[OCR ERROR: {e}]")
            
            if not quiet:
                # Report error to stderr (unless in quiet mode)
                print(f"Error on page {page_num}: {e}", file=sys.stderr)

    # Write output file
    try:
        if output_format == "json":
            # JSON format: structured output with metadata
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source": str(pdf_path),
                        "engine": engine,
                        "language": lang,
                        "pages": all_results,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        else:
            # Text format: concatenate page texts with separators
            output_file.write_text("\n\n".join(all_text), encoding="utf-8")
        
        return output_file
        
    except Exception as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        return None


def ocr_image_file(
    image_path: Path,
    output_dir: Path,
    engine: str = "tesseract",
    lang: str = "eng",
    enhance: bool = True,
    force: bool = False,
    quiet: bool = False,
    output_format: str = "text",
    gpu: bool = False,
    batch_size: int = 1,
) -> Optional[Path]:
    """OCR an image file and save the extracted text to a file.
    
    This function provides a complete pipeline for OCR'ing image files:
        1. Loads the image from disk
        2. OCRs the image using the specified engine
        3. Saves results to output file
    
    The function handles errors gracefully and supports both plain text and
    JSON output formats. It checks for existing output files and prompts for
    confirmation before overwriting (unless --force or --quiet is used).
    
    Output formats:
        - "text": Plain text file with extracted text
        - "json": Structured JSON file with metadata
          Format: {"source": "...", "engine": "...", "language": "...", "text": "..."}
          Or with bounding boxes: {"source": "...", "results": [...]}
    
    Args:
        image_path: Path to the image file to process.
        output_dir: Directory where output file will be saved. Must exist.
        engine: OCR engine to use. One of: "tesseract", "easyocr", "trocr",
            "trocr-handwritten", "paddleocr", "doctr". Default is "tesseract".
        lang: Language code in Tesseract format (e.g., "eng", "deu", "fra").
            Default is "eng". Automatically converted to engine-specific format.
        enhance: Apply CLAHE preprocessing to improve OCR accuracy. Default is True.
        force: Force overwrite of existing output files without prompting.
            Default is False.
        quiet: Suppress prompts. If True, existing files are skipped without
            prompting. Default is False.
        output_format: Output format. "text" for plain text (default) or "json"
            for structured JSON with metadata.
        gpu: Use GPU acceleration when supported by the engine. Default is False.
        batch_size: Batch size for PaddleOCR. Default is 1 (minimal memory).
    
    Returns:
        Path to the output file if successful, None if failed or skipped.
        Possible reasons for None return:
            - User declined overwrite prompt
            - Image loading failed
            - OCR failed
            - File write failed
    
    Raises:
        No exceptions are raised. All errors are caught and reported to stderr.
        
    Examples:
        >>> from pathlib import Path
        >>> from pdfocr.core import ocr_image_file
        >>> 
        >>> # Simple OCR of image
        >>> output = ocr_image_file(
        ...     Path("scan.png"),
        ...     Path("output/"),
        ...     engine="tesseract"
        ... )
        >>> print(f"Saved to: {output}")
        
        >>> # Use EasyOCR with GPU and JSON output
        >>> output = ocr_image_file(
        ...     Path("scan.png"),
        ...     Path("output/"),
        ...     engine="easyocr",
        ...     output_format="json",
        ...     gpu=True
        ... )
        
        >>> # German language with PaddleOCR
        >>> output = ocr_image_file(
        ...     Path("scan.png"),
        ...     Path("output/"),
        ...     engine="paddleocr",
        ...     lang="deu",
        ...     force=True
        ... )
    """
    # Determine output file name based on input image stem and format
    stem = image_path.stem
    ext = ".json" if output_format == "json" else ".txt"
    output_file = output_dir / f"{stem}{ext}"

    # Check if output file already exists and handle overwrite logic
    if output_file.exists() and not force:
        if quiet:
            # Quiet mode: skip without prompting
            print(
                f"Skipping existing file '{output_file}' (use --force to overwrite).",
                file=sys.stderr,
            )
            return None
        
        # Interactive mode: prompt user for confirmation
        print(f"File '{output_file}' exists. Overwrite? (Y/N): ", end="", flush=True)
        try:
            response = input().strip().lower()
            if response != "y":
                # User declined, skip this file
                return None
        except EOFError:
            # Input stream closed (e.g., in non-interactive environment)
            return None

    try:
        # Load the image from disk
        image = Image.open(image_path)

        # Determine if we need structured output (bounding boxes)
        # Currently only EasyOCR supports this in JSON format
        if output_format == "json" and engine == "easyocr":
            # Get structured results with bounding boxes
            results = ocr_image(
                image,
                engine=engine,
                lang=lang,
                enhance=enhance,
                gpu=gpu,
                return_boxes=True,
                batch_size=batch_size,
            )
            
            # Write JSON output with bounding boxes
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source": str(image_path),
                        "engine": engine,
                        "language": lang,
                        "results": results,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        else:
            # Get plain text or convert structured to text
            text = ocr_image(
                image,
                engine=engine,
                lang=lang,
                enhance=enhance,
                gpu=gpu,
                batch_size=batch_size,
            )
            
            if output_format == "json":
                # Write JSON output with text only
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "source": str(image_path),
                            "engine": engine,
                            "language": lang,
                            "text": text,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
            else:
                # Write plain text output
                output_file.write_text(text, encoding="utf-8")

        return output_file
        
    except Exception as e:
        print(f"Error processing '{image_path}': {e}", file=sys.stderr)
        return None
