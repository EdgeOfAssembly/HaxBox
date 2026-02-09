#!/usr/bin/env python3
"""
pdfocr - OCR tool for extracting text from PDFs and images.

Features:
- Extract text from scanned PDFs and images using OCR
- Support multiple OCR engines: tesseract (fast, default), easyocr (higher accuracy), trocr (transformer-based)
- Multiple input files and batch directory processing
- Image preprocessing for better OCR accuracy
- Page selection for OCR (e.g., only first 5 pages)
- Output to configurable directory
- JSON output format with bounding boxes
- Searchable PDF output

Author: EdgeOfAssembly
Date: 2025-12-19

License: GPLv3 / Commercial dual-license

Dependencies:
- pytesseract (default engine): pip install pytesseract
  Also requires tesseract binary: apt install tesseract-ocr (Ubuntu)
- easyocr (higher accuracy engine): pip install easyocr
- transformers (trocr engine, line-level OCR): pip install transformers torch
  NOTE: TrOCR works best on text lines, not full pages. Use tesseract/easyocr for full pages.
- paddleocr (state-of-the-art): pip install paddleocr paddlepaddle
  NOTE: Requires PaddleOCR 3.0+ which uses device='gpu'/'cpu' parameter
- python-doctr (document-focused): pip install python-doctr[torch]
- pdf2image: pip install pdf2image
  Also requires poppler: apt install poppler-utils (Ubuntu)
- opencv-python-headless: pip install opencv-python-headless
- pillow: pip install pillow
"""

from __future__ import annotations

import sys
import argparse
import json
import threading
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, FrozenSet, Iterable, TypeVar

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is not installed. Install with: pip install pillow")
    sys.exit(1)

# TypeVar for tqdm fallback
_T = TypeVar("_T")

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

    def tqdm(  # type: ignore[no-redef]
        iterable: Iterable[_T], *args: Any, **kwargs: Any
    ) -> Iterable[_T]:
        """Fallback tqdm that passes through the iterable unchanged."""
        return iterable


# OCR engines - lazily imported with thread safety
_pytesseract: Any = None
_easyocr_reader: Any = None
_easyocr_reader_langs: Optional[FrozenSet[str]] = None
_cv2: Any = None
_np: Any = None
_trocr_cache: Dict[Tuple[str, str], Tuple[Any, Any]] = {}  # Cache by (model_name, device)
_paddleocr: Any = None
_doctr_model: Any = None
_easyocr_lock = threading.Lock()
_pytesseract_lock = threading.Lock()
_cv2_lock = threading.Lock()
_numpy_lock = threading.Lock()
_trocr_lock = threading.Lock()
_paddleocr_lock = threading.Lock()
_doctr_lock = threading.Lock()


def _get_numpy() -> Any:
    """Lazy import numpy (thread-safe).

    Returns:
        The numpy module if available, otherwise None.
    """
    global _np
    if _np is None:
        with _numpy_lock:
            if _np is None:
                try:
                    import numpy as np

                    _np = np
                except ImportError:
                    # numpy is an optional dependency; if not installed, return None
                    pass
    return _np


__version__ = "2.0.0"
DEFAULT_OUTPUT_DIR = "ocr_out"
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
MIN_DPI = 72
MAX_DPI = 600

# Mapping from tesseract language codes to easyocr language codes
# Not all tesseract languages are supported by easyocr
TESSERACT_TO_EASYOCR_LANG = {
    "eng": "en",
    "deu": "de",
    "fra": "fr",
    "spa": "es",
    "ita": "it",
    "por": "pt",
    "nld": "nl",
    "pol": "pl",
    "rus": "ru",
    "ukr": "uk",
    "chi_sim": "ch_sim",
    "chi_tra": "ch_tra",
    "jpn": "ja",
    "kor": "ko",
    "ara": "ar",
    "hin": "hi",
    "tha": "th",
    "vie": "vi",
    "tur": "tr",
    "heb": "he",
}

# Mapping from tesseract language codes to paddleocr language codes
TESSERACT_TO_PADDLEOCR_LANG = {
    "eng": "en",
    "deu": "german",
    "fra": "french",
    "spa": "spanish",
    "ita": "italian",
    "por": "portuguese",
    "rus": "ru",
    "ukr": "ukrainian",
    "chi_sim": "ch",
    "chi_tra": "chinese_cht",
    "jpn": "japan",
    "kor": "korean",
    "ara": "ar",
    "hin": "hi",
    "tha": "th",
    "vie": "vi",
    "tur": "tr",
}


def _get_pytesseract() -> Any:
    """Lazy import pytesseract (thread-safe).

    Returns:
        The pytesseract module if available, otherwise None.
    """
    global _pytesseract
    if _pytesseract is None:
        with _pytesseract_lock:
            if _pytesseract is None:
                try:
                    import pytesseract

                    _pytesseract = pytesseract
                except ImportError:
                    # pytesseract is an optional dependency; if not installed, return None
                    pass
    return _pytesseract


def _get_easyocr_reader(
    langs: Optional[List[str]] = None, gpu: bool = False
) -> Any:
    """Lazy import and initialize easyocr reader (thread-safe).

    Note: Creates a new reader if languages differ from the cached one.

    Args:
        langs: List of language codes (default: ['en']).
        gpu: Whether to use GPU acceleration (default: False).

    Returns:
        EasyOCR Reader instance, or None if not available.
    """
    global _easyocr_reader, _easyocr_reader_langs
    if langs is None:
        langs = ["en"]

    requested_langs: FrozenSet[str] = frozenset(langs)

    with _easyocr_lock:
        # Check if we need to reinitialize with different languages
        if _easyocr_reader is not None and _easyocr_reader_langs == requested_langs:
            return _easyocr_reader

        try:
            import easyocr

            _easyocr_reader = easyocr.Reader(list(langs), gpu=gpu)
            _easyocr_reader_langs = requested_langs
        except ImportError:
            # easyocr is an optional dependency; if not installed, return None
            pass

    return _easyocr_reader


def _get_cv2() -> Any:
    """Lazy import OpenCV (thread-safe).

    Returns:
        The cv2 (OpenCV) module if available, otherwise None.
    """
    global _cv2, _np
    if _cv2 is None:
        with _cv2_lock:
            if _cv2 is None:
                try:
                    import cv2
                    import numpy as np

                    _cv2 = cv2
                    _np = np
                except ImportError:
                    # OpenCV and NumPy are optional; if unavailable, return None so callers
                    # can skip CV-based preprocessing gracefully.
                    pass
    return _cv2


def _get_trocr(model_variant: str = "printed", gpu: bool = False) -> Tuple[Any, Any]:
    """Lazy import and initialize TrOCR processor and model (thread-safe).
    
    Args:
        model_variant: 'printed' or 'handwritten'
        gpu: Whether to use GPU acceleration
    
    Returns:
        Tuple of (processor, model)
    """
    global _trocr_cache
    
    # Model names
    model_map = {
        "printed": "microsoft/trocr-base-printed",
        "handwritten": "microsoft/trocr-base-handwritten",
    }
    model_name = model_map.get(model_variant, model_map["printed"])
    device = 'cuda' if gpu else 'cpu'
    cache_key = (model_name, device)
    
    with _trocr_lock:
        # Return cached model if already initialized with same variant and device
        if cache_key in _trocr_cache:
            return _trocr_cache[cache_key]
        
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            processor = TrOCRProcessor.from_pretrained(model_name)
            model = VisionEncoderDecoderModel.from_pretrained(model_name)
            
            # Move model to device
            model = model.to(device)
            model.eval()
            
            # Cache the model and processor
            _trocr_cache[cache_key] = (processor, model)
            
        except ImportError as e:
            return None, None
    
    return _trocr_cache[cache_key]


def _get_paddleocr(lang: str = "en", gpu: bool = False) -> Any:
    """Lazy import and initialize PaddleOCR (thread-safe).
    
    Note: Creates a new instance if language or GPU settings differ from cached one.
    
    Args:
        lang: Language code for PaddleOCR
        gpu: Whether to use GPU acceleration
    
    Returns:
        PaddleOCR instance or None if not available
    """
    global _paddleocr
    
    def _create_paddleocr_instance(lang: str, gpu: bool) -> Any:
        """Helper to create PaddleOCR instance."""
        try:
            from paddleocr import PaddleOCR
            
            # PaddleOCR 3.0+ uses 'device' instead of deprecated 'use_gpu'
            # Also 'use_textline_orientation' replaces deprecated 'use_angle_cls'
            try:
                return PaddleOCR(
                    use_textline_orientation=True,
                    lang=lang,
                    device='gpu' if gpu else 'cpu'
                )
            except TypeError:
                # PaddleOCR <3.0 doesn't support these parameters
                raise ImportError(
                    "PaddleOCR 3.0+ is required. Please upgrade: "
                    "pip install --upgrade paddleocr paddlepaddle"
                )
        except ImportError:
            return None
    
    # We cache by parameters since PaddleOCR initialization is expensive
    # In practice, most use cases will use the same language throughout
    cache_key = (lang, gpu)
    
    with _paddleocr_lock:
        # For simplicity, we'll reinitialize if different parameters are requested
        # In most use cases, parameters remain constant
        if _paddleocr is None:
            instance = _create_paddleocr_instance(lang, gpu)
            if instance is not None:
                _paddleocr = (instance, cache_key)
        else:
            # Check if cached parameters match
            instance, cached_key = _paddleocr
            if cached_key != cache_key:
                # Parameters changed, reinitialize
                instance = _create_paddleocr_instance(lang, gpu)
                if instance is not None:
                    _paddleocr = (instance, cache_key)
        
        return _paddleocr[0] if _paddleocr else None


def _get_doctr_model(gpu: bool = False) -> Any:
    """Lazy import and initialize docTR model (thread-safe).
    
    Note: Creates a new model if GPU setting differs from cached one.
    
    Args:
        gpu: Whether to use GPU acceleration
    
    Returns:
        docTR predictor or None if not available
    """
    global _doctr_model
    
    def _create_doctr_model(gpu: bool) -> Any:
        """Helper to create docTR model."""
        try:
            from doctr.models import ocr_predictor
            
            device = 'cuda' if gpu else 'cpu'
            return ocr_predictor(pretrained=True).to(device)
        except ImportError:
            return None
    
    with _doctr_lock:
        # We cache by GPU parameter
        if _doctr_model is None:
            instance = _create_doctr_model(gpu)
            if instance is not None:
                _doctr_model = (instance, gpu)
        else:
            # Check if cached GPU setting matches
            instance, cached_gpu = _doctr_model
            if cached_gpu != gpu:
                # GPU setting changed, reinitialize
                instance = _create_doctr_model(gpu)
                if instance is not None:
                    _doctr_model = (instance, gpu)
        
        return _doctr_model[0] if _doctr_model else None


def preprocess_image_for_ocr(image: Image.Image, enhance: bool = True) -> Image.Image:
    """
    Preprocess image to improve OCR accuracy.

    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    and optional denoising.

    Args:
        image: PIL Image to preprocess.
        enhance: Whether to apply enhancement (default True).

    Returns:
        Preprocessed PIL Image.
    """
    if not enhance:
        return image

    cv2 = _get_cv2()
    if cv2 is None:
        return image  # No opencv, return as-is

    # _get_cv2 also loads numpy into _np
    if _np is None:
        return image  # No numpy, return as-is

    # Convert PIL to OpenCV format
    img_array = _np.array(image)

    # Convert to BGR if RGB
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif img_array.shape[2] == 3:  # RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Convert to LAB color space for CLAHE on lightness channel
    if len(img_array.shape) == 3:
        lab = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)
        lang, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(lang)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        result = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    else:
        # Grayscale
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        result = clahe.apply(img_array)

    return Image.fromarray(result)


def ocr_with_tesseract(image: Image.Image, lang: str = "eng") -> str:
    """Perform OCR using Tesseract.

    Args:
        image: PIL Image to OCR.
        lang: Tesseract language code (default: 'eng').

    Returns:
        Extracted text.
    """
    pytesseract = _get_pytesseract()
    if pytesseract is None:
        raise ImportError("pytesseract not installed. Install: pip install pytesseract")

    result: str = pytesseract.image_to_string(image, lang=lang)
    return result


def ocr_with_easyocr(
    image: Image.Image,
    langs: Optional[List[str]] = None,
    gpu: bool = False,
    return_boxes: bool = False,
) -> Any:
    """
    Perform OCR using EasyOCR.

    Args:
        image: PIL Image to OCR.
        langs: List of language codes (default: ['en']).
        gpu: Whether to use GPU acceleration.
        return_boxes: If True, return list of dicts with text and bounding boxes.

    Returns:
        Extracted text (str), or list of dicts with boxes if return_boxes=True.
    """
    reader = _get_easyocr_reader(langs, gpu=gpu)
    if reader is None:
        raise ImportError("easyocr not installed. Install: pip install easyocr")

    np = _get_numpy()
    if np is None:
        raise ImportError("numpy not installed. Install: pip install numpy")

    img_array = np.array(image)
    results = reader.readtext(img_array)

    if return_boxes:
        # Return structured data with bounding boxes
        structured_results = []
        for bbox, text, confidence in results:
            # bbox is a list of 4 points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            structured_results.append(
                {
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": [[int(p[0]), int(p[1])] for p in bbox],
                }
            )
        return structured_results

    # Extract just the text from results
    text_parts = [result[1] for result in results]
    return "\n".join(text_parts)


def ocr_with_trocr(
    image: Image.Image,
    model_variant: str = "printed",
    gpu: bool = False,
    return_boxes: bool = False,
) -> Any:
    """Perform OCR using Microsoft TrOCR.
    
    NOTE: TrOCR is designed for line-level OCR (single lines of text), not full-page documents.
    For best results, use TrOCR on cropped text line images. Full-page documents will be 
    resized to 384x384 pixels, causing distortion and poor recognition quality.
    
    For full-page document OCR, consider using tesseract or easyocr engines instead.
    
    Args:
        image: PIL Image to OCR. Best results with single text lines (cropped regions).
        model_variant: 'printed' or 'handwritten'
        gpu: Whether to use GPU acceleration.
        return_boxes: If True, return structured data (TrOCR doesn't provide boxes natively)
    
    Returns:
        Extracted text (str), or dict with text if return_boxes=True.
    """
    processor, model = _get_trocr(model_variant, gpu=gpu)
    
    if processor is None or model is None:
        raise ImportError(
            "TrOCR dependencies are missing or failed to import. "
            "Ensure that the 'transformers' library and its dependencies such as 'torch' "
            "are installed (for example: pip install transformers torch)."
        )
    
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Warn if image is too large (likely a full page)
    # TrOCR works best on text lines, not full pages
    width, height = image.size
    if width > 1000 or height > 1000:
        import sys
        print(
            f"Warning: Image size ({width}x{height}) is large for TrOCR. "
            f"TrOCR is designed for line-level OCR and will resize to 384x384, "
            f"causing distortion. For full-page OCR, use tesseract or easyocr instead.",
            file=sys.stderr
        )
    
    # Get device from model
    device = next(model.parameters()).device
    
    # Process image - processor returns pixel_values
    pixel_values = processor(images=image, return_tensors='pt').pixel_values.to(device)
    
    # Generate text
    generated_ids = model.generate(pixel_values, max_new_tokens=256)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    if return_boxes:
        # TrOCR doesn't provide bounding boxes natively
        return {"text": text, "confidence": None, "bbox": None}
    
    return text


def ocr_with_paddleocr(
    image: Image.Image,
    lang: str = "en",
    gpu: bool = False,
    return_boxes: bool = False,
) -> Any:
    """Perform OCR using PaddleOCR.
    
    Args:
        image: PIL Image to OCR
        lang: PaddleOCR language code (default: 'en')
        gpu: Whether to use GPU acceleration
        return_boxes: If True, return list of dicts with text and bounding boxes
    
    Returns:
        Extracted text (str), or list of dicts with boxes if return_boxes=True
    """
    paddleocr = _get_paddleocr(lang=lang, gpu=gpu)
    if paddleocr is None:
        raise ImportError(
            "PaddleOCR not installed. Install: pip install paddleocr paddlepaddle"
        )
    
    np = _get_numpy()
    if np is None:
        raise ImportError("numpy not installed. Install: pip install numpy")
    
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Run PaddleOCR
    result = paddleocr.ocr(img_array, cls=True)
    
    if result is None or len(result) == 0:
        return [] if return_boxes else ""
    
    if return_boxes:
        # Return structured data with bounding boxes
        structured_results = []
        for line in result:
            if line is None:
                continue
            for box_info in line:
                bbox, (text, confidence) = box_info
                structured_results.append({
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": [[int(p[0]), int(p[1])] for p in bbox],
                })
        return structured_results
    
    # Extract just the text from results
    text_parts = []
    for line in result:
        if line is None:
            continue
        for box_info in line:
            _, (text, _) = box_info
            text_parts.append(text)
    
    return "\n".join(text_parts)


def ocr_with_doctr(
    image: Image.Image,
    gpu: bool = False,
    return_boxes: bool = False,
) -> Any:
    """Perform OCR using docTR.
    
    Args:
        image: PIL Image to OCR
        gpu: Whether to use GPU acceleration
        return_boxes: If True, return list of dicts with text and bounding boxes
    
    Returns:
        Extracted text (str), or list of dicts with boxes if return_boxes=True
    """
    # Constants for horizontal spacing detection
    WIDE_GAP_THRESHOLD_PIXELS = 30  # Threshold for detecting wide gaps between words
    PIXELS_PER_SPACE = 10  # Conversion factor for gap pixels to spaces
    
    doctr_model = _get_doctr_model(gpu=gpu)
    if doctr_model is None:
        raise ImportError(
            "docTR not installed. Install: pip install python-doctr[torch]"
        )
    
    np = _get_numpy()
    if np is None:
        raise ImportError("numpy not installed. Install: pip install numpy")
    
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Run docTR
    result = doctr_model([img_array])
    
    if return_boxes:
        # Return structured data with bounding boxes
        structured_results = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        # Get bounding box coordinates
                        bbox = word.geometry
                        # docTR returns normalized coordinates [0, 1]
                        h, w = img_array.shape[:2]
                        x1, y1 = int(bbox[0][0] * w), int(bbox[0][1] * h)
                        x2, y2 = int(bbox[1][0] * w), int(bbox[1][1] * h)
                        
                        structured_results.append({
                            "text": word.value,
                            "confidence": float(word.confidence),
                            "bbox": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        })
        return structured_results
    
    # Extract just the text from results, preserving horizontal spacing
    text_parts = []
    for page in result.pages:
        _, page_width = img_array.shape[:2]
        
        for block in page.blocks:
            for line in block.lines:
                if not line.words:
                    continue
                
                # Sort words by horizontal position
                sorted_words = sorted(line.words, key=lambda w: w.geometry[0][0])
                
                line_parts = []
                prev_end_x = None
                
                for word in sorted_words:
                    # Get word position (normalized 0-1)
                    word_start_x = word.geometry[0][0]
                    
                    if prev_end_x is not None:
                        # Calculate gap between words in pixels
                        gap = (word_start_x - prev_end_x) * page_width
                        
                        # If gap is large (> ~3 average char widths), add extra spaces
                        if gap > WIDE_GAP_THRESHOLD_PIXELS:
                            # Add proportional spaces
                            num_spaces = max(1, int(gap / PIXELS_PER_SPACE))
                            line_parts.append(" " * num_spaces)
                        else:
                            line_parts.append(" ")
                    
                    line_parts.append(word.value)
                    prev_end_x = word.geometry[1][0]  # End x of current word
                
                line_text = "".join(line_parts)
                if line_text.strip():
                    text_parts.append(line_text)
    
    return "\n".join(text_parts)


def pdf_to_images(
    pdf_path: Path, dpi: int = 300, pages: Optional[List[int]] = None
) -> List[Tuple[int, Image.Image]]:
    """
    Convert PDF pages to PIL Images.

    Args:
        pdf_path: Path to PDF file.
        dpi: DPI for rendering (default 300).
        pages: Optional list of 1-indexed page numbers to convert.
               If None, converts all pages.

    Returns:
        List of (page_number, PIL Image) tuples.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError("pdf2image not installed. Install: pip install pdf2image")

    if pages is not None and len(pages) > 0:
        # Convert specific pages only
        result = []
        for page_num in sorted(set(pages)):
            try:
                images = convert_from_path(
                    str(pdf_path), dpi=dpi, first_page=page_num, last_page=page_num
                )
                if images:
                    result.append((page_num, images[0]))
            except Exception:
                pass
        return result
    else:
        # Convert all pages
        images = convert_from_path(str(pdf_path), dpi=dpi)
        return [(i + 1, img) for i, img in enumerate(images)]


def parse_page_spec(spec: str, total_pages: int) -> List[int]:
    """
    Parse page specification string into list of page numbers.

    Supports formats like:
    - "1,7,67" - individual pages
    - "1-10" - page range
    - "56-" - from page 56 to end
    - "-10" - from page 1 to 10
    - "1-5,10,20-25" - combined

    Args:
        spec: Page specification string.
        total_pages: Total number of pages in the PDF.

    Returns:
        List of 1-indexed page numbers.

    Raises:
        ValueError: If specification is invalid.
    """
    pages: List[int] = []
    parts = spec.split(",")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        try:
            if "-" in part:
                hyphen_count = part.count("-")

                if hyphen_count == 1:
                    idx = part.index("-")
                    start_str = part[:idx]
                    end_str = part[idx + 1 :]
                    start = int(start_str) if start_str else 1
                    end = int(end_str) if end_str else total_pages
                elif hyphen_count > 1:
                    raise ValueError(
                        f"Invalid range format (multiple hyphens): '{part}'"
                    )
                else:
                    raise ValueError(f"Invalid range format: '{part}'")
            else:
                start = int(part)
                end = start
        except ValueError as e:
            if "Invalid range format" in str(e) or "multiple hyphens" in str(e):
                raise
            raise ValueError(f"Invalid page number in '{part}': not a valid integer")

        if start < 1 or end > total_pages or start > end:
            raise ValueError(
                f"Page range out of bounds: {start}-{end} (PDF has {total_pages} pages)"
            )

        pages.extend(range(start, end + 1))

    return sorted(set(pages))


def validate_dpi(dpi: int) -> int:
    """
    Validate DPI value and return a valid value.

    Args:
        dpi: DPI value to validate.

    Returns:
        Valid DPI value (clamped to MIN_DPI-MAX_DPI range).

    Raises:
        ValueError: If DPI is invalid (zero or negative).
    """
    if dpi <= 0:
        raise ValueError(f"DPI must be positive, got {dpi}")
    if dpi < MIN_DPI:
        return MIN_DPI
    if dpi > MAX_DPI:
        return MAX_DPI
    return dpi


def ocr_image(
    image: Image.Image,
    engine: str = "tesseract",
    lang: str = "eng",
    enhance: bool = True,
    gpu: bool = False,
    return_boxes: bool = False,
) -> Any:
    """
    OCR a single image.

    Args:
        image: PIL Image to OCR.
        engine: OCR engine ('tesseract', 'easyocr', 'trocr', 'trocr-handwritten', 'paddleocr', or 'doctr').
        lang: Language code(s).
        enhance: Apply preprocessing.
        gpu: Use GPU acceleration for EasyOCR, TrOCR, PaddleOCR, and docTR.
        return_boxes: Return structured data with bounding boxes (EasyOCR, PaddleOCR, docTR) or text-only dict (TrOCR).

    Returns:
        Extracted text (str), or list of dicts with boxes for EasyOCR/PaddleOCR/docTR, or dict with text for TrOCR if return_boxes=True.
    """
    processed = preprocess_image_for_ocr(image, enhance=enhance)

    if engine == "trocr":
        return ocr_with_trocr(processed, model_variant="printed", gpu=gpu, return_boxes=return_boxes)
    elif engine == "trocr-handwritten":
        return ocr_with_trocr(processed, model_variant="handwritten", gpu=gpu, return_boxes=return_boxes)
    elif engine == "easyocr":
        # Convert tesseract lang code to easyocr using mapping
        if lang in TESSERACT_TO_EASYOCR_LANG:
            easyocr_lang = TESSERACT_TO_EASYOCR_LANG[lang]
        else:
            # Fallback: try first 2 chars or use as-is
            easyocr_lang = lang[:2] if len(lang) > 2 else lang
        langs = [easyocr_lang] if easyocr_lang else ["en"]
        return ocr_with_easyocr(processed, langs, gpu=gpu, return_boxes=return_boxes)
    elif engine == "paddleocr":
        # Convert tesseract lang code to paddleocr using mapping
        paddleocr_lang = TESSERACT_TO_PADDLEOCR_LANG.get(lang, "en")
        return ocr_with_paddleocr(processed, paddleocr_lang, gpu=gpu, return_boxes=return_boxes)
    elif engine == "doctr":
        return ocr_with_doctr(processed, gpu=gpu, return_boxes=return_boxes)
    else:
        return ocr_with_tesseract(processed, lang)


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
) -> Optional[Path]:
    """
    OCR a PDF file and save the extracted text.

    Args:
        pdf_path: Path to PDF file.
        output_dir: Output directory.
        engine: OCR engine to use.
        lang: Language code.
        dpi: DPI for rendering.
        enhance: Apply image preprocessing.
        save_images: Also save page images.
        force: Force overwrite.
        quiet: Suppress output.
        pages: Optional list of 1-indexed page numbers to OCR.
        output_format: Output format ('text' or 'json').
        gpu: Use GPU acceleration for EasyOCR.

    Returns:
        Path to output text file, or None if failed.
    """
    stem = pdf_path.stem
    ext = ".json" if output_format == "json" else ".txt"
    output_file = output_dir / f"{stem}{ext}"

    if output_file.exists() and not force:
        if quiet:
            print(
                f"Skipping existing file '{output_file}' (use --force to overwrite).",
                file=sys.stderr,
            )
            return None
        print(f"File '{output_file}' exists. Overwrite? (Y/N): ", end="", flush=True)
        try:
            response = input().strip().lower()
            if response != "y":
                return None
        except EOFError:
            return None

    try:
        page_images = pdf_to_images(pdf_path, dpi=dpi, pages=pages)
    except Exception as e:
        print(f"Error converting PDF to images: {e}", file=sys.stderr)
        return None

    if output_format == "json":
        all_results: List[Dict[str, Any]] = []
    else:
        all_text: List[str] = []

    pages_iter: Iterable[Tuple[int, Image.Image]]
    if not quiet and _HAS_TQDM:
        pages_iter = tqdm(page_images, desc=f"OCR {pdf_path.name}", unit="page")
    else:
        pages_iter = page_images

    for page_num, image in pages_iter:
        try:
            if output_format == "json" and engine == "easyocr":
                # Get structured results with bounding boxes
                results = ocr_image(
                    image,
                    engine=engine,
                    lang=lang,
                    enhance=enhance,
                    gpu=gpu,
                    return_boxes=True,
                )
                all_results.append({"page": page_num, "results": results})
            else:
                text = ocr_image(
                    image, engine=engine, lang=lang, enhance=enhance, gpu=gpu
                )
                if output_format == "json":
                    all_results.append({"page": page_num, "text": text})
                else:
                    all_text.append(f"--- Page {page_num} ---\n{text}")

            if save_images:
                img_path = output_dir / f"{stem}_page_{page_num:03d}.png"
                image.save(str(img_path), "PNG")

        except Exception as e:
            if output_format == "json":
                all_results.append({"page": page_num, "error": str(e)})
            else:
                all_text.append(f"--- Page {page_num} ---\n[OCR ERROR: {e}]")
            if not quiet:
                print(f"Error on page {page_num}: {e}", file=sys.stderr)

    # Write output
    try:
        if output_format == "json":
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
) -> Optional[Path]:
    """
    OCR an image file and save the extracted text.

    Args:
        image_path: Path to image file.
        output_dir: Output directory.
        engine: OCR engine to use.
        lang: Language code.
        enhance: Apply image preprocessing.
        force: Force overwrite.
        quiet: Suppress output.
        output_format: Output format ('text' or 'json').
        gpu: Use GPU acceleration for EasyOCR.

    Returns:
        Path to output text file, or None if failed.
    """
    stem = image_path.stem
    ext = ".json" if output_format == "json" else ".txt"
    output_file = output_dir / f"{stem}{ext}"

    if output_file.exists() and not force:
        if quiet:
            print(
                f"Skipping existing file '{output_file}' (use --force to overwrite).",
                file=sys.stderr,
            )
            return None
        print(f"File '{output_file}' exists. Overwrite? (Y/N): ", end="", flush=True)
        try:
            response = input().strip().lower()
            if response != "y":
                return None
        except EOFError:
            return None

    try:
        image = Image.open(image_path)

        if output_format == "json" and engine == "easyocr":
            results = ocr_image(
                image,
                engine=engine,
                lang=lang,
                enhance=enhance,
                gpu=gpu,
                return_boxes=True,
            )
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
            text = ocr_image(image, engine=engine, lang=lang, enhance=enhance, gpu=gpu)
            if output_format == "json":
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
                output_file.write_text(text, encoding="utf-8")

        return output_file
    except Exception as e:
        print(f"Error processing '{image_path}': {e}", file=sys.stderr)
        return None


def process_inputs(inputs: List[str]) -> List[Path]:
    """
    Process input arguments - can be files or directories.

    Args:
        inputs: List of input paths.

    Returns:
        List of file paths to process.
    """
    files: List[Path] = []

    for inp in inputs:
        path = Path(inp)
        if path.is_dir():
            # Batch mode: get all PDFs and images (case-insensitive)
            # Use case-insensitive glob pattern to avoid duplicates on
            # case-insensitive filesystems (Windows, macOS default)
            files.extend(sorted(path.glob("*.[pP][dD][fF]")))

            # Collect image files with case-insensitive patterns
            for ext in SUPPORTED_IMAGE_EXTENSIONS:
                # Create case-insensitive glob pattern
                # e.g., '.png' -> '*.[pP][nN][gG]'
                ext_no_dot = ext[1:]  # Remove leading dot
                pattern = "*." + "".join(
                    f"[{c.lower()}{c.upper()}]" for c in ext_no_dot
                )
                files.extend(sorted(path.glob(pattern)))
        elif path.is_file():
            suffix = path.suffix.lower()
            if suffix == ".pdf" or suffix in SUPPORTED_IMAGE_EXTENSIONS:
                files.append(path)
            else:
                print(f"Warning: Unsupported file type: {path}", file=sys.stderr)
        else:
            print(f"Warning: Path not found: {path}", file=sys.stderr)

    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in files:
        f_resolved = f.resolve()
        if f_resolved not in seen:
            seen.add(f_resolved)
            unique_files.append(f)

    return unique_files


def check_engine_available(engine: str) -> bool:
    """Check if OCR engine is available."""
    if engine == "easyocr":
        try:
            import easyocr  # noqa: F401

            return True
        except ImportError:
            return False
    elif engine in ("trocr", "trocr-handwritten"):
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # noqa: F401

            return True
        except ImportError:
            return False
    elif engine == "paddleocr":
        try:
            import paddleocr  # noqa: F401

            return True
        except ImportError:
            return False
    elif engine == "doctr":
        try:
            from doctr.models import ocr_predictor  # noqa: F401

            return True
        except ImportError:
            return False
    else:  # tesseract
        try:
            import pytesseract

            # Check if tesseract binary is available
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False


def main() -> None:
    """Main entry point for pdfocr."""
    parser = argparse.ArgumentParser(
        prog="pdfocr",
        description="OCR tool for extracting text from PDFs and images.",
        epilog="""
Examples:
  pdfocr scanned.pdf                      # OCR a PDF with tesseract
  pdfocr scanned.pdf -e easyocr           # OCR with easyocr (better quality)
  pdfocr scanned.pdf -e easyocr --gpu     # OCR with easyocr using GPU
  pdfocr scanned.pdf -e trocr --gpu       # OCR with TrOCR transformer model
  pdfocr image.png                        # OCR an image
  pdfocr /path/to/files/                  # Batch process directory
  pdfocr a.pdf b.png -d output            # Process multiple files
  pdfocr scanned.pdf --save-images        # Also save page images
  pdfocr scanned.pdf -l deu               # OCR in German
  pdfocr scanned.pdf -p 1-5               # OCR only first 5 pages
  pdfocr scanned.pdf --format json        # Output as JSON with bounding boxes

Supported OCR engines:
  tesseract        - Fast, default, requires tesseract-ocr binary
  easyocr          - Better quality, slower, pure Python (use --gpu for speed)
  trocr            - Transformer OCR for line-level printed text (use --gpu)
                     NOTE: Best for text lines, not full pages
  trocr-handwritten - Transformer OCR for line-level handwriting (use --gpu)
                     NOTE: Best for text lines, not full pages
  paddleocr        - State-of-the-art accuracy, medium speed (use --gpu for speed)
  doctr            - Document-focused OCR, best for complex layouts (use --gpu)
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "inputs",
        nargs="*",
        help="PDF/image file(s) or directory. If directory, processes all PDFs and images.",
    )
    parser.add_argument(
        "-e",
        "--engine",
        choices=["tesseract", "easyocr", "trocr", "trocr-handwritten", "paddleocr", "doctr"],
        default="tesseract",
        help="OCR engine to use (default: tesseract)",
    )
    parser.add_argument(
        "-l",
        "--lang",
        default="eng",
        help="Language code for OCR (default: eng). For tesseract: eng, deu, fra, etc. For easyocr: en, de, fr, etc.",
    )
    parser.add_argument(
        "-d",
        "--directory",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "-p",
        "--pages",
        help='Page specification for PDFs: "1,7,67" or "1-10" or "56-" or "-10" (combined: "1-5,10,20-25")',
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help=f"DPI for PDF rendering (default: 300, range: {MIN_DPI}-{MAX_DPI})",
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Disable image preprocessing (CLAHE contrast enhancement)",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Also save rendered page images (for PDFs)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format: text (default) or json (includes bounding boxes with easyocr)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration for EasyOCR, TrOCR, PaddleOCR, and docTR (requires CUDA)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite existing output files",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress non-error output"
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if not args.inputs:
        print("Error: No input files specified.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    # Validate DPI
    try:
        args.dpi = validate_dpi(args.dpi)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Check if engine is available
    if not check_engine_available(args.engine):
        if args.engine == "easyocr":
            print(
                "Error: easyocr not available. Install: pip install easyocr",
                file=sys.stderr,
            )
            if not args.quiet:
                print("Falling back to tesseract...", file=sys.stderr)
            args.engine = "tesseract"
            if not check_engine_available("tesseract"):
                print("Error: tesseract also not available.", file=sys.stderr)
                print(
                    "Install: pip install pytesseract && apt install tesseract-ocr",
                    file=sys.stderr,
                )
                sys.exit(1)
        elif args.engine in ("trocr", "trocr-handwritten"):
            print(
                f"Error: {args.engine} not available. Install: pip install transformers",
                file=sys.stderr,
            )
            if not args.quiet:
                print("Falling back to tesseract...", file=sys.stderr)
            args.engine = "tesseract"
            if not check_engine_available("tesseract"):
                print("Error: tesseract also not available.", file=sys.stderr)
                print(
                    "Install: pip install pytesseract && apt install tesseract-ocr",
                    file=sys.stderr,
                )
                sys.exit(1)
        elif args.engine == "paddleocr":
            print(
                "Error: paddleocr not available. Install: pip install paddleocr paddlepaddle",
                file=sys.stderr,
            )
            if not args.quiet:
                print("Falling back to tesseract...", file=sys.stderr)
            args.engine = "tesseract"
            if not check_engine_available("tesseract"):
                print("Error: tesseract also not available.", file=sys.stderr)
                print(
                    "Install: pip install pytesseract && apt install tesseract-ocr",
                    file=sys.stderr,
                )
                sys.exit(1)
        elif args.engine == "doctr":
            print(
                "Error: doctr not available. Install: pip install python-doctr[torch]",
                file=sys.stderr,
            )
            if not args.quiet:
                print("Falling back to tesseract...", file=sys.stderr)
            args.engine = "tesseract"
            if not check_engine_available("tesseract"):
                print("Error: tesseract also not available.", file=sys.stderr)
                print(
                    "Install: pip install pytesseract && apt install tesseract-ocr",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            print("Error: tesseract not available.", file=sys.stderr)
            print(
                "Install: pip install pytesseract && apt install tesseract-ocr",
                file=sys.stderr,
            )
            sys.exit(1)

    # Warn if --gpu specified with tesseract
    if args.gpu and args.engine == "tesseract":
        print("Warning: --gpu is only supported with easyocr, trocr, trocr-handwritten, paddleocr, and doctr engines.", file=sys.stderr)

    # Process inputs
    files = process_inputs(args.inputs)

    if not files:
        print("Error: No supported files found.", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.directory)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory '{output_dir}': {e}", file=sys.stderr)
        sys.exit(1)

    successful = 0

    for file_path in files:
        if not args.quiet:
            print(f"\nProcessing: {file_path}")

        if file_path.suffix.lower() == ".pdf":
            # Parse page specification if provided
            pages_to_ocr = None
            if args.pages:
                try:
                    # Get total pages for validation
                    from pdf2image import pdfinfo_from_path

                    info = pdfinfo_from_path(str(file_path))
                    total_pages = info.get("Pages", 0)
                    if total_pages > 0:
                        pages_to_ocr = parse_page_spec(args.pages, total_pages)
                except ImportError:
                    print(
                        "Warning: Cannot validate page specification without pdf2image.",
                        file=sys.stderr,
                    )
                    # Try to use the pages anyway
                    pages_to_ocr = [
                        int(p) for p in args.pages.split(",") if p.strip().isdigit()
                    ]
                except ValueError as e:
                    print(f"Error: {e}", file=sys.stderr)
                    continue

            result = ocr_pdf(
                file_path,
                output_dir,
                engine=args.engine,
                lang=args.lang,
                dpi=args.dpi,
                enhance=not args.no_enhance,
                save_images=args.save_images,
                force=args.force,
                quiet=args.quiet,
                pages=pages_to_ocr,
                output_format=args.format,
                gpu=args.gpu,
            )
        else:
            result = ocr_image_file(
                file_path,
                output_dir,
                engine=args.engine,
                lang=args.lang,
                enhance=not args.no_enhance,
                force=args.force,
                quiet=args.quiet,
                output_format=args.format,
                gpu=args.gpu,
            )

        if result:
            successful += 1
            if not args.quiet:
                print(f"  Output: {result}")

    if not args.quiet:
        print(f"\nSuccessfully processed {successful}/{len(files)} file(s)")
        print(f"Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
