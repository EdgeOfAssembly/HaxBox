#!/usr/bin/env python3
"""Package initialization and public API for pdfocr.

This module exports the public API for the pdfocr package, making it usable
both as a CLI tool and as a Python library.

Public API:
    # Core OCR functions
    ocr_image(image, engine, lang, ...) -> str or dict
    ocr_pdf(pdf_path, output_dir, ...) -> None
    ocr_image_file(image_path, output_dir, ...) -> None
    
    # Engine management
    get_engine(name, **kwargs) -> OCREngine
    available_engines() -> List[str]
    
    # Utilities
    preprocess_image_for_ocr(image, enhance) -> Image
    validate_dpi(dpi) -> int
    parse_page_spec(spec, total_pages) -> List[int]
    
    # Font detection
    detect_fonts(input_path, dpi) -> List[dict]
    detect_fonts_native(pdf_path) -> List[dict]
    detect_fonts_scanned(input_path, dpi) -> List[dict]
    estimate_redacted_chars(image, redaction_bbox, dpi, reference_text_bboxes) -> dict
    
    # Constants
    __version__: str
    MIN_DPI, MAX_DPI: int
    DEFAULT_OUTPUT_DIR: str
    SUPPORTED_IMAGE_EXTENSIONS: Set[str]

Usage as a library:
    ```python
    from pdfocr import ocr_image, get_engine, available_engines
    from PIL import Image
    
    # Check available engines
    print(available_engines())  # ['tesseract', 'easyocr', ...]
    
    # Use high-level API
    image = Image.open("scan.png")
    text = ocr_image(image, engine="easyocr", lang="eng", gpu=True)
    print(text)
    
    # Or use low-level engine API
    engine = get_engine("easyocr", lang="eng", gpu=True)
    text = engine.ocr(image, return_boxes=False)
    print(text)
    ```

Usage as CLI:
    ```bash
    python -m pdfocr document.pdf
    pdfocr document.pdf -e easyocr --gpu
    ```
"""

from __future__ import annotations

# Version
from pdfocr.types import (
    __version__,
    DEFAULT_OUTPUT_DIR,
    SUPPORTED_IMAGE_EXTENSIONS,
    MIN_DPI,
    MAX_DPI,
    WIDE_GAP_THRESHOLD_PIXELS,
    PIXELS_PER_SPACE,
    TESSERACT_TO_EASYOCR_LANG,
    TESSERACT_TO_PADDLEOCR_LANG,
    OCRResult,
)

# Core OCR functions
from pdfocr.core import (
    ocr_image,
    ocr_pdf,
    ocr_image_file,
    preprocess_image_for_ocr,
)

# Utilities
from pdfocr.utils import (
    validate_dpi,
    parse_page_spec,
    process_inputs,
    pdf_to_images,
)

# Engine management
from pdfocr.engines import (
    get_engine,
    available_engines,
    get_all_engines,
    OCREngine,
)

# CLI functions
from pdfocr.cli import (
    main,
    check_engine_available,
)

# Font detection
from pdfocr.font_detect import (
    detect_fonts,
    detect_fonts_native,
    detect_fonts_scanned,
    estimate_redacted_chars,
)

__all__ = [
    # Version and constants
    "__version__",
    "DEFAULT_OUTPUT_DIR",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "MIN_DPI",
    "MAX_DPI",
    "WIDE_GAP_THRESHOLD_PIXELS",
    "PIXELS_PER_SPACE",
    "TESSERACT_TO_EASYOCR_LANG",
    "TESSERACT_TO_PADDLEOCR_LANG",
    "OCRResult",
    # Core OCR functions
    "ocr_image",
    "ocr_pdf",
    "ocr_image_file",
    "preprocess_image_for_ocr",
    # Utilities
    "validate_dpi",
    "parse_page_spec",
    "process_inputs",
    "pdf_to_images",
    # Engine management
    "get_engine",
    "available_engines",
    "get_all_engines",
    "OCREngine",
    # CLI functions
    "main",
    "check_engine_available",
    # Font detection
    "detect_fonts",
    "detect_fonts_native",
    "detect_fonts_scanned",
    "estimate_redacted_chars",
]
