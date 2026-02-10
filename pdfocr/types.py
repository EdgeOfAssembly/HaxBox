#!/usr/bin/env python3
"""Shared types, constants, and type definitions for the pdfocr package.

This module contains all shared type definitions, constants, and TypedDicts
used throughout the pdfocr package. Centralizing these allows for consistent
typing across all modules and engines.

Type Definitions:
    OCRResult: Union type for OCR function return values
    
Constants:
    __version__: Package version string
    DEFAULT_OUTPUT_DIR: Default output directory name
    SUPPORTED_IMAGE_EXTENSIONS: Set of supported image file extensions
    MIN_DPI: Minimum DPI for PDF-to-image conversion
    MAX_DPI: Maximum DPI for PDF-to-image conversion
    
Language Mappings:
    TESSERACT_TO_EASYOCR_LANG: Map tesseract → EasyOCR language codes
    TESSERACT_TO_PADDLEOCR_LANG: Map tesseract → PaddleOCR language codes
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Union

# ============================================================================
# VERSION AND METADATA
# ============================================================================

__version__ = "3.0.0"

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

DEFAULT_OUTPUT_DIR: str = "ocr_out"

# ============================================================================
# SUPPORTED FILE FORMATS
# ============================================================================

SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
    ".webp",
}

# ============================================================================
# DPI LIMITS FOR PDF-TO-IMAGE CONVERSION
# ============================================================================

MIN_DPI: int = 72  # Minimum DPI for readable text
MAX_DPI: int = 600  # Maximum DPI to prevent excessive memory usage

# ============================================================================
# DOCTR-SPECIFIC CONSTANTS FOR TEXT SPACING
# ============================================================================

# docTR returns word-level bounding boxes but doesn't preserve horizontal
# spacing between words. These constants help reconstruct spacing for
# tabular data and formatted text.
WIDE_GAP_THRESHOLD_PIXELS: int = 30  # Gap width to consider as wide separation
PIXELS_PER_SPACE: int = 10  # Approximate pixels per space character

# ============================================================================
# LANGUAGE CODE MAPPINGS
# ============================================================================

# Mapping from tesseract language codes to EasyOCR language codes.
# EasyOCR uses ISO 639-1 two-letter codes for most languages, with some
# exceptions for Chinese variants.
# Not all tesseract languages are supported by EasyOCR.
TESSERACT_TO_EASYOCR_LANG: Dict[str, str] = {
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
    "chi_sim": "ch_sim",  # Simplified Chinese
    "chi_tra": "ch_tra",  # Traditional Chinese
    "jpn": "ja",
    "kor": "ko",
    "ara": "ar",
    "hin": "hi",
    "tha": "th",
    "vie": "vi",
    "tur": "tr",
    "heb": "he",
}

# Mapping from tesseract language codes to PaddleOCR language codes.
# PaddleOCR uses a mix of ISO codes and full language names.
# Default fallback is "en" for unsupported languages.
TESSERACT_TO_PADDLEOCR_LANG: Dict[str, str] = {
    "eng": "en",
    "deu": "german",
    "fra": "french",
    "spa": "spanish",
    "ita": "italian",
    "por": "portuguese",
    "rus": "ru",
    "ukr": "ukrainian",
    "chi_sim": "ch",  # Simplified Chinese
    "chi_tra": "chinese_cht",  # Traditional Chinese
    "jpn": "japan",
    "kor": "korean",
    "ara": "ar",
    "hin": "hi",
    "tha": "th",
    "vie": "vi",
    "tur": "tr",
}

# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

# OCR result can be either plain text (str) or structured data with boxes
# Structured data format varies by engine:
# - EasyOCR: List[List[List[int], str, float]] (bbox, text, confidence)
# - PaddleOCR: List[List[bbox, Tuple[str, float]]]
# - docTR: Dict with nested structure
OCRResult = Union[str, List[Any], Dict[str, Any]]
