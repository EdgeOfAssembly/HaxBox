#!/usr/bin/env python3
"""Font detection module for PDFs and images.

This module provides font type and font point size detection for both native
PDFs (with text layers) and scanned PDFs/images (OCR-based detection).

Features:
    - Native PDF font extraction using PyMuPDF (fitz) or pdfminer.six
    - Scanned image font metrics using pytesseract image_to_data
    - Character width/height estimation for redaction analysis
    - Unified API that auto-detects document type

Functions:
    detect_fonts: Unified API for font detection (auto-detects native vs scanned)
    detect_fonts_native: Extract fonts from native PDFs with text layers
    detect_fonts_scanned: Extract font metrics from scanned PDFs/images via OCR
    estimate_redacted_chars: Estimate character count under redaction boxes

Usage Example:
    ```python
    from pdfocr.font_detect import detect_fonts
    from pathlib import Path
    
    # Auto-detect and extract font info
    fonts = detect_fonts(Path("document.pdf"))
    
    for font_info in fonts:
        print(f"Text: {font_info['text']}")
        print(f"Font: {font_info.get('font_name', 'unknown')}")
        print(f"Size: {font_info.get('font_size_pt', font_info.get('est_font_size_pt'))}")
    ```

Dependencies:
    Required: pytesseract, PIL, pdf2image
    Optional: PyMuPDF (fitz), pdfminer.six
    
    Install optional dependencies:
        pip install PyMuPDF pdfminer.six
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

# Try to import pytesseract (required for scanned image detection)
try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False

# Try to import PyMuPDF (optional, for native PDF extraction)
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# Try to import pdfminer (optional, fallback for native PDF extraction)
try:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextBox, LTTextLine, LTChar, LTPage
    HAS_PDFMINER = True
except ImportError:
    HAS_PDFMINER = False

# Import pdf2image for PDF to image conversion
try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False


__all__ = [
    "detect_fonts",
    "detect_fonts_native",
    "detect_fonts_scanned",
    "estimate_redacted_chars",
]


# ============================================================================
# NATIVE PDF FONT DETECTION (using PyMuPDF or pdfminer.six)
# ============================================================================


def detect_fonts_native(pdf_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Extract font information from native PDFs with text layers.
    
    Uses PyMuPDF (fitz) as the primary extraction method, falling back to
    pdfminer.six if PyMuPDF is not available. Extracts per-span font name,
    font size, text content, bounding boxes, and bold/italic flags.
    
    Args:
        pdf_path: Path to the PDF file to analyze.
        
    Returns:
        List of dictionaries with keys:
            - text (str): Text content
            - font_name (str): Font family name (e.g., "Helvetica", "Times-Roman")
            - font_size_pt (float): Font size in points
            - bbox (Tuple[float, float, float, float]): Bounding box (x0, y0, x1, y1)
            - bold (bool): Whether text is bold (if available)
            - italic (bool): Whether text is italic (if available)
            - page (int): 1-indexed page number
            
    Raises:
        ImportError: If neither PyMuPDF nor pdfminer.six is installed.
        FileNotFoundError: If the PDF file does not exist.
        
    Examples:
        >>> fonts = detect_fonts_native("document.pdf")
        >>> for font_info in fonts:
        ...     print(f"{font_info['text']}: {font_info['font_name']} {font_info['font_size_pt']}pt")
        Hello World: Helvetica 12.0pt
        Subtitle: Times-Roman 10.0pt
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Try PyMuPDF first (faster and more accurate)
    if HAS_PYMUPDF:
        return _detect_fonts_pymupdf(pdf_path)
    
    # Fall back to pdfminer.six
    if HAS_PDFMINER:
        return _detect_fonts_pdfminer(pdf_path)
    
    # Neither library is available
    raise ImportError(
        "Font detection from native PDFs requires either PyMuPDF or pdfminer.six.\n"
        "Install with: pip install PyMuPDF  (recommended)\n"
        "         or: pip install pdfminer.six"
    )


def _detect_fonts_pymupdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract fonts using PyMuPDF (fitz).
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        List of font information dictionaries.
    """
    results: List[Dict[str, Any]] = []
    
    try:
        doc = fitz.open(str(pdf_path))
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get text with detailed span information
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        
                        # Extract font information
                        font_name = span.get("font", "unknown")
                        font_size_pt = span.get("size", 0.0)
                        
                        # Get bounding box
                        bbox = tuple(span.get("bbox", (0, 0, 0, 0)))
                        
                        # Detect bold/italic from font name (common naming conventions)
                        font_lower = font_name.lower()
                        bold = "bold" in font_lower or "-bold" in font_lower
                        italic = "italic" in font_lower or "oblique" in font_lower
                        
                        results.append({
                            "text": text,
                            "font_name": font_name,
                            "font_size_pt": float(font_size_pt),
                            "bbox": bbox,
                            "bold": bold,
                            "italic": italic,
                            "page": page_num + 1,
                        })
        
        doc.close()
        
    except Exception as e:
        print(f"Warning: PyMuPDF extraction failed: {e}", file=sys.stderr)
        return []
    
    return results


def _detect_fonts_pdfminer(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract fonts using pdfminer.six (fallback).
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        List of font information dictionaries.
    """
    results: List[Dict[str, Any]] = []
    
    try:
        page_num = 0
        for page_layout in extract_pages(str(pdf_path)):
            page_num += 1
            
            # Recursively extract characters from page
            chars = _extract_chars_from_layout(page_layout)
            
            for char_data in chars:
                text = char_data["text"].strip()
                if not text:
                    continue
                
                results.append({
                    "text": text,
                    "font_name": char_data["font_name"],
                    "font_size_pt": char_data["font_size_pt"],
                    "bbox": char_data["bbox"],
                    "bold": char_data["bold"],
                    "italic": char_data["italic"],
                    "page": page_num,
                })
                
    except Exception as e:
        print(f"Warning: pdfminer.six extraction failed: {e}", file=sys.stderr)
        return []
    
    return results


def _extract_chars_from_layout(layout_obj: Any) -> List[Dict[str, Any]]:
    """Recursively extract character information from pdfminer layout objects.
    
    Args:
        layout_obj: A pdfminer layout object (LTPage, LTTextBox, etc.)
        
    Returns:
        List of character information dictionaries.
    """
    chars = []
    
    if isinstance(layout_obj, LTChar):
        # Extract character-level information
        font_name = layout_obj.fontname or "unknown"
        font_size_pt = layout_obj.height  # Approximate font size from character height
        
        # Detect bold/italic from font name
        font_lower = font_name.lower()
        bold = "bold" in font_lower or "-bold" in font_lower
        italic = "italic" in font_lower or "oblique" in font_lower
        
        chars.append({
            "text": layout_obj.get_text(),
            "font_name": font_name,
            "font_size_pt": float(font_size_pt),
            "bbox": (layout_obj.x0, layout_obj.y0, layout_obj.x1, layout_obj.y1),
            "bold": bold,
            "italic": italic,
        })
    
    # Recursively process child objects
    if hasattr(layout_obj, "__iter__"):
        for child in layout_obj:
            chars.extend(_extract_chars_from_layout(child))
    
    return chars


# ============================================================================
# SCANNED PDF/IMAGE FONT DETECTION (using pytesseract)
# ============================================================================


def detect_fonts_scanned(
    input_path: Union[str, Path], dpi: int = 300
) -> List[Dict[str, Any]]:
    """Extract font metrics from scanned PDFs/images using OCR.
    
    Uses pytesseract's image_to_data() to get per-word bounding boxes, heights,
    widths, and confidence scores. Calculates approximate font size in points
    from pixel height and DPI.
    
    For PDFs, converts pages to images first using pdf2image.
    
    Args:
        input_path: Path to PDF or image file.
        dpi: DPI for PDF rendering (default: 300). Higher DPI = more accurate
            font size estimation but slower processing.
            
    Returns:
        List of dictionaries with keys:
            - text (str): Text content (word-level)
            - est_font_size_pt (float): Estimated font size in points
            - char_height_px (int): Character height in pixels
            - avg_char_width_px (float): Average character width in pixels
            - bbox (Tuple[int, int, int, int]): Bounding box (left, top, width, height)
            - confidence (float): OCR confidence score (0-100)
            - page (int): 1-indexed page number
            - char_metrics (Dict): Additional metrics (height_px, width_px, avg_char_width_px)
            - font_name (str): Always "unknown (scanned)" for scanned images
            
    Raises:
        ImportError: If pytesseract is not installed.
        FileNotFoundError: If the input file does not exist.
        
    Examples:
        >>> fonts = detect_fonts_scanned("scan.pdf", dpi=300)
        >>> for font_info in fonts:
        ...     print(f"{font_info['text']}: ~{font_info['est_font_size_pt']:.1f}pt")
        Hello: ~12.5pt
        World: ~12.3pt
    """
    if not HAS_PYTESSERACT:
        raise ImportError(
            "Font detection from scanned images requires pytesseract.\n"
            "Install with: pip install pytesseract"
        )
    
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Check if input is a PDF or image
    suffix = input_path.suffix.lower()
    
    if suffix == ".pdf":
        # Convert PDF to images
        if not HAS_PDF2IMAGE:
            raise ImportError(
                "PDF processing requires pdf2image.\n"
                "Install with: pip install pdf2image"
            )
        
        images = convert_from_path(str(input_path), dpi=dpi)
        results = []
        
        for page_num, image in enumerate(images, start=1):
            page_results = _extract_font_metrics_from_image(image, dpi, page_num)
            results.extend(page_results)
        
        return results
    else:
        # Process as single image
        image = Image.open(input_path)
        return _extract_font_metrics_from_image(image, dpi, page=1)


def _extract_font_metrics_from_image(
    image: Image.Image, dpi: int, page: int
) -> List[Dict[str, Any]]:
    """Extract font metrics from a single image using pytesseract.
    
    Args:
        image: PIL Image to analyze.
        dpi: DPI of the image for point size calculation.
        page: Page number (1-indexed).
        
    Returns:
        List of font metric dictionaries.
    """
    # Get detailed OCR data with bounding boxes
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    results: List[Dict[str, Any]] = []
    
    # Iterate through detected words
    n_boxes = len(data["text"])
    
    for i in range(n_boxes):
        text = data["text"][i].strip()
        if not text:
            continue
        
        # Extract bounding box and dimensions
        left = data["left"][i]
        top = data["top"][i]
        width = data["width"][i]
        height = data["height"][i]
        conf = data["conf"][i]
        
        # Skip low-confidence detections
        if conf < 0:
            continue
        
        # Calculate font size in points: pt_size = (pixel_height * 72) / dpi
        # The height of the bounding box approximates the font size
        est_font_size_pt = (height * 72.0) / dpi
        
        # Calculate average character width
        char_count = len(text)
        avg_char_width_px = width / char_count if char_count > 0 else 0
        
        results.append({
            "text": text,
            "est_font_size_pt": float(est_font_size_pt),
            "char_height_px": int(height),
            "avg_char_width_px": float(avg_char_width_px),
            "bbox": (left, top, width, height),
            "confidence": float(conf),
            "page": page,
            "char_metrics": {
                "height_px": int(height),
                "width_px": int(width),
                "avg_char_width_px": float(avg_char_width_px),
            },
            "font_name": "unknown (scanned)",
        })
    
    return results


# ============================================================================
# UNIFIED FONT DETECTION API
# ============================================================================


def detect_fonts(
    input_path: Union[str, Path], dpi: int = 300
) -> List[Dict[str, Any]]:
    """Detect fonts from PDFs or images (auto-detects native vs scanned).
    
    Unified API that automatically determines whether the input is a native
    PDF with text layers or a scanned PDF/image, then routes to the appropriate
    detection method.
    
    For native PDFs:
        - Uses PyMuPDF or pdfminer.six to extract exact font information
        - Returns font_name, font_size_pt, bbox, bold, italic
        
    For scanned PDFs/images:
        - Uses pytesseract OCR to estimate font metrics from pixels
        - Returns est_font_size_pt, char_height_px, avg_char_width_px
        
    Args:
        input_path: Path to PDF or image file.
        dpi: DPI for rendering (used for scanned PDFs and font size calculation).
            Default: 300. Range: 72-600.
            
    Returns:
        List of font information dictionaries. Format varies based on whether
        the document is native or scanned (see detect_fonts_native and
        detect_fonts_scanned for details).
        
    Examples:
        >>> # Works with both native and scanned PDFs
        >>> fonts = detect_fonts("document.pdf")
        >>> for font in fonts:
        ...     text = font["text"]
        ...     size = font.get("font_size_pt") or font.get("est_font_size_pt")
        ...     print(f"{text}: {size:.1f}pt")
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    suffix = input_path.suffix.lower()
    
    # Handle images directly (always scanned)
    if suffix in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}:
        return detect_fonts_scanned(input_path, dpi)
    
    # Handle PDFs (check if native or scanned)
    if suffix == ".pdf":
        # Try to detect if PDF has text layer
        has_text_layer = _pdf_has_text_layer(input_path)
        
        if has_text_layer:
            # Native PDF with text layer
            return detect_fonts_native(input_path)
        else:
            # Scanned PDF (no text layer)
            return detect_fonts_scanned(input_path, dpi)
    
    # Unsupported file type
    raise ValueError(f"Unsupported file type: {suffix}")


def _pdf_has_text_layer(pdf_path: Path) -> bool:
    """Check if a PDF has a text layer (native) or is scanned.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        True if PDF has extractable text, False if it's image-only.
    """
    # Try PyMuPDF first (fastest)
    if HAS_PYMUPDF:
        try:
            doc = fitz.open(str(pdf_path))
            has_text = False
            
            # Check first few pages for text
            for page_num in range(min(3, len(doc))):
                page = doc[page_num]
                text = page.get_text().strip()
                if text:
                    has_text = True
                    break
            
            doc.close()
            return has_text
            
        except Exception:
            pass
    
    # Try pdfminer.six as fallback
    if HAS_PDFMINER:
        try:
            # Check first page for text
            for page_layout in extract_pages(str(pdf_path), maxpages=1):
                text = ""
                for element in page_layout:
                    if hasattr(element, "get_text"):
                        text += element.get_text()
                
                if text.strip():
                    return True
                    
        except Exception:
            pass
    
    # If we can't check, assume scanned (safer default)
    return False


# ============================================================================
# REDACTION CHARACTER ESTIMATION
# ============================================================================


def estimate_redacted_chars(
    image: Union[Image.Image, np.ndarray],
    redaction_bbox: Tuple[int, int, int, int],
    dpi: int = 300,
    reference_text_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> Dict[str, Any]:
    """Estimate the number of characters hidden under a redaction box.
    
    Analyzes nearby visible text to measure average character width, then
    estimates how many characters fit under the redaction box. Useful for
    constraining candidate generation when attempting to recover redacted
    content.
    
    Args:
        image: PIL Image or numpy array of the document page.
        redaction_bbox: Bounding box of the redaction (left, top, width, height).
        dpi: DPI of the image (default: 300). Used for font size estimation.
        reference_text_bboxes: Optional list of bounding boxes for nearby
            visible text. If None, automatically detects text near redaction.
            
    Returns:
        Dictionary with keys:
            - redaction_width_px (int): Width of redaction box in pixels
            - avg_char_width_px (float): Average character width from nearby text
            - est_font_size_pt (float): Estimated font size in points
            - est_char_count (int): Estimated number of characters under redaction
            - min_char_count (int): Minimum plausible character count
            - max_char_count (int): Maximum plausible character count
            - confidence (str): Confidence level ("high", "medium", "low")
            
    Raises:
        ImportError: If pytesseract is not installed.
        
    Examples:
        >>> from PIL import Image
        >>> img = Image.open("redacted_email.png")
        >>> # Redaction box at (100, 200) with size 300x20
        >>> result = estimate_redacted_chars(img, (100, 200, 300, 20))
        >>> print(f"Estimated {result['est_char_count']} characters")
        >>> print(f"Range: {result['min_char_count']}-{result['max_char_count']}")
        Estimated 25 characters
        Range: 7-35
    """
    if not HAS_PYTESSERACT:
        raise ImportError(
            "Character estimation requires pytesseract.\n"
            "Install with: pip install pytesseract"
        )
    
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Extract redaction dimensions
    redaction_left, redaction_top, redaction_width, redaction_height = redaction_bbox
    
    # If reference text boxes not provided, detect nearby text automatically
    if reference_text_bboxes is None:
        reference_text_bboxes = _find_nearby_text(
            image, redaction_bbox, dpi, search_radius_px=200
        )
    
    # Measure average character width from reference text
    if reference_text_bboxes:
        avg_char_width_px, est_font_size_pt = _measure_avg_char_width(
            image, reference_text_bboxes, dpi
        )
        confidence = "high"
    else:
        # No reference text found - use fallback estimates
        # Assume typical 12pt font at given DPI
        est_font_size_pt = 12.0
        # Typical character width is ~60% of height for most fonts
        char_height_px = (est_font_size_pt * dpi) / 72.0
        avg_char_width_px = char_height_px * 0.6
        confidence = "low"
    
    # Estimate character count
    if avg_char_width_px > 0:
        est_char_count = int(redaction_width / avg_char_width_px)
    else:
        est_char_count = 0
    
    # Calculate plausible range
    # Assume +/- 30% variation in character width (narrow vs wide characters)
    min_char_width_px = avg_char_width_px * 1.3  # Wider chars = fewer total
    max_char_width_px = avg_char_width_px * 0.7  # Narrower chars = more total
    
    min_char_count = max(7, int(redaction_width / min_char_width_px))  # At least 7 (e.g., "x@y.com")
    max_char_count = int(redaction_width / max_char_width_px)
    
    return {
        "redaction_width_px": int(redaction_width),
        "avg_char_width_px": float(avg_char_width_px),
        "est_font_size_pt": float(est_font_size_pt),
        "est_char_count": int(est_char_count),
        "min_char_count": int(min_char_count),
        "max_char_count": int(max_char_count),
        "confidence": confidence,
    }


def _find_nearby_text(
    image: Image.Image,
    redaction_bbox: Tuple[int, int, int, int],
    dpi: int,
    search_radius_px: int = 200,
) -> List[Tuple[int, int, int, int]]:
    """Find text bounding boxes near a redaction box.
    
    Args:
        image: PIL Image to search.
        redaction_bbox: Redaction bounding box (left, top, width, height).
        dpi: DPI of the image.
        search_radius_px: Search radius in pixels around redaction.
        
    Returns:
        List of text bounding boxes (left, top, width, height).
    """
    redaction_left, redaction_top, redaction_width, redaction_height = redaction_bbox
    redaction_center_x = redaction_left + redaction_width // 2
    redaction_center_y = redaction_top + redaction_height // 2
    
    # Get all text with OCR
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    nearby_boxes = []
    n_boxes = len(data["text"])
    
    for i in range(n_boxes):
        text = data["text"][i].strip()
        if not text:
            continue
        
        # Skip low-confidence detections
        conf = data["conf"][i]
        if conf < 60:
            continue
        
        # Get bounding box
        left = data["left"][i]
        top = data["top"][i]
        width = data["width"][i]
        height = data["height"][i]
        
        # Calculate distance from redaction center
        box_center_x = left + width // 2
        box_center_y = top + height // 2
        
        distance = ((box_center_x - redaction_center_x) ** 2 + 
                   (box_center_y - redaction_center_y) ** 2) ** 0.5
        
        # Check if within search radius and not overlapping redaction
        if distance < search_radius_px:
            # Check for overlap
            if not _boxes_overlap(
                (redaction_left, redaction_top, redaction_width, redaction_height),
                (left, top, width, height)
            ):
                nearby_boxes.append((left, top, width, height))
    
    return nearby_boxes


def _boxes_overlap(
    box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]
) -> bool:
    """Check if two bounding boxes overlap.
    
    Args:
        box1: First box (left, top, width, height).
        box2: Second box (left, top, width, height).
        
    Returns:
        True if boxes overlap, False otherwise.
    """
    left1, top1, width1, height1 = box1
    left2, top2, width2, height2 = box2
    
    right1 = left1 + width1
    bottom1 = top1 + height1
    right2 = left2 + width2
    bottom2 = top2 + height2
    
    # Check for no overlap (then invert)
    if right1 <= left2 or right2 <= left1:
        return False
    if bottom1 <= top2 or bottom2 <= top1:
        return False
    
    return True


def _measure_avg_char_width(
    image: Image.Image,
    text_bboxes: List[Tuple[int, int, int, int]],
    dpi: int,
) -> Tuple[float, float]:
    """Measure average character width from text bounding boxes.
    
    Args:
        image: PIL Image.
        text_bboxes: List of text bounding boxes (left, top, width, height).
        dpi: DPI of the image.
        
    Returns:
        Tuple of (average character width in pixels, estimated font size in points).
    """
    total_char_width = 0.0
    total_chars = 0
    total_height = 0.0
    total_boxes = 0
    
    # Get text content for each box to count characters
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    for bbox in text_bboxes:
        left, top, width, height = bbox
        
        # Find matching OCR data
        n_boxes = len(data["text"])
        for i in range(n_boxes):
            if (data["left"][i] == left and data["top"][i] == top and 
                data["width"][i] == width and data["height"][i] == height):
                
                text = data["text"][i].strip()
                if text:
                    char_count = len(text)
                    total_chars += char_count
                    total_char_width += width
                    total_height += height
                    total_boxes += 1
                break
    
    if total_chars > 0 and total_boxes > 0:
        avg_char_width_px = total_char_width / total_chars
        avg_height_px = total_height / total_boxes
        est_font_size_pt = (avg_height_px * 72.0) / dpi
        return avg_char_width_px, est_font_size_pt
    
    # Fallback
    return 8.0, 12.0
