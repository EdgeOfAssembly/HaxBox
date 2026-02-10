#!/usr/bin/env python3
"""Utility functions for the pdfocr package.

This module contains pure utility functions that are used throughout the
pdfocr package. These functions are self-contained with no side effects
and handle tasks such as:
- DPI validation and range clamping
- Page specification parsing (e.g., "1-5,10,20-25")
- Input file/directory processing
- PDF to image conversion

All functions use comprehensive type hints compatible with mypy --strict
and include detailed Google-style docstrings.

Functions:
    validate_dpi: Validate and clamp DPI values to acceptable range
    parse_page_spec: Parse page specification strings into page numbers
    process_inputs: Process input paths (files/directories) into file list
    pdf_to_images: Convert PDF pages to PIL Image objects
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from pdfocr.types import MAX_DPI, MIN_DPI, SUPPORTED_IMAGE_EXTENSIONS


def validate_dpi(dpi: int) -> int:
    """Validate DPI value and return a valid clamped value.

    This function ensures that the DPI value is within the acceptable range
    defined by MIN_DPI and MAX_DPI constants. If the value is outside this
    range, it will be clamped to the nearest boundary. Zero or negative
    values are rejected as invalid.

    The DPI (dots per inch) value controls the resolution when converting
    PDFs to images. Higher DPI values produce sharper images but consume
    more memory and processing time. Lower DPI values are faster but may
    result in poor OCR accuracy if text becomes unreadable.

    Args:
        dpi: The DPI value to validate. Must be a positive integer.

    Returns:
        A valid DPI value, clamped to the range [MIN_DPI, MAX_DPI].
        - If dpi < MIN_DPI: returns MIN_DPI
        - If dpi > MAX_DPI: returns MAX_DPI
        - Otherwise: returns dpi unchanged

    Raises:
        ValueError: If dpi is zero or negative.

    Examples:
        >>> validate_dpi(300)
        300
        >>> validate_dpi(50)  # Below MIN_DPI (72)
        72
        >>> validate_dpi(1000)  # Above MAX_DPI (600)
        600
        >>> validate_dpi(0)
        Traceback (most recent call last):
            ...
        ValueError: DPI must be positive, got 0
    """
    # Reject invalid (non-positive) DPI values
    if dpi <= 0:
        raise ValueError(f"DPI must be positive, got {dpi}")
    
    # Clamp to minimum DPI if below threshold
    if dpi < MIN_DPI:
        return MIN_DPI
    
    # Clamp to maximum DPI if above threshold
    if dpi > MAX_DPI:
        return MAX_DPI
    
    # Return valid DPI unchanged
    return dpi


def parse_page_spec(spec: str, total_pages: int) -> List[int]:
    """Parse page specification string into a list of page numbers.

    This function parses flexible page specification strings and converts
    them into a sorted list of unique 1-indexed page numbers. It supports
    individual pages, ranges, and combinations thereof.

    Supported formats:
        - Individual pages: "1,7,67" → [1, 7, 67]
        - Page ranges: "1-10" → [1, 2, 3, ..., 10]
        - Open-ended ranges: "56-" → [56, 57, ..., total_pages]
        - Starting ranges: "-10" → [1, 2, 3, ..., 10]
        - Combinations: "1-5,10,20-25" → [1, 2, 3, 4, 5, 10, 20, 21, ..., 25]

    Page numbers are 1-indexed (first page is 1, not 0). Duplicate page
    numbers in the specification are automatically deduplicated in the result.

    Args:
        spec: Page specification string. Can contain individual page numbers
            separated by commas, ranges separated by hyphens, or a mix of both.
            Whitespace around commas is ignored.
        total_pages: Total number of pages in the document. Used to validate
            page numbers and resolve open-ended ranges.

    Returns:
        Sorted list of unique 1-indexed page numbers specified by the input.
        Returns an empty list if the specification is empty or contains only
        whitespace.

    Raises:
        ValueError: If the specification is invalid, including:
            - Invalid number formats (non-numeric values)
            - Multiple hyphens in a single range part
            - Page numbers less than 1
            - Page numbers greater than total_pages
            - Invalid ranges where start > end

    Examples:
        >>> parse_page_spec("1,3,5", total_pages=10)
        [1, 3, 5]
        >>> parse_page_spec("1-5", total_pages=10)
        [1, 2, 3, 4, 5]
        >>> parse_page_spec("8-", total_pages=10)
        [8, 9, 10]
        >>> parse_page_spec("-3", total_pages=10)
        [1, 2, 3]
        >>> parse_page_spec("1-3,7,9-10", total_pages=10)
        [1, 2, 3, 7, 9, 10]
        >>> parse_page_spec("1,1,2,2", total_pages=10)  # Duplicates removed
        [1, 2]
        >>> parse_page_spec("1-15", total_pages=10)
        Traceback (most recent call last):
            ...
        ValueError: Page range out of bounds: 1-15 (PDF has 10 pages)
    """
    pages: List[int] = []
    # Split by commas to get individual parts (pages or ranges)
    parts = spec.split(",")

    for part in parts:
        # Remove leading/trailing whitespace from each part
        part = part.strip()
        
        # Skip empty parts (e.g., from trailing commas or multiple spaces)
        if not part:
            continue

        try:
            # Check if this part is a range (contains hyphen)
            if "-" in part:
                # Count hyphens to detect invalid formats like "1-5-10"
                hyphen_count = part.count("-")

                if hyphen_count == 1:
                    # Single hyphen: valid range format
                    idx = part.index("-")
                    start_str = part[:idx]  # Everything before hyphen
                    end_str = part[idx + 1 :]  # Everything after hyphen
                    
                    # Handle open-ended ranges:
                    # "-10" means start=1, end=10
                    # "5-" means start=5, end=total_pages
                    # "5-10" means start=5, end=10
                    start = int(start_str) if start_str else 1
                    end = int(end_str) if end_str else total_pages
                elif hyphen_count > 1:
                    # Multiple hyphens: invalid format
                    raise ValueError(
                        f"Invalid range format (multiple hyphens): '{part}'"
                    )
                else:
                    # This shouldn't happen since we checked "-" in part
                    raise ValueError(f"Invalid range format: '{part}'")
            else:
                # No hyphen: single page number
                start = int(part)
                end = start
        except ValueError as e:
            # Check if this is one of our custom ValueError messages
            if "Invalid range format" in str(e) or "multiple hyphens" in str(e):
                # Re-raise our custom error messages as-is
                raise
            # Otherwise, it's likely an int() conversion error
            raise ValueError(f"Invalid page number in '{part}': not a valid integer")

        # Validate page range bounds
        # - start must be >= 1 (pages are 1-indexed)
        # - end must be <= total_pages
        # - start must be <= end (valid range)
        if start < 1 or end > total_pages or start > end:
            raise ValueError(
                f"Page range out of bounds: {start}-{end} (PDF has {total_pages} pages)"
            )

        # Add all pages in range [start, end] (inclusive)
        pages.extend(range(start, end + 1))

    # Remove duplicates and sort the result
    # Use set() to remove duplicates, then sorted() to order them
    return sorted(set(pages))


def process_inputs(inputs: List[str]) -> List[Path]:
    """Process input arguments into a list of file paths to process.

    This function handles both individual files and directories as input.
    When a directory is provided, it recursively searches for all supported
    PDF and image files within that directory. The function performs
    case-insensitive file extension matching to handle files with various
    casing conventions (e.g., .PDF, .pdf, .Pdf).

    Supported file types:
        - PDF files (.pdf)
        - Image files (defined by SUPPORTED_IMAGE_EXTENSIONS constant)

    The function automatically:
        - Resolves all paths to absolute paths
        - Deduplicates files (same file not processed twice)
        - Preserves the order of unique files
        - Warns about unsupported files or missing paths to stderr

    Args:
        inputs: List of input path strings. Each can be either:
            - A path to a single file (PDF or supported image format)
            - A path to a directory (will process all PDFs/images inside)

    Returns:
        List of Path objects representing files to process. Duplicates are
        removed while preserving the order of first occurrence. Empty list
        if no valid files are found.

    Warns:
        Prints warnings to stderr for:
            - Files with unsupported extensions
            - Paths that don't exist (not a file or directory)

    Examples:
        >>> process_inputs(['document.pdf'])
        [Path('document.pdf')]
        >>> process_inputs(['scans/'])  # Directory with PDFs and images
        [Path('scans/doc1.pdf'), Path('scans/image1.png'), ...]
        >>> process_inputs(['doc1.pdf', 'doc2.pdf', 'images/'])
        [Path('doc1.pdf'), Path('doc2.pdf'), Path('images/scan1.jpg'), ...]

    Note:
        Case-insensitive matching is performed to avoid duplicates on
        case-insensitive filesystems (Windows, macOS with default settings).
        On case-sensitive filesystems (Linux), 'file.PDF' and 'file.pdf'
        would be treated as different files by the filesystem, but this
        function will process both if they exist.
    """
    files: List[Path] = []

    for inp in inputs:
        # Convert string path to Path object
        path = Path(inp)
        
        if path.is_dir():
            # Batch mode: process all PDFs and images in directory
            # Use case-insensitive glob pattern to match files regardless
            # of extension casing (e.g., .PDF, .pdf, .Pdf)
            # This avoids duplicate processing on case-insensitive filesystems
            files.extend(sorted(path.glob("*.[pP][dD][fF]")))

            # Collect image files with case-insensitive patterns
            for ext in SUPPORTED_IMAGE_EXTENSIONS:
                # Remove leading dot from extension (e.g., ".png" -> "png")
                ext_no_dot = ext[1:]
                
                # Create case-insensitive glob pattern
                # e.g., "png" -> "*.[pP][nN][gG]"
                # Each character gets [lowercase][uppercase] pattern
                pattern = "*." + "".join(
                    f"[{c.lower()}{c.upper()}]" for c in ext_no_dot
                )
                files.extend(sorted(path.glob(pattern)))
                
        elif path.is_file():
            # Single file mode: check if file type is supported
            suffix = path.suffix.lower()  # Normalize to lowercase for comparison
            
            if suffix == ".pdf" or suffix in SUPPORTED_IMAGE_EXTENSIONS:
                # Supported file type: add to processing list
                files.append(path)
            else:
                # Unsupported file type: warn user
                print(f"Warning: Unsupported file type: {path}", file=sys.stderr)
        else:
            # Path doesn't exist or is neither file nor directory
            print(f"Warning: Path not found: {path}", file=sys.stderr)

    # Remove duplicates while preserving order of first occurrence
    # This handles cases where the same file might be specified multiple
    # times or found through different directory traversals
    seen = set()  # Track resolved paths we've already added
    unique_files = []
    
    for f in files:
        # Resolve to absolute path to catch duplicates that might be
        # specified with different relative paths
        f_resolved = f.resolve()
        
        if f_resolved not in seen:
            # First time seeing this file: add it
            seen.add(f_resolved)
            unique_files.append(f)  # Append original path, not resolved

    return unique_files


def pdf_to_images(
    pdf_path: Path, dpi: int = 300, pages: Optional[List[int]] = None
) -> List[Tuple[int, Image.Image]]:
    """Convert PDF pages to PIL Image objects.

    This function uses the pdf2image library to render PDF pages as images.
    Each page is converted at the specified DPI resolution. You can optionally
    specify which pages to convert, or convert all pages by default.

    The function requires the pdf2image Python package and the poppler-utils
    system library to be installed. On Ubuntu/Debian, install poppler with:
        sudo apt-get install poppler-utils

    Args:
        pdf_path: Path to the PDF file to convert.
        dpi: Resolution in dots per inch for rendering. Higher DPI produces
            sharper images but uses more memory. Default is 300 DPI, which
            provides good quality for OCR. Range typically 72-600 DPI.
        pages: Optional list of 1-indexed page numbers to convert. If None,
            all pages in the PDF are converted. Page numbers should be within
            the valid range [1, total_pages]. Duplicates are automatically
            handled (each unique page converted once).

    Returns:
        List of tuples, where each tuple contains:
            - page_number (int): 1-indexed page number
            - image (PIL.Image.Image): The rendered page as a PIL Image

        Pages are returned in ascending order of page number. If a specific
        page fails to convert, it is silently skipped (no tuple for that page).
        Returns an empty list if no pages could be converted.

    Raises:
        ImportError: If pdf2image is not installed. Install with:
            pip install pdf2image

    Examples:
        >>> # Convert all pages at default 300 DPI
        >>> images = pdf_to_images(Path("document.pdf"))
        >>> print(f"Converted {len(images)} pages")
        Converted 5 pages

        >>> # Convert specific pages at higher resolution
        >>> images = pdf_to_images(Path("document.pdf"), dpi=600, pages=[1, 3, 5])
        >>> for page_num, img in images:
        ...     print(f"Page {page_num}: {img.size}")
        Page 1: (5100, 6600)
        Page 3: (5100, 6600)
        Page 5: (5100, 6600)

        >>> # Convert only first page
        >>> images = pdf_to_images(Path("document.pdf"), pages=[1])
        >>> page_num, img = images[0]
        >>> img.save(f"page_{page_num}.png")

    Note:
        - Page numbers in the input list are 1-indexed (first page is 1)
        - If pages parameter contains duplicates, each unique page is
          converted only once
        - Failed page conversions are silently skipped and don't raise errors
        - For large PDFs, consider converting specific pages rather than all
          pages to reduce memory usage
    """
    # Lazy import pdf2image - only needed when converting PDFs
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError("pdf2image not installed. Install: pip install pdf2image")

    # Check if we need to convert specific pages or all pages
    if pages is not None and len(pages) > 0:
        # Specific pages mode: convert each requested page individually
        # This is more memory-efficient for large PDFs when only a few
        # pages are needed
        result = []
        
        # Sort and deduplicate page numbers for efficient processing
        for page_num in sorted(set(pages)):
            try:
                # Convert single page by specifying first_page=last_page
                # pdf2image uses 1-indexed page numbers
                images = convert_from_path(
                    str(pdf_path), dpi=dpi, first_page=page_num, last_page=page_num
                )
                
                # convert_from_path returns a list; we expect one image
                if images:
                    result.append((page_num, images[0]))
            except Exception:
                # Silently skip pages that fail to convert
                # This could happen due to corrupted pages, unsupported
                # PDF features, or out-of-bounds page numbers
                pass
        
        return result
    else:
        # All pages mode: convert entire PDF in one operation
        # More efficient when processing most/all pages of a document
        images = convert_from_path(str(pdf_path), dpi=dpi)
        
        # Return list of (1-indexed page number, image) tuples
        # enumerate() starts at 0, so add 1 to get 1-indexed page numbers
        return [(i + 1, img) for i, img in enumerate(images)]
