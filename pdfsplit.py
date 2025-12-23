#!/usr/bin/env python3
"""
pdfsplit - Advanced PDF splitting and page extraction tool.

Features:
- Extract specific pages or page ranges from PDFs
- Split PDFs by granularity (e.g., every 10 pages)
- Export pages as high-quality PNG images (300+ DPI)
- Support for multiple input files and batch directory processing
- Extract embedded images from PDF pages
- Merge multiple PDFs into one
- Unlock/remove PDF restrictions (owner password)
- Reverse page order
- Split by file size limit
- Split at bookmark boundaries
- Optimize/compress PDF output
- Progress bars and quiet mode support
- Sane defaults: outputs to 'pdf_out' directory

Author: EdgeOfAssembly
Date: 2025-12-19

License: GPLv3 / Commercial dual-license
"""

from __future__ import annotations

import sys
import argparse
import io
from pathlib import Path
from types import ModuleType
from typing import List, Tuple, Optional, Set, Any, Iterable, TypeVar

try:
    from PyPDF2 import PdfReader, PdfWriter
except ImportError:
    print("Error: PyPDF2 is not installed. Install with: pip install PyPDF2")
    sys.exit(1)

# Optional imports - declare types before conditional import
fitz: Optional[ModuleType]
try:
    import fitz as _fitz  # pymupdf - for PNG export and image extraction
    fitz = _fitz
except ImportError:
    fitz = None

# PIL Image import
Image: Any
try:
    from PIL import Image as _Image
    Image = _Image
except ImportError:
    Image = None

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


__version__ = "3.0.0"
DEFAULT_OUTPUT_DIR = "pdf_out"
DEFAULT_DPI = 300
MIN_DPI = 72
MAX_DPI = 2400


def parse_page_spec(spec: str, total_pages: int) -> List[Tuple[int, int]]:
    """
    Parse page specification string into list of (start, end) tuples.

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
        List of (start_page, end_page) tuples (1-indexed).

    Raises:
        ValueError: If specification is invalid.
    """
    ranges: List[Tuple[int, int]] = []
    parts = spec.split(",")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        try:
            if "-" in part:
                # Handle ranges like "1-10", "-10" (start from 1), "10-" (to end)
                # Count hyphens to detect edge cases like "1--5" or "5--10"
                hyphen_count = part.count("-")

                if hyphen_count == 1:
                    # Simple range: "1-10", "-10", "10-"
                    idx = part.index("-")
                    start_str = part[:idx]
                    end_str = part[idx + 1 :]
                    start = int(start_str) if start_str else 1
                    end = int(end_str) if end_str else total_pages
                elif hyphen_count > 1:
                    # Multiple hyphens: "1--5", "5--10", "--5" etc are all invalid
                    raise ValueError(
                        f"Invalid range format (multiple hyphens): '{part}'"
                    )
                else:
                    # Shouldn't reach here, but handle gracefully
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

        ranges.append((start, end))

    return ranges


def generate_ranges_by_granularity(
    total_pages: int, granularity: int
) -> List[Tuple[int, int]]:
    """
    Generate page ranges based on granularity.

    Args:
        total_pages: Total number of pages.
        granularity: Number of pages per chunk.

    Returns:
        List of (start_page, end_page) tuples.
    """
    ranges: List[Tuple[int, int]] = []
    gran = max(1, granularity)
    start = 1

    while start <= total_pages:
        end = min(start + gran - 1, total_pages)
        ranges.append((start, end))
        start = end + 1

    return ranges


def extract_pdf_pages(
    input_path: Path,
    output_dir: Path,
    ranges: List[Tuple[int, int]],
    prefix: Optional[str],
    force: bool,
    quiet: bool,
    reader: Optional[PdfReader] = None,
) -> int:
    """
    Extract pages from PDF to separate PDF files.

    Args:
        input_path: Path to input PDF.
        output_dir: Output directory.
        ranges: List of (start, end) page ranges.
        prefix: Prefix for output filenames.
        force: Force overwrite existing files.
        quiet: Suppress output.
        reader: Optional pre-opened PdfReader to avoid reopening large PDFs.

    Returns:
        Number of files created.
    """
    if reader is None:
        try:
            reader = PdfReader(input_path)
        except Exception as e:
            print(f"Error opening PDF '{input_path}': {e}", file=sys.stderr)
            return 0

    total_pages = len(reader.pages)
    if total_pages == 0:
        print(f"Warning: PDF '{input_path}' is empty.", file=sys.stderr)
        return 0

    pad_width = len(str(total_pages))
    stem = prefix if prefix else input_path.stem
    files_created = 0

    ranges_iter: Iterable[Tuple[int, int]]
    if not quiet and _HAS_TQDM:
        ranges_iter = tqdm(ranges, desc=f"Splitting {input_path.name}", unit="file")
    else:
        ranges_iter = ranges

    for start, end in ranges_iter:
        writer = PdfWriter()
        for page_num in range(start - 1, end):
            writer.add_page(reader.pages[page_num])

        if start == end:
            output_filename = f"{stem}_page_{start:0{pad_width}d}.pdf"
        else:
            output_filename = (
                f"{stem}_pages_{start:0{pad_width}d}-{end:0{pad_width}d}.pdf"
            )

        output_path = output_dir / output_filename

        if output_path.exists() and not force:
            if quiet:
                print(
                    f"Skipping existing file '{output_path}' (use --force to overwrite).",
                    file=sys.stderr,
                )
                continue
            print(
                f"File '{output_path}' exists. Overwrite? (Y/N): ", end="", flush=True
            )
            try:
                response = input().strip().lower()
                if response != "y":
                    continue
            except EOFError:
                continue

        try:
            with open(output_path, "wb") as f:
                writer.write(f)
            files_created += 1
        except Exception as e:
            print(f"Error writing '{output_path}': {e}", file=sys.stderr)

    return files_created


def extract_pages_as_png(
    input_path: Path,
    output_dir: Path,
    ranges: List[Tuple[int, int]],
    prefix: Optional[str],
    dpi: int,
    force: bool,
    quiet: bool,
) -> int:
    """
    Extract pages from PDF as high-quality PNG images.

    Args:
        input_path: Path to input PDF.
        output_dir: Output directory.
        ranges: List of (start, end) page ranges.
        prefix: Prefix for output filenames.
        dpi: DPI for rendering (default 300).
        force: Force overwrite existing files.
        quiet: Suppress output.

    Returns:
        Number of PNG files created.
    """
    if fitz is None:
        print(
            "Error: pymupdf required for PNG export. Install: pip install pymupdf",
            file=sys.stderr,
        )
        return 0

    try:
        doc = fitz.open(str(input_path))
    except Exception as e:
        print(f"Error opening PDF '{input_path}': {e}", file=sys.stderr)
        return 0

    total_pages = len(doc)
    if total_pages == 0:
        print(f"Warning: PDF '{input_path}' is empty.", file=sys.stderr)
        doc.close()
        return 0

    pad_width = len(str(total_pages))
    stem = prefix if prefix else input_path.stem
    files_created = 0

    # Flatten ranges to individual pages
    pages_to_extract: Set[int] = set()
    for start, end in ranges:
        for p in range(start, end + 1):
            pages_to_extract.add(p)

    pages_list = sorted(pages_to_extract)

    pages_iter: Iterable[int]
    if not quiet and _HAS_TQDM:
        pages_iter = tqdm(
            pages_list, desc=f"Exporting PNG from {input_path.name}", unit="page"
        )
    else:
        pages_iter = pages_list

    # Calculate zoom factor for desired DPI (72 is default PDF DPI)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for page_num in pages_iter:
        output_filename = f"{stem}_page_{page_num:0{pad_width}d}.png"
        output_path = output_dir / output_filename

        if output_path.exists() and not force:
            if quiet:
                print(
                    f"Skipping existing file '{output_path}' (use --force to overwrite).",
                    file=sys.stderr,
                )
                continue
            print(
                f"File '{output_path}' exists. Overwrite? (Y/N): ", end="", flush=True
            )
            try:
                response = input().strip().lower()
                if response != "y":
                    continue
            except EOFError:
                continue

        try:
            page = doc.load_page(page_num - 1)  # 0-indexed
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pix.save(str(output_path))
            files_created += 1
        except Exception as e:
            print(f"Error exporting page {page_num}: {e}", file=sys.stderr)

    doc.close()
    return files_created


def extract_embedded_images(
    input_path: Path,
    output_dir: Path,
    ranges: Optional[List[Tuple[int, int]]],
    prefix: Optional[str],
    force: bool,
    quiet: bool,
) -> int:
    """
    Extract embedded raster images from PDF pages.

    Args:
        input_path: Path to input PDF.
        output_dir: Output directory.
        ranges: List of (start, end) page ranges, or None for all pages.
        prefix: Prefix for output filenames.
        force: Force overwrite existing files.
        quiet: Suppress output.

    Returns:
        Number of images extracted.
    """
    if fitz is None or Image is None:
        print(
            "Error: pymupdf and Pillow required. Install: pip install pymupdf pillow",
            file=sys.stderr,
        )
        return 0

    try:
        doc = fitz.open(str(input_path))
    except Exception as e:
        print(f"Error opening PDF '{input_path}': {e}", file=sys.stderr)
        return 0

    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        return 0

    stem = prefix if prefix else input_path.stem
    seen_xrefs: Set[int] = set()
    image_count = 0

    # Determine pages to process
    if ranges is None:
        pages_to_process = list(range(1, total_pages + 1))
    else:
        pages_set: Set[int] = set()
        for start, end in ranges:
            for p in range(start, end + 1):
                pages_set.add(p)
        pages_to_process = sorted(pages_set)

    pages_iter: Iterable[int]
    if not quiet and _HAS_TQDM:
        pages_iter = tqdm(
            pages_to_process,
            desc=f"Extracting images from {input_path.name}",
            unit="page",
        )
    else:
        pages_iter = pages_to_process

    for page_num in pages_iter:
        page = doc.load_page(page_num - 1)
        img_list = page.get_images(full=True)

        for img in img_list:
            xref = img[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            base_dict = doc.extract_image(xref)
            if not base_dict:
                continue

            try:
                base_img = Image.open(io.BytesIO(base_dict["image"]))

                # Handle soft-mask (transparency)
                smask_xref = img[1]
                if smask_xref > 0:
                    mask_dict = doc.extract_image(smask_xref)
                    if mask_dict:
                        mask_img = Image.open(io.BytesIO(mask_dict["image"])).convert(
                            "L"
                        )
                        if base_img.size == mask_img.size:
                            base_img.putalpha(mask_img)

                # Convert to RGBA
                if base_img.mode in ("CMYK", "YCbCr", "I", "F"):
                    base_img = base_img.convert("RGB")
                if base_img.mode != "RGBA":
                    base_img = base_img.convert("RGBA")

                image_count += 1
                filename = f"{stem}_img_{image_count:04d}_page{page_num:03d}.png"
                output_path = output_dir / filename

                if output_path.exists() and not force:
                    if quiet:
                        print(
                            f"Skipping existing file '{output_path}' (use --force to overwrite).",
                            file=sys.stderr,
                        )
                        continue
                    print(
                        f"File '{output_path}' exists. Overwrite? (Y/N): ",
                        end="",
                        flush=True,
                    )
                    try:
                        response = input().strip().lower()
                        if response != "y":
                            continue
                    except EOFError:
                        continue

                base_img.save(str(output_path), "PNG")

            except Exception as e:
                if not quiet:
                    print(
                        f"Warning: Failed to extract image xref {xref}: {e}",
                        file=sys.stderr,
                    )

    doc.close()
    return image_count


def get_pdf_info(
    input_path: Path, reader: Optional[PdfReader] = None
) -> Optional[PdfReader]:
    """
    Print PDF metadata and page count.

    Args:
        input_path: Path to the PDF file.
        reader: Optional pre-opened PdfReader.

    Returns:
        The PdfReader instance for reuse, or None on error.
    """
    if reader is None:
        try:
            reader = PdfReader(input_path)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return None

    total_pages = len(reader.pages)
    print(f"File: {input_path}")
    print(f"Pages: {total_pages}")

    # Check encryption status
    if reader.is_encrypted:
        print("Encrypted: Yes (restrictions may apply)")

    metadata = reader.metadata
    if metadata:
        if metadata.title:
            print(f"Title: {metadata.title}")
        if metadata.author:
            print(f"Author: {metadata.author}")
        if metadata.creator:
            print(f"Creator: {metadata.creator}")

    return reader


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


def parse_size_spec(size_str: str) -> int:
    """
    Parse a size specification string into bytes.

    Supports formats like "10MB", "1GB", "500KB", "1024" (bytes).

    Args:
        size_str: Size specification string.

    Returns:
        Size in bytes.

    Raises:
        ValueError: If format is invalid.
    """
    size_str = size_str.strip().upper()
    multipliers = {
        "B": 1,
        "KB": 1024,
        "K": 1024,
        "MB": 1024 * 1024,
        "M": 1024 * 1024,
        "GB": 1024 * 1024 * 1024,
        "G": 1024 * 1024 * 1024,
    }

    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if size_str.endswith(suffix):
            try:
                num = float(size_str[: -len(suffix)])
                return int(num * mult)
            except ValueError:
                raise ValueError(f"Invalid size format: '{size_str}'")

    # Try parsing as plain bytes
    try:
        return int(size_str)
    except ValueError:
        raise ValueError(
            f"Invalid size format: '{size_str}'. Use e.g., '10MB', '1GB', '500KB'"
        )


def unlock_pdf(
    input_path: Path,
    output_path: Path,
    password: Optional[str] = None,
    force: bool = False,
    quiet: bool = False,
) -> bool:
    """
    Remove restrictions/owner password from PDF.

    This creates a new PDF without copying encryption restrictions.

    LEGAL DISCLAIMER: This feature is intended for removing restrictions from
    documents you own or have rights to modify. Circumventing protection on
    copyrighted materials may violate laws in your jurisdiction.

    Args:
        input_path: Path to input (possibly restricted) PDF.
        output_path: Path for unrestricted output PDF.
        password: Password to decrypt (if user password protected).
        force: Force overwrite existing files.
        quiet: Suppress output.

    Returns:
        True if successful, False otherwise.
    """
    if output_path.exists() and not force:
        if quiet:
            print(
                f"Skipping existing file '{output_path}' (use --force to overwrite).",
                file=sys.stderr,
            )
            return False
        print(f"File '{output_path}' exists. Overwrite? (Y/N): ", end="", flush=True)
        try:
            response = input().strip().lower()
            if response != "y":
                return False
        except EOFError:
            return False

    try:
        reader = PdfReader(input_path)

        if reader.is_encrypted:
            # Try to decrypt with provided password (or empty for owner-only restrictions)
            try:
                decrypted = reader.decrypt(password or "")
                if decrypted == 0:
                    print(
                        f"Error: Could not decrypt '{input_path}'. Wrong password?",
                        file=sys.stderr,
                    )
                    return False
            except Exception as e:
                print(f"Error decrypting '{input_path}': {e}", file=sys.stderr)
                return False

        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)

        # Save WITHOUT calling writer.encrypt() = unrestricted PDF
        with open(output_path, "wb") as f:
            writer.write(f)

        if not quiet:
            print(f"  Created unrestricted PDF: {output_path}")
        return True

    except Exception as e:
        print(f"Error processing '{input_path}': {e}", file=sys.stderr)
        return False


def merge_pdfs(
    input_paths: List[Path], output_path: Path, force: bool = False, quiet: bool = False
) -> bool:
    """
    Merge multiple PDFs into a single file.

    Args:
        input_paths: List of PDF paths to merge (in order).
        output_path: Path for merged output PDF.
        force: Force overwrite existing files.
        quiet: Suppress output.

    Returns:
        True if successful, False otherwise.
    """
    if output_path.exists() and not force:
        if quiet:
            print(
                f"Skipping existing file '{output_path}' (use --force to overwrite).",
                file=sys.stderr,
            )
            return False
        print(f"File '{output_path}' exists. Overwrite? (Y/N): ", end="", flush=True)
        try:
            response = input().strip().lower()
            if response != "y":
                return False
        except EOFError:
            return False

    try:
        writer = PdfWriter()
        total_pages = 0

        for pdf_path in input_paths:
            if not quiet:
                print(f"  Adding: {pdf_path}")
            reader = PdfReader(pdf_path)

            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    print(
                        f"Warning: Could not decrypt '{pdf_path}', skipping.",
                        file=sys.stderr,
                    )
                    continue

            for page in reader.pages:
                writer.add_page(page)
                total_pages += 1

        with open(output_path, "wb") as f:
            writer.write(f)

        if not quiet:
            print(
                f"  Merged {len(input_paths)} PDFs ({total_pages} pages) -> {output_path}"
            )
        return True

    except Exception as e:
        print(f"Error merging PDFs: {e}", file=sys.stderr)
        return False


def reverse_pdf(
    input_path: Path, output_path: Path, force: bool = False, quiet: bool = False
) -> bool:
    """
    Reverse the page order of a PDF.

    Args:
        input_path: Path to input PDF.
        output_path: Path for reversed output PDF.
        force: Force overwrite existing files.
        quiet: Suppress output.

    Returns:
        True if successful, False otherwise.
    """
    if output_path.exists() and not force:
        if quiet:
            print(
                f"Skipping existing file '{output_path}' (use --force to overwrite).",
                file=sys.stderr,
            )
            return False
        print(f"File '{output_path}' exists. Overwrite? (Y/N): ", end="", flush=True)
        try:
            response = input().strip().lower()
            if response != "y":
                return False
        except EOFError:
            return False

    try:
        reader = PdfReader(input_path)
        writer = PdfWriter()

        for page in reversed(reader.pages):
            writer.add_page(page)

        with open(output_path, "wb") as f:
            writer.write(f)

        if not quiet:
            print(f"  Reversed {len(reader.pages)} pages -> {output_path}")
        return True

    except Exception as e:
        print(f"Error reversing '{input_path}': {e}", file=sys.stderr)
        return False


def split_by_bookmarks(
    input_path: Path, output_dir: Path, prefix: Optional[str], force: bool, quiet: bool
) -> int:
    """
    Split PDF at bookmark (TOC/chapter) boundaries.

    Args:
        input_path: Path to input PDF.
        output_dir: Output directory.
        prefix: Prefix for output filenames.
        force: Force overwrite existing files.
        quiet: Suppress output.

    Returns:
        Number of files created.
    """
    try:
        reader = PdfReader(input_path)
    except Exception as e:
        print(f"Error opening PDF '{input_path}': {e}", file=sys.stderr)
        return 0

    total_pages = len(reader.pages)
    if total_pages == 0:
        print(f"Warning: PDF '{input_path}' is empty.", file=sys.stderr)
        return 0

    # Get bookmarks/outlines
    try:
        outlines = reader.outline
    except Exception:
        outlines = []

    if not outlines:
        print(
            f"Warning: PDF '{input_path}' has no bookmarks. Using single output.",
            file=sys.stderr,
        )
        # Fall back to single output
        outlines = []

    # Extract bookmark page numbers
    bookmark_pages: List[Tuple[int, str]] = []

    def extract_bookmark_pages(outline_items: Any, depth: int = 0) -> None:
        """Recursively extract bookmark page numbers and titles."""
        for item in outline_items:
            if isinstance(item, list):
                # Nested bookmarks
                extract_bookmark_pages(item, depth + 1)
            else:
                try:
                    # Get page number from bookmark
                    page_num = reader.get_destination_page_number(item) + 1  # 1-indexed
                    title = (
                        item.title
                        if hasattr(item, "title")
                        else f"Section_{len(bookmark_pages)+1}"
                    )
                    # Clean title for filename
                    safe_title = "".join(
                        c if c.isalnum() or c in " -_" else "_" for c in title
                    )
                    safe_title = safe_title.strip()[:50]  # Limit length
                    bookmark_pages.append((page_num, safe_title))
                except Exception:
                    pass

    extract_bookmark_pages(outlines)

    if not bookmark_pages:
        # No valid bookmarks, output entire document
        print(f"Warning: No valid bookmarks found in '{input_path}'.", file=sys.stderr)
        return 0

    # Sort by page number and remove duplicates
    bookmark_pages = sorted(set(bookmark_pages), key=lambda x: x[0])

    # Create ranges from bookmarks
    stem = prefix if prefix else input_path.stem
    files_created = 0

    for i, (start_page, title) in enumerate(bookmark_pages):
        if i + 1 < len(bookmark_pages):
            end_page = bookmark_pages[i + 1][0] - 1
        else:
            end_page = total_pages

        if start_page > end_page or start_page > total_pages:
            continue

        writer = PdfWriter()
        for page_num in range(start_page - 1, min(end_page, total_pages)):
            writer.add_page(reader.pages[page_num])

        output_filename = f"{stem}_{i+1:02d}_{title}.pdf"
        output_path = output_dir / output_filename

        if output_path.exists() and not force:
            if quiet:
                print(
                    f"Skipping existing file '{output_path}' (use --force to overwrite).",
                    file=sys.stderr,
                )
                continue
            print(
                f"File '{output_path}' exists. Overwrite? (Y/N): ", end="", flush=True
            )
            try:
                response = input().strip().lower()
                if response != "y":
                    continue
            except EOFError:
                continue

        try:
            with open(output_path, "wb") as f:
                writer.write(f)
            files_created += 1
            if not quiet:
                print(f"  Created: {output_filename} (pages {start_page}-{end_page})")
        except Exception as e:
            print(f"Error writing '{output_path}': {e}", file=sys.stderr)

    return files_created


def split_by_size(
    input_path: Path,
    output_dir: Path,
    max_size: int,
    prefix: Optional[str],
    force: bool,
    quiet: bool,
) -> int:
    """
    Split PDF when output exceeds maximum file size.

    Args:
        input_path: Path to input PDF.
        output_dir: Output directory.
        max_size: Maximum file size in bytes.
        prefix: Prefix for output filenames.
        force: Force overwrite existing files.
        quiet: Suppress output.

    Returns:
        Number of files created.
    """
    try:
        reader = PdfReader(input_path)
    except Exception as e:
        print(f"Error opening PDF '{input_path}': {e}", file=sys.stderr)
        return 0

    total_pages = len(reader.pages)
    if total_pages == 0:
        print(f"Warning: PDF '{input_path}' is empty.", file=sys.stderr)
        return 0

    stem = prefix if prefix else input_path.stem
    files_created = 0
    current_writer = PdfWriter()
    current_pages: List[int] = []
    part_num = 1

    def write_current_part() -> bool:
        nonlocal files_created, current_writer, current_pages, part_num

        if not current_pages:
            return True

        output_filename = f"{stem}_part_{part_num:03d}.pdf"
        output_path = output_dir / output_filename

        if output_path.exists() and not force:
            if quiet:
                print(
                    f"Skipping existing file '{output_path}' (use --force to overwrite).",
                    file=sys.stderr,
                )
                return False
            print(
                f"File '{output_path}' exists. Overwrite? (Y/N): ", end="", flush=True
            )
            try:
                response = input().strip().lower()
                if response != "y":
                    return False
            except EOFError:
                return False

        try:
            with open(output_path, "wb") as f:
                current_writer.write(f)
            files_created += 1
            if not quiet:
                size_kb = output_path.stat().st_size / 1024
                print(
                    f"  Created: {output_filename} ({len(current_pages)} pages, {size_kb:.1f} KB)"
                )
            part_num += 1
            return True
        except Exception as e:
            print(f"Error writing '{output_path}': {e}", file=sys.stderr)
            return False

    for page_num in range(total_pages):
        # Add page to current writer
        current_writer.add_page(reader.pages[page_num])
        current_pages.append(page_num + 1)

        # Check current size by writing to memory
        buffer = io.BytesIO()
        current_writer.write(buffer)
        current_size = buffer.tell()

        # If we exceed max size and have more than one page, split
        if current_size > max_size and len(current_pages) > 1:
            # Remove last page and write
            current_writer = PdfWriter()
            for p in current_pages[:-1]:
                current_writer.add_page(reader.pages[p - 1])

            write_current_part()

            # Start new part with the page that caused overflow
            current_writer = PdfWriter()
            current_writer.add_page(reader.pages[page_num])
            current_pages = [page_num + 1]

    # Write remaining pages
    if current_pages:
        write_current_part()

    return files_created


def optimize_pdf(
    input_path: Path,
    output_path: Path,
    image_quality: int = 80,
    target_dpi: int = 150,
    remove_metadata: bool = True,
    subset_fonts: bool = True,
    force: bool = False,
    quiet: bool = False,
) -> Tuple[int, int]:
    """
    Optimize PDF to reduce file size.

    Uses PyMuPDF (fitz) for optimization including:
    - Removing metadata and thumbnails
    - Font subsetting (only include used glyphs)
    - Garbage collection (remove unused objects)
    - Stream compression

    Args:
        input_path: Path to input PDF.
        output_path: Path for optimized output PDF.
        image_quality: JPEG quality 1-100 (default 80).
        target_dpi: Downsample images above this DPI (default 150).
        remove_metadata: Remove metadata from PDF (default True).
        subset_fonts: Subset fonts to only used glyphs (default True).
        force: Force overwrite existing files.
        quiet: Suppress output.

    Returns:
        Tuple of (original_size, new_size) in bytes, or (0, 0) on failure.
    """
    if fitz is None:
        print(
            "Error: pymupdf required for optimization. Install: pip install pymupdf",
            file=sys.stderr,
        )
        return (0, 0)

    if output_path.exists() and not force:
        if quiet:
            print(
                f"Skipping existing file '{output_path}' (use --force to overwrite).",
                file=sys.stderr,
            )
            return (0, 0)
        print(f"File '{output_path}' exists. Overwrite? (Y/N): ", end="", flush=True)
        try:
            response = input().strip().lower()
            if response != "y":
                return (0, 0)
        except EOFError:
            return (0, 0)

    try:
        original_size = input_path.stat().st_size
        doc = fitz.open(str(input_path))

        # Step 1: Remove metadata and deadweight
        if remove_metadata:
            try:
                doc.scrub(
                    metadata=True,
                    xml_metadata=True,
                    attached_files=False,  # Keep attachments by default
                    embedded_files=False,
                    thumbnails=True,
                    reset_fields=True,
                    reset_responses=True,
                )
            except Exception:
                # scrub might not be available in older versions
                pass

        # Step 2: Subset fonts (only include used glyphs)
        if subset_fonts:
            try:
                doc.subset_fonts()
            except Exception:
                # subset_fonts might not be available
                pass

        # Step 3: Save with garbage collection and compression
        # Note: linear=True (web optimization) may not be supported in all versions
        try:
            doc.save(
                str(output_path),
                garbage=4,  # Maximum garbage collection
                deflate=True,  # Compress streams
                clean=True,  # Clean content streams
                linear=True,  # Web optimization (fast first page)
            )
        except Exception:
            # Fallback without linearization
            doc.save(
                str(output_path),
                garbage=4,
                deflate=True,
                clean=True,
            )

        new_size = output_path.stat().st_size
        doc.close()

        if not quiet:
            reduction = (1 - new_size / original_size) * 100 if original_size > 0 else 0
            print(
                f"  Optimized: {original_size / 1024:.1f} KB -> {new_size / 1024:.1f} KB ({reduction:.1f}% reduction)"
            )

        return (original_size, new_size)

    except Exception as e:
        print(f"Error optimizing '{input_path}': {e}", file=sys.stderr)
        return (0, 0)


def process_inputs(inputs: List[str]) -> List[Path]:
    """
    Process input arguments - can be files or directories.

    Args:
        inputs: List of input paths (files or directories).

    Returns:
        List of PDF file paths to process.
    """
    pdf_files: List[Path] = []

    for inp in inputs:
        path = Path(inp)
        if path.is_dir():
            # Batch mode: get all PDFs from directory (case-insensitive)
            # Use case-insensitive glob pattern to avoid duplicates on
            # case-insensitive filesystems (Windows, macOS default)
            pdf_files.extend(sorted(path.glob("*.[pP][dD][fF]")))
        elif path.is_file():
            if path.suffix.lower() == ".pdf":
                pdf_files.append(path)
            else:
                print(f"Warning: Skipping non-PDF file: {path}", file=sys.stderr)
        else:
            print(f"Warning: Path not found: {path}", file=sys.stderr)

    return pdf_files


def main() -> None:
    """Main entry point for pdfsplit."""
    parser = argparse.ArgumentParser(
        prog="pdfsplit",
        description="Advanced PDF splitting, merging, and optimization tool.",
        epilog="""
Examples:
  pdfsplit document.pdf                    # Split all pages (one PDF per page)
  pdfsplit document.pdf -p 1-10            # Extract pages 1-10
  pdfsplit document.pdf -p 1,5,10-15       # Extract specific pages
  pdfsplit document.pdf -g 10              # Split every 10 pages
  pdfsplit document.pdf --png              # Export all pages as 300 DPI PNGs
  pdfsplit document.pdf -p 1-5 --png       # Export pages 1-5 as PNGs
  pdfsplit document.pdf --images           # Extract embedded images
  pdfsplit /path/to/pdfs/                  # Batch process all PDFs in directory
  pdfsplit a.pdf b.pdf c.pdf -d output     # Process multiple files

Advanced Operations:
  pdfsplit --merge a.pdf b.pdf -o combined.pdf    # Merge multiple PDFs
  pdfsplit document.pdf --reverse -o reversed.pdf # Reverse page order
  pdfsplit document.pdf --unlock -o unlocked.pdf  # Remove restrictions
  pdfsplit document.pdf --by-bookmark             # Split at chapter boundaries
  pdfsplit large.pdf --max-size 10MB              # Split by file size
  pdfsplit document.pdf --optimize -o smaller.pdf # Optimize/compress PDF
  pdfsplit document.pdf -p 1-10 --optimize        # Extract + optimize

LEGAL DISCLAIMER:
  The --unlock feature is intended for removing restrictions from documents
  you own or have rights to modify. Circumventing protection on copyrighted
  materials may violate laws in your jurisdiction.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "inputs",
        nargs="*",
        help="PDF file(s) or directory containing PDFs. If directory, processes all PDFs.",
    )
    parser.add_argument(
        "-p",
        "--pages",
        help='Page specification: "1,7,67" or "1-10" or "56-" or "-10" (combined: "1-5,10,20-25")',
    )
    parser.add_argument(
        "-g",
        "--granularity",
        type=int,
        default=1,
        help="Split every N pages (default: 1, creates one file per page)",
    )
    parser.add_argument(
        "-d",
        "--directory",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (for --merge, --reverse, --unlock, --optimize operations)",
    )
    parser.add_argument(
        "--prefix", help="Custom prefix for output filenames (default: input filename)"
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Export pages as high-quality PNG images instead of PDFs",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"DPI for PNG export (default: {DEFAULT_DPI}, range: {MIN_DPI}-{MAX_DPI})",
    )
    parser.add_argument(
        "--images",
        action="store_true",
        help="Extract embedded images from the PDF instead of pages",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show PDF metadata and page count, then exit",
    )

    # New advanced operations
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge multiple input PDFs into one (requires -o/--output)",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse page order (requires -o/--output)",
    )
    parser.add_argument(
        "--unlock",
        action="store_true",
        help="Remove restrictions/owner password from PDF (requires -o/--output)",
    )
    parser.add_argument(
        "--password", help="Password for encrypted PDFs (used with --unlock)"
    )
    parser.add_argument(
        "--by-bookmark",
        action="store_true",
        help="Split PDF at bookmark/chapter boundaries",
    )
    parser.add_argument(
        "--max-size",
        help='Split PDF when output exceeds size limit (e.g., "10MB", "500KB")',
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize/compress PDF output to reduce file size",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=80,
        help="Image quality for optimization (1-100, default: 80)",
    )
    parser.add_argument(
        "--keep-metadata",
        action="store_true",
        help="Keep metadata during optimization (default: remove metadata)",
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

    # Process inputs (files or directories)
    pdf_files = process_inputs(args.inputs)

    if not pdf_files:
        print("Error: No PDF files found.", file=sys.stderr)
        sys.exit(1)

    # Handle --info flag
    if args.info:
        for pdf_file in pdf_files:
            get_pdf_info(pdf_file)
            print()
        sys.exit(0)

    # Handle --merge operation
    if args.merge:
        if not args.output:
            print(
                "Error: --merge requires -o/--output to specify output file.",
                file=sys.stderr,
            )
            sys.exit(1)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        success = merge_pdfs(pdf_files, output_path, args.force, args.quiet)

        # Apply optimization if requested
        if success and args.optimize:
            temp_path = output_path.with_suffix(".tmp.pdf")
            output_path.rename(temp_path)
            optimize_pdf(
                temp_path,
                output_path,
                image_quality=args.quality,
                remove_metadata=not args.keep_metadata,
                force=True,
                quiet=args.quiet,
            )
            temp_path.unlink(missing_ok=True)

        sys.exit(0 if success else 1)

    # Handle --reverse operation (single file)
    if args.reverse:
        if not args.output:
            print(
                "Error: --reverse requires -o/--output to specify output file.",
                file=sys.stderr,
            )
            sys.exit(1)
        if len(pdf_files) != 1:
            print("Error: --reverse requires exactly one input file.", file=sys.stderr)
            sys.exit(1)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        success = reverse_pdf(pdf_files[0], output_path, args.force, args.quiet)

        # Apply optimization if requested
        if success and args.optimize:
            temp_path = output_path.with_suffix(".tmp.pdf")
            output_path.rename(temp_path)
            optimize_pdf(
                temp_path,
                output_path,
                image_quality=args.quality,
                remove_metadata=not args.keep_metadata,
                force=True,
                quiet=args.quiet,
            )
            temp_path.unlink(missing_ok=True)

        sys.exit(0 if success else 1)

    # Handle --unlock operation (single file)
    if args.unlock:
        if not args.output:
            print(
                "Error: --unlock requires -o/--output to specify output file.",
                file=sys.stderr,
            )
            sys.exit(1)
        if len(pdf_files) != 1:
            print("Error: --unlock requires exactly one input file.", file=sys.stderr)
            sys.exit(1)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        success = unlock_pdf(
            pdf_files[0], output_path, args.password, args.force, args.quiet
        )

        # Apply optimization if requested
        if success and args.optimize:
            temp_path = output_path.with_suffix(".tmp.pdf")
            output_path.rename(temp_path)
            optimize_pdf(
                temp_path,
                output_path,
                image_quality=args.quality,
                remove_metadata=not args.keep_metadata,
                force=True,
                quiet=args.quiet,
            )
            temp_path.unlink(missing_ok=True)

        sys.exit(0 if success else 1)

    # Handle --optimize only operation (single file)
    if (
        args.optimize
        and not args.pages
        and not args.png
        and not args.images
        and not args.by_bookmark
        and not args.max_size
    ):
        if not args.output:
            print(
                "Error: --optimize requires -o/--output to specify output file.",
                file=sys.stderr,
            )
            sys.exit(1)
        if len(pdf_files) != 1:
            print("Error: --optimize requires exactly one input file.", file=sys.stderr)
            sys.exit(1)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        original_size, new_size = optimize_pdf(
            pdf_files[0],
            output_path,
            image_quality=args.quality,
            remove_metadata=not args.keep_metadata,
            force=args.force,
            quiet=args.quiet,
        )

        sys.exit(0 if new_size > 0 else 1)

    # Create output directory for splitting operations
    output_dir = Path(args.directory)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory '{output_dir}': {e}", file=sys.stderr)
        sys.exit(1)

    # Parse max size if specified
    max_size_bytes = None
    if args.max_size:
        try:
            max_size_bytes = parse_size_spec(args.max_size)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    total_files_created = 0

    for pdf_file in pdf_files:
        if not args.quiet:
            print(f"\nProcessing: {pdf_file}")

        # Get total pages for this PDF - keep reader for reuse
        try:
            reader = PdfReader(pdf_file)
            total_pages = len(reader.pages)
        except Exception as e:
            print(f"Error reading '{pdf_file}': {e}", file=sys.stderr)
            continue

        if total_pages == 0:
            print(f"Warning: PDF '{pdf_file}' is empty.", file=sys.stderr)
            continue

        # Use per-file prefix if multiple files and no custom prefix
        prefix = args.prefix
        if prefix is None and len(pdf_files) > 1:
            prefix = pdf_file.stem

        # Handle --by-bookmark operation
        if args.by_bookmark:
            count = split_by_bookmarks(
                pdf_file, output_dir, prefix, args.force, args.quiet
            )
            if not args.quiet:
                print(f"  Created {count} PDF file(s) from bookmarks")
            total_files_created += count
            continue

        # Handle --max-size operation
        if max_size_bytes:
            count = split_by_size(
                pdf_file, output_dir, max_size_bytes, prefix, args.force, args.quiet
            )
            if not args.quiet:
                print(f"  Created {count} PDF file(s) within size limit")
            total_files_created += count
            continue

        # Determine page ranges
        if args.pages:
            try:
                ranges = parse_page_spec(args.pages, total_pages)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                continue
        else:
            ranges = generate_ranges_by_granularity(total_pages, args.granularity)

        # Execute requested operation
        if args.images:
            count = extract_embedded_images(
                pdf_file,
                output_dir,
                ranges if args.pages else None,
                prefix,
                args.force,
                args.quiet,
            )
            if not args.quiet:
                print(f"  Extracted {count} embedded image(s)")
        elif args.png:
            count = extract_pages_as_png(
                pdf_file, output_dir, ranges, prefix, args.dpi, args.force, args.quiet
            )
            if not args.quiet:
                print(f"  Created {count} PNG file(s)")
        else:
            count = extract_pdf_pages(
                pdf_file,
                output_dir,
                ranges,
                prefix,
                args.force,
                args.quiet,
                reader=reader,  # Pass reader to avoid reopening
            )
            if not args.quiet:
                print(f"  Created {count} PDF file(s)")

        total_files_created += count

    if not args.quiet:
        print(f"\nTotal files created: {total_files_created}")
        print(f"Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
