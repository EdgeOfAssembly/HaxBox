#!/usr/bin/env python3
"""Command-line interface for pdfocr.

This module contains the argument parser and main() function for the pdfocr
CLI tool. It handles user input, engine selection, and orchestrates the OCR
workflow.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import List, Optional

from pdfocr.types import __version__, DEFAULT_OUTPUT_DIR
from pdfocr.engines import get_engine, available_engines
from pdfocr.core import ocr_pdf, ocr_image_file
from pdfocr.utils import process_inputs, parse_page_spec

# Lazy import for pdf2image - optional dependency
try:
    from pdf2image import pdfinfo_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False


def main() -> None:
    """Main entry point for the pdfocr CLI."""
    parser = argparse.ArgumentParser(
        description="OCR tool for extracting text from PDFs and images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OCR a PDF with default engine (tesseract)
  pdfocr document.pdf
  
  # Use EasyOCR with GPU acceleration
  pdfocr document.pdf -e easyocr --gpu
  
  # OCR specific pages with German language
  pdfocr document.pdf -l deu -p 1-5,10
  
  # Output as JSON with bounding boxes
  pdfocr document.pdf --format json
  
  # Process multiple files
  pdfocr file1.pdf file2.jpg file3.png
  
  # Process entire directory
  pdfocr /path/to/documents/
""",
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input PDF/image files or directories to process",
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
        help="Language code in tesseract format (default: eng). Examples: eng, deu, fra, spa, chi_sim",
    )

    parser.add_argument(
        "-d",
        "--directory",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for extracted text files (default: {DEFAULT_OUTPUT_DIR})",
    )

    parser.add_argument(
        "-p",
        "--pages",
        help="Page specification for PDFs (e.g., '1-5', '1,3,5', '1-3,7-9'). Only for PDF files.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for rendering PDF pages (default: 300, range: 72-600)",
    )

    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Disable CLAHE image preprocessing for better OCR",
    )

    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save rendered page images alongside text output",
    )

    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format: plain text or JSON with bounding boxes (default: text)",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration (if supported by engine)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for PaddleOCR (default: 1, higher = faster but more GPU memory)",
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing output files",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"pdfocr {__version__}",
    )

    args = parser.parse_args()

    # Handle TrOCR variants
    model_variant: Optional[str] = None
    if args.engine == "trocr-handwritten":
        args.engine = "trocr"
        model_variant = "handwritten"
    elif args.engine == "trocr":
        model_variant = "printed"

    # Check if engine is available, fall back to tesseract if not
    if not check_engine_available(args.engine):
        available = available_engines()
        print(
            f"Warning: {args.engine} is not available. Available engines: {', '.join(available)}",
            file=sys.stderr,
        )
        if "tesseract" in available:
            print(f"Falling back to tesseract", file=sys.stderr)
            args.engine = "tesseract"
        else:
            print("Error: No OCR engines are available. Please install at least one engine.", file=sys.stderr)
            sys.exit(1)

    # Process inputs
    try:
        files = process_inputs(args.inputs)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not files:
        print("Error: No files to process", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each file
    enhance = not args.no_enhance
    output_format = "json" if args.format == "json" else "text"

    for file_path in files:
        try:
            if file_path.suffix.lower() == ".pdf":
                # Parse page specification for PDFs
                pages_to_process = None
                if args.pages:
                    if not HAS_PDF2IMAGE:
                        print(
                            "Warning: Cannot parse page specification without pdf2image. Processing all pages.",
                            file=sys.stderr,
                        )
                    else:
                        try:
                            # Need to get total pages to parse the spec
                            info = pdfinfo_from_path(str(file_path))
                            total_pages = info.get("Pages", 0)
                            if total_pages > 0:
                                pages_to_process = parse_page_spec(args.pages, total_pages)
                        except ValueError as e:
                            print(f"Error: Invalid page specification: {e}", file=sys.stderr)
                            continue

                ocr_pdf(
                    file_path,
                    output_dir,
                    engine=args.engine,
                    lang=args.lang,
                    pages=pages_to_process,
                    dpi=args.dpi,
                    enhance=enhance,
                    output_format=output_format,
                    save_images=args.save_images,
                    gpu=args.gpu,
                    batch_size=args.batch_size,
                    force=args.force,
                    quiet=args.quiet,
                )
            else:
                ocr_image_file(
                    file_path,
                    output_dir,
                    engine=args.engine,
                    lang=args.lang,
                    enhance=enhance,
                    output_format=output_format,
                    gpu=args.gpu,
                    batch_size=args.batch_size,
                    force=args.force,
                    quiet=args.quiet,
                )
        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)
            if not args.quiet:
                import traceback
                traceback.print_exc()


def check_engine_available(engine: str) -> bool:
    """Check if an OCR engine is available.

    Args:
        engine: Engine name to check.

    Returns:
        True if engine dependencies are installed, False otherwise.
    """
    available = available_engines()
    return engine in available


if __name__ == "__main__":
    main()
