#!/usr/bin/env python3
"""Test and compare OCR engines against ground truth data.

This script tests all available OCR engines using the test data provided in the
repository and ranks them by accuracy. It uses:
- Ground truth: PPC_example_data.txt
- Test PDF: PPC_example_data_pages_001-010.pdf

The script computes accuracy metrics for each engine and provides a ranked list
of results to help users choose the best OCR engine for their needs.

Note: TrOCR is not tested because it is designed for single-line OCR only, not
full-page document OCR.

Usage:
    python test_ocr_engines.py [--gpu] [--verbose]
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import difflib

# Set PaddleOCR environment variables before imports
os.environ['FLAGS_use_mkldnn'] = 'False'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_use_onednn'] = '0'

from pdfocr.engines import available_engines, get_engine
from pdfocr.core import ocr_pdf


def load_ground_truth(ground_truth_path: Path) -> str:
    """Load and normalize ground truth text.
    
    Args:
        ground_truth_path: Path to ground truth text file.
        
    Returns:
        Normalized ground truth text.
    """
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Normalize whitespace for comparison
    return text.strip()


def normalize_text(text: str) -> str:
    """Normalize text for comparison.
    
    Args:
        text: Raw OCR text output.
        
    Returns:
        Normalized text.
    """
    # Remove extra whitespace and normalize line endings
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(line for line in lines if line)


def compute_accuracy(ground_truth: str, ocr_output: str) -> Dict[str, float]:
    """Compute accuracy metrics between ground truth and OCR output.
    
    Args:
        ground_truth: Ground truth text.
        ocr_output: OCR engine output text.
        
    Returns:
        Dictionary with accuracy metrics:
        - char_accuracy: Character-level accuracy (0-100)
        - word_accuracy: Word-level accuracy (0-100)
        - sequence_similarity: Sequence similarity ratio (0-100)
    """
    # Normalize both texts
    gt_norm = normalize_text(ground_truth)
    ocr_norm = normalize_text(ocr_output)
    
    # Character-level accuracy using difflib
    char_matcher = difflib.SequenceMatcher(None, gt_norm, ocr_norm)
    char_accuracy = char_matcher.ratio() * 100
    
    # Word-level accuracy
    gt_words = gt_norm.split()
    ocr_words = ocr_norm.split()
    
    word_matcher = difflib.SequenceMatcher(None, gt_words, ocr_words)
    word_accuracy = word_matcher.ratio() * 100
    
    # Overall sequence similarity
    sequence_similarity = char_matcher.quick_ratio() * 100
    
    return {
        'char_accuracy': char_accuracy,
        'word_accuracy': word_accuracy,
        'sequence_similarity': sequence_similarity,
    }


def test_engine(
    engine_name: str,
    pdf_path: Path,
    gpu: bool = False,
    verbose: bool = False
) -> Tuple[str, Dict[str, float], str]:
    """Test a single OCR engine.
    
    Args:
        engine_name: Name of the OCR engine to test.
        pdf_path: Path to test PDF file.
        gpu: Whether to use GPU acceleration.
        verbose: Whether to print verbose output.
        
    Returns:
        Tuple of (engine_name, accuracy_metrics, error_message).
        If successful, error_message is empty string.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing engine: {engine_name}")
        print(f"{'='*60}")
    
    try:
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_output = Path(temp_dir)
            
            if verbose:
                print(f"Running OCR with {engine_name}...")
            
            # Run OCR
            result = ocr_pdf(
                pdf_path=pdf_path,
                output_dir=temp_output,
                engine=engine_name,
                lang="eng",
                output_format="text",
                enhance=True,
                save_images=False,
                force=True,
                quiet=not verbose,
                gpu=gpu,
            )
            
            # Read the output file
            output_files = list(temp_output.glob("*.txt"))
            if not output_files:
                return engine_name, {}, "No output file generated"
            
            with open(output_files[0], 'r', encoding='utf-8') as f:
                ocr_text = f.read()
            
            if verbose:
                print(f"OCR completed. Output length: {len(ocr_text)} characters")
            
            return engine_name, {}, ocr_text
            
    except Exception as e:
        error_msg = f"Error testing {engine_name}: {str(e)}"
        if verbose:
            print(error_msg)
        return engine_name, {}, error_msg


def print_results_table(results: List[Tuple[str, Dict[str, float], str]]) -> None:
    """Print results in a formatted table.
    
    Args:
        results: List of (engine_name, accuracy_metrics, status) tuples.
    """
    print("\n" + "="*80)
    print("OCR ENGINE COMPARISON RESULTS")
    print("="*80)
    
    # Separate successful and failed tests
    successful = [(name, metrics) for name, metrics, status in results if isinstance(metrics, dict) and metrics]
    failed = [(name, status) for name, metrics, status in results if not (isinstance(metrics, dict) and metrics)]
    
    if successful:
        print("\n" + "-"*80)
        print(f"{'Rank':<6} {'Engine':<15} {'Char Acc %':<12} {'Word Acc %':<12} {'Similarity %':<12}")
        print("-"*80)
        
        for rank, (engine, metrics) in enumerate(successful, 1):
            print(f"{rank:<6} {engine:<15} "
                  f"{metrics['char_accuracy']:<12.2f} "
                  f"{metrics['word_accuracy']:<12.2f} "
                  f"{metrics['sequence_similarity']:<12.2f}")
    
    if failed:
        print("\n" + "-"*80)
        print("ENGINES NOT TESTED:")
        print("-"*80)
        for engine, status in failed:
            print(f"  {engine}: {status}")
    
    print("\n" + "="*80)
    print("Notes:")
    print("  - Char Acc: Character-level accuracy (higher is better)")
    print("  - Word Acc: Word-level accuracy (higher is better)")
    print("  - Similarity: Overall sequence similarity (higher is better)")
    print("  - TrOCR is not tested as it's designed for single-line OCR only")
    print("="*80 + "\n")


def main():
    """Main entry point for the OCR engine testing script."""
    parser = argparse.ArgumentParser(
        description="Test and compare OCR engines against ground truth data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script tests all available OCR engines and ranks them by accuracy.
It uses the test data files:
  - Ground truth: PPC_example_data.txt
  - Test PDF: PPC_example_data_pages_001-010.pdf

Note: TrOCR is not tested as it is designed for single-line OCR only,
not full-page document OCR.

Example:
  python test_ocr_engines.py --verbose
  python test_ocr_engines.py --gpu
"""
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration for engines that support it"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output during testing"
    )
    
    args = parser.parse_args()
    
    # Find repository root and test files
    script_dir = Path(__file__).parent
    ground_truth_path = script_dir / "PPC_example_data.txt"
    pdf_path = script_dir / "PPC_example_data_pages_001-010.pdf"
    
    # Verify files exist
    if not ground_truth_path.exists():
        print(f"Error: Ground truth file not found: {ground_truth_path}", file=sys.stderr)
        sys.exit(1)
    
    if not pdf_path.exists():
        print(f"Error: Test PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load ground truth
    ground_truth = load_ground_truth(ground_truth_path)
    
    if args.verbose:
        print(f"Loaded ground truth: {len(ground_truth)} characters")
        print(f"Test PDF: {pdf_path}")
    
    # Get available engines
    all_available = available_engines()
    
    # Filter out TrOCR engines (single-line only)
    engines_to_test = [e for e in all_available if not e.startswith("trocr")]
    
    if not engines_to_test:
        print("Error: No OCR engines available for testing", file=sys.stderr)
        print("Please install at least one engine (tesseract, easyocr, paddleocr, doctr)", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nTesting {len(engines_to_test)} engine(s): {', '.join(engines_to_test)}")
    if args.gpu:
        print("GPU acceleration enabled")
    
    # Test each engine
    results = []
    for engine_name in engines_to_test:
        engine_name_display, metrics_or_error, ocr_output = test_engine(
            engine_name, pdf_path, args.gpu, args.verbose
        )
        
        # Check if we got an error (ocr_output will be an error string)
        if isinstance(ocr_output, str) and (ocr_output.startswith("Error") or ocr_output.startswith("No output")):
            results.append((engine_name_display, {}, ocr_output))
        else:
            # Compute accuracy
            metrics = compute_accuracy(ground_truth, ocr_output)
            results.append((engine_name_display, metrics, "Success"))
    
    # Sort by character accuracy (descending)
    results.sort(key=lambda x: x[1].get('char_accuracy', -1), reverse=True)
    
    # Print results
    print_results_table(results)
    
    # Return exit code based on whether any engines succeeded
    successful = [r for r in results if r[1]]
    if successful:
        return 0
    else:
        print("Error: All engine tests failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
