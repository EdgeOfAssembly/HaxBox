#!/usr/bin/env python3
"""Validation script to test PaddleOCR in CPU mode.

This script validates that PaddleOCR works correctly in CPU mode
by performing OCR on PPC_example_data_pages_001-010.pdf and comparing
the results to the ground truth in PPC_example_data.txt.

Supports both PaddleOCR 2.x and 3.x with automatic version detection.

Requirements:
- PaddleOCR (2.x or 3.x)
- PaddlePaddle (2.x or 3.x, matching PaddleOCR version)
- pdf2image
- Pillow

Usage:
    python validate_paddleocr_cpu.py
"""

from __future__ import annotations

# ============================================================================
# CRITICAL: ENV VAR WORKAROUND MUST COME BEFORE PADDLEOCR IMPORTS
# ============================================================================
# PaddlePaddle 3.0+ has a bug (GitHub issue #59989) where PIR CPU execution
# crashes with "NotImplementedError: set_tensor is not implemented on oneDNN".
# The workaround is to disable MKL-DNN/oneDNN BEFORE importing paddle.

import os

# Set FLAGS_use_mkldnn to '0' to disable MKL-DNN/oneDNN
# This is a workaround for PaddlePaddle 3.0+ PIR CPU execution bug
# Multiple settings attempted as paranoid defense (last one takes effect)
os.environ['FLAGS_use_mkldnn'] = '0'      # Disable MKL-DNN
os.environ['FLAGS_use_onednn'] = '0'      # Disable oneDNN (new name for MKL-DNN)

# Additional flags that may help
os.environ['PADDLE_USE_ONEDNN'] = '0'     # Alternative OneDNN flag
os.environ['FLAGS_use_cudnn'] = '0'       # Disable CUDNN on CPU
os.environ['CPU_NUM'] = '1'               # Limit CPU threads

# ============================================================================
# NOW SAFE TO IMPORT OTHER MODULES
# ============================================================================

import sys
from pathlib import Path
from difflib import SequenceMatcher
import numpy as np
from pdf2image import convert_from_path
import paddleocr
from paddleocr import PaddleOCR


def similarity_ratio(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two texts using SequenceMatcher.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Similarity ratio between 0.0 and 1.0
    """
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def extract_text_from_paddleocr_results(results: list) -> str:
    """Extract plain text from PaddleOCR results.
    
    PaddleOCR can return results in different formats:
    
    Format 1 (batched - list of pages):
    [
        [  # First page
            [
                [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],  # bounding box
                ('text', confidence)                      # text and confidence
            ],
            ...
        ],
        [  # Second page (if any)
            ...
        ]
    ]
    
    Format 2 (single page - flat list):
    [
        [
            [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
            ('text', confidence)
        ],
        ...
    ]
    
    This function handles both formats by checking if results[0] contains
    line entries (batched) or is itself a line entry (flat).
    
    Args:
        results: PaddleOCR prediction results
        
    Returns:
        Extracted plain text string
    """
    if not results:
        return ""
    
    # Check if results is empty or None
    first = results[0]
    if first is None:
        return ""
    
    # Detect format: batched (list of pages) vs flat (single page)
    # In batched format, results[0] is a list of line entries
    # In flat format, results[0] is a single line entry [bbox, (text, conf)]
    if (
        isinstance(first, list)
        and first
        and len(first) == 2
        and isinstance(first[0], list)  # bbox is a list
        and isinstance(first[1], tuple)  # (text, confidence) is a tuple
    ):
        # Flat format: results is already the list of line entries
        iterable = results
    else:
        # Batched format: results[0] is the first page's line entries
        iterable = first
    
    text_lines = []
    for line in iterable:
        if line and len(line) > 1:
            text, confidence = line[1]
            text_lines.append(text)
    
    return '\n'.join(text_lines)


def main():
    """Main validation function."""
    print("=" * 80)
    print("PaddleOCR CPU Mode Validation Script")
    print("=" * 80)
    
    # Define file paths
    pdf_path = Path(__file__).parent / "PPC_example_data_pages_001-010.pdf"
    ground_truth_path = Path(__file__).parent / "PPC_example_data.txt"
    
    # Verify files exist
    if not pdf_path.exists():
        print(f"ERROR: PDF file not found at {pdf_path}")
        sys.exit(1)
    
    if not ground_truth_path.exists():
        print(f"ERROR: Ground truth file not found at {ground_truth_path}")
        sys.exit(1)
    
    print(f"\nPDF file: {pdf_path}")
    print(f"Ground truth file: {ground_truth_path}")
    
    # Read ground truth
    print("\n[1/5] Reading ground truth...")
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth = f.read()
    print(f"Ground truth length: {len(ground_truth)} characters")
    
    # Convert PDF to images
    print("\n[2/5] Converting PDF to images...")
    try:
        images = convert_from_path(str(pdf_path))
        print(f"Converted {len(images)} pages")
    except Exception as e:
        print(f"ERROR: Failed to convert PDF to images: {e}")
        sys.exit(1)
    
    # Check PaddleOCR version
    version = paddleocr.__version__
    print(f"\n[3/5] Detected PaddleOCR version: {version}")
    major_version = int(version.split('.')[0])
    is_v3_plus = major_version >= 3
    
    # Test GPU mode support (even without GPU hardware)
    print("\n[3a/5] Testing GPU mode initialization (without GPU hardware)...")
    gpu_supported = False
    try:
        if is_v3_plus:
            test_ocr = PaddleOCR(
                lang='en',
                device='gpu',
                text_recognition_batch_size=1,
                use_textline_orientation=True,
            )
        else:
            test_ocr = PaddleOCR(
                lang='en',
                use_gpu=True,
                use_angle_cls=True,
                rec_batch_num=1,
            )
        print("✓ GPU mode initialization succeeded (will use GPU if available, CPU otherwise)")
        gpu_supported = True
        del test_ocr  # Clean up
    except Exception as e:
        print(f"✗ GPU mode initialization failed: {e}")
        print("  This is expected if no GPU is available, but the flag should still be accepted")
        gpu_supported = False
    
    # Initialize PaddleOCR in CPU mode
    print("\n[4/5] Initializing PaddleOCR in CPU mode...")
    try:
        if is_v3_plus:
            print("Using PaddleOCR 3.0+ API...")
            # PaddleOCR 3.0+ uses device parameter
            ocr = PaddleOCR(
                lang='en',
                device='cpu',  # Force CPU mode
                text_recognition_batch_size=1,  # Minimal batch size for stability
                use_textline_orientation=True,  # Enable text orientation detection
            )
        else:
            print("Using PaddleOCR 2.x API...")
            # PaddleOCR 2.x uses use_gpu parameter
            ocr = PaddleOCR(
                lang='en',
                use_gpu=False,  # Disable GPU (CPU mode)
                use_angle_cls=True,  # Enable text angle classification
                rec_batch_num=1,  # Minimal batch size for stability
            )
        
        print("PaddleOCR initialized successfully!")
    except Exception as e:
        print(f"ERROR: Failed to initialize PaddleOCR: {e}")
        print("\nMake sure you have installed:")
        print("  pip install paddleocr paddlepaddle")
        sys.exit(1)
    
    # Perform OCR on all pages
    print("\n[5/5] Performing OCR on all pages...")
    all_text = []
    
    # Determine which method to use based on version
    
    for i, image in enumerate(images, 1):
        print(f"  Processing page {i}/{len(images)}...")
        try:
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            if is_v3_plus:
                # Use predict() method (PaddleOCR 3.0+ API)
                # Note: ocr() method is deprecated in 3.0+, predict() is the new standard
                result = ocr.predict(img_array)
            else:
                # Use ocr() method (PaddleOCR 2.x API)
                result = ocr.ocr(img_array, cls=True)
            
            # Extract text from results
            page_text = extract_text_from_paddleocr_results(result)
            all_text.append(f"--- Page {i} ---\n\n{page_text}")
            
        except Exception as e:
            print(f"  ERROR on page {i}: {e}")
            all_text.append(f"--- Page {i} ---\n\n[OCR FAILED]")
    
    # Combine all OCR text
    ocr_output = '\n\n'.join(all_text)
    
    # Save OCR output for inspection
    output_path = Path(__file__).parent / "paddleocr_validation_output.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(ocr_output)
    print(f"\nOCR output saved to: {output_path}")
    
    # Calculate similarity
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    similarity = similarity_ratio(ocr_output, ground_truth)
    
    print(f"\nOCR output length: {len(ocr_output)} characters")
    print(f"Ground truth length: {len(ground_truth)} characters")
    print(f"Similarity ratio: {similarity:.2%}")
    print(f"GPU mode support: {'✓ Yes' if gpu_supported else '✗ No (or not available)'}")
    
    # Define success threshold
    # OCR is not perfect, so we accept 60% similarity as reasonable
    success_threshold = 0.60
    
    if similarity >= success_threshold:
        print(f"\n✓ SUCCESS: Similarity {similarity:.2%} >= {success_threshold:.2%}")
        print(f"\nPaddleOCR {version} is working correctly in CPU mode!")
        return 0
    else:
        print(f"\n✗ FAILURE: Similarity {similarity:.2%} < {success_threshold:.2%}")
        print("\nPaddleOCR output does not match ground truth sufficiently.")
        print(f"Please inspect {output_path} for details.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
