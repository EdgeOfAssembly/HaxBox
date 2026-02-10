#!/usr/bin/env python3
"""Validation script to test PaddleOCR 3.0+ in CPU mode.

This script validates that PaddleOCR 3.0+ works correctly in CPU mode
by performing OCR on PPC_example_data_pages_001-010.pdf and comparing
the results to the ground truth in PPC_example_data.txt.

Requirements:
- PaddleOCR 3.0+
- PaddlePaddle 3.0+ (CPU version)
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

os.environ['FLAGS_use_mkldnn'] = 'False'  # String 'False' for compatibility
os.environ['FLAGS_use_mkldnn'] = '0'      # Numeric '0' as backup
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
    
    PaddleOCR returns results in format:
    [
        [
            [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],  # bounding box
            ('text', confidence)                      # text and confidence
        ],
        ...
    ]
    
    Args:
        results: PaddleOCR prediction results
        
    Returns:
        Extracted plain text string
    """
    text_lines = []
    
    if not results or results[0] is None:
        return ""
    
    for line in results[0]:
        if line and len(line) > 1:
            text, confidence = line[1]
            text_lines.append(text)
    
    return '\n'.join(text_lines)


def main():
    """Main validation function."""
    print("=" * 80)
    print("PaddleOCR 3.0+ CPU Mode Validation Script")
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
    print("\n[1/4] Reading ground truth...")
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth = f.read()
    print(f"Ground truth length: {len(ground_truth)} characters")
    
    # Convert PDF to images
    print("\n[2/4] Converting PDF to images...")
    try:
        images = convert_from_path(str(pdf_path))
        print(f"Converted {len(images)} pages")
    except Exception as e:
        print(f"ERROR: Failed to convert PDF to images: {e}")
        sys.exit(1)
    
    # Initialize PaddleOCR in CPU mode
    print("\n[3/4] Initializing PaddleOCR 3.0+ in CPU mode...")
    try:
        # Use PaddleOCR 3.0+ API with CPU device
        ocr = PaddleOCR(
            lang='en',
            device='cpu',  # Force CPU mode
            text_recognition_batch_size=1,  # Minimal batch size for stability
            use_textline_orientation=True,  # PaddleOCR 3.0+ parameter (replaces use_angle_cls)
        )
        print("PaddleOCR initialized successfully!")
    except Exception as e:
        print(f"ERROR: Failed to initialize PaddleOCR: {e}")
        print("\nMake sure you have installed:")
        print("  pip install paddleocr>=3.0.0 paddlepaddle>=3.0.0")
        sys.exit(1)
    
    # Perform OCR on all pages
    print("\n[4/4] Performing OCR on all pages...")
    all_text = []
    
    for i, image in enumerate(images, 1):
        print(f"  Processing page {i}/{len(images)}...")
        try:
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            # Use predict() method (PaddleOCR 3.0+ API)
            result = ocr.predict(img_array)
            
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
    
    # Define success threshold
    # OCR is not perfect, so we accept 60% similarity as reasonable
    success_threshold = 0.60
    
    if similarity >= success_threshold:
        print(f"\n✓ SUCCESS: Similarity {similarity:.2%} >= {success_threshold:.2%}")
        print("\nPaddleOCR 3.0+ is working correctly in CPU mode!")
        return 0
    else:
        print(f"\n✗ FAILURE: Similarity {similarity:.2%} < {success_threshold:.2%}")
        print("\nPaddleOCR output does not match ground truth sufficiently.")
        print(f"Please inspect {output_path} for details.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
