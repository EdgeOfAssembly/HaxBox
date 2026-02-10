#!/usr/bin/env python3
"""Backward-compatible entry point for pdfocr.

This file provides backward compatibility for users who run:
    python pdfocr.py [arguments]
    
Instead of:
    python -m pdfocr [arguments]

The actual implementation is in the pdfocr/ package. This file is a thin
wrapper that imports and delegates to the new modular structure.

Note: The preferred way to run pdfocr is now:
    python -m pdfocr [arguments]
    
Or if installed as a package:
    pdfocr [arguments]
"""

import os

# CRITICAL: Set PaddleOCR environment variables BEFORE any imports
# This must be at the very top to fix PaddlePaddle 3.0+ PIR CPU errors
# See: https://github.com/PaddlePaddle/Paddle/issues/59989
os.environ['FLAGS_use_mkldnn'] = 'False'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_use_onednn'] = '0'

# Import and run the CLI
from pdfocr.cli import main

if __name__ == "__main__":
    main()
