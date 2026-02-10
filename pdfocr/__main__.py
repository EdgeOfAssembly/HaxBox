#!/usr/bin/env python3
"""Entry point for running pdfocr as a module.

This allows the package to be invoked with:
    python -m pdfocr [arguments]
    
Instead of:
    python pdfocr.py [arguments]
"""

from pdfocr.cli import main

if __name__ == "__main__":
    main()
