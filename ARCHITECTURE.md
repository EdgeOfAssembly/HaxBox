# pdfocr Architecture Documentation

## Overview

pdfocr is a modular OCR tool that extracts text from PDFs and images using multiple OCR engines. Version 3.0.0 represents a complete architectural refactor from a monolithic 1537-line file into a clean, modular package structure.

## Package Structure

```
pdfocr/
├── __init__.py          # Public API exports
├── __main__.py          # CLI entry point (python -m pdfocr)
├── cli.py               # Argument parsing and main() function
├── core.py              # OCR orchestration: ocr_image(), ocr_pdf(), ocr_image_file()
├── utils.py             # Utilities: validate_dpi(), parse_page_spec(), pdf_to_images()
├── types.py             # Shared types, constants, language mappings
└── engines/
    ├── __init__.py      # Engine registry and get_engine() factory
    ├── base.py          # Abstract OCREngine base class
    ├── tesseract.py     # TesseractEngine
    ├── easyocr.py       # EasyOCREngine
    ├── trocr.py         # TrOCREngine (handles printed and handwritten)
    ├── paddleocr.py     # PaddleOCREngine (complete rewrite)
    └── doctr.py         # DocTREngine
```

## Design Principles

### 1. Plugin Architecture

Each OCR engine is a self-contained module that implements the `OCREngine` abstract base class. Engines register themselves using the `@register_engine` decorator, making them automatically discoverable.

### 2. Lazy Loading

All engines use lazy initialization with thread-safe double-checked locking. Dependencies are imported only when an engine is first used, not at package import time.

### 3. Consistent Interface

All engines follow the same interface:
- `__init__(lang, gpu, **kwargs)` - Initialize engine
- `ocr(image, return_boxes)` - Perform OCR
- `is_available()` - Check if dependencies are installed
- `convert_language(tesseract_lang)` - Convert language codes

### 4. Type Safety

Full type hints throughout for `mypy --strict` compliance. All functions have complete parameter and return type annotations.

### 5. Documentation

Every module, class, and function has comprehensive Google-style docstrings explaining purpose, parameters, returns, raises, and examples.

## Adding a New OCR Engine

Adding a new OCR engine is straightforward:

### Step 1: Create the Engine Module

Create a new file `pdfocr/engines/myengine.py`:

```python
#!/usr/bin/env python3
"""My Custom OCR Engine implementation."""

from __future__ import annotations

import threading
from typing import Any, Optional

from PIL import Image
from pdfocr.engines.base import OCREngine, register_engine

# Global cache and lock for thread-safe lazy loading
_myengine_instance: Any = None
_myengine_lock = threading.Lock()


@register_engine
class MyEngine(OCREngine):
    """My Custom OCR Engine.
    
    Comprehensive docstring explaining what this engine does,
    when to use it, and any special considerations.
    """
    
    # Required class attributes
    name = "myengine"
    display_name = "My Custom OCR Engine"
    supports_gpu = True
    supports_boxes = True
    install_hint = "pip install my-ocr-library"
    
    def __init__(self, lang: str = "eng", gpu: bool = False, **kwargs: Any) -> None:
        """Initialize the engine with lazy loading.
        
        Args:
            lang: Language code (tesseract format, will be converted)
            gpu: Whether to use GPU acceleration
            **kwargs: Engine-specific options
        """
        self.lang = self.convert_language(lang)
        self.gpu = gpu
        
        # Lazy initialization happens here
        global _myengine_instance
        if _myengine_instance is None:
            with _myengine_lock:
                if _myengine_instance is None:
                    try:
                        import my_ocr_library
                        _myengine_instance = my_ocr_library.OCR(
                            lang=self.lang,
                            use_gpu=gpu
                        )
                    except ImportError as e:
                        raise ImportError(
                            f"my-ocr-library not installed. {self.install_hint}"
                        ) from e
        
        self.engine = _myengine_instance
    
    def ocr(self, image: Image.Image, return_boxes: bool = False) -> Any:
        """Perform OCR on the image.
        
        Args:
            image: PIL Image to OCR
            return_boxes: Whether to return structured data with boxes
            
        Returns:
            str if return_boxes=False, structured data if True
        """
        result = self.engine.recognize(image)
        
        if return_boxes:
            return result  # Return structured format
        else:
            # Extract text only
            return "\n".join([item["text"] for item in result])
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if dependencies are installed."""
        try:
            import my_ocr_library
            return True
        except ImportError:
            return False
    
    @classmethod
    def convert_language(cls, tesseract_lang: str) -> str:
        """Convert tesseract language code to engine format.
        
        Override this if your engine uses different codes.
        """
        # Example mapping
        mapping = {
            "eng": "en",
            "deu": "de",
            "fra": "fr",
        }
        return mapping.get(tesseract_lang, tesseract_lang[:2])
```

### Step 2: Register in engines/__init__.py

Add an import to `pdfocr/engines/__init__.py`:

```python
from pdfocr.engines import myengine  # noqa: F401
```

### Step 3: Add CLI Choice

Update `pdfocr/cli.py` argument parser:

```python
parser.add_argument(
    "-e",
    "--engine",
    choices=["tesseract", "easyocr", "trocr", "trocr-handwritten", 
             "paddleocr", "doctr", "myengine"],  # Add here
    default="tesseract",
    help="OCR engine to use (default: tesseract)",
)
```

### Step 4: Test

```python
from pdfocr import get_engine, available_engines

# Check it's registered
print(available_engines())  # Should include 'myengine'

# Use it
engine = get_engine("myengine", lang="eng", gpu=True)
result = engine.ocr(image)
```

That's it! Your engine is now fully integrated.

## Engine Details

### Tesseract (tesseract.py)

- **Type**: Classical OCR (binary required)
- **Speed**: Fast
- **Accuracy**: Good for printed text
- **GPU**: Not supported
- **Boxes**: Basic support
- **Install**: `pip install pytesseract && apt install tesseract-ocr`

Uses the pytesseract Python wrapper around the Tesseract C++ binary.

### EasyOCR (easyocr.py)

- **Type**: Deep learning OCR
- **Speed**: Medium (fast with GPU)
- **Accuracy**: Very high
- **GPU**: Supported (recommended)
- **Boxes**: Full support with confidence scores
- **Install**: `pip install easyocr`

Pure Python implementation with excellent accuracy. Supports 80+ languages. Reader reinitializes if language set changes.

### TrOCR (trocr.py)

- **Type**: Transformer-based OCR
- **Speed**: Slow (medium with GPU)
- **Accuracy**: Excellent for line-level text
- **GPU**: Supported (recommended)
- **Boxes**: Not supported
- **Install**: `pip install transformers torch`
- **Variants**: `printed` and `handwritten`

**Important**: TrOCR is designed for line-level OCR, not full-page documents. It resizes images to 384×384, which causes severe distortion for full pages. Best for extracting text from individual text lines or regions.

### PaddleOCR (paddleocr.py)

- **Type**: State-of-the-art deep learning OCR
- **Speed**: Fast (very fast with GPU)
- **Accuracy**: Excellent
- **GPU**: Supported with OOM fallback
- **Boxes**: Full support
- **Install**: `pip install paddleocr paddlepaddle`

**Completely rewritten in v3.0.0** to fix persistent bugs from PRs #9, #11, #12, #13, #14.

#### PaddleOCR Workarounds

This engine includes several critical workarounds for PaddlePaddle 3.0+ issues:

1. **oneDNN CPU Bug**: Sets environment variables before imports:
   ```python
   os.environ['FLAGS_use_mkldnn'] = '0'
   os.environ['FLAGS_use_onednn'] = '0'
   ```
   Fixes PIR (Paddle Intermediate Representation) errors when using CPU.
   See: https://github.com/PaddlePaddle/Paddle/issues/59989

2. **GPU OOM Fallback**: Automatically detects "out of memory" errors and falls back to CPU with a clean instance recreation.

3. **API Version Compatibility**: Only uses PaddleOCR 3.0+ API:
   - `device='gpu'/'cpu'` (not `use_gpu`)
   - `use_textline_orientation=True` (not `use_angle_cls`)
   - `text_recognition_batch_size` (not `rec_batch_num`)
   - `.predict()` method (not `.ocr()`)

4. **Version Detection**: Catches `TypeError` during initialization to detect old PaddleOCR versions and raises a clear error message.

### docTR (doctr.py)

- **Type**: Document-focused deep learning OCR
- **Speed**: Medium (fast with GPU)
- **Accuracy**: Excellent for complex layouts
- **GPU**: Supported
- **Boxes**: Full hierarchical structure
- **Install**: `pip install python-doctr[torch]`

Specialized for document understanding with complex layouts. Includes horizontal spacing preservation using gap analysis between word bounding boxes.

## Core Functions

### ocr_image()

Main dispatcher for single-image OCR. Instantiates the requested engine and performs OCR.

```python
from pdfocr import ocr_image
from PIL import Image

image = Image.open("scan.png")
text = ocr_image(
    image,
    engine="easyocr",
    lang="eng",
    gpu=True,
    enhance=True,
    return_boxes=False
)
```

### ocr_pdf()

Process an entire PDF with page filtering, progress tracking, and JSON/text output.

```python
from pdfocr import ocr_pdf
from pathlib import Path

ocr_pdf(
    Path("document.pdf"),
    Path("output_dir"),
    engine="paddleocr",
    lang="eng",
    pages_spec="1-10",  # First 10 pages
    dpi=300,
    gpu=True,
    return_boxes=True,  # JSON output with boxes
)
```

### ocr_image_file()

Process a single image file with I/O handling.

```python
from pdfocr import ocr_image_file
from pathlib import Path

ocr_image_file(
    Path("scan.jpg"),
    Path("output_dir"),
    engine="tesseract",
    lang="deu",  # German
)
```

## Utilities

### validate_dpi()

Validates and clamps DPI values to the acceptable range (72-600).

### parse_page_spec()

Parses page specification strings like "1-5,10,20-25" into page number lists.

### pdf_to_images()

Converts PDF pages to PIL Images using pdf2image.

### process_inputs()

Discovers and validates input files from paths or directories.

## Type System

### OCRResult

Union type for OCR results:
```python
OCRResult = Union[str, List[Any], Dict[str, Any]]
```

Different engines return different structured formats when `return_boxes=True`.

### Language Mappings

- `TESSERACT_TO_EASYOCR_LANG`: Map tesseract → EasyOCR codes
- `TESSERACT_TO_PADDLEOCR_LANG`: Map tesseract → PaddleOCR codes

These allow users to specify languages in tesseract format and have them automatically converted.

## Thread Safety

All engines use thread-safe lazy initialization with double-checked locking:

```python
_engine_instance = None
_engine_lock = threading.Lock()

if _engine_instance is None:
    with _engine_lock:
        if _engine_instance is None:
            # Initialize engine
            _engine_instance = ...
```

This pattern:
- Minimizes lock contention (only lock on first access)
- Prevents race conditions (double-check inside lock)
- Allows safe concurrent usage

## Testing

Run tests with:
```bash
pytest tests/test_pdfocr.py -v
```

Test structure:
- Unit tests for utilities (validate_dpi, parse_page_spec)
- Integration tests with mocked OCR engines
- Engine availability checks
- Registry tests

## Backward Compatibility

The root `pdfocr.py` file is a thin shim that imports from the package:

```python
from pdfocr.cli import main

if __name__ == "__main__":
    main()
```

This maintains compatibility for users who run:
```bash
python pdfocr.py document.pdf
```

## CLI Usage

```bash
# Preferred: Use as module
python -m pdfocr document.pdf -e easyocr --gpu

# Backward compatible: Direct execution
python pdfocr.py document.pdf -e easyocr --gpu

# If installed as package
pdfocr document.pdf -e easyocr --gpu
```

## Library Usage

```python
# High-level API
from pdfocr import ocr_image, available_engines
from PIL import Image

engines = available_engines()
print(f"Available: {engines}")

image = Image.open("scan.png")
text = ocr_image(image, engine="easyocr", lang="eng", gpu=True)

# Low-level engine API
from pdfocr import get_engine

engine = get_engine("paddleocr", lang="chi_sim", gpu=True, batch_size=4)
result = engine.ocr(image, return_boxes=True)
```

## Version History

- **v3.0.0**: Complete architectural refactor into modular package
  - Split monolithic file into 13 modules
  - Plugin-based engine architecture
  - PaddleOCR complete rewrite with all fixes
  - Full type hints and documentation

- **v2.0.0**: Added PaddleOCR and docTR engines, type hints

- **v1.0.0**: Initial release with Tesseract, EasyOCR, TrOCR

## Contributing

When adding features:
1. Maintain the plugin architecture pattern
2. Add comprehensive docstrings
3. Include full type hints
4. Write tests
5. Update this documentation

## License

GPLv3 / Commercial dual-license
