# PDF Tools

Advanced PDF splitting, merging, OCR, and optimization tools.

This package contains two powerful command-line tools:
- **pdfsplit** - PDF manipulation (split, merge, extract, optimize)
- **pdfocr** - OCR extraction from PDFs and images

## License

This project is dual-licensed:
- **GPLv3** for open-source use
- **Commercial license** available for proprietary applications

Contact: haxbox2000@gmail.com

## Installation

### Requirements

- Python 3.8+
- pip (Python package manager)

### Install Dependencies

```bash
# Core dependencies
pip install PyPDF2 pymupdf pillow tqdm

# Additional dependencies for pdfocr
pip install pytesseract pdf2image opencv-python-headless numpy

# For EasyOCR engine (optional, better quality OCR)
pip install easyocr

# System dependencies (Ubuntu/Debian)
sudo apt install tesseract-ocr poppler-utils
```

### For GPU-accelerated OCR (optional)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install easyocr
```

## pdfsplit

Advanced PDF splitting, merging, and optimization tool.

### Basic Usage

```bash
# Split all pages (one PDF per page)
pdfsplit document.pdf

# Extract specific pages
pdfsplit document.pdf -p 1-10
pdfsplit document.pdf -p 1,5,10-15
pdfsplit document.pdf -p 56-      # From page 56 to end
pdfsplit document.pdf -p -10       # First 10 pages

# Split every N pages
pdfsplit document.pdf -g 10

# Export as PNG images
pdfsplit document.pdf --png
pdfsplit document.pdf -p 1-5 --png --dpi 600

# Extract embedded images
pdfsplit document.pdf --images

# Batch process directory
pdfsplit /path/to/pdfs/

# Multiple files
pdfsplit a.pdf b.pdf c.pdf -d output
```

### Advanced Operations

```bash
# Merge PDFs
pdfsplit --merge a.pdf b.pdf c.pdf -o combined.pdf

# Reverse page order
pdfsplit document.pdf --reverse -o reversed.pdf

# Remove restrictions/unlock PDF
pdfsplit document.pdf --unlock -o unlocked.pdf
pdfsplit encrypted.pdf --unlock --password "secret" -o unlocked.pdf

# Split at bookmark/chapter boundaries
pdfsplit document.pdf --by-bookmark

# Split by file size (great for email attachments)
pdfsplit large.pdf --max-size 10MB
pdfsplit large.pdf --max-size 500KB

# Optimize/compress PDF
pdfsplit document.pdf --optimize -o smaller.pdf
pdfsplit document.pdf --optimize --quality 60 -o compressed.pdf

# Combined operations
pdfsplit document.pdf -p 1-10 --optimize -o extract_optimized.pdf
pdfsplit --merge a.pdf b.pdf --optimize -o merged_optimized.pdf
```

### Options

| Option | Description |
|--------|-------------|
| `-p, --pages` | Page specification (e.g., "1-10", "1,5,10-15") |
| `-g, --granularity` | Split every N pages (default: 1) |
| `-d, --directory` | Output directory (default: pdf_out) |
| `-o, --output` | Output file for merge/reverse/unlock/optimize |
| `--prefix` | Custom prefix for output filenames |
| `--png` | Export as PNG images |
| `--dpi` | DPI for PNG export (default: 300, range: 72-2400) |
| `--images` | Extract embedded images |
| `--info` | Show PDF metadata and exit |
| `--merge` | Merge multiple PDFs |
| `--reverse` | Reverse page order |
| `--unlock` | Remove restrictions |
| `--password` | Password for encrypted PDFs |
| `--by-bookmark` | Split at bookmark boundaries |
| `--max-size` | Split by file size limit |
| `--optimize` | Compress/optimize output |
| `--quality` | Image quality for optimization (1-100) |
| `--keep-metadata` | Keep metadata during optimization |
| `-f, --force` | Force overwrite existing files |
| `-q, --quiet` | Suppress output |
| `-v, --version` | Show version |

## pdfocr

OCR tool for extracting text from scanned PDFs and images.

### Basic Usage

```bash
# OCR a PDF (using tesseract)
pdfocr scanned.pdf

# Use EasyOCR engine (better quality)
pdfocr scanned.pdf -e easyocr

# OCR an image
pdfocr scan.png

# Batch process directory
pdfocr /path/to/files/

# Multiple files
pdfocr a.pdf b.png c.jpg -d output
```

### Advanced Usage

```bash
# OCR specific pages only
pdfocr scanned.pdf -p 1-5
pdfocr scanned.pdf -p 1,10,20-25

# Use different language
pdfocr document.pdf -l deu    # German
pdfocr document.pdf -l fra    # French
pdfocr document.pdf -l jpn    # Japanese

# JSON output with bounding boxes
pdfocr scanned.pdf --format json
pdfocr scanned.pdf -e easyocr --format json  # Full bounding box data

# GPU acceleration (EasyOCR only)
pdfocr scanned.pdf -e easyocr --gpu

# Save rendered page images
pdfocr scanned.pdf --save-images

# High-quality rendering
pdfocr scanned.pdf --dpi 600

# Disable image preprocessing
pdfocr scanned.pdf --no-enhance
```

### Options

| Option | Description |
|--------|-------------|
| `-e, --engine` | OCR engine: tesseract (default) or easyocr |
| `-l, --lang` | Language code (eng, deu, fra, etc.) |
| `-d, --directory` | Output directory (default: ocr_out) |
| `-p, --pages` | Page specification for PDFs |
| `--dpi` | DPI for PDF rendering (default: 300) |
| `--no-enhance` | Disable CLAHE contrast enhancement |
| `--save-images` | Save rendered page images |
| `--format` | Output format: text or json |
| `--gpu` | Use GPU acceleration (EasyOCR only) |
| `-f, --force` | Force overwrite existing files |
| `-q, --quiet` | Suppress output |
| `-v, --version` | Show version |

### Supported Languages

| Tesseract | EasyOCR | Language |
|-----------|---------|----------|
| eng | en | English |
| deu | de | German |
| fra | fr | French |
| spa | es | Spanish |
| ita | it | Italian |
| por | pt | Portuguese |
| rus | ru | Russian |
| chi_sim | ch_sim | Chinese (Simplified) |
| chi_tra | ch_tra | Chinese (Traditional) |
| jpn | ja | Japanese |
| kor | ko | Korean |
| ara | ar | Arabic |

## Examples

### Workflow: Scan → OCR → Archive

```bash
# 1. OCR scanned documents
pdfocr scans/*.pdf -e easyocr --gpu -d ocr_output

# 2. Optimize originals for storage
for f in scans/*.pdf; do
    pdfsplit "$f" --optimize -o "archive/$(basename $f)"
done
```

### Workflow: Split Large Document

```bash
# Split a large PDF for email attachments (max 10MB each)
pdfsplit large_report.pdf --max-size 10MB -d email_parts

# Or split by chapters
pdfsplit book.pdf --by-bookmark -d chapters
```

### Workflow: Merge and Optimize

```bash
# Merge multiple PDFs and optimize
pdfsplit --merge scan1.pdf scan2.pdf scan3.pdf --optimize -o combined.pdf
```

### Workflow: Unlock and Process

```bash
# Remove restrictions and extract specific pages
pdfsplit restricted.pdf --unlock -o temp.pdf -f
pdfsplit temp.pdf -p 10-20 --optimize -o final.pdf
rm temp.pdf
```

## Legal Disclaimer

The `--unlock` feature is intended for removing restrictions from documents you own or have rights to modify. Circumventing protection on copyrighted materials may violate laws in your jurisdiction.

## Author

- **EdgeOfAssembly**
- Email: haxbox2000@gmail.com

## Version History

### v3.0.0 (pdfsplit)
- Added --merge for combining PDFs
- Added --reverse for page order reversal
- Added --unlock for removing restrictions
- Added --by-bookmark for chapter splitting
- Added --max-size for size-based splitting
- Added --optimize for PDF compression
- Fixed duplicate file discovery on case-insensitive filesystems
- Fixed memory issue with large PDFs
- Added DPI validation

### v2.0.0 (pdfocr)
- Added --pages for selective OCR
- Added --format json for structured output
- Added --gpu for EasyOCR acceleration
- Fixed race condition in lazy loading
- Fixed duplicate file discovery
- Added DPI validation
