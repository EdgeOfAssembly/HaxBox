[![codecov](https://codecov.io/github/EdgeOfAssembly/HaxBox/graph/badge.svg?token=OQAB6V87AL)](https://codecov.io/github/EdgeOfAssembly/HaxBox)

# HaxBox

A collection of useful command-line tools for Linux.

## Tools

| Tool | Description |
|------|-------------|
| **pdfsplit** | PDF manipulation (split, merge, extract, optimize) |
| **pdfocr** | OCR extraction from PDFs and images |
| **stegpng** | Steganographic file encoder/decoder for PNG/JPEG images |
| **llmprep** | Prepare codebases for LLM analysis |
| **screenrec** | Flexible screen and audio recorder |

## License

This project is dual-licensed:
- **GPLv3** for open-source use
- **Commercial license** available for proprietary applications

Contact: haxbox2000@gmail.com

## Installation

### Requirements

- Python 3.8+
- pip (Python package manager)

### Install All Dependencies

```bash
# Core dependencies for all tools
pip install pypdf pymupdf pillow tqdm numpy

# For pdfocr
pip install pytesseract pdf2image opencv-python-headless
pip install easyocr  # Optional, higher accuracy
pip install transformers torch  # Optional, TrOCR transformer-based OCR
pip install paddleocr paddlepaddle  # Optional, state-of-the-art OCR
pip install python-doctr[torch]  # Optional, document-focused OCR

# For screenrec
pip install opencv-python-headless mss pynput

# System dependencies (Ubuntu/Debian)
sudo apt install tesseract-ocr poppler-utils xdotool wmctrl ffmpeg
```

---

## pdfsplit

Advanced PDF splitting, merging, and optimization tool.

### Basic Usage

```bash
# Split all pages (one PDF per page)
pdfsplit document.pdf

# Extract specific pages
pdfsplit document.pdf -p 1-10
pdfsplit document.pdf -p 1,5,10-15

# Split every N pages
pdfsplit document.pdf -g 10

# Export as PNG images
pdfsplit document.pdf --png --dpi 600

# Merge PDFs
pdfsplit --merge a.pdf b.pdf c.pdf -o combined.pdf

# Optimize/compress PDF
pdfsplit document.pdf --optimize -o smaller.pdf
```

### Options

| Option | Description |
|--------|-------------|
| `-p, --pages` | Page specification (e.g., "1-10", "1,5,10-15") |
| `-g, --granularity` | Split every N pages |
| `-d, --directory` | Output directory |
| `-o, --output` | Output file for merge/optimize |
| `--png` | Export as PNG images |
| `--dpi` | DPI for PNG export (default: 300) |
| `--merge` | Merge multiple PDFs |
| `--optimize` | Compress/optimize output |
| `--unlock` | Remove PDF restrictions |

---

## pdfocr

OCR tool for extracting text from scanned PDFs and images.

### Basic Usage

```bash
# OCR a PDF (using tesseract)
pdfocr scanned.pdf

# Use EasyOCR engine (better quality)
pdfocr scanned.pdf -e easyocr

# Use TrOCR engine (transformer-based, good for printed text)
pdfocr scanned.pdf -e trocr --gpu

# Use TrOCR for handwritten text
pdfocr notes.pdf -e trocr-handwritten --gpu

# Use PaddleOCR (state-of-the-art accuracy)
pdfocr scanned.pdf -e paddleocr --gpu

# Use docTR (document-focused OCR)
pdfocr scanned.pdf -e doctr --gpu

# OCR specific pages
pdfocr scanned.pdf -p 1-5

# Different language
pdfocr document.pdf -l deu  # German

# JSON output with bounding boxes
pdfocr scanned.pdf --format json

# GPU acceleration (EasyOCR, TrOCR, PaddleOCR, docTR)
pdfocr scanned.pdf -e easyocr --gpu
```

### Options

| Option | Description |
|--------|-------------|
| `-e, --engine` | OCR engine: tesseract, easyocr, trocr, trocr-handwritten, paddleocr, doctr |
| `-l, --lang` | Language code (eng, deu, fra, etc.) |
| `-p, --pages` | Page specification for PDFs |
| `--dpi` | DPI for PDF rendering (default: 300) |
| `--format` | Output format: text or json |
| `--gpu` | Use GPU acceleration (EasyOCR, TrOCR, PaddleOCR, docTR) |

### OCR Engines

| Engine | Speed | Accuracy | GPU | Best For |
|--------|-------|----------|-----|----------|
| tesseract | Fast | Good | No | General use, default |
| easyocr | Medium | Better | Yes | Multiple languages |
| trocr | Slow | Best | Yes | Printed text |
| trocr-handwritten | Slow | Best | Yes | Handwritten text |
| paddleocr | Medium | Best | Yes | State-of-the-art accuracy |
| doctr | Medium | Best | Yes | Document layouts |

---

## stegpng

Steganographic file encoder/decoder for hiding files in PNG and JPEG images.

### Basic Usage

```bash
# Encode a file into a PNG
stegpng encode secret.txt -o hidden.png

# Encode with XOR obfuscation
stegpng encode secret.txt -o hidden.png -k "mypassword"

# Use metadata embedding (survives some image processors)
stegpng encode secret.txt -o hidden.png -m metadata

# Use existing image as cover
stegpng encode secret.txt -o hidden.png -b photo.png

# Decode from image
stegpng decode hidden.png -o recovered.txt
stegpng decode hidden.png -o recovered.txt -k "mypassword"

# Check for hidden data
stegpng info hidden.png
```

### Commands

| Command | Description |
|---------|-------------|
| `encode` | Hide a file inside an image |
| `decode` | Extract hidden file from image |
| `info` | Show image info and detect hidden data |

### Encode Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output image file |
| `-b, --base` | Use existing image as cover |
| `-k, --key` | XOR key for obfuscation |
| `-m, --method` | append (default) or metadata |

### Decode Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output file for extracted data |
| `-k, --key` | XOR key (must match encode key) |
| `-m, --method` | Must match encode method |

---

## llmprep

Prepare a codebase for LLM (Large Language Model) analysis. Generates
documentation, statistics, and context files.

### Basic Usage

```bash
# Analyze current directory
llmprep .

# Analyze a project
llmprep /path/to/project

# Custom output directory
llmprep /path/to/project -o analysis

# Skip heavy processing
llmprep . --no-doxygen --no-pyreverse

# Quiet mode
llmprep . -q
```

### Output Files

| File | Description |
|------|-------------|
| `codebase_overview.md` | Main overview (use as LLM context) |
| `codebase_structure.txt` | Directory tree |
| `codebase_stats.txt` | Code statistics |
| `llm_system_prompt.md` | Suggested system prompt |
| `project_guidance.md` | Best practices |
| `tags` | ctags symbol index |
| `dot_graphs_*/` | Dependency graphs |

### Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output directory name (default: llm_prep) |
| `-d, --depth` | Tree depth (default: 4) |
| `--exclude` | Exclusion pattern for tree |
| `--no-doxygen` | Skip Doxygen |
| `--no-pyreverse` | Skip pyreverse |
| `--no-ctags` | Skip ctags |
| `-q, --quiet` | Suppress output |

### Optional Dependencies

- `tree` - Directory listing
- `cloc` - Code statistics
- `ctags` - Symbol indexing
- `doxygen` - C/C++ documentation
- `pyreverse` - Python UML diagrams (via pylint)

---

## screenrec

Flexible screen and audio recorder with multiple capture modes.

### Basic Usage

```bash
# Select region by drawing (default)
screenrec

# Record full screen for 60 seconds
screenrec --fullscreen -d 60

# Record with audio
screenrec --fullscreen --audio -d 60

# Record primary monitor for 2 minutes
screenrec --monitor 1 -d 2m

# Click on a window to record
screenrec --window -d 30

# Record specific region
screenrec --region 100,100,800,600 -d 1m

# With countdown and custom output
screenrec --fullscreen --countdown 3 -o ~/Videos/capture.mp4 -d 30
```

### Capture Modes

| Mode | Description |
|------|-------------|
| `--fullscreen` | Record entire screen (all monitors) |
| `--monitor N` | Record monitor N (1=primary) |
| `--window` | Click on a window to record |
| `--window-id ID` | Record specific window ID |
| `--select` | Draw rectangle (default) |
| `--region X,Y,W,H` | Record specific coordinates |

### Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output file (default: /tmp/screenrec.mp4) |
| `-d, --duration` | Duration: 30, 30s, 1m, 1h (default: 30) |
| `--fps` | Frames per second (default: 30) |
| `-a, --audio` | Record desktop audio |
| `--audio-source` | Specific PulseAudio source |
| `--countdown` | Countdown before recording |
| `--list-monitors` | List available monitors |
| `--list-windows` | List available windows |
| `--list-audio` | List audio sources |
| `-v, --verbose` | Show progress |

### Dependencies

```bash
pip install opencv-python-headless mss pynput numpy
sudo apt install xdotool wmctrl ffmpeg
```

---

## Man Pages

Man pages are provided for all tools. Install them with:

```bash
sudo cp *.1 /usr/local/share/man/man1/
sudo mandb
```

Then access with `man pdfsplit`, `man pdfocr`, etc.

---

## Examples

### Workflow: Scan → OCR → Archive

```bash
# OCR scanned documents
pdfocr scans/*.pdf -e easyocr -d ocr_output

# Optimize originals for storage
for f in scans/*.pdf; do
    pdfsplit "$f" --optimize -o "archive/$(basename $f)"
done
```

### Workflow: Hide and Transfer Files

```bash
# Hide a file in an image
stegpng encode sensitive.pdf -o vacation.png -k "secretkey"

# Later, extract it
stegpng decode vacation.png -o sensitive.pdf -k "secretkey"
```

### Workflow: Record Tutorial

```bash
# Record window with audio and countdown
screenrec --window --audio --countdown 5 -d 5m -o tutorial.mp4 -v
```

### Workflow: Analyze Codebase for AI Review

```bash
# Generate LLM context
llmprep /path/to/project

# Copy overview to clipboard
cat llm_prep/codebase_overview.md | xclip -selection clipboard
```

---

## Author

**EdgeOfAssembly**
Email: haxbox2000@gmail.com

## Version History

### v3.0.0 (pdfsplit)
- Added --merge, --reverse, --unlock, --by-bookmark, --max-size, --optimize

### v2.0.0 (pdfocr)
- Added --pages, --format json, --gpu

### v1.0.0 (stegpng)
- Initial release: PNG/JPEG support, XOR obfuscation, metadata embedding

### v1.0.0 (llmprep)
- Initial release: Directory tree, cloc, doxygen, pyreverse, ctags

### v1.0.0 (screenrec)
- Initial release: Multiple capture modes, audio recording
