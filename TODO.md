# HaxBox TODO - Redaction Recovery & New OCR Engines

**Last Updated:** February 10, 2026

**Origin:** This roadmap is based on redaction recovery research from PR #113 and PR #114 in EdgeOfAssembly/RetroCodeMess, which analyzed vulnerabilities in OmniPage CSDK 21.1 redactions in document EFTA01743628.pdf.

---

## Section 1: Redaction Recovery Methods

Ordered from easiest to hardest, with implementation time estimates and required packages.

### Tier 1 — Quick Wins (Hours)

These methods can be implemented quickly and may yield immediate results with minimal effort.

- [ ] **PDF metadata & hidden text layer extraction**
  - **Description:** Check if redacted text exists under annotations or in PDF metadata using PyPDF2/pypdf
  - **Effort:** 2-4 hours
  - **Packages:** `pypdf` or `PyPDF2`
  - **Implementation:** Parse PDF content streams, check for text under redaction annotations, extract metadata fields
  - **Success Rate:** Low (~5%), but worth checking first

- [ ] **PDF content stream parsing**
  - **Description:** Search for Tj/TJ text operators in PDF content streams that may contain redacted text
  - **Effort:** 3-6 hours
  - **Packages:** `pypdf`, `PyPDF2`
  - **Implementation:** Parse PDF operators, look for text-showing operators (Tj, TJ, ', ") that may precede redaction rectangles
  - **Success Rate:** Low-Medium (~10-15%), depends on redaction tool

- [ ] **Multi-engine OCR on redaction boundary edges**
  - **Description:** Use all pdfocr engines with heavy preprocessing on cropped regions around redaction edges to catch partial character leaks
  - **Effort:** 4-8 hours
  - **Packages:** All existing pdfocr engines (tesseract, easyocr, paddleocr, doctr, trocr)
  - **Implementation:** 
    - Crop image to redaction boundary + 50px margin
    - Apply extreme contrast, brightness adjustments
    - Run all available engines with ensemble voting
    - Check for partial characters at edges
  - **Success Rate:** Medium (~20-30%), effective if redaction doesn't fully cover text

- [ ] **Font metrics estimation from visible text**
  - **Description:** Use pytesseract image_to_data() to measure character widths/heights near redacted areas, estimate character count under black box
  - **Effort:** 4-6 hours
  - **Packages:** `pytesseract`, `pdfocr.font_detect` (implemented)
  - **Implementation:** Already implemented in `pdfocr/font_detect.py` via `estimate_redacted_chars()`
  - **Success Rate:** Medium (~30-40%), provides strong constraint for candidate generation
  - **Status:** ✅ **IMPLEMENTED** (see `pdfocr/font_detect.py`)

### Tier 2 — Moderate Effort (Days)

These methods require more development time and may involve image processing pipelines.

- [ ] **Redaction box detection & pixel forensics**
  - **Description:** Detect solid black rectangles via contour analysis (OpenCV), measure exact dimensions, check for non-uniform black pixels
  - **Effort:** 2-3 days
  - **Packages:** `opencv-python`, `numpy`
  - **Implementation:**
    - Use Canny edge detection + contour finding to locate rectangular regions
    - Check pixel uniformity (std dev of RGB values)
    - Analyze histogram of pixel values under redaction
    - Look for slight variations that might indicate text underneath
  - **Success Rate:** Medium (~25-35%), depends on redaction opacity

- [ ] **RGB channel separation + individual channel OCR**
  - **Description:** Split R/G/B channels and OCR each separately (sometimes redactions aren't perfectly opaque in all channels)
  - **Effort:** 1-2 days
  - **Packages:** `PIL`, `opencv-python`, existing OCR engines
  - **Implementation:**
    - Split image into R, G, B channels
    - Apply contrast enhancement to each channel
    - Run OCR on each channel separately
    - Combine results using voting or confidence scores
  - **Success Rate:** Medium (~20-30%), effective for certain PDF renderers

- [ ] **Image enhancement pipeline on redacted regions**
  - **Description:** Extreme contrast (5x), brightness lifting, unsharp masking, binary thresholding at multiple levels (50, 100, 150, 200), edge detection
  - **Effort:** 2-3 days
  - **Packages:** `PIL`, `opencv-python`, `scikit-image`
  - **Implementation:**
    - Create pipeline: CLAHE → Unsharp Mask → Multi-level thresholding
    - Apply edge detection (Sobel, Canny) to reveal ghosted text
    - Use morphological operations (dilation, erosion) to enhance faint text
    - Run OCR on each enhanced variant
  - **Success Rate:** Medium-High (~30-45%), particularly effective for scanned documents

- [ ] **Alpha channel / transparency layer examination**
  - **Description:** Check if PDF/image has RGBA with alpha data under redaction
  - **Effort:** 1-2 days
  - **Packages:** `PIL`, `PyMuPDF`, `pypdf`
  - **Implementation:**
    - Extract image with alpha channel from PDF
    - Check alpha values under redaction regions
    - Look for partial transparency or layering artifacts
  - **Success Rate:** Low-Medium (~15-25%), depends on PDF structure

- [ ] **PDF annotation layer analysis**
  - **Description:** Some redaction tools add annotations that can be removed to reveal text underneath (especially non-flattened PDFs)
  - **Effort:** 2-3 days
  - **Packages:** `PyMuPDF`, `pypdf`, `PyPDF2`
  - **Implementation:**
    - Parse PDF annotation structures
    - Identify redaction annotations (/Subtype /Redact or /Square with black fill)
    - Remove annotations and re-render to check for underlying text
    - Check if PDF is flattened or has separate layers
  - **Success Rate:** Low-Medium (~10-20%), highly dependent on redaction tool behavior

### Tier 3 — Advanced (Weeks)

These methods require significant development effort and may involve AI/ML components.

- [ ] **Font type recognition on nearby visible text**
  - **Description:** Identify exact font (e.g., Calibri, Helvetica, San Francisco) using WhatTheFont API or deep learning font classifier
  - **Effort:** 1-2 weeks
  - **Packages:** Font matching API or custom model, `fontTools`, `pdfocr.font_detect`
  - **Implementation:**
    - Extract visible text samples near redaction
    - Use WhatTheFont API or train/use a font classification CNN
    - Load precise character width tables for identified font
    - Calculate exact character count with font metrics
  - **Success Rate:** Medium-High (~40-50%) for character count, doesn't recover text directly

- [ ] **LLM-assisted contextual inference (local Ollama)**
  - **Description:** Feed visible email context to local LLM (e.g., llama3.2:3b via Ollama) to generate candidate email addresses
  - **Effort:** 1-2 weeks
  - **Packages:** `ollama`, `langchain` or direct API
  - **Implementation:**
    - Extract all visible text from document (To, From, Date, Subject, body, signature)
    - Craft prompt with character count constraint and visible context
    - Generate N candidate email addresses (N=100-1000)
    - Filter candidates by character count, domain patterns, known names
    - Rank by plausibility score
  - **Success Rate:** High (~60-80%) for emails with strong context
  - **Requirements:** Local Ollama installation, 8-16 GB RAM

- [ ] **LLM-assisted contextual inference (cloud API)**
  - **Description:** Same as local LLM but using OpenAI/Anthropic/Google APIs for higher quality reasoning
  - **Effort:** 1-2 weeks
  - **Packages:** `openai`, `anthropic`, `google-generativeai`
  - **Implementation:**
    - Same as local approach but with GPT-4, Claude 3.5, or Gemini Pro
    - Use both local and cloud, compare/ensemble results
    - Implement cost controls (max tokens, batch processing)
  - **Success Rate:** Very High (~70-90%) for emails with strong context
  - **Requirements:** API keys, cost budget (~$0.01-$0.10 per inference)

- [ ] **VLM (Vision-Language Model) direct analysis**
  - **Description:** Send entire scanned page image to VLM (Qwen2.5-VL, GOT-OCR2, GPT-4V) and ask it to analyze redacted region
  - **Effort:** 2-3 weeks
  - **Packages:** `transformers`, `torch`, `openai` (for GPT-4V)
  - **Implementation:**
    - Load VLM (Qwen2.5-VL-7B or GOT-OCR2)
    - Prompt: "Analyze this document image. There is a black redaction box at coordinates [x,y,w,h]. Describe what you can see at the edges of the redaction and infer the hidden content."
    - Parse VLM response for candidates
    - Cross-reference with character count constraints
  - **Success Rate:** Medium-High (~50-70%), highly model-dependent
  - **Requirements:** 8+ GB VRAM for local models, or cloud API access

- [ ] **Cross-document correlation**
  - **Description:** Search for the same document ID, sender, or thread in other unredacted documents/emails in the same collection
  - **Effort:** 2-3 weeks
  - **Packages:** `whoosh`, `elasticsearch`, or custom indexing
  - **Implementation:**
    - Index all documents in collection by metadata (subject, date, participants, document IDs)
    - Extract unique identifiers from redacted document
    - Search for matching documents with same thread/ID
    - Compare redacted vs unredacted versions
  - **Success Rate:** High (~80-95%) if matching documents exist
  - **Requirements:** Access to full document collection

- [ ] **OmniPage CSDK 21.1 specific weaknesses analysis**
  - **Description:** Research known vulnerabilities or incomplete redaction behaviors in OmniPage Capture SDK 21.1
  - **Effort:** 2-4 weeks (research-heavy)
  - **Packages:** N/A (research + testing)
  - **Implementation:**
    - Obtain OmniPage CSDK 21.1 (legacy version)
    - Create test documents with redactions
    - Analyze output for patterns:
      - Does it use PDF annotations or content stream overlays?
      - Are redactions flattened or removable?
      - Are there metadata leaks?
    - Document reproducible weaknesses
  - **Success Rate:** Unknown, potentially High if specific vulnerabilities found
  - **Requirements:** Access to OmniPage CSDK 21.1, legal/ethical considerations

### Tier 4 — Research-Grade (Months)

These methods require extensive R&D and may be suitable for academic research or advanced projects.

- [ ] **Custom deep learning model for redacted text recovery**
  - **Description:** Train a model on synthetic data (render text, apply synthetic redaction with varying opacity/coverage, train to predict original text)
  - **Effort:** 2-4 months
  - **Packages:** `pytorch`, `transformers`, `datasets`, synthetic data generation
  - **Implementation:**
    - Generate synthetic training data:
      - Render text in various fonts, sizes, colors
      - Apply synthetic redactions (black boxes) with varying opacity (0-100%), coverage (50-100%), and blur
    - Train encoder-decoder model (transformer or CNN-RNN)
    - Train on task: [redacted image] → [original text]
    - Fine-tune on real redaction examples if available
  - **Success Rate:** Unknown, potentially High (~70-90%) depending on training data quality
  - **Requirements:** GPU cluster, large corpus of text samples, 1000+ hours of training

- [ ] **Steganographic analysis**
  - **Description:** Check for hidden data using LSB analysis, frequency domain analysis (FFT/DCT), statistical anomalies
  - **Effort:** 1-3 months (requires deep expertise)
  - **Packages:** `numpy`, `scipy`, `opencv-python`, `pillow`
  - **Implementation:**
    - LSB (Least Significant Bit) analysis: Extract low bits from pixels
    - DCT/FFT analysis: Look for patterns in frequency domain
    - Chi-square test: Detect statistical anomalies
    - Histogram analysis: Look for non-natural distributions
  - **Success Rate:** Very Low (~1-5%), only if steganography was used (unlikely in redactions)
  - **Requirements:** Deep knowledge of steganography and information theory

- [ ] **Print-scan cycle artifact analysis**
  - **Description:** If document was printed then scanned, analyze printer halftone patterns, scanner noise for text ghost artifacts
  - **Effort:** 2-4 months (highly specialized)
  - **Packages:** `opencv-python`, `scipy`, `scikit-image`
  - **Implementation:**
    - Detect halftone patterns (dithering) characteristic of printers
    - Analyze scanner noise characteristics (Gaussian, salt-and-pepper)
    - Look for ghosting artifacts from incomplete print coverage
    - Use frequency domain analysis to isolate halftone patterns
    - Apply deconvolution to recover underlying text
  - **Success Rate:** Very Low (~5-10%), requires specific print-scan chain and incomplete redaction
  - **Requirements:** Expertise in printer forensics, scanner characteristics, signal processing

---

## Section 2: Recommended New OCR Engine Integrations

Prioritized by compatibility with GTX 1050 4GB VRAM constraints. All engines should be integrated as `pdfocr/engines/<name>.py` following existing patterns.

### Priority 1 — Fits GTX 1050 4GB VRAM ✅

These engines can run on a GTX 1050 4GB GPU with full or partial GPU acceleration.

| Engine | Parameters | Quantization | VRAM | Accuracy | CPU Support | Install Command | Target File |
|--------|-----------|--------------|------|----------|-------------|----------------|------------|
| **Qwen2.5-VL-3B Q4** | 3B | Q4_K_M | ~2-2.6 GB | 96% | ✅ Yes | `pip install transformers torch` or use llama.cpp GGUF | `pdfocr/engines/qwen_vl.py` |
| **GOT-OCR2 Q4** | ~1B | Q4 | ~2-3 GB | 90-93% | ✅ Yes | `pip install transformers accelerate bitsandbytes` | `pdfocr/engines/got_ocr.py` |
| **Florence-2-base** | 232M | FP16 | <1 GB | 85-90% | ✅ Yes | `pip install transformers torch` | `pdfocr/engines/florence.py` |

**Implementation Notes:**
- **Qwen2.5-VL-3B Q4:** Best accuracy-to-VRAM ratio. Run via llama.cpp with `--n-gpu-layers ALL` for full GPU offload. Also supports CPU-only mode (slower). Excellent for document understanding and OCR.
- **GOT-OCR2 Q4:** Specialized for OCR with document understanding. Has official CPU support. Good balance of speed and accuracy.
- **Florence-2-base:** Lightweight Microsoft model. Excellent for captioning and OCR. Very fast inference. Good fallback option.

### Priority 2 — CPU-Only on GTX 1050 ⚠️

These engines require 8-12 GB VRAM for GPU acceleration but can run on CPU (slow).

| Engine | Parameters | Quantization | VRAM (GPU) | CPU Support | Inference Time (CPU) | Accuracy | Install Command | Target File |
|--------|-----------|--------------|------------|-------------|---------------------|----------|----------------|------------|
| **Surya OCR** | ~300M | FP16 | 8-12 GB | ✅ Yes | ~5-15 sec/page | 95% | `pip install surya-ocr` | `pdfocr/engines/surya.py` |

**Implementation Notes:**
- **Surya OCR:** High accuracy multilingual OCR. Requires 8-12 GB VRAM for GPU mode, but works on CPU (slow). Add clear CLI warning about CPU-only mode on low-VRAM GPUs. Consider adding `--cpu-only` flag to force CPU mode.

### Priority 3 — Requires GPU Upgrade (>4GB VRAM) 🚫

These engines require more than 4 GB VRAM and won't fit on GTX 1050. Add with GPU detection + warning.

| Engine | Parameters | Quantization | VRAM | Accuracy | Notes | Install Command | Target File |
|--------|-----------|--------------|------|----------|-------|----------------|------------|
| **Qwen2.5-VL-7B Q4** | 7B | Q4 | ~4+ GB | 98% | Too tight for 4GB card, excellent accuracy | `pip install transformers torch` | `pdfocr/engines/qwen_vl_7b.py` |
| **DeepSeek-OCR-3B Q4** | 3B | Q4 | ~3-4 GB | 92-95% | May fit 4GB with careful memory management | `pip install transformers torch` | `pdfocr/engines/deepseek_ocr.py` |
| **LLaVA-1.6-7B Q4** | 7B | Q4 | ~4+ GB | 93% | Risky on 4GB, good multimodal understanding | `pip install transformers torch` | `pdfocr/engines/llava.py` |
| **CogVLM2-Llama3-8B** | 8B | FP16/Q4 | ~8-12 GB | 96% | Definitely too large for 4GB GPU, skip unless GPU upgrade | `pip install transformers torch` | `pdfocr/engines/cogvlm.py` |

**Implementation Notes:**
- Add GPU memory detection in `is_available()` method using `torch.cuda.get_device_properties(0).total_memory`
- Show clear warning message if VRAM is insufficient
- Consider adding `--allow-oom` flag to attempt loading anyway (user accepts risk)

---

## Implementation Guidelines

### For Redaction Recovery Methods:
1. Start with Tier 1 methods (quick wins)
2. Implement one method at a time, test thoroughly on sample redacted documents
3. Document success rate and failure modes for each method
4. Create a unified redaction analysis pipeline that chains multiple methods
5. Add CLI subcommand: `pdfocr redact-analyze <input> --methods <method1,method2,...>`

### For OCR Engine Integrations:
1. Follow existing engine patterns in `pdfocr/engines/base.py`
2. Implement `is_available()` with proper dependency checking and GPU memory detection
3. Add to `pdfocr/engines/__init__.py` and CLI choices
4. Add comprehensive docstrings and type hints (mypy --strict compatible)
5. Handle graceful fallbacks (GPU → CPU, or skip if insufficient resources)
6. Test on PPC_example_data files and document accuracy metrics

### Testing:
- Add tests to `tests/test_pdfocr.py` for each new method/engine
- Use mock objects to avoid requiring optional dependencies in CI
- Document test data requirements (sample redacted PDFs, etc.)

---

## Progress Tracking

Use GitHub Issues and PRs to track progress on individual items. Reference this TODO.md in relevant issues.

**Legend:**
- `- [ ]` Not started
- `- [x]` Completed
- `✅` Implemented/Available
- `⚠️` Warning/Limitation
- `🚫` Not compatible/Skip

---

## References

- **PR #113 (EdgeOfAssembly/RetroCodeMess):** Initial redaction analysis of EFTA01743628.pdf
- **PR #114 (EdgeOfAssembly/RetroCodeMess):** OmniPage CSDK 21.1 vulnerability research
- **PaddlePaddle Issue #59989:** CPU bug in PaddlePaddle 3.x (reference for careful dependency management)

---

**Maintainer Notes:**
- Keep this TODO.md updated as items are completed
- Add new methods/engines as they are discovered or requested
- Link to relevant issues/PRs for each item
- Update "Last Updated" date when making changes
