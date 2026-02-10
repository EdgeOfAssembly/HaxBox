# LLM-Ready Codebase Overview — 2026-02-10

**Project:** HaxBox

## Directory Structure

```text
.
├── llm_prep
│   ├── dot_graphs_doxygen
│   ├── dot_graphs_pyreverse
│   └── codebase_structure.txt
├── pdfocr
│   ├── engines
│   │   ├── base.py
│   │   ├── doctr.py
│   │   ├── easyocr.py
│   │   ├── __init__.py
│   │   ├── paddleocr.py
│   │   ├── tesseract.py
│   │   └── trocr.py
│   ├── cli.py
│   ├── core.py
│   ├── __init__.py
│   ├── __main__.py
│   ├── types.py
│   └── utils.py
├── testi
│   └── PPC_example_data_pages_001-010.txt
├── tests
│   ├── conftest.py
│   ├── __init__.py
│   ├── test_dummy.py
│   ├── test_llmprep.py
│   ├── test_pdfocr.py
│   ├── test_pdfsplit.py
│   ├── test_screenrec.py
│   └── test_stegpng.py
├── ARCHITECTURE.md
├── __init__.py
├── llmprep.1
├── llmprep.py
├── pdfocr.1
├── pdfocr_old.py
├── pdfocr.py
├── pdfsplit.1
├── pdfsplit.py
├── PPC_example_data_pages_001-010.pdf
├── PPC_example_data.txt
├── pyproject.toml
├── README.md
├── REFACTOR_VERIFICATION.md
├── requirements.txt
├── screenrec.1
├── screenrec.py
├── stegpng.1
└── stegpng.py

8 directories, 42 files
```

## Code Statistics

```text
github.com/AlDanial/cloc v 2.00  T=0.09 s (409.4 files/s, 166528.5 lines/s)
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
Python                          27           2402           4396           6624
Markdown                         3            300              0            823
Text                             4             93              0            312
YAML                             2             10              0             46
TOML                             1              2              1             41
-------------------------------------------------------------------------------
SUM:                            37           2807           4397           7846
-------------------------------------------------------------------------------
```

## Python Class Diagrams (pyreverse)

- `classes.dot`
- `packages.dot`

## Symbol Index

- `llm_prep/tags` - ctags file for symbol navigation

## LLM Context Files

- `llm_system_prompt.md` - System prompt for LLM sessions
- `project_guidance.md` - Best practices and guidelines

## How to Use

1. Copy this file as initial context for your LLM
2. Paste relevant DOT graphs for architecture questions
3. Reference specific files when asking about code
