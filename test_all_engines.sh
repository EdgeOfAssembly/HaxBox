#!/bin/bash
# Test all 6 OCR engines with the example PDF

set -e

cd /home/runner/work/HaxBox/HaxBox
TEST_PDF="test_data/test.pdf"
GROUND_TRUTH="test_data/ground_truth.txt"

echo "=========================================="
echo "Testing All OCR Engines"
echo "=========================================="
echo ""

# Test 1: Tesseract (already installed)
echo "1. Testing tesseract..."
python pdfocr.py "$TEST_PDF" -e tesseract -d test_data/out_tesseract -q --no-enhance 2>&1 | tail -3
if [ -f test_data/out_tesseract/test.txt ]; then
    echo "   ✓ Output created: $(wc -l < test_data/out_tesseract/test.txt) lines"
else
    echo "   ✗ Failed to create output"
fi
echo ""

# Test 2: EasyOCR
echo "2. Testing easyocr..."
pip install easyocr -q 2>&1 | tail -1
python pdfocr.py "$TEST_PDF" -e easyocr -d test_data/out_easyocr -q -p 1 2>&1 | tail -3
if [ -f test_data/out_easyocr/test.txt ]; then
    echo "   ✓ Output created: $(wc -l < test_data/out_easyocr/test.txt) lines"
else
    echo "   ✗ Failed to create output"
fi
echo ""

# Test 3: TrOCR (printed)
echo "3. Testing trocr (printed)..."
pip install transformers torch --index-url https://download.pytorch.org/whl/cpu -q 2>&1 | tail -1
python pdfocr.py "$TEST_PDF" -e trocr -d test_data/out_trocr -q -p 1 2>&1 | tail -5
if [ -f test_data/out_trocr/test.txt ]; then
    echo "   ✓ Output created: $(wc -l < test_data/out_trocr/test.txt) lines"
else
    echo "   ✗ Failed to create output"
fi
echo ""

# Test 4: TrOCR (handwritten)
echo "4. Testing trocr-handwritten..."
python pdfocr.py "$TEST_PDF" -e trocr-handwritten -d test_data/out_trocr_hw -q -p 1 2>&1 | tail -5
if [ -f test_data/out_trocr_hw/test.txt ]; then
    echo "   ✓ Output created: $(wc -l < test_data/out_trocr_hw/test.txt) lines"
else
    echo "   ✗ Failed to create output"
fi
echo ""

# Test 5: PaddleOCR
echo "5. Testing paddleocr..."
pip install paddleocr paddlepaddle -q 2>&1 | tail -1
python pdfocr.py "$TEST_PDF" -e paddleocr -d test_data/out_paddleocr -q -p 1 2>&1 | tail -3
if [ -f test_data/out_paddleocr/test.txt ]; then
    echo "   ✓ Output created: $(wc -l < test_data/out_paddleocr/test.txt) lines"
else
    echo "   ✗ Failed to create output"
fi
echo ""

# Test 6: docTR
echo "6. Testing doctr..."
pip install python-doctr[torch] -q 2>&1 | tail -1
python pdfocr.py "$TEST_PDF" -e doctr -d test_data/out_doctr -q -p 1 2>&1 | tail -3
if [ -f test_data/out_doctr/test.txt ]; then
    echo "   ✓ Output created: $(wc -l < test_data/out_doctr/test.txt) lines"
else
    echo "   ✗ Failed to create output"
fi
echo ""

echo "=========================================="
echo "Test Summary"
echo "=========================================="
ls -lh test_data/out_*/test.txt 2>/dev/null | awk '{print $9, $5}'
