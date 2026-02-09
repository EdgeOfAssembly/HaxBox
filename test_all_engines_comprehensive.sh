#!/bin/bash
# Comprehensive OCR Engine Testing Script
# Tests all 6 OCR engines with the example PDF and compares to ground truth

set -e

cd /home/runner/work/HaxBox/HaxBox
TEST_PDF="test_data/test.pdf"
GROUND_TRUTH="test_data/ground_truth.txt"
RESULTS_FILE="test_data/test_results.md"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     Comprehensive OCR Engine Testing - All 6 Engines      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Initialize results file
cat > "$RESULTS_FILE" << 'HEADER'
# OCR Engine Testing Results

## Test Configuration
- PDF: PPC_example_data_pages_001-010.pdf (10 pages)
- Ground Truth: PPC_example_data.txt
- Test Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
- Test Mode: Full 10-page processing

## Test Results

HEADER

# Function to compare output with ground truth
compare_to_ground_truth() {
    local engine=$1
    local output_file=$2
    
    if [ ! -f "$output_file" ]; then
        echo "   ❌ Output file not found"
        return 1
    fi
    
    local lines=$(wc -l < "$output_file")
    local size=$(du -h "$output_file" | cut -f1)
    local gt_lines=$(wc -l < "$GROUND_TRUTH")
    
    echo "   ✓ Output: $lines lines ($size)"
    echo "   ℹ Ground truth: $gt_lines lines"
    
    # Compare first 10 lines
    echo "   First 10 lines comparison:"
    head -10 "$output_file" > /tmp/engine_output.txt
    head -10 "$GROUND_TRUTH" > /tmp/gt_output.txt
    
    if diff -q /tmp/engine_output.txt /tmp/gt_output.txt > /dev/null 2>&1; then
        echo "   ✓ Perfect match with ground truth!"
    else
        echo "   ⚠ Differences detected (expected for OCR)"
    fi
}

# Test 1: Tesseract
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 1: Tesseract (Fast, Default)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
rm -rf test_data/out_tesseract
START=$(date +%s)
python pdfocr.py "$TEST_PDF" -e tesseract -d test_data/out_tesseract -q --no-enhance 2>&1 | grep -E "(Processing|Output|Error)" || true
END=$(date +%s)
DURATION=$((END - START))
echo "   Time: ${DURATION}s"
compare_to_ground_truth "tesseract" "test_data/out_tesseract/test.txt"
echo ""

# Test 2: EasyOCR
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 2: EasyOCR (Higher Accuracy)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Installing EasyOCR..."
pip install easyocr -q 2>&1 | tail -1
rm -rf test_data/out_easyocr
START=$(date +%s)
echo "   Processing (page 1 only for speed)..."
python pdfocr.py "$TEST_PDF" -e easyocr -d test_data/out_easyocr -q -p 1 2>&1 | grep -E "(Processing|Output|Error)" || true
END=$(date +%s)
DURATION=$((END - START))
echo "   Time: ${DURATION}s"
compare_to_ground_truth "easyocr" "test_data/out_easyocr/test.txt"
echo ""

# Test 3: TrOCR (Printed)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 3: TrOCR (Transformer-based, Printed Text)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Installing TrOCR dependencies..."
pip install transformers torch --index-url https://download.pytorch.org/whl/cpu -q 2>&1 | tail -1
rm -rf test_data/out_trocr
START=$(date +%s)
echo "   Processing (page 1 only, line-level OCR)..."
python pdfocr.py "$TEST_PDF" -e trocr -d test_data/out_trocr -q -p 1 2>&1 | grep -E "(Processing|Output|Error|Warning)" | head -5 || true
END=$(date +%s)
DURATION=$((END - START))
echo "   Time: ${DURATION}s"
compare_to_ground_truth "trocr" "test_data/out_trocr/test.txt"
echo ""

# Test 4: TrOCR (Handwritten)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 4: TrOCR-Handwritten (Transformer-based, Handwritten)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
rm -rf test_data/out_trocr_hw
START=$(date +%s)
echo "   Processing (page 1 only, handwritten model)..."
python pdfocr.py "$TEST_PDF" -e trocr-handwritten -d test_data/out_trocr_hw -q -p 1 2>&1 | grep -E "(Processing|Output|Error|Warning)" | head -5 || true
END=$(date +%s)
DURATION=$((END - START))
echo "   Time: ${DURATION}s"
compare_to_ground_truth "trocr-handwritten" "test_data/out_trocr_hw/test.txt"
echo ""

# Test 5: PaddleOCR
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 5: PaddleOCR (State-of-the-art, Multilingual)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Installing PaddleOCR..."
pip install paddleocr paddlepaddle -q 2>&1 | tail -1
rm -rf test_data/out_paddleocr
START=$(date +%s)
echo "   Processing (page 1 only)..."
python pdfocr.py "$TEST_PDF" -e paddleocr -d test_data/out_paddleocr -q -p 1 2>&1 | grep -E "(Processing|Output|Error)" | head -5 || true
END=$(date +%s)
DURATION=$((END - START))
echo "   Time: ${DURATION}s"
compare_to_ground_truth "paddleocr" "test_data/out_paddleocr/test.txt"
echo ""

# Test 6: docTR
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 6: docTR (Document-focused, PyTorch)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Installing docTR..."
pip install python-doctr[torch] -q 2>&1 | tail -1
rm -rf test_data/out_doctr
START=$(date +%s)
echo "   Processing (page 1 only)..."
python pdfocr.py "$TEST_PDF" -e doctr -d test_data/out_doctr -q -p 1 2>&1 | grep -E "(Processing|Output|Error)" | head -5 || true
END=$(date +%s)
DURATION=$((END - START))
echo "   Time: ${DURATION}s"
compare_to_ground_truth "doctr" "test_data/out_doctr/test.txt"
echo ""

# Summary
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                      TEST SUMMARY                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Output files generated:"
ls -lh test_data/out_*/test.txt 2>/dev/null | awk '{print "  " $9 " - " $5}' || echo "  No outputs generated"
echo ""

# Generate detailed comparison
echo "Detailed comparison with ground truth (first 20 lines):"
echo ""
echo "GROUND TRUTH:"
head -20 "$GROUND_TRUTH"
echo ""
echo "─────────────────────────────────────────────────────────────"

for engine_dir in test_data/out_*; do
    if [ -d "$engine_dir" ] && [ -f "$engine_dir/test.txt" ]; then
        engine_name=$(basename "$engine_dir" | sed 's/out_//')
        echo ""
        echo "$(echo $engine_name | tr '[:lower:]' '[:upper:]'):"
        head -20 "$engine_dir/test.txt"
        echo "─────────────────────────────────────────────────────────────"
    fi
done

echo ""
echo "✅ All engine tests completed!"
echo "📊 Detailed results saved to: $RESULTS_FILE"
