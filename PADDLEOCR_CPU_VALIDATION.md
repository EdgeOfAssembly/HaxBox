# PaddleOCR 3.0+ CPU Validation Results

## Summary

This document summarizes the validation testing of PaddleOCR 3.0+ in CPU mode on the HaxBox repository test data.

## Test Configuration

- **PaddleOCR Version:** 3.4.0
- **PaddlePaddle Version:** 3.3.0
- **Test Platform:** CPU only (no GPU)
- **Test Data:** PPC_example_data_pages_001-010.pdf (10 pages)
- **Ground Truth:** PPC_example_data.txt

## Test Results

**Status:** ❌ FAILED

All 10 pages failed OCR with the same error:

```
(Unimplemented) ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]
  (at /paddle/paddle/fluid/framework/new_executor/instruction/onednn/onednn_instruction.cc:116)
```

## Error Analysis

### Root Cause

This is a known bug in PaddlePaddle 3.0+ related to PIR (Paddle Intermediate Representation) CPU execution. The error occurs when PaddlePaddle attempts to convert PIR attributes to runtime attributes in OneDNN (Intel's Deep Neural Network Library, formerly MKL-DNN) operations.

### Related Issues

- **GitHub Issue:** PaddlePaddle/Paddle#59989
- **Error Location:** `paddle/fluid/framework/new_executor/instruction/onednn/onednn_instruction.cc:116`
- **Component:** OneDNN instruction converter

### Environment Variable Workarounds Attempted

The following environment variables were set before importing PaddleOCR (as recommended in various workarounds):

```python
os.environ['FLAGS_use_mkldnn'] = 'False'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_use_onednn'] = '0'
os.environ['PADDLE_USE_ONEDNN'] = '0'
os.environ['FLAGS_use_cudnn'] = '0'
os.environ['CPU_NUM'] = '1'
```

**Result:** None of these workarounds resolved the issue. The error persists regardless of the environment variable configuration.

## PaddleOCR Initialization

PaddleOCR was successfully initialized with the following configuration:

```python
ocr = PaddleOCR(
    lang='en',
    device='cpu',
    text_recognition_batch_size=1,
    use_textline_orientation=True,
)
```

The initialization downloads all required models successfully:
- PP-LCNet_x1_0_doc_ori (6.87 MB)
- UVDoc (32.3 MB)
- PP-LCNet_x1_0_textline_ori (6.86 MB)
- PP-OCRv5_server_det (88.4 MB)
- en_PP-OCRv5_mobile_rec (8.01 MB)

However, the `predict()` method fails on every image with the PIR conversion error.

## Conclusion

**PaddleOCR 3.0+ (specifically version 3.4.0) with PaddlePaddle 3.3.0 is NOT functional in CPU mode** due to the PIR/OneDNN conversion bug. The error occurs consistently across all test pages.

### Recommendations

1. **Do NOT proceed with modifying `pdfocr/engines/paddleocr.py`** until PaddlePaddle fixes this CPU execution bug
2. **Consider downgrading to PaddleOCR 2.x** if CPU support is required
3. **Test with GPU mode** to determine if the issue is CPU-specific
4. **Monitor PaddlePaddle/Paddle#59989** for updates on the fix

### Next Steps

- Wait for PaddlePaddle team to fix issue #59989
- Test with alternative OCR engines (EasyOCR, Tesseract, docTR, TrOCR) which are known to work
- Consider using PaddleOCR only in GPU mode if GPU is available

## Test Script

The validation script `validate_paddleocr_cpu.py` can be run from the repository root:

```bash
python validate_paddleocr_cpu.py
```

This script:
1. Converts the PDF to images using pdf2image
2. Initializes PaddleOCR in CPU mode with environment workarounds
3. Attempts OCR on all 10 pages
4. Compares results to ground truth using sequence similarity
5. Reports success/failure with detailed error logs

## Dependencies

The following dependencies were added to `requirements.txt`:

```
paddleocr>=3.0.0
paddlepaddle>=3.0.0
```

Additionally, the system requires:
- `poppler-utils` for PDF to image conversion (pdf2image dependency)

Install with:
```bash
pip install -r requirements.txt
sudo apt-get install poppler-utils
```
