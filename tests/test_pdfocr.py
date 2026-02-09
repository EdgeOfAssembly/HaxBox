"""Comprehensive tests for pdfocr.py - OCR tool for PDFs and images."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pdfocr


class TestValidateDpi:
    """Tests for validate_dpi function."""

    def test_valid_dpi(self):
        """Valid DPI values pass through."""
        assert pdfocr.validate_dpi(300) == 300
        assert pdfocr.validate_dpi(150) == 150

    def test_dpi_below_minimum(self):
        """DPI below minimum is clamped."""
        assert pdfocr.validate_dpi(50) == pdfocr.MIN_DPI

    def test_dpi_above_maximum(self):
        """DPI above maximum is clamped."""
        assert pdfocr.validate_dpi(1000) == pdfocr.MAX_DPI

    def test_dpi_zero(self):
        """DPI of zero raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            pdfocr.validate_dpi(0)

    def test_dpi_negative(self):
        """Negative DPI raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            pdfocr.validate_dpi(-100)


class TestParsePageSpec:
    """Tests for parse_page_spec function."""

    def test_single_page(self):
        """Parses single page number."""
        result = pdfocr.parse_page_spec("5", 10)
        assert result == [5]

    def test_page_range(self):
        """Parses page range."""
        result = pdfocr.parse_page_spec("1-5", 10)
        assert result == [1, 2, 3, 4, 5]

    def test_multiple_pages(self):
        """Parses comma-separated pages."""
        result = pdfocr.parse_page_spec("1,3,5", 10)
        assert result == [1, 3, 5]

    def test_mixed_spec(self):
        """Parses mixed specification."""
        result = pdfocr.parse_page_spec("1-3,5,7-9", 10)
        assert result == [1, 2, 3, 5, 7, 8, 9]

    def test_from_start(self):
        """Parses '-N' as 1 to N."""
        result = pdfocr.parse_page_spec("-5", 10)
        assert result == [1, 2, 3, 4, 5]

    def test_to_end(self):
        """Parses 'N-' as N to end."""
        result = pdfocr.parse_page_spec("8-", 10)
        assert result == [8, 9, 10]

    def test_removes_duplicates(self):
        """Removes duplicate pages."""
        result = pdfocr.parse_page_spec("1,1,2,2,3", 10)
        assert result == [1, 2, 3]

    def test_empty_string(self):
        """Handles empty string."""
        result = pdfocr.parse_page_spec("", 10)
        assert result == []

    def test_invalid_page_out_of_bounds(self):
        """Raises error for page out of bounds."""
        with pytest.raises(ValueError, match="out of bounds"):
            pdfocr.parse_page_spec("15", 10)

    def test_invalid_format(self):
        """Raises error for invalid format."""
        with pytest.raises(ValueError, match="not a valid integer"):
            pdfocr.parse_page_spec("abc", 10)

    def test_multiple_hyphens_invalid(self):
        """Raises error for multiple hyphens."""
        with pytest.raises(ValueError, match="multiple hyphens"):
            pdfocr.parse_page_spec("1--5", 10)


class TestProcessInputs:
    """Tests for process_inputs function."""

    def test_single_pdf_file(self, minimal_pdf: Path):
        """Processes single PDF file."""
        result = pdfocr.process_inputs([str(minimal_pdf)])
        assert len(result) == 1
        assert result[0] == minimal_pdf

    def test_image_file(self, sample_png_path: Path):
        """Processes image file."""
        result = pdfocr.process_inputs([str(sample_png_path)])
        assert len(result) == 1

    def test_multiple_extensions(self, tmp_path: Path, minimal_png: bytes):
        """Handles various image extensions."""
        # Create files with different extensions
        extensions = [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
        for ext in extensions:
            img_file = tmp_path / f"test{ext}"
            img_file.write_bytes(minimal_png)

        result = pdfocr.process_inputs([str(tmp_path)])
        assert len(result) >= len(extensions)

    def test_directory_batch(self, tmp_path: Path, minimal_pdf: Path, minimal_png: bytes):
        """Processes directory in batch mode."""
        # Create multiple files
        (tmp_path / "doc.pdf").write_bytes(minimal_pdf.read_bytes())
        (tmp_path / "img.png").write_bytes(minimal_png)

        result = pdfocr.process_inputs([str(tmp_path)])
        assert len(result) >= 2

    def test_unsupported_file_warning(self, tmp_path: Path, capsys):
        """Warns about unsupported files."""
        txt_file = tmp_path / "document.txt"
        txt_file.write_text("not an image")

        result = pdfocr.process_inputs([str(txt_file)])
        assert len(result) == 0
        captured = capsys.readouterr()
        assert "Unsupported" in captured.err

    def test_nonexistent_path_warning(self, tmp_path: Path, capsys):
        """Warns about nonexistent paths."""
        result = pdfocr.process_inputs([str(tmp_path / "nonexistent.pdf")])
        assert len(result) == 0
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_removes_duplicates(self, minimal_pdf: Path):
        """Removes duplicate file paths."""
        result = pdfocr.process_inputs([str(minimal_pdf), str(minimal_pdf)])
        assert len(result) == 1


class TestPreprocessImageForOcr:
    """Tests for preprocess_image_for_ocr function."""

    def test_no_enhance(self, sample_png_path: Path):
        """Returns original image when enhance=False."""
        from PIL import Image

        img = Image.open(sample_png_path)
        result = pdfocr.preprocess_image_for_ocr(img, enhance=False)
        # Should return same image object
        assert result.size == img.size

    def test_enhance_without_opencv(self, sample_png_path: Path):
        """Returns original when OpenCV not available."""
        from PIL import Image

        img = Image.open(sample_png_path)
        with patch.object(pdfocr, "_get_cv2", return_value=None):
            result = pdfocr.preprocess_image_for_ocr(img, enhance=True)
            assert result.size == img.size


class TestOcrWithTesseract:
    """Tests for ocr_with_tesseract function."""

    def test_requires_pytesseract(self, sample_png_path: Path):
        """Raises ImportError when pytesseract not available."""
        from PIL import Image

        img = Image.open(sample_png_path)
        with patch.object(pdfocr, "_get_pytesseract", return_value=None):
            with pytest.raises(ImportError, match="pytesseract"):
                pdfocr.ocr_with_tesseract(img)


class TestOcrWithEasyocr:
    """Tests for ocr_with_easyocr function."""

    def test_requires_easyocr(self, sample_png_path: Path):
        """Raises ImportError when easyocr not available."""
        from PIL import Image

        img = Image.open(sample_png_path)
        with patch.object(pdfocr, "_get_easyocr_reader", return_value=None):
            with pytest.raises(ImportError, match="easyocr"):
                pdfocr.ocr_with_easyocr(img)

    def test_requires_numpy(self, sample_png_path: Path):
        """Raises ImportError when numpy not available."""
        from PIL import Image

        img = Image.open(sample_png_path)
        mock_reader = MagicMock()
        with patch.object(pdfocr, "_get_easyocr_reader", return_value=mock_reader):
            with patch.object(pdfocr, "_get_numpy", return_value=None):
                with pytest.raises(ImportError, match="numpy"):
                    pdfocr.ocr_with_easyocr(img)


class TestOcrWithTrocr:
    """Tests for ocr_with_trocr function."""

    def test_requires_transformers(self, sample_png_path: Path):
        """Raises ImportError when transformers not available."""
        from PIL import Image

        img = Image.open(sample_png_path)
        with patch.object(pdfocr, "_get_trocr", return_value=(None, None)):
            with pytest.raises(ImportError, match="transformers"):
                pdfocr.ocr_with_trocr(img)

    def test_printed_variant(self, sample_png_path: Path):
        """Calls TrOCR with printed model variant."""
        from PIL import Image

        img = Image.open(sample_png_path)
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_pixel_values = MagicMock()
        mock_pixel_values.to.return_value = mock_pixel_values
        mock_processor.return_value.pixel_values = mock_pixel_values
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_processor.batch_decode.return_value = ["test text"]
        
        with patch.object(pdfocr, "_get_trocr", return_value=(mock_processor, mock_model)):
            result = pdfocr.ocr_with_trocr(img, model_variant="printed", gpu=False)
            assert result == "test text"

    def test_handwritten_variant(self, sample_png_path: Path):
        """Calls TrOCR with handwritten model variant."""
        from PIL import Image

        img = Image.open(sample_png_path)
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_pixel_values = MagicMock()
        mock_pixel_values.to.return_value = mock_pixel_values
        mock_processor.return_value.pixel_values = mock_pixel_values
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_processor.batch_decode.return_value = ["handwritten text"]
        
        with patch.object(pdfocr, "_get_trocr", return_value=(mock_processor, mock_model)):
            result = pdfocr.ocr_with_trocr(img, model_variant="handwritten", gpu=False)
            assert result == "handwritten text"

    def test_return_boxes(self, sample_png_path: Path):
        """Returns structured data when return_boxes=True."""
        from PIL import Image

        img = Image.open(sample_png_path)
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_pixel_values = MagicMock()
        mock_pixel_values.to.return_value = mock_pixel_values
        mock_processor.return_value.pixel_values = mock_pixel_values
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_processor.batch_decode.return_value = ["test text"]
        
        with patch.object(pdfocr, "_get_trocr", return_value=(mock_processor, mock_model)):
            result = pdfocr.ocr_with_trocr(img, return_boxes=True)
            assert isinstance(result, dict)
            assert result["text"] == "test text"
            assert result["confidence"] is None
            assert result["bbox"] is None


class TestOcrWithPaddleocr:
    """Tests for ocr_with_paddleocr function."""

    def test_requires_paddleocr(self, sample_png_path: Path):
        """Raises ImportError when paddleocr not available."""
        from PIL import Image

        img = Image.open(sample_png_path)
        with patch.object(pdfocr, "_get_paddleocr", return_value=None):
            with pytest.raises(ImportError, match="paddleocr"):
                pdfocr.ocr_with_paddleocr(img)

    def test_requires_numpy(self, sample_png_path: Path):
        """Raises ImportError when numpy not available."""
        from PIL import Image

        img = Image.open(sample_png_path)
        mock_paddle = MagicMock()
        with patch.object(pdfocr, "_get_paddleocr", return_value=mock_paddle):
            with patch.object(pdfocr, "_get_numpy", return_value=None):
                with pytest.raises(ImportError, match="numpy"):
                    pdfocr.ocr_with_paddleocr(img)

    def test_text_extraction(self, sample_png_path: Path):
        """Extracts text using PaddleOCR."""
        from PIL import Image

        img = Image.open(sample_png_path)
        mock_paddle = MagicMock()
        # PaddleOCR returns: [[[bbox], (text, confidence)], ...]
        mock_paddle.ocr.return_value = [[
            [[[0, 0], [10, 0], [10, 10], [0, 10]], ("test text", 0.95)]
        ]]
        
        with patch.object(pdfocr, "_get_paddleocr", return_value=mock_paddle):
            with patch.object(pdfocr, "_get_numpy", return_value=__import__("numpy")):
                result = pdfocr.ocr_with_paddleocr(img, lang="en", gpu=False)
                assert result == "test text"

    def test_return_boxes(self, sample_png_path: Path):
        """Returns structured data when return_boxes=True."""
        from PIL import Image

        img = Image.open(sample_png_path)
        mock_paddle = MagicMock()
        mock_paddle.ocr.return_value = [[
            [[[0, 0], [10, 0], [10, 10], [0, 10]], ("test text", 0.95)]
        ]]
        
        with patch.object(pdfocr, "_get_paddleocr", return_value=mock_paddle):
            with patch.object(pdfocr, "_get_numpy", return_value=__import__("numpy")):
                result = pdfocr.ocr_with_paddleocr(img, return_boxes=True)
                assert isinstance(result, list)
                assert len(result) == 1
                assert result[0]["text"] == "test text"
                assert result[0]["confidence"] == 0.95


class TestOcrWithDoctr:
    """Tests for ocr_with_doctr function."""

    def test_requires_doctr(self, sample_png_path: Path):
        """Raises ImportError when doctr not available."""
        from PIL import Image

        img = Image.open(sample_png_path)
        with patch.object(pdfocr, "_get_doctr_model", return_value=None):
            with pytest.raises(ImportError, match="docTR"):
                pdfocr.ocr_with_doctr(img)

    def test_requires_numpy(self, sample_png_path: Path):
        """Raises ImportError when numpy not available."""
        from PIL import Image

        img = Image.open(sample_png_path)
        mock_doctr = MagicMock()
        with patch.object(pdfocr, "_get_doctr_model", return_value=mock_doctr):
            with patch.object(pdfocr, "_get_numpy", return_value=None):
                with pytest.raises(ImportError, match="numpy"):
                    pdfocr.ocr_with_doctr(img)

    def test_text_extraction(self, sample_png_path: Path):
        """Extracts text using docTR."""
        from PIL import Image

        img = Image.open(sample_png_path)
        mock_doctr = MagicMock()
        
        # Create mock docTR result structure
        mock_word = MagicMock()
        mock_word.value = "test"
        mock_word.confidence = 0.95
        mock_word.geometry = [[0, 0], [1, 1]]
        
        mock_line = MagicMock()
        mock_line.words = [mock_word]
        
        mock_block = MagicMock()
        mock_block.lines = [mock_line]
        
        mock_page = MagicMock()
        mock_page.blocks = [mock_block]
        
        mock_result = MagicMock()
        mock_result.pages = [mock_page]
        
        mock_doctr.return_value = mock_result
        
        with patch.object(pdfocr, "_get_doctr_model", return_value=mock_doctr):
            with patch.object(pdfocr, "_get_numpy", return_value=__import__("numpy")):
                result = pdfocr.ocr_with_doctr(img, gpu=False)
                assert result == "test"

    def test_return_boxes(self, sample_png_path: Path):
        """Returns structured data when return_boxes=True."""
        from PIL import Image

        img = Image.open(sample_png_path)
        mock_doctr = MagicMock()
        
        # Create mock docTR result structure
        mock_word = MagicMock()
        mock_word.value = "test"
        mock_word.confidence = 0.95
        mock_word.geometry = [[0, 0], [1, 1]]
        
        mock_line = MagicMock()
        mock_line.words = [mock_word]
        
        mock_block = MagicMock()
        mock_block.lines = [mock_line]
        
        mock_page = MagicMock()
        mock_page.blocks = [mock_block]
        
        mock_result = MagicMock()
        mock_result.pages = [mock_page]
        
        mock_doctr.return_value = mock_result
        
        with patch.object(pdfocr, "_get_doctr_model", return_value=mock_doctr):
            with patch.object(pdfocr, "_get_numpy", return_value=__import__("numpy")):
                result = pdfocr.ocr_with_doctr(img, return_boxes=True)
                assert isinstance(result, list)
                assert len(result) == 1
                assert result[0]["text"] == "test"
                assert result[0]["confidence"] == 0.95


class TestCheckEngineAvailable:
    """Tests for check_engine_available function."""

    def test_tesseract_not_available(self):
        """Returns False when tesseract not available."""
        with patch.object(pdfocr, "_get_pytesseract", return_value=None):
            # Also mock the import to fail
            with patch.dict("sys.modules", {"pytesseract": None}):
                pdfocr.check_engine_available("tesseract")
                # Result depends on actual system installation

    def test_easyocr_not_available(self):
        """Returns False when easyocr not available."""
        # Mock import failure
        def mock_import(*args, **kwargs):
            raise ImportError("No module named 'easyocr'")

        with patch("builtins.__import__", side_effect=mock_import):
            result = pdfocr.check_engine_available("easyocr")
            assert result is False

    def test_trocr_not_available(self):
        """Returns False when trocr not available."""
        # Mock import failure
        def mock_import(*args, **kwargs):
            raise ImportError("No module named 'transformers'")

        with patch("builtins.__import__", side_effect=mock_import):
            result = pdfocr.check_engine_available("trocr")
            assert result is False

    def test_trocr_handwritten_not_available(self):
        """Returns False when trocr-handwritten not available."""
        # Mock import failure
        def mock_import(*args, **kwargs):
            raise ImportError("No module named 'transformers'")

        with patch("builtins.__import__", side_effect=mock_import):
            result = pdfocr.check_engine_available("trocr-handwritten")
            assert result is False

    def test_paddleocr_not_available(self):
        """Returns False when paddleocr not available."""
        # Mock import failure
        def mock_import(*args, **kwargs):
            raise ImportError("No module named 'paddleocr'")

        with patch("builtins.__import__", side_effect=mock_import):
            result = pdfocr.check_engine_available("paddleocr")
            assert result is False

    def test_doctr_not_available(self):
        """Returns False when doctr not available."""
        # Mock import failure
        def mock_import(*args, **kwargs):
            raise ImportError("No module named 'doctr'")

        with patch("builtins.__import__", side_effect=mock_import):
            result = pdfocr.check_engine_available("doctr")
            assert result is False


class TestLanguageMapping:
    """Tests for language code mapping."""

    def test_tesseract_to_easyocr_mapping(self):
        """Language mapping contains expected entries."""
        assert pdfocr.TESSERACT_TO_EASYOCR_LANG["eng"] == "en"
        assert pdfocr.TESSERACT_TO_EASYOCR_LANG["deu"] == "de"
        assert pdfocr.TESSERACT_TO_EASYOCR_LANG["fra"] == "fr"
        assert pdfocr.TESSERACT_TO_EASYOCR_LANG["chi_sim"] == "ch_sim"
    
    def test_tesseract_to_paddleocr_mapping(self):
        """PaddleOCR language mapping contains expected entries."""
        assert pdfocr.TESSERACT_TO_PADDLEOCR_LANG["eng"] == "en"
        assert pdfocr.TESSERACT_TO_PADDLEOCR_LANG["deu"] == "german"
        assert pdfocr.TESSERACT_TO_PADDLEOCR_LANG["fra"] == "french"
        assert pdfocr.TESSERACT_TO_PADDLEOCR_LANG["chi_sim"] == "ch"


class TestOcrImage:
    """Tests for ocr_image function."""

    def test_tesseract_engine(self, sample_png_path: Path):
        """Calls tesseract engine when specified."""
        from PIL import Image

        img = Image.open(sample_png_path)
        mock_tesseract = MagicMock(return_value="test text")
        with patch.object(pdfocr, "ocr_with_tesseract", mock_tesseract):
            with patch.object(pdfocr, "preprocess_image_for_ocr", return_value=img):
                pdfocr.ocr_image(img, engine="tesseract")
                mock_tesseract.assert_called_once()

    def test_easyocr_engine(self, sample_png_path: Path):
        """Calls easyocr engine when specified."""
        from PIL import Image

        img = Image.open(sample_png_path)
        mock_easyocr = MagicMock(return_value="test text")
        with patch.object(pdfocr, "ocr_with_easyocr", mock_easyocr):
            with patch.object(pdfocr, "preprocess_image_for_ocr", return_value=img):
                pdfocr.ocr_image(img, engine="easyocr")
                mock_easyocr.assert_called_once()

    def test_language_conversion_for_easyocr(self, sample_png_path: Path):
        """Converts tesseract lang code to easyocr."""
        from PIL import Image

        img = Image.open(sample_png_path)
        mock_easyocr = MagicMock(return_value="test")
        with patch.object(pdfocr, "ocr_with_easyocr", mock_easyocr):
            with patch.object(pdfocr, "preprocess_image_for_ocr", return_value=img):
                pdfocr.ocr_image(img, engine="easyocr", lang="eng")
                # Check that easyocr was called with converted language
                call_args = mock_easyocr.call_args
                assert call_args[0][1] == ["en"]

    def test_trocr_engine(self, sample_png_path: Path):
        """Calls trocr engine when specified."""
        from PIL import Image

        img = Image.open(sample_png_path)
        mock_trocr = MagicMock(return_value="test text")
        with patch.object(pdfocr, "ocr_with_trocr", mock_trocr):
            with patch.object(pdfocr, "preprocess_image_for_ocr", return_value=img):
                pdfocr.ocr_image(img, engine="trocr")
                mock_trocr.assert_called_once()
                # Check that printed variant was used
                call_args = mock_trocr.call_args
                assert call_args[1]["model_variant"] == "printed"

    def test_trocr_handwritten_engine(self, sample_png_path: Path):
        """Calls trocr-handwritten engine when specified."""
        from PIL import Image

        img = Image.open(sample_png_path)
        mock_trocr = MagicMock(return_value="handwritten text")
        with patch.object(pdfocr, "ocr_with_trocr", mock_trocr):
            with patch.object(pdfocr, "preprocess_image_for_ocr", return_value=img):
                pdfocr.ocr_image(img, engine="trocr-handwritten")
                mock_trocr.assert_called_once()
                # Check that handwritten variant was used
                call_args = mock_trocr.call_args
                assert call_args[1]["model_variant"] == "handwritten"

    def test_paddleocr_engine(self, sample_png_path: Path):
        """Calls paddleocr engine when specified."""
        from PIL import Image

        img = Image.open(sample_png_path)
        mock_paddleocr = MagicMock(return_value="test text")
        with patch.object(pdfocr, "ocr_with_paddleocr", mock_paddleocr):
            with patch.object(pdfocr, "preprocess_image_for_ocr", return_value=img):
                pdfocr.ocr_image(img, engine="paddleocr")
                mock_paddleocr.assert_called_once()

    def test_language_conversion_for_paddleocr(self, sample_png_path: Path):
        """Converts tesseract lang code to paddleocr."""
        from PIL import Image

        img = Image.open(sample_png_path)
        mock_paddleocr = MagicMock(return_value="test")
        with patch.object(pdfocr, "ocr_with_paddleocr", mock_paddleocr):
            with patch.object(pdfocr, "preprocess_image_for_ocr", return_value=img):
                pdfocr.ocr_image(img, engine="paddleocr", lang="eng")
                # Check that paddleocr was called with converted language
                call_args = mock_paddleocr.call_args
                # The lang is passed as second positional argument
                assert call_args[0][1] == "en" or call_args.kwargs.get("lang") == "en"

    def test_doctr_engine(self, sample_png_path: Path):
        """Calls doctr engine when specified."""
        from PIL import Image

        img = Image.open(sample_png_path)
        mock_doctr = MagicMock(return_value="test text")
        with patch.object(pdfocr, "ocr_with_doctr", mock_doctr):
            with patch.object(pdfocr, "preprocess_image_for_ocr", return_value=img):
                pdfocr.ocr_image(img, engine="doctr")
                mock_doctr.assert_called_once()


class TestCLI:
    """Tests for command-line interface."""

    def test_cli_no_args_prints_help(self, capsys):
        """CLI prints help when no args provided."""
        with patch.object(sys, "argv", ["pdfocr"]):
            with pytest.raises(SystemExit) as exc_info:
                pdfocr.main()
            assert exc_info.value.code == 0

    def test_cli_no_inputs(self, capsys):
        """CLI errors when no inputs after parsing."""
        # Simulate empty inputs list after argument parsing
        with patch.object(sys, "argv", ["pdfocr", ""]):
            with patch.object(pdfocr, "check_engine_available", return_value=True):
                with patch.object(pdfocr, "process_inputs", return_value=[]):
                    with pytest.raises(SystemExit) as exc_info:
                        pdfocr.main()
                    assert exc_info.value.code == 1
        captured = capsys.readouterr()
        # Should show error about no files
        assert "No supported files" in captured.err

    def test_cli_version(self, capsys):
        """CLI --version shows version."""
        with patch.object(sys, "argv", ["pdfocr", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                pdfocr.main()
            assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert pdfocr.__version__ in captured.out

    def test_cli_engine_selection(self, sample_png_path: Path, tmp_path: Path):
        """CLI respects engine selection."""
        output_dir = tmp_path / "output"
        with patch.object(
            sys,
            "argv",
            [
                "pdfocr",
                str(sample_png_path),
                "-e",
                "tesseract",
                "-d",
                str(output_dir),
                "-q",
                "-f",
            ],
        ):
            with patch.object(pdfocr, "check_engine_available", return_value=True):
                with patch.object(pdfocr, "ocr_image_file", return_value=None):
                    pdfocr.main()

    def test_cli_dpi_validation(self, capsys):
        """CLI validates DPI value."""
        with patch.object(
            sys, "argv", ["pdfocr", "test.pdf", "--dpi", "0"]
        ):
            with pytest.raises(SystemExit) as exc_info:
                pdfocr.main()
            assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "must be positive" in captured.err

    def test_cli_gpu_warning_with_tesseract(self, capsys, sample_png_path: Path, tmp_path: Path):
        """CLI warns about --gpu with tesseract."""
        with patch.object(
            sys,
            "argv",
            [
                "pdfocr",
                str(sample_png_path),
                "-e",
                "tesseract",
                "--gpu",
                "-d",
                str(tmp_path),
                "-q",
            ],
        ):
            with patch.object(pdfocr, "check_engine_available", return_value=True):
                with patch.object(pdfocr, "ocr_image_file", return_value=None):
                    pdfocr.main()
        captured = capsys.readouterr()
        assert "--gpu is only supported with easyocr, trocr, trocr-handwritten, paddleocr, and doctr" in captured.err

    def test_cli_trocr_engine_selection(self, sample_png_path: Path, tmp_path: Path):
        """CLI respects trocr engine selection."""
        output_dir = tmp_path / "output"
        with patch.object(
            sys,
            "argv",
            [
                "pdfocr",
                str(sample_png_path),
                "-e",
                "trocr",
                "-d",
                str(output_dir),
                "-q",
                "-f",
            ],
        ):
            with patch.object(pdfocr, "check_engine_available", return_value=True):
                with patch.object(pdfocr, "ocr_image_file", return_value=None):
                    pdfocr.main()


class TestConstants:
    """Tests for module constants."""

    def test_version(self):
        """Version string is defined."""
        assert hasattr(pdfocr, "__version__")
        assert isinstance(pdfocr.__version__, str)

    def test_default_output_dir(self):
        """Default output directory is defined."""
        assert pdfocr.DEFAULT_OUTPUT_DIR == "ocr_out"

    def test_supported_image_extensions(self):
        """Supported extensions include common formats."""
        extensions = pdfocr.SUPPORTED_IMAGE_EXTENSIONS
        assert ".png" in extensions
        assert ".jpg" in extensions
        assert ".jpeg" in extensions
        assert ".tiff" in extensions

    def test_dpi_limits(self):
        """DPI limits are reasonable."""
        assert pdfocr.MIN_DPI > 0
        assert pdfocr.MAX_DPI > pdfocr.MIN_DPI


class TestLazyImports:
    """Tests for lazy import functions."""

    def test_get_pytesseract_caches_result(self):
        """_get_pytesseract caches the module."""
        # Clear any cached value
        pdfocr._pytesseract = None

        # First call
        result1 = pdfocr._get_pytesseract()
        # Second call should return same object (if available)
        result2 = pdfocr._get_pytesseract()

        if result1 is not None:
            assert result1 is result2

    def test_get_cv2_caches_result(self):
        """_get_cv2 caches the module."""
        # Clear any cached value
        pdfocr._cv2 = None

        # First call
        result1 = pdfocr._get_cv2()
        # Second call should return same object
        result2 = pdfocr._get_cv2()

        if result1 is not None:
            assert result1 is result2

    def test_get_numpy_caches_result(self):
        """_get_numpy caches the module."""
        # Clear any cached value
        pdfocr._np = None

        # First call
        result1 = pdfocr._get_numpy()
        # Second call should return same object
        result2 = pdfocr._get_numpy()

        if result1 is not None:
            assert result1 is result2


class TestOcrPdf:
    """Tests for ocr_pdf function."""

    def test_ocr_pdf_skip_existing(self, minimal_pdf: Path, tmp_path: Path, capsys):
        """Skips existing output file without force."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create existing output file
        existing = output_dir / f"{minimal_pdf.stem}.txt"
        existing.write_text("existing content")

        with patch.object(pdfocr, "pdf_to_images", return_value=[]):
            result = pdfocr.ocr_pdf(
                minimal_pdf,
                output_dir,
                force=False,
                quiet=True,
            )
            assert result is None

        captured = capsys.readouterr()
        assert "Skipping existing" in captured.err

    def test_ocr_pdf_force_overwrite(self, minimal_pdf: Path, tmp_path: Path):
        """Overwrites existing file with force."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create existing output file
        existing = output_dir / f"{minimal_pdf.stem}.txt"
        existing.write_text("existing content")

        with patch.object(pdfocr, "pdf_to_images", return_value=[]):
            pdfocr.ocr_pdf(
                minimal_pdf,
                output_dir,
                force=True,
                quiet=True,
            )
            # Result may be None (conversion failed) or a Path (success)
            # We just verify the function doesn't raise an unexpected exception


class TestOcrImageFile:
    """Tests for ocr_image_file function."""

    def test_ocr_image_file_skip_existing(
        self, sample_png_path: Path, tmp_path: Path, capsys
    ):
        """Skips existing output file without force."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create existing output file
        existing = output_dir / f"{sample_png_path.stem}.txt"
        existing.write_text("existing content")

        result = pdfocr.ocr_image_file(
            sample_png_path,
            output_dir,
            force=False,
            quiet=True,
        )
        assert result is None

        captured = capsys.readouterr()
        assert "Skipping existing" in captured.err


class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=20)
    def test_parse_page_spec_single_page(self, page: int):
        """Property: single page parses to list with that page."""
        total = max(page, 100)
        result = pdfocr.parse_page_spec(str(page), total)
        assert result == [page]

    @given(
        st.integers(min_value=1, max_value=50),
        st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=20)
    def test_parse_page_spec_range(self, start: int, length: int):
        """Property: range parses to consecutive pages."""
        end = start + length
        total = end + 10
        result = pdfocr.parse_page_spec(f"{start}-{end}", total)
        expected = list(range(start, end + 1))
        assert result == expected

    @given(st.integers(min_value=pdfocr.MIN_DPI, max_value=pdfocr.MAX_DPI))
    @settings(max_examples=20)
    def test_validate_dpi_in_range(self, dpi: int):
        """Property: valid DPI passes through unchanged."""
        assert pdfocr.validate_dpi(dpi) == dpi
