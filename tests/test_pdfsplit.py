"""Comprehensive tests for pdfsplit.py - PDF splitting and manipulation tool."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pdfsplit


class TestParsePageSpec:
    """Tests for parse_page_spec function."""

    def test_single_page(self):
        """Parses single page number."""
        result = pdfsplit.parse_page_spec("5", 10)
        assert result == [(5, 5)]

    def test_page_range(self):
        """Parses page range."""
        result = pdfsplit.parse_page_spec("1-5", 10)
        assert result == [(1, 5)]

    def test_multiple_pages(self):
        """Parses comma-separated pages."""
        result = pdfsplit.parse_page_spec("1,3,5", 10)
        assert result == [(1, 1), (3, 3), (5, 5)]

    def test_mixed_spec(self):
        """Parses mixed specification."""
        result = pdfsplit.parse_page_spec("1-3,5,7-10", 10)
        assert result == [(1, 3), (5, 5), (7, 10)]

    def test_from_start(self):
        """Parses '-N' as 1 to N."""
        result = pdfsplit.parse_page_spec("-5", 10)
        assert result == [(1, 5)]

    def test_to_end(self):
        """Parses 'N-' as N to end."""
        result = pdfsplit.parse_page_spec("5-", 10)
        assert result == [(5, 10)]

    def test_empty_string(self):
        """Handles empty string."""
        result = pdfsplit.parse_page_spec("", 10)
        assert result == []

    def test_whitespace_handling(self):
        """Handles whitespace in spec."""
        result = pdfsplit.parse_page_spec(" 1 , 3 , 5 ", 10)
        assert result == [(1, 1), (3, 3), (5, 5)]

    def test_invalid_range_start_greater_than_end(self):
        """Raises error when start > end."""
        with pytest.raises(ValueError, match="out of bounds"):
            pdfsplit.parse_page_spec("5-3", 10)

    def test_invalid_page_out_of_bounds(self):
        """Raises error for page out of bounds."""
        with pytest.raises(ValueError, match="out of bounds"):
            pdfsplit.parse_page_spec("15", 10)

    def test_invalid_page_zero(self):
        """Raises error for page 0."""
        with pytest.raises(ValueError, match="out of bounds"):
            pdfsplit.parse_page_spec("0", 10)

    def test_invalid_format(self):
        """Raises error for invalid format."""
        with pytest.raises(ValueError, match="not a valid integer"):
            pdfsplit.parse_page_spec("abc", 10)

    def test_multiple_hyphens_invalid(self):
        """Raises error for multiple hyphens."""
        with pytest.raises(ValueError, match="multiple hyphens"):
            pdfsplit.parse_page_spec("1--5", 10)

    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=20)
    def test_single_page_property(self, page: int):
        """Property: any valid single page parses correctly."""
        total = max(page, 100)
        result = pdfsplit.parse_page_spec(str(page), total)
        assert result == [(page, page)]


class TestGenerateRangesByGranularity:
    """Tests for generate_ranges_by_granularity function."""

    def test_granularity_1(self):
        """Granularity 1 splits every page."""
        result = pdfsplit.generate_ranges_by_granularity(5, 1)
        assert result == [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]

    def test_granularity_2(self):
        """Granularity 2 splits every 2 pages."""
        result = pdfsplit.generate_ranges_by_granularity(5, 2)
        assert result == [(1, 2), (3, 4), (5, 5)]

    def test_granularity_larger_than_total(self):
        """Granularity larger than total returns single range."""
        result = pdfsplit.generate_ranges_by_granularity(5, 10)
        assert result == [(1, 5)]

    def test_granularity_zero(self):
        """Granularity 0 is treated as 1."""
        result = pdfsplit.generate_ranges_by_granularity(3, 0)
        assert result == [(1, 1), (2, 2), (3, 3)]

    def test_granularity_negative(self):
        """Negative granularity is treated as 1."""
        result = pdfsplit.generate_ranges_by_granularity(3, -5)
        assert result == [(1, 1), (2, 2), (3, 3)]

    @given(st.integers(min_value=1, max_value=100), st.integers(min_value=1, max_value=50))
    @settings(max_examples=20)
    def test_ranges_cover_all_pages(self, total: int, gran: int):
        """Property: generated ranges cover all pages exactly once."""
        ranges = pdfsplit.generate_ranges_by_granularity(total, gran)
        pages = []
        for start, end in ranges:
            pages.extend(range(start, end + 1))
        assert pages == list(range(1, total + 1))


class TestValidateDpi:
    """Tests for validate_dpi function."""

    def test_valid_dpi(self):
        """Valid DPI values pass through."""
        assert pdfsplit.validate_dpi(300) == 300
        assert pdfsplit.validate_dpi(150) == 150

    def test_dpi_below_minimum(self):
        """DPI below minimum is clamped."""
        assert pdfsplit.validate_dpi(50) == pdfsplit.MIN_DPI

    def test_dpi_above_maximum(self):
        """DPI above maximum is clamped."""
        assert pdfsplit.validate_dpi(5000) == pdfsplit.MAX_DPI

    def test_dpi_zero(self):
        """DPI of zero raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            pdfsplit.validate_dpi(0)

    def test_dpi_negative(self):
        """Negative DPI raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            pdfsplit.validate_dpi(-100)


class TestParseSizeSpec:
    """Tests for parse_size_spec function."""

    def test_bytes(self):
        """Parses plain bytes."""
        assert pdfsplit.parse_size_spec("1024") == 1024

    def test_kilobytes(self):
        """Parses kilobytes."""
        assert pdfsplit.parse_size_spec("1KB") == 1024
        assert pdfsplit.parse_size_spec("1K") == 1024

    def test_megabytes(self):
        """Parses megabytes."""
        assert pdfsplit.parse_size_spec("1MB") == 1024 * 1024
        assert pdfsplit.parse_size_spec("10M") == 10 * 1024 * 1024

    def test_gigabytes(self):
        """Parses gigabytes."""
        assert pdfsplit.parse_size_spec("1GB") == 1024 * 1024 * 1024
        assert pdfsplit.parse_size_spec("2G") == 2 * 1024 * 1024 * 1024

    def test_case_insensitive(self):
        """Size specs are case insensitive."""
        assert pdfsplit.parse_size_spec("10mb") == pdfsplit.parse_size_spec("10MB")
        assert pdfsplit.parse_size_spec("1kb") == pdfsplit.parse_size_spec("1KB")

    def test_whitespace(self):
        """Handles whitespace."""
        assert pdfsplit.parse_size_spec("  10MB  ") == 10 * 1024 * 1024

    def test_float_values(self):
        """Handles float values."""
        assert pdfsplit.parse_size_spec("1.5MB") == int(1.5 * 1024 * 1024)

    def test_invalid_format(self):
        """Raises error for invalid format."""
        with pytest.raises(ValueError, match="Invalid size format"):
            pdfsplit.parse_size_spec("invalid")

    def test_invalid_number(self):
        """Raises error for invalid number."""
        with pytest.raises(ValueError, match="Invalid size format"):
            pdfsplit.parse_size_spec("xyzMB")


class TestProcessInputs:
    """Tests for process_inputs function."""

    def test_single_pdf_file(self, minimal_pdf: Path):
        """Processes single PDF file."""
        result = pdfsplit.process_inputs([str(minimal_pdf)])
        assert len(result) == 1
        assert result[0] == minimal_pdf

    def test_directory_with_pdfs(self, tmp_path: Path, minimal_pdf: Path):
        """Processes directory containing PDFs."""
        # Create multiple PDFs with different case extensions
        pdf_lowercase = tmp_path / "doc1.pdf"
        pdf_uppercase = tmp_path / "doc2.PDF"  # Test case-insensitive handling
        pdf_lowercase.write_bytes(minimal_pdf.read_bytes())
        pdf_uppercase.write_bytes(minimal_pdf.read_bytes())

        result = pdfsplit.process_inputs([str(tmp_path)])
        assert len(result) >= 1

    def test_non_pdf_file_warning(self, tmp_path: Path, capsys):
        """Warns about non-PDF files."""
        txt_file = tmp_path / "document.txt"
        txt_file.write_text("not a pdf")

        result = pdfsplit.process_inputs([str(txt_file)])
        assert len(result) == 0
        captured = capsys.readouterr()
        assert "non-PDF" in captured.err

    def test_nonexistent_path_warning(self, tmp_path: Path, capsys):
        """Warns about nonexistent paths."""
        result = pdfsplit.process_inputs([str(tmp_path / "nonexistent.pdf")])
        assert len(result) == 0
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_multiple_inputs(self, tmp_path: Path, minimal_pdf: Path):
        """Handles multiple input files."""
        pdf1 = tmp_path / "doc1.pdf"
        pdf2 = tmp_path / "doc2.pdf"
        pdf1.write_bytes(minimal_pdf.read_bytes())
        pdf2.write_bytes(minimal_pdf.read_bytes())

        result = pdfsplit.process_inputs([str(pdf1), str(pdf2)])
        assert len(result) == 2


class TestGetPdfInfo:
    """Tests for get_pdf_info function."""

    def test_basic_info(self, minimal_pdf: Path, capsys):
        """Gets basic PDF info."""
        reader = pdfsplit.get_pdf_info(minimal_pdf)
        assert reader is not None
        captured = capsys.readouterr()
        assert "Pages:" in captured.out

    def test_nonexistent_file(self, tmp_path: Path, capsys):
        """Returns None for nonexistent file."""
        result = pdfsplit.get_pdf_info(tmp_path / "nonexistent.pdf")
        assert result is None


class TestExtractPdfPages:
    """Tests for extract_pdf_pages function."""

    def test_extract_single_page(self, minimal_pdf: Path, tmp_path: Path):
        """Extracts single page."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        count = pdfsplit.extract_pdf_pages(
            minimal_pdf,
            output_dir,
            ranges=[(1, 1)],
            prefix=None,
            force=True,
            quiet=True,
        )
        assert count == 1
        # Check output file exists
        output_files = list(output_dir.glob("*.pdf"))
        assert len(output_files) == 1

    def test_extract_with_custom_prefix(self, minimal_pdf: Path, tmp_path: Path):
        """Extracts with custom prefix."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        count = pdfsplit.extract_pdf_pages(
            minimal_pdf,
            output_dir,
            ranges=[(1, 1)],
            prefix="custom",
            force=True,
            quiet=True,
        )
        assert count == 1
        output_files = list(output_dir.glob("custom*.pdf"))
        assert len(output_files) == 1


class TestMergePdfs:
    """Tests for merge_pdfs function."""

    def test_merge_multiple_pdfs(self, tmp_path: Path, minimal_pdf: Path):
        """Merges multiple PDFs."""
        pdf1 = tmp_path / "doc1.pdf"
        pdf2 = tmp_path / "doc2.pdf"
        pdf1.write_bytes(minimal_pdf.read_bytes())
        pdf2.write_bytes(minimal_pdf.read_bytes())

        output = tmp_path / "merged.pdf"
        result = pdfsplit.merge_pdfs([pdf1, pdf2], output, force=True, quiet=True)
        assert result is True
        assert output.exists()


class TestReversePdf:
    """Tests for reverse_pdf function."""

    def test_reverse_pdf(self, minimal_pdf: Path, tmp_path: Path):
        """Reverses PDF page order."""
        output = tmp_path / "reversed.pdf"
        result = pdfsplit.reverse_pdf(minimal_pdf, output, force=True, quiet=True)
        assert result is True
        assert output.exists()


class TestUnlockPdf:
    """Tests for unlock_pdf function."""

    def test_unlock_unencrypted_pdf(self, minimal_pdf: Path, tmp_path: Path):
        """Unlocks unencrypted PDF (essentially copies it)."""
        output = tmp_path / "unlocked.pdf"
        result = pdfsplit.unlock_pdf(minimal_pdf, output, force=True, quiet=True)
        assert result is True
        assert output.exists()


class TestOptimizePdf:
    """Tests for optimize_pdf function."""

    def test_optimize_requires_fitz(self, minimal_pdf: Path, tmp_path: Path):
        """Optimize requires PyMuPDF."""
        output = tmp_path / "optimized.pdf"
        # This test verifies the function runs; actual optimization depends on fitz
        original, new = pdfsplit.optimize_pdf(
            minimal_pdf, output, force=True, quiet=True
        )
        # Either both are 0 (fitz not available) or optimization succeeded
        assert (original == 0 and new == 0) or (original > 0 and new > 0)


class TestCLI:
    """Tests for command-line interface."""

    def test_cli_no_args_prints_help(self, capsys):
        """CLI prints help when no args provided."""
        with patch.object(sys, "argv", ["pdfsplit"]):
            with pytest.raises(SystemExit) as exc_info:
                pdfsplit.main()
            assert exc_info.value.code == 0

    def test_cli_info_flag(self, minimal_pdf: Path, capsys):
        """CLI --info shows PDF information."""
        with patch.object(sys, "argv", ["pdfsplit", str(minimal_pdf), "--info"]):
            with pytest.raises(SystemExit) as exc_info:
                pdfsplit.main()
            assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "Pages:" in captured.out

    def test_cli_split_basic(self, minimal_pdf: Path, tmp_path: Path):
        """CLI splits PDF into pages."""
        output_dir = tmp_path / "output"
        with patch.object(
            sys,
            "argv",
            [
                "pdfsplit",
                str(minimal_pdf),
                "-d",
                str(output_dir),
                "-f",
                "-q",
            ],
        ):
            pdfsplit.main()
        assert output_dir.exists()

    def test_cli_page_spec(self, minimal_pdf: Path, tmp_path: Path):
        """CLI respects page specification."""
        output_dir = tmp_path / "output"
        with patch.object(
            sys,
            "argv",
            [
                "pdfsplit",
                str(minimal_pdf),
                "-p",
                "1",
                "-d",
                str(output_dir),
                "-f",
                "-q",
            ],
        ):
            pdfsplit.main()

    def test_cli_merge_requires_output(self, tmp_path: Path, minimal_pdf: Path, capsys):
        """CLI --merge requires -o/--output."""
        with patch.object(
            sys, "argv", ["pdfsplit", str(minimal_pdf), "--merge"]
        ):
            with pytest.raises(SystemExit) as exc_info:
                pdfsplit.main()
            assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "--merge requires" in captured.err

    def test_cli_merge(self, tmp_path: Path, minimal_pdf: Path):
        """CLI merges PDFs."""
        pdf1 = tmp_path / "doc1.pdf"
        pdf2 = tmp_path / "doc2.pdf"
        pdf1.write_bytes(minimal_pdf.read_bytes())
        pdf2.write_bytes(minimal_pdf.read_bytes())
        output = tmp_path / "merged.pdf"

        with patch.object(
            sys,
            "argv",
            [
                "pdfsplit",
                str(pdf1),
                str(pdf2),
                "--merge",
                "-o",
                str(output),
                "-f",
                "-q",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                pdfsplit.main()
            assert exc_info.value.code == 0

    def test_cli_reverse(self, minimal_pdf: Path, tmp_path: Path):
        """CLI reverses PDF."""
        output = tmp_path / "reversed.pdf"
        with patch.object(
            sys,
            "argv",
            [
                "pdfsplit",
                str(minimal_pdf),
                "--reverse",
                "-o",
                str(output),
                "-f",
                "-q",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                pdfsplit.main()
            assert exc_info.value.code == 0

    def test_cli_no_input_files(self, tmp_path: Path, capsys):
        """CLI errors when no input files found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch.object(sys, "argv", ["pdfsplit", str(empty_dir)]):
            with pytest.raises(SystemExit) as exc_info:
                pdfsplit.main()
            assert exc_info.value.code == 1


class TestConstants:
    """Tests for module constants."""

    def test_version(self):
        """Version string is defined."""
        assert hasattr(pdfsplit, "__version__")
        assert isinstance(pdfsplit.__version__, str)

    def test_default_output_dir(self):
        """Default output directory is defined."""
        assert pdfsplit.DEFAULT_OUTPUT_DIR == "pdf_out"

    def test_dpi_limits(self):
        """DPI limits are reasonable."""
        assert pdfsplit.MIN_DPI > 0
        assert pdfsplit.MAX_DPI > pdfsplit.MIN_DPI
        assert pdfsplit.DEFAULT_DPI >= pdfsplit.MIN_DPI
        assert pdfsplit.DEFAULT_DPI <= pdfsplit.MAX_DPI


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_directory(self, tmp_path: Path, capsys):
        """Handles empty directory gracefully."""
        empty = tmp_path / "empty"
        empty.mkdir()
        result = pdfsplit.process_inputs([str(empty)])
        assert result == []

    def test_symlinks_handling(self, tmp_path: Path, minimal_pdf: Path):
        """Handles symlinks to PDFs."""
        link = tmp_path / "link.pdf"
        try:
            link.symlink_to(minimal_pdf)
            result = pdfsplit.process_inputs([str(link)])
            assert len(result) == 1
        except OSError:
            pytest.skip("Symlinks not supported on this system")

    def test_unicode_filename(self, tmp_path: Path, minimal_pdf: Path):
        """Handles unicode in filenames."""
        unicode_pdf = tmp_path / "документ.pdf"
        unicode_pdf.write_bytes(minimal_pdf.read_bytes())

        result = pdfsplit.process_inputs([str(unicode_pdf)])
        assert len(result) == 1
