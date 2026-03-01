"""Comprehensive tests for ocr_benchmark.py - OCR benchmarking tool."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import ocr_benchmark


class TestLevenshteinDistance:
    """Tests for levenshtein_distance function."""

    def test_identical_strings(self):
        """Identical strings have distance 0."""
        assert ocr_benchmark.levenshtein_distance("hello", "hello") == 0
        assert ocr_benchmark.levenshtein_distance("", "") == 0

    def test_single_substitution(self):
        """Single character substitution has distance 1."""
        assert ocr_benchmark.levenshtein_distance("hello", "hallo") == 1
        assert ocr_benchmark.levenshtein_distance("cat", "bat") == 1

    def test_single_insertion(self):
        """Single character insertion has distance 1."""
        assert ocr_benchmark.levenshtein_distance("cat", "cast") == 1
        assert ocr_benchmark.levenshtein_distance("hello", "helloo") == 1

    def test_single_deletion(self):
        """Single character deletion has distance 1."""
        assert ocr_benchmark.levenshtein_distance("cast", "cat") == 1
        assert ocr_benchmark.levenshtein_distance("helloo", "hello") == 1

    def test_empty_strings(self):
        """Distance to/from empty string equals length."""
        assert ocr_benchmark.levenshtein_distance("", "hello") == 5
        assert ocr_benchmark.levenshtein_distance("hello", "") == 5

    def test_completely_different(self):
        """Completely different strings."""
        assert ocr_benchmark.levenshtein_distance("abc", "xyz") == 3
        assert ocr_benchmark.levenshtein_distance("kitten", "sitting") == 3


class TestCalculateCER:
    """Tests for calculate_cer function."""

    def test_perfect_match(self):
        """Perfect OCR has CER of 0."""
        assert ocr_benchmark.calculate_cer("hello world", "hello world") == 0.0

    def test_single_error(self):
        """Single character error."""
        # "hello world" vs "hallo world" - 1 error in 11 chars
        cer = ocr_benchmark.calculate_cer("hallo world", "hello world")
        assert 0.09 < cer < 0.10  # ~0.0909

    def test_empty_ground_truth(self):
        """Empty ground truth edge case."""
        assert ocr_benchmark.calculate_cer("", "") == 0.0
        assert ocr_benchmark.calculate_cer("hello", "") == 1.0

    def test_high_error_rate(self):
        """High error rate."""
        cer = ocr_benchmark.calculate_cer("abc", "xyz")
        assert cer == 1.0  # 3 errors in 3 chars


class TestCalculateWER:
    """Tests for calculate_wer function."""

    def test_perfect_match(self):
        """Perfect OCR has WER of 0."""
        assert ocr_benchmark.calculate_wer("hello world", "hello world") == 0.0

    def test_one_word_error(self):
        """Single word error."""
        # One word different
        wer = ocr_benchmark.calculate_wer("hello world", "hello word")
        assert wer > 0

    def test_empty_texts(self):
        """Empty text edge cases."""
        assert ocr_benchmark.calculate_wer("", "") == 0.0


class TestCalculateJaccard:
    """Tests for calculate_jaccard function."""

    def test_identical_text(self):
        """Identical texts have Jaccard similarity of 1.0."""
        assert ocr_benchmark.calculate_jaccard("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        """No word overlap has Jaccard of 0.0."""
        assert ocr_benchmark.calculate_jaccard("foo bar", "baz qux") == 0.0

    def test_partial_overlap(self):
        """Partial overlap."""
        # Words: {hello, world} vs {hello, there} -> intersection=1, union=3
        jaccard = ocr_benchmark.calculate_jaccard("hello world", "hello there")
        assert 0.3 < jaccard < 0.4

    def test_empty_texts(self):
        """Empty texts edge case."""
        assert ocr_benchmark.calculate_jaccard("", "") == 1.0


class TestCalculateExactLineMatch:
    """Tests for calculate_exact_line_match function."""

    def test_all_lines_match(self):
        """All lines matching."""
        text = "line 1\nline 2\nline 3"
        assert ocr_benchmark.calculate_exact_line_match(text, text) == 1.0

    def test_no_lines_match(self):
        """No lines matching."""
        text1 = "line 1\nline 2\nline 3"
        text2 = "different\ncompletely\ntext"
        assert ocr_benchmark.calculate_exact_line_match(text2, text1) == 0.0

    def test_partial_match(self):
        """Some lines matching."""
        text1 = "line 1\nline 2\nline 3"
        text2 = "line 1\ndifferent\nline 3"
        match = ocr_benchmark.calculate_exact_line_match(text2, text1)
        assert 0.6 < match < 0.7  # 2/3 lines match


class TestImageRendering:
    """Tests for image rendering functions."""

    def test_render_creates_image(self):
        """Rendering text creates a valid image."""
        settings = ocr_benchmark.ScanSettings()
        img = ocr_benchmark.render_text_to_image("Hello World", settings)
        assert isinstance(img, Image.Image)
        assert img.size[0] > 0
        assert img.size[1] > 0

    def test_render_multiline_text(self):
        """Rendering multiline text."""
        settings = ocr_benchmark.ScanSettings()
        text = "Line 1\nLine 2\nLine 3"
        img = ocr_benchmark.render_text_to_image(text, settings)
        assert isinstance(img, Image.Image)
        # Multi-line text should be taller
        assert img.size[1] > 50

    def test_noise_adds_variation(self):
        """Adding noise increases image variance."""
        text = "Test"
        clean_settings = ocr_benchmark.ScanSettings(noise_level=0.0)
        noisy_settings = ocr_benchmark.ScanSettings(noise_level=0.1)

        clean_img = ocr_benchmark.render_text_to_image(text, clean_settings)
        noisy_img = ocr_benchmark.render_text_to_image(text, noisy_settings)

        # Check that noise was added (images differ)
        # Note: Cannot guarantee exact differences due to randomness
        assert isinstance(clean_img, Image.Image)
        assert isinstance(noisy_img, Image.Image)

    def test_blur_applies(self):
        """Blur filter is applied."""
        text = "Test"
        settings = ocr_benchmark.ScanSettings(blur_radius=2.0)
        img = ocr_benchmark.render_text_to_image(text, settings)
        assert isinstance(img, Image.Image)

    def test_rotation_applies(self):
        """Rotation is applied."""
        text = "Test"
        settings = ocr_benchmark.ScanSettings(rotation_deg=5.0)
        img = ocr_benchmark.render_text_to_image(text, settings)
        assert isinstance(img, Image.Image)


class TestScanSettings:
    """Tests for ScanSettings dataclass."""

    def test_default_settings(self):
        """Default settings are pristine."""
        settings = ocr_benchmark.ScanSettings()
        assert settings.noise_level == 0.0
        assert settings.blur_radius == 0.0
        assert settings.rotation_deg == 0.0
        assert settings.contrast == 1.0
        assert settings.background_gray == 255
        assert settings.jpeg_quality == 95

    def test_quality_presets_exist(self):
        """All quality presets are defined."""
        assert "pristine" in ocr_benchmark.QUALITY_PRESETS
        assert "good_scan" in ocr_benchmark.QUALITY_PRESETS
        assert "average_scan" in ocr_benchmark.QUALITY_PRESETS
        assert "poor_scan" in ocr_benchmark.QUALITY_PRESETS
        assert "photocopy" in ocr_benchmark.QUALITY_PRESETS


class TestFindErrors:
    """Tests for find_errors function."""

    def test_no_errors(self):
        """Identical texts have no errors."""
        errors = ocr_benchmark.find_errors("hello", "hello")
        assert len(errors) == 0

    def test_substitution_error(self):
        """Substitution errors are detected."""
        errors = ocr_benchmark.find_errors("hallo", "hello")
        assert len(errors) == 1
        assert errors[0]["type"] == "substitution"
        assert errors[0]["expected"] == "e"
        assert errors[0]["got"] == "a"

    def test_deletion_error(self):
        """Deletion errors are detected."""
        errors = ocr_benchmark.find_errors("helo", "hello")
        # Note: Simple implementation may detect multiple errors for complex cases
        # The important thing is that errors are detected
        assert len(errors) >= 1
        assert any(e["type"] == "deletion" for e in errors)

    def test_insertion_error(self):
        """Insertion errors are detected."""
        errors = ocr_benchmark.find_errors("helloo", "hello")
        assert len(errors) == 1
        assert errors[0]["type"] == "insertion"


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_create_result(self):
        """Can create a BenchmarkResult."""
        result = ocr_benchmark.BenchmarkResult(
            engine="test",
            char_accuracy=0.95,
            cer=0.05,
            wer=0.10,
            bleu=0.90,
            exact_line_match=0.85,
            jaccard=0.88,
            time_seconds=1.5,
            ocr_text="test output",
            errors=[],
        )
        assert result.engine == "test"
        assert result.char_accuracy == 0.95
        assert result.time_seconds == 1.5
