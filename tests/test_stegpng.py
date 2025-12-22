"""Comprehensive tests for stegpng.py - Steganographic file encoder/decoder."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import stegpng


class TestXorData:
    """Tests for XOR obfuscation function."""

    def test_xor_empty_data(self):
        """XOR with empty data returns empty bytes."""
        assert stegpng.xor_data(b"", "key") == b""

    def test_xor_empty_key(self):
        """XOR with empty key returns original data."""
        data = b"hello world"
        assert stegpng.xor_data(data, "") == data

    def test_xor_roundtrip(self):
        """XOR is reversible (encryption = decryption)."""
        data = b"secret message"
        key = "mypassword"
        encrypted = stegpng.xor_data(data, key)
        decrypted = stegpng.xor_data(encrypted, key)
        assert decrypted == data

    def test_xor_different_keys_different_output(self):
        """Different keys produce different output."""
        data = b"test data"
        result1 = stegpng.xor_data(data, "key1")
        result2 = stegpng.xor_data(data, "key2")
        assert result1 != result2

    @given(st.binary(min_size=1, max_size=1000), st.text(min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_xor_roundtrip_property(self, data: bytes, key: str):
        """Property test: XOR roundtrip always returns original data."""
        # Filter out keys with null bytes which can cause issues with encoding
        if "\x00" in key:
            return
        encrypted = stegpng.xor_data(data, key)
        decrypted = stegpng.xor_data(encrypted, key)
        assert decrypted == data


class TestMinimalImageCreation:
    """Tests for minimal image creation functions."""

    def test_create_minimal_png_valid_signature(self):
        """Minimal PNG has valid PNG signature."""
        png_data = stegpng.create_minimal_png()
        assert png_data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_create_minimal_png_has_iend(self):
        """Minimal PNG contains IEND chunk."""
        png_data = stegpng.create_minimal_png()
        assert b"IEND" in png_data

    def test_create_minimal_jpeg_valid_markers(self):
        """Minimal JPEG has valid SOI and EOI markers."""
        prefix, eoi = stegpng.create_minimal_jpeg()
        assert prefix[:2] == b"\xff\xd8"  # SOI marker
        assert eoi == b"\xff\xd9"  # EOI marker


class TestFormatDetection:
    """Tests for image format detection."""

    def test_detect_png(self, minimal_png: bytes):
        """Detects PNG format correctly."""
        assert stegpng.detect_format(minimal_png) == "png"

    def test_detect_jpeg(self, minimal_jpeg: bytes):
        """Detects JPEG format correctly."""
        assert stegpng.detect_format(minimal_jpeg) == "jpeg"

    def test_detect_unknown(self):
        """Returns None for unknown format."""
        assert stegpng.detect_format(b"not an image") is None

    def test_detect_empty(self):
        """Returns None for empty data."""
        assert stegpng.detect_format(b"") is None


class TestPngChunkParsing:
    """Tests for PNG chunk parsing."""

    def test_find_png_iend(self, minimal_png: bytes):
        """Finds IEND position in valid PNG."""
        pos = stegpng.find_png_iend(minimal_png)
        assert pos > 0
        # find_png_iend returns position AFTER the IEND chunk (end of valid PNG)
        assert pos <= len(minimal_png)

    def test_find_png_iend_invalid(self):
        """Returns -1 for invalid PNG."""
        assert stegpng.find_png_iend(b"not a png") == -1

    def test_find_png_iend_truncated(self):
        """Returns -1 for truncated PNG."""
        assert stegpng.find_png_iend(b"\x89PNG\r\n\x1a\n") == -1


class TestJpegParsing:
    """Tests for JPEG parsing."""

    def test_find_jpeg_eoi(self, minimal_jpeg: bytes):
        """Finds EOI position in valid JPEG."""
        pos = stegpng.find_jpeg_eoi(minimal_jpeg)
        assert pos > 0
        assert minimal_jpeg[pos : pos + 2] == b"\xff\xd9"

    def test_find_jpeg_eoi_invalid(self):
        """Returns -1 for invalid JPEG."""
        assert stegpng.find_jpeg_eoi(b"\xff\xd8not valid") == -1


class TestEncodeAppend:
    """Tests for append encoding method."""

    def test_encode_append_png_with_base(self, minimal_png: bytes):
        """Append method adds payload after PNG."""
        payload = b"hidden data"
        result = stegpng.encode_append(payload, minimal_png, "png")
        assert result.startswith(minimal_png[:8])
        assert result.endswith(payload)

    def test_encode_append_png_no_base(self):
        """Append method creates minimal PNG when no base provided."""
        payload = b"hidden data"
        result = stegpng.encode_append(payload, None, "png")
        assert result[:8] == b"\x89PNG\r\n\x1a\n"
        assert result.endswith(payload)

    def test_encode_append_jpeg_with_base(self, minimal_jpeg: bytes):
        """Append method adds payload after JPEG."""
        payload = b"hidden data"
        result = stegpng.encode_append(payload, minimal_jpeg, "jpeg")
        assert result[:2] == b"\xff\xd8"
        assert result.endswith(payload)

    def test_encode_append_invalid_png_base(self):
        """Raises error for invalid PNG base image."""
        with pytest.raises(ValueError, match="not a valid PNG"):
            stegpng.encode_append(b"data", b"not a png", "png")

    def test_encode_append_invalid_jpeg_base(self):
        """Raises error for invalid JPEG base image."""
        with pytest.raises(ValueError, match="not a valid JPEG"):
            stegpng.encode_append(b"data", b"not a jpeg", "jpeg")


class TestDecodeAppend:
    """Tests for append decoding method."""

    def test_decode_append_png(self, minimal_png: bytes):
        """Decodes payload appended to PNG."""
        payload = b"hidden data"
        encoded = minimal_png + payload
        result = stegpng.decode_append(encoded, "png")
        assert result == payload

    def test_decode_append_jpeg(self, minimal_jpeg: bytes):
        """Decodes payload appended to JPEG."""
        payload = b"hidden data"
        encoded = minimal_jpeg + payload
        result = stegpng.decode_append(encoded, "jpeg")
        assert result == payload

    def test_decode_append_empty_payload(self, minimal_png: bytes):
        """Decodes empty payload from PNG."""
        result = stegpng.decode_append(minimal_png, "png")
        assert result == b""


class TestEncodeMetadata:
    """Tests for metadata encoding method."""

    def test_encode_metadata_png_creates_ztxt(self, minimal_png: bytes):
        """Metadata method creates zTXt chunk in PNG."""
        payload = b"hidden data"
        result = stegpng.encode_metadata(payload, minimal_png, "png")
        assert b"zTXt" in result
        assert result[:8] == b"\x89PNG\r\n\x1a\n"

    def test_encode_metadata_jpeg_creates_com(self, minimal_jpeg: bytes):
        """Metadata method creates COM marker in JPEG."""
        payload = b"hidden data"
        result = stegpng.encode_metadata(payload, minimal_jpeg, "jpeg")
        assert b"\xff\xfe" in result  # COM marker
        assert result[:2] == b"\xff\xd8"


class TestDecodeMetadata:
    """Tests for metadata decoding method."""

    def test_decode_metadata_png_roundtrip(self, minimal_png: bytes):
        """Metadata encoding/decoding roundtrip for PNG."""
        payload = b"secret message"
        encoded = stegpng.encode_metadata(payload, minimal_png, "png")
        decoded = stegpng.decode_metadata(encoded, "png")
        assert decoded == payload

    def test_decode_metadata_jpeg_roundtrip(self, minimal_jpeg: bytes):
        """Metadata encoding/decoding roundtrip for JPEG."""
        payload = b"secret message"
        encoded = stegpng.encode_metadata(payload, minimal_jpeg, "jpeg")
        decoded = stegpng.decode_metadata(encoded, "jpeg")
        assert decoded == payload

    def test_decode_metadata_png_not_found(self, minimal_png: bytes):
        """Raises error when no metadata found in PNG."""
        with pytest.raises(ValueError, match="No stegpng metadata found"):
            stegpng.decode_metadata(minimal_png, "png")

    def test_decode_metadata_jpeg_not_found(self, minimal_jpeg: bytes):
        """Raises error when no metadata found in JPEG."""
        with pytest.raises(ValueError, match="No stegpng metadata found"):
            stegpng.decode_metadata(minimal_jpeg, "jpeg")


class TestEndToEndEncoding:
    """End-to-end tests for file encoding/decoding."""

    def test_encode_decode_file_append(self, tmp_path: Path, sample_text_file: Path):
        """Full encode/decode cycle with append method."""
        output_path = tmp_path / "encoded.png"

        stegpng.encode_file(
            str(sample_text_file),
            str(output_path),
            base_path=None,
            key=None,
            method="append",
            verbose=False,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Decode
        decoded_path = tmp_path / "decoded.txt"
        stegpng.decode_file(
            str(output_path),
            str(decoded_path),
            key=None,
            method="append",
            verbose=False,
        )

        assert decoded_path.exists()
        assert decoded_path.read_text() == sample_text_file.read_text()

    def test_encode_decode_file_with_xor(self, tmp_path: Path, sample_text_file: Path):
        """Full encode/decode cycle with XOR obfuscation."""
        output_path = tmp_path / "encoded.png"
        key = "secretkey123"

        stegpng.encode_file(
            str(sample_text_file),
            str(output_path),
            base_path=None,
            key=key,
            method="append",
            verbose=False,
        )

        # Decode with correct key
        decoded_path = tmp_path / "decoded.txt"
        stegpng.decode_file(
            str(output_path),
            str(decoded_path),
            key=key,
            method="append",
            verbose=False,
        )

        assert decoded_path.read_text() == sample_text_file.read_text()

    def test_encode_decode_metadata_method(
        self, tmp_path: Path, sample_text_file: Path
    ):
        """Full encode/decode cycle with metadata method."""
        output_path = tmp_path / "encoded.png"

        stegpng.encode_file(
            str(sample_text_file),
            str(output_path),
            base_path=None,
            key=None,
            method="metadata",
            verbose=False,
        )

        decoded_path = tmp_path / "decoded.txt"
        stegpng.decode_file(
            str(output_path),
            str(decoded_path),
            key=None,
            method="metadata",
            verbose=False,
        )

        assert decoded_path.read_text() == sample_text_file.read_text()

    def test_encode_with_base_image(
        self, tmp_path: Path, sample_text_file: Path, sample_png_path: Path
    ):
        """Encoding with a base image preserves image start."""
        output_path = tmp_path / "encoded.png"

        stegpng.encode_file(
            str(sample_text_file),
            str(output_path),
            base_path=str(sample_png_path),
            key=None,
            method="append",
            verbose=False,
        )

        assert output_path.exists()
        output_data = output_path.read_bytes()
        base_data = sample_png_path.read_bytes()
        assert output_data.startswith(base_data[:8])

    def test_encode_default_output_name(self, tmp_path: Path):
        """Default output name is input name with .png extension."""
        input_file = tmp_path / "myfile.txt"
        input_file.write_text("test content")

        # Change to tmp_path so output goes there
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            stegpng.encode_file(
                str(input_file),
                None,
                base_path=None,
                key=None,
                method="append",
                verbose=False,
            )
            expected_output = tmp_path / "myfile.png"
            assert expected_output.exists()
        finally:
            os.chdir(original_cwd)


class TestInfoFunction:
    """Tests for info_file function."""

    def test_info_png(self, sample_png_path: Path, capsys):
        """Shows info for PNG file."""
        stegpng.info_file(str(sample_png_path))
        captured = capsys.readouterr()
        assert "Format: PNG" in captured.out

    def test_info_jpeg(self, sample_jpeg_path: Path, capsys):
        """Shows info for JPEG file."""
        stegpng.info_file(str(sample_jpeg_path))
        captured = capsys.readouterr()
        assert "Format: JPEG" in captured.out

    def test_info_with_hidden_data(self, tmp_path: Path, capsys):
        """Shows hidden data info when present."""
        png_path = tmp_path / "test.png"
        payload = b"hidden"
        png_data = stegpng.create_minimal_png() + payload
        png_path.write_bytes(png_data)

        stegpng.info_file(str(png_path))
        captured = capsys.readouterr()
        assert "Hidden data (append method)" in captured.out

    def test_info_invalid_file(self, tmp_path: Path, capsys):
        """Shows error for invalid file."""
        invalid = tmp_path / "invalid.bin"
        invalid.write_bytes(b"not an image")

        stegpng.info_file(str(invalid))
        captured = capsys.readouterr()
        assert "not a valid PNG or JPEG" in captured.out


class TestCLI:
    """Tests for command-line interface."""

    def test_cli_encode_basic(self, tmp_path: Path, sample_text_file: Path):
        """CLI encode command works."""
        output = tmp_path / "output.png"
        with patch.object(
            sys,
            "argv",
            ["stegpng", "encode", str(sample_text_file), "-o", str(output)],
        ):
            result = stegpng.main()
            assert result == 0
            assert output.exists()

    def test_cli_decode_basic(self, tmp_path: Path, sample_text_file: Path):
        """CLI decode command works."""
        # First encode
        encoded = tmp_path / "encoded.png"
        stegpng.encode_file(
            str(sample_text_file), str(encoded), None, None, "append", False
        )

        # Then decode via CLI
        output = tmp_path / "decoded.txt"
        with patch.object(
            sys, "argv", ["stegpng", "decode", str(encoded), "-o", str(output)]
        ):
            result = stegpng.main()
            assert result == 0
            assert output.read_text() == sample_text_file.read_text()

    def test_cli_info(self, sample_png_path: Path, capsys):
        """CLI info command works."""
        with patch.object(sys, "argv", ["stegpng", "info", str(sample_png_path)]):
            result = stegpng.main()
            assert result == 0

    def test_cli_file_not_found(self, tmp_path: Path):
        """CLI returns error for missing file."""
        with patch.object(
            sys, "argv", ["stegpng", "encode", str(tmp_path / "nonexistent.txt")]
        ):
            result = stegpng.main()
            assert result == 1

    def test_cli_encode_with_key(self, tmp_path: Path, sample_text_file: Path):
        """CLI encode with XOR key."""
        output = tmp_path / "output.png"
        with patch.object(
            sys,
            "argv",
            [
                "stegpng",
                "encode",
                str(sample_text_file),
                "-o",
                str(output),
                "-k",
                "mykey",
            ],
        ):
            result = stegpng.main()
            assert result == 0

    def test_cli_encode_metadata_method(self, tmp_path: Path, sample_text_file: Path):
        """CLI encode with metadata method."""
        output = tmp_path / "output.png"
        with patch.object(
            sys,
            "argv",
            [
                "stegpng",
                "encode",
                str(sample_text_file),
                "-o",
                str(output),
                "-m",
                "metadata",
            ],
        ):
            result = stegpng.main()
            assert result == 0


class TestErrorHandling:
    """Tests for error handling."""

    def test_encode_format_mismatch(
        self, tmp_path: Path, sample_text_file: Path, sample_png_path: Path
    ):
        """Raises error when base format doesn't match output."""
        output = tmp_path / "output.jpg"  # JPEG output
        with pytest.raises(ValueError, match="doesn't match output format"):
            stegpng.encode_file(
                str(sample_text_file),
                str(output),
                base_path=str(sample_png_path),  # PNG base
                key=None,
                method="append",
                verbose=False,
            )

    def test_decode_no_hidden_data_append(self, tmp_path: Path, minimal_png: bytes):
        """Decode append raises error when no data and payload is empty."""
        png_path = tmp_path / "clean.png"
        png_path.write_bytes(minimal_png)

        output = tmp_path / "output.txt"
        # decode_file raises ValueError when no hidden data found (empty payload)
        with pytest.raises(ValueError, match="No hidden data found"):
            stegpng.decode_file(str(png_path), str(output), None, "append", False)


class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(st.binary(min_size=1, max_size=10000))
    @settings(max_examples=20)
    def test_png_append_roundtrip_property(self, payload: bytes):
        """Property: any binary payload can be roundtripped via PNG append."""
        encoded = stegpng.encode_append(payload, None, "png")
        decoded = stegpng.decode_append(encoded, "png")
        assert decoded == payload

    @given(st.binary(min_size=1, max_size=10000))
    @settings(max_examples=20)
    def test_jpeg_append_roundtrip_property(self, payload: bytes):
        """Property: any binary payload can be roundtripped via JPEG append."""
        encoded = stegpng.encode_append(payload, None, "jpeg")
        decoded = stegpng.decode_append(encoded, "jpeg")
        assert decoded == payload

    @given(st.binary(min_size=1, max_size=5000))
    @settings(max_examples=10)
    def test_png_metadata_roundtrip_property(self, payload: bytes):
        """Property: any binary payload can be roundtripped via PNG metadata."""
        encoded = stegpng.encode_metadata(payload, None, "png")
        decoded = stegpng.decode_metadata(encoded, "png")
        assert decoded == payload
