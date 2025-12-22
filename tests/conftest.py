"""Shared pytest fixtures for HaxBox tests."""

import struct
import zlib
from pathlib import Path

import pytest


@pytest.fixture
def sample_text() -> str:
    """Sample text for testing."""
    return "Hello, World! This is sample text for testing."


@pytest.fixture
def sample_binary() -> bytes:
    """Sample binary data for testing."""
    return b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f"


@pytest.fixture
def minimal_png() -> bytes:
    """Create a minimal 1x1 transparent PNG image."""
    signature = b"\x89PNG\r\n\x1a\n"

    # IHDR chunk: 1x1 pixel, RGBA
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    ihdr = (
        struct.pack(">I", 13)
        + b"IHDR"
        + ihdr_data
        + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF)
    )

    # IDAT chunk: minimal compressed pixel data (transparent black)
    pixel = b"\x00\x00\x00\x00\x00"  # filter byte + RGBA
    compressed = zlib.compress(pixel)
    idat = (
        struct.pack(">I", len(compressed))
        + b"IDAT"
        + compressed
        + struct.pack(">I", zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF)
    )

    # IEND chunk
    iend = (
        struct.pack(">I", 0)
        + b"IEND"
        + struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
    )

    return signature + ihdr + idat + iend


@pytest.fixture
def minimal_jpeg() -> bytes:
    """Create a minimal 1x1 black JPEG image."""
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00"
        b"\xff\xdb\x00C\x00" + b"\x10" * 64
        + b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
        b"\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\x7f"
        b"\xff\xd9"
    )


@pytest.fixture
def sample_png_path(tmp_path: Path, minimal_png: bytes) -> Path:
    """Create a sample PNG file in a temp directory."""
    png_path = tmp_path / "sample.png"
    png_path.write_bytes(minimal_png)
    return png_path


@pytest.fixture
def sample_jpeg_path(tmp_path: Path, minimal_jpeg: bytes) -> Path:
    """Create a sample JPEG file in a temp directory."""
    jpeg_path = tmp_path / "sample.jpg"
    jpeg_path.write_bytes(minimal_jpeg)
    return jpeg_path


@pytest.fixture
def sample_text_file(tmp_path: Path, sample_text: str) -> Path:
    """Create a sample text file in a temp directory."""
    text_path = tmp_path / "sample.txt"
    text_path.write_text(sample_text)
    return text_path


@pytest.fixture
def minimal_pdf(tmp_path: Path) -> Path:
    """Create a minimal valid PDF file using PyPDF2."""
    from PyPDF2 import PdfWriter

    writer = PdfWriter()
    # Add a blank page (standard letter size)
    writer.add_blank_page(width=612, height=792)

    pdf_path = tmp_path / "sample.pdf"
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return pdf_path


@pytest.fixture
def empty_dir(tmp_path: Path) -> Path:
    """Create an empty directory."""
    empty = tmp_path / "empty"
    empty.mkdir()
    return empty
