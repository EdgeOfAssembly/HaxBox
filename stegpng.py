#!/usr/bin/env python3
"""
stegpng - Steganographic file encoder/decoder for PNG and JPEG images

Encode any file into a PNG or JPEG image, or extract hidden data from images.
Supports XOR obfuscation and multiple embedding methods.

Author: EdgeOfAssembly
Email: haxbox2000@gmail.com
License: GPLv3 / Commercial
"""

import argparse
import os
import struct
import sys
import zlib
import base64
from itertools import cycle
from typing import Optional, Tuple

__version__ = "1.0.0"

# Magic marker for metadata embedding
MAGIC_MARKER = b'STEGPNG'


def xor_data(data: bytes, key: str) -> bytes:
    """XOR data with a key for simple obfuscation.
    
    WARNING: XOR is NOT secure encryption! It is trivially breakable and should
    NOT be relied upon for actual security. For sensitive data, encrypt files
    with GPG or similar tools before encoding.
    """
    if not key:
        return data
    key_bytes = key.encode('utf-8')
    return bytes(a ^ b for a, b in zip(data, cycle(key_bytes)))


def create_minimal_png() -> bytes:
    """Create a minimal 1x1 transparent PNG image."""
    signature = b'\x89PNG\r\n\x1a\n'
    
    # IHDR chunk: 1x1 pixel, RGBA
    ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 6, 0, 0, 0)
    ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff)
    
    # IDAT chunk: minimal compressed pixel data (transparent black)
    pixel = b'\x00\x00\x00\x00\x00'  # filter byte + RGBA
    compressed = zlib.compress(pixel)
    idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + struct.pack('>I', zlib.crc32(b'IDAT' + compressed) & 0xffffffff)
    
    # IEND chunk
    iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', zlib.crc32(b'IEND') & 0xffffffff)
    
    return signature + ihdr + idat + iend


def create_minimal_jpeg() -> Tuple[bytes, bytes]:
    """Create a minimal 1x1 black JPEG image (prefix and EOI marker)."""
    minimal_prefix = (
        b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00'
        b'\xff\xdb\x00C\x00' + b'\x10' * 64 +
        b'\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00'
        b'\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        b'\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        b'\xff\xda\x00\x08\x01\x01\x00\x00?\x00\x7f'
    )
    eoi = b'\xff\xd9'
    return minimal_prefix, eoi


def detect_format(data: bytes) -> Optional[str]:
    """Detect image format from magic bytes."""
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    elif data[:2] == b'\xff\xd8':
        return 'jpeg'
    return None


def find_png_iend(data: bytes) -> int:
    """Find the position after the IEND chunk in a PNG."""
    pos = 8  # Skip signature
    while pos < len(data):
        if pos + 8 > len(data):
            break
        chunk_length = struct.unpack('>I', data[pos:pos+4])[0]
        chunk_type = data[pos+4:pos+8]
        if chunk_type == b'IEND':
            return pos + 12  # length(4) + type(4) + crc(4)
        pos += 12 + chunk_length
    return -1


def find_jpeg_eoi(data: bytes) -> int:
    """Find the position of the EOI marker in a JPEG."""
    pos = 2  # Skip SOI
    while pos < len(data) - 1:
        if data[pos] == 0xff:
            marker = data[pos + 1]
            if marker == 0xd9:  # EOI
                return pos
            elif marker in (0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0x01, 0x00):
                pos += 2
            elif marker == 0xff:
                pos += 1
            else:
                if pos + 4 > len(data):
                    break
                seg_length = struct.unpack('>H', data[pos+2:pos+4])[0]
                pos += 2 + seg_length
        else:
            pos += 1
    return -1


def encode_append(payload: bytes, base_image: Optional[bytes], fmt: str) -> bytes:
    """Encode payload by appending after image end marker."""
    if base_image:
        if fmt == 'png':
            if base_image[:8] != b'\x89PNG\r\n\x1a\n':
                raise ValueError('Base file is not a valid PNG.')
            iend_pos = find_png_iend(base_image)
            if iend_pos == -1:
                raise ValueError('Invalid PNG structure: IEND not found.')
        else:
            if not (base_image[:2] == b'\xff\xd8'):
                raise ValueError('Base file is not a valid JPEG.')
        return base_image + payload
    else:
        if fmt == 'png':
            return create_minimal_png() + payload
        else:
            prefix, eoi = create_minimal_jpeg()
            return prefix + eoi + payload


def encode_metadata(payload: bytes, base_image: Optional[bytes], fmt: str) -> bytes:
    """Encode payload in image metadata (zTXt for PNG, COM for JPEG)."""
    if fmt == 'png':
        # Compress payload and create zTXt chunk
        compressed = zlib.compress(payload)
        ztxt_data = MAGIC_MARKER + b'\x00\x00' + compressed  # keyword + null + compression method + data
        ztxt_chunk = (struct.pack('>I', len(ztxt_data)) + b'zTXt' + ztxt_data + 
                      struct.pack('>I', zlib.crc32(b'zTXt' + ztxt_data) & 0xffffffff))
        
        if base_image:
            if base_image[:8] != b'\x89PNG\r\n\x1a\n':
                raise ValueError('Base file is not a valid PNG.')
            # Insert after IHDR
            ihdr_end = 8 + 4 + 4 + struct.unpack('>I', base_image[8:12])[0] + 4
            return base_image[:ihdr_end] + ztxt_chunk + base_image[ihdr_end:]
        else:
            base = create_minimal_png()
            ihdr_end = 8 + 4 + 4 + 13 + 4  # signature + length + type + IHDR data + CRC
            return base[:ihdr_end] + ztxt_chunk + base[ihdr_end:]
    else:
        # JPEG: Use COM markers, split into chunks if needed
        chunk_size = 65530  # Max COM segment size minus header
        chunks = []
        for i in range(0, len(payload), chunk_size):
            part = payload[i:i+chunk_size]
            b64_part = base64.b64encode(part)
            com_data = MAGIC_MARKER + struct.pack('>H', i // chunk_size) + b64_part
            com_chunk = b'\xff\xfe' + struct.pack('>H', len(com_data) + 2) + com_data
            chunks.append(com_chunk)
        
        if base_image:
            if not base_image[:2] == b'\xff\xd8':
                raise ValueError('Base file is not a valid JPEG.')
            eoi_pos = find_jpeg_eoi(base_image)
            if eoi_pos == -1:
                raise ValueError('Invalid JPEG: EOI not found.')
            return base_image[:eoi_pos] + b''.join(chunks) + base_image[eoi_pos:]
        else:
            prefix, eoi = create_minimal_jpeg()
            return prefix + b''.join(chunks) + eoi


def decode_append(data: bytes, fmt: str) -> bytes:
    """Extract payload appended after image end marker."""
    if fmt == 'png':
        iend_pos = find_png_iend(data)
        if iend_pos == -1:
            raise ValueError('Invalid PNG: IEND not found.')
        return data[iend_pos:]
    else:
        eoi_pos = find_jpeg_eoi(data)
        if eoi_pos == -1:
            raise ValueError('Invalid JPEG: EOI not found.')
        return data[eoi_pos + 2:]


def decode_metadata(data: bytes, fmt: str) -> bytes:
    """Extract payload from image metadata."""
    if fmt == 'png':
        pos = 8  # Skip signature
        while pos < len(data):
            if pos + 8 > len(data):
                break
            chunk_length = struct.unpack('>I', data[pos:pos+4])[0]
            chunk_type = data[pos+4:pos+8]
            chunk_data = data[pos+8:pos+8+chunk_length]
            
            if chunk_type == b'zTXt':
                null_pos = chunk_data.find(b'\x00')
                if null_pos != -1 and chunk_data[:null_pos] == MAGIC_MARKER:
                    # Skip compression method byte
                    compressed = chunk_data[null_pos+2:]
                    return zlib.decompress(compressed)
            pos += 12 + chunk_length
        raise ValueError('No stegpng metadata found in PNG.')
    else:
        # JPEG: Extract from COM markers
        parts = {}
        pos = 2
        while pos < len(data) - 1:
            if data[pos] == 0xff:
                marker = data[pos + 1]
                if marker == 0xd9:  # EOI
                    break
                elif marker == 0xfe:  # COM
                    if pos + 4 > len(data):
                        break
                    seg_length = struct.unpack('>H', data[pos+2:pos+4])[0]
                    com_data = data[pos+4:pos+2+seg_length]
                    if com_data.startswith(MAGIC_MARKER):
                        part_num = struct.unpack('>H', com_data[len(MAGIC_MARKER):len(MAGIC_MARKER)+2])[0]
                        b64_data = com_data[len(MAGIC_MARKER)+2:]
                        parts[part_num] = base64.b64decode(b64_data)
                    pos += 2 + seg_length
                elif marker in (0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0x01, 0x00):
                    pos += 2
                elif marker == 0xff:
                    pos += 1
                else:
                    if pos + 4 > len(data):
                        break
                    seg_length = struct.unpack('>H', data[pos+2:pos+4])[0]
                    pos += 2 + seg_length
            else:
                pos += 1
        
        if not parts:
            raise ValueError('No stegpng metadata found in JPEG.')
        return b''.join(parts[k] for k in sorted(parts.keys()))


def encode_file(input_path: str, output_path: Optional[str], base_path: Optional[str],
                key: Optional[str], method: str, verbose: bool) -> None:
    """Encode a file into an image."""
    # Read input file
    with open(input_path, 'rb') as f:
        payload = f.read()
    
    if verbose:
        print(f"Input file: {input_path} ({len(payload)} bytes)")
    
    # Apply XOR obfuscation if key provided
    if key:
        payload = xor_data(payload, key)
        if verbose:
            print("Applied XOR obfuscation with key")
    
    # Determine output format
    if output_path is None:
        output_path = os.path.splitext(os.path.basename(input_path))[0] + '.png'
    
    ext = os.path.splitext(output_path)[1].lower()
    fmt = 'jpeg' if ext in ('.jpg', '.jpeg') else 'png'
    
    # Load base image if provided
    base_image = None
    if base_path:
        with open(base_path, 'rb') as f:
            base_image = f.read()
        base_fmt = detect_format(base_image)
        if base_fmt != fmt:
            raise ValueError(f"Base image format ({base_fmt}) doesn't match output format ({fmt})")
        if verbose:
            print(f"Using base image: {base_path}")
    
    # Encode payload
    if method == 'append':
        output_data = encode_append(payload, base_image, fmt)
    else:
        output_data = encode_metadata(payload, base_image, fmt)
    
    # Write output
    with open(output_path, 'wb') as f:
        f.write(output_data)
    
    print(f"Encoded {len(payload)} bytes → {output_path} ({len(output_data)} bytes)")
    if key:
        print(f"XOR key used: {key}")


def decode_file(input_path: str, output_path: Optional[str], key: Optional[str],
                method: str, verbose: bool) -> None:
    """Decode a file from an image."""
    # Read input image
    with open(input_path, 'rb') as f:
        data = f.read()
    
    fmt = detect_format(data)
    if fmt is None:
        raise ValueError('Input file is not a valid PNG or JPEG.')
    
    if verbose:
        print(f"Input image: {input_path} ({fmt.upper()}, {len(data)} bytes)")
    
    # Extract payload
    if method == 'append':
        payload = decode_append(data, fmt)
    else:
        payload = decode_metadata(data, fmt)
    
    if not payload:
        raise ValueError('No hidden data found in image.')
    
    # Apply XOR decryption if key provided
    if key:
        payload = xor_data(payload, key)
        if verbose:
            print("Applied XOR decryption with key")
    
    # Determine output filename
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"{base_name}.extracted"
    
    # Write output
    with open(output_path, 'wb') as f:
        f.write(payload)
    
    print(f"Extracted {len(payload)} bytes → {output_path}")
    if key:
        print(f"XOR key used: {key}")


def info_file(input_path: str) -> None:
    """Show information about an image file."""
    with open(input_path, 'rb') as f:
        data = f.read()
    
    fmt = detect_format(data)
    if fmt is None:
        print(f"Error: {input_path} is not a valid PNG or JPEG.")
        return
    
    print(f"File: {input_path}")
    print(f"Format: {fmt.upper()}")
    print(f"Size: {len(data)} bytes")
    
    # Check for hidden data
    try:
        payload = decode_append(data, fmt)
        if payload:
            print(f"Hidden data (append method): {len(payload)} bytes")
    except ValueError:
        # No hidden data found with append method, continue checking metadata
        pass
    
    try:
        payload = decode_metadata(data, fmt)
        if payload:
            print(f"Hidden data (metadata method): {len(payload)} bytes")
    except ValueError:
        # No hidden data found with metadata method
        pass


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Steganographic file encoder/decoder for PNG and JPEG images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Encode a file into a PNG
  stegpng encode secret.txt -o hidden.png
  
  # Encode with XOR obfuscation
  stegpng encode secret.txt -o hidden.png -k "mypassword"
  
  # Encode using metadata method (survives some image processors)
  stegpng encode secret.txt -o hidden.png -m metadata
  
  # Use existing image as cover
  stegpng encode secret.txt -o hidden.png -b cover.png
  
  # Decode a file from an image
  stegpng decode hidden.png -o recovered.txt
  
  # Decode with XOR key
  stegpng decode hidden.png -o recovered.txt -k "mypassword"
  
  # Show image info
  stegpng info hidden.png
''')
    
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {__version__}')
    
    subparsers = parser.add_subparsers(dest='command', required=True, help='Command to run')
    
    # Encode subcommand
    encode_parser = subparsers.add_parser('encode', help='Encode a file into an image')
    encode_parser.add_argument('input', help='Input file to encode')
    encode_parser.add_argument('-o', '--output', help='Output image file (default: <input>.png)')
    encode_parser.add_argument('-b', '--base', help='Base image to use as cover')
    encode_parser.add_argument('-k', '--key', help='XOR key for obfuscation')
    encode_parser.add_argument('-m', '--method', choices=['append', 'metadata'], default='append',
                               help='Embedding method (default: append)')
    encode_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Decode subcommand
    decode_parser = subparsers.add_parser('decode', help='Decode a file from an image')
    decode_parser.add_argument('input', help='Input image file')
    decode_parser.add_argument('-o', '--output', help='Output file (default: <input>.extracted)')
    decode_parser.add_argument('-k', '--key', help='XOR key for decryption')
    decode_parser.add_argument('-m', '--method', choices=['append', 'metadata'], default='append',
                               help='Extraction method (default: append)')
    decode_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Info subcommand
    info_parser = subparsers.add_parser('info', help='Show image information')
    info_parser.add_argument('input', help='Input image file')
    
    args = parser.parse_args()
    
    try:
        if args.command == 'encode':
            if not os.path.exists(args.input):
                print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
                return 1
            encode_file(args.input, args.output, args.base, args.key, args.method, args.verbose)
        
        elif args.command == 'decode':
            if not os.path.exists(args.input):
                print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
                return 1
            decode_file(args.input, args.output, args.key, args.method, args.verbose)
        
        elif args.command == 'info':
            if not os.path.exists(args.input):
                print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
                return 1
            info_file(args.input)
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
