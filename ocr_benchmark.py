#!/usr/bin/env python3
"""
OCR Engine Benchmark Tool

Evaluates OCR engine accuracy by comparing output against ground truth text.
Renders text as synthetic "scanned" documents with configurable degradation.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# Import pdfocr module for OCR functions
import pdfocr


@dataclass
class BenchmarkResult:
    """Results from benchmarking a single OCR engine."""

    engine: str
    char_accuracy: float
    cer: float  # Character Error Rate
    wer: float  # Word Error Rate
    bleu: float
    exact_line_match: float
    jaccard: float
    time_seconds: float
    ocr_text: str
    errors: List[Dict[str, Any]]  # List of specific errors found


@dataclass
class ScanSettings:
    """Settings for simulating scanned document degradation."""

    noise_level: float = 0.0
    blur_radius: float = 0.0
    rotation_deg: float = 0.0
    contrast: float = 1.0
    background_gray: int = 255
    jpeg_quality: int = 95


# Quality presets for different scan conditions
QUALITY_PRESETS = {
    "pristine": ScanSettings(),
    "good_scan": ScanSettings(
        noise_level=0.02, blur_radius=0.3, background_gray=252
    ),
    "average_scan": ScanSettings(
        noise_level=0.05, blur_radius=0.5, background_gray=248, contrast=0.95
    ),
    "poor_scan": ScanSettings(
        noise_level=0.1,
        blur_radius=0.8,
        rotation_deg=0.5,
        background_gray=240,
        contrast=0.9,
    ),
    "photocopy": ScanSettings(
        noise_level=0.15,
        blur_radius=1.0,
        rotation_deg=1.0,
        background_gray=235,
        contrast=0.85,
        jpeg_quality=75,
    ),
}


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance between two strings using dynamic programming.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Minimum number of single-character edits needed to transform s1 into s2
    """
    # Try to use rapidfuzz for performance if available
    try:
        from rapidfuzz.distance import Levenshtein

        return Levenshtein.distance(s1, s2)
    except ImportError:
        pass

    # Fallback to pure Python implementation
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    # Create distance matrix
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_cer(ocr_text: str, ground_truth: str) -> float:
    """Calculate Character Error Rate.

    Args:
        ocr_text: OCR output text
        ground_truth: Ground truth text

    Returns:
        Character Error Rate (0.0 = perfect, 1.0 = completely wrong)
    """
    if len(ground_truth) == 0:
        return 0.0 if len(ocr_text) == 0 else 1.0

    distance = levenshtein_distance(ocr_text, ground_truth)
    return distance / len(ground_truth)


def calculate_wer(ocr_text: str, ground_truth: str) -> float:
    """Calculate Word Error Rate.

    Args:
        ocr_text: OCR output text
        ground_truth: Ground truth text

    Returns:
        Word Error Rate (0.0 = perfect, 1.0 = completely wrong)
    """
    ocr_words = ocr_text.split()
    truth_words = ground_truth.split()

    if len(truth_words) == 0:
        return 0.0 if len(ocr_words) == 0 else 1.0

    # Calculate word-level edit distance
    distance = levenshtein_distance(" ".join(ocr_words), " ".join(truth_words))
    # Normalize by character count as an approximation (word-level distance would require alignment)
    # This gives a reasonable estimate of word-level errors
    avg_word_length = len(ground_truth) / max(len(truth_words), 1)
    return min(distance / (len(truth_words) * avg_word_length), 1.0)


def calculate_bleu(ocr_text: str, ground_truth: str) -> float:
    """Calculate BLEU score.

    Args:
        ocr_text: OCR output text
        ground_truth: Ground truth text

    Returns:
        BLEU score (0.0 = worst, 1.0 = perfect)
    """
    try:
        from sacrebleu import sentence_bleu

        score = sentence_bleu(ocr_text, [ground_truth])
        return score.score / 100.0
    except ImportError:
        pass

    try:
        from nltk.translate.bleu_score import sentence_bleu as nltk_bleu

        reference = [ground_truth.split()]
        candidate = ocr_text.split()
        score = nltk_bleu(reference, candidate)
        return score
    except ImportError:
        pass

    # Simple fallback: use character-level precision
    ocr_chars = set(ocr_text)
    truth_chars = set(ground_truth)
    if len(truth_chars) == 0:
        return 1.0 if len(ocr_chars) == 0 else 0.0
    precision = len(ocr_chars & truth_chars) / len(truth_chars)
    return precision


def calculate_jaccard(ocr_text: str, ground_truth: str) -> float:
    """Calculate Jaccard similarity coefficient.

    Args:
        ocr_text: OCR output text
        ground_truth: Ground truth text

    Returns:
        Jaccard similarity (0.0 = no overlap, 1.0 = identical word sets)
    """
    ocr_words = set(ocr_text.split())
    truth_words = set(ground_truth.split())

    if len(ocr_words) == 0 and len(truth_words) == 0:
        return 1.0

    intersection = len(ocr_words & truth_words)
    union = len(ocr_words | truth_words)

    if union == 0:
        return 0.0

    return intersection / union


def calculate_exact_line_match(ocr_text: str, ground_truth: str) -> float:
    """Calculate percentage of lines that match exactly.

    Args:
        ocr_text: OCR output text
        ground_truth: Ground truth text

    Returns:
        Percentage of exact line matches (0.0 to 1.0)
    """
    ocr_lines = ocr_text.split("\n")
    truth_lines = ground_truth.split("\n")

    if len(truth_lines) == 0:
        return 1.0 if len(ocr_lines) == 0 else 0.0

    matches = sum(1 for o, t in zip(ocr_lines, truth_lines) if o == t)
    return matches / len(truth_lines)


def find_errors(ocr_text: str, ground_truth: str) -> List[Dict[str, Any]]:
    """Find and categorize specific OCR errors.

    Note: This is a simplified implementation that does character-by-character comparison.
    For more accurate error detection, a proper alignment algorithm should be used.

    Args:
        ocr_text: OCR output text
        ground_truth: Ground truth text

    Returns:
        List of error dictionaries with details
    """
    errors = []

    # Character-level comparison (note: this is approximate and doesn't handle alignment)
    for i, (o, t) in enumerate(zip(ocr_text, ground_truth)):
        if o != t:
            errors.append(
                {"position": i, "expected": t, "got": o, "type": "substitution"}
            )

    # Handle length differences
    if len(ocr_text) < len(ground_truth):
        for i in range(len(ocr_text), len(ground_truth)):
            errors.append(
                {"position": i, "expected": ground_truth[i], "got": "", "type": "deletion"}
            )
    elif len(ocr_text) > len(ground_truth):
        for i in range(len(ground_truth), len(ocr_text)):
            errors.append(
                {"position": i, "expected": "", "got": ocr_text[i], "type": "insertion"}
            )

    return errors


def render_text_to_image(
    text: str,
    settings: ScanSettings,
    font_path: Optional[str] = None,
    font_size: int = 14,
    dpi: int = 300,
) -> Image.Image:
    """Render text to a realistic 'scanned' image with degradation.

    Args:
        text: Text to render
        settings: Scan simulation settings
        font_path: Path to font file (uses default if None)
        font_size: Font size in points
        dpi: Image resolution

    Returns:
        PIL Image with rendered and degraded text
    """
    # Load font
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            # Try common monospace fonts
            for font_name in [
                "DejaVuSansMono.ttf",
                "LiberationMono-Regular.ttf",
                "FreeMono.ttf",
                "DejaVuSans.ttf",
            ]:
                try:
                    font = ImageFont.truetype(font_name, font_size)
                    break
                except (OSError, IOError):
                    continue
            else:
                # Fall back to default font
                font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Calculate image dimensions
    lines = text.split("\n")
    
    # Use textbbox to get accurate text dimensions
    temp_img = Image.new("RGB", (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    
    max_width = 0
    total_height = 0
    line_heights = []
    
    for line in lines:
        if line:  # Only measure non-empty lines
            bbox = temp_draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            line_height = bbox[3] - bbox[1]
        else:
            bbox = temp_draw.textbbox((0, 0), "X", font=font)
            line_width = 0
            line_height = bbox[3] - bbox[1]
        
        max_width = max(max_width, line_width)
        line_heights.append(line_height)
        total_height += line_height
    
    # Add margins
    margin = 40
    width = max_width + 2 * margin
    height = total_height + 2 * margin

    # Create image with background
    image = Image.new("RGB", (width, height), (settings.background_gray,) * 3)
    draw = ImageDraw.Draw(image)

    # Draw text
    y = margin
    for line, line_height in zip(lines, line_heights):
        draw.text((margin, y), line, fill=(0, 0, 0), font=font)
        y += line_height

    # Apply degradations
    if settings.contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(settings.contrast)

    if settings.blur_radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=settings.blur_radius))

    if settings.rotation_deg != 0:
        image = image.rotate(settings.rotation_deg, fillcolor=(settings.background_gray,) * 3, expand=True)

    if settings.noise_level > 0:
        image = add_scan_noise(image, settings.noise_level)

    if settings.jpeg_quality < 100:
        image = apply_jpeg_artifacts(image, settings.jpeg_quality)

    return image


def add_scan_noise(image: Image.Image, noise_level: float) -> Image.Image:
    """Add Gaussian noise to simulate scanner artifacts.

    Args:
        image: Input image
        noise_level: Noise intensity (0.0 to 1.0)

    Returns:
        Noisy image
    """
    img_array = np.array(image, dtype=np.float32)
    noise = np.random.normal(0, noise_level * 255, img_array.shape)
    noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_jpeg_artifacts(image: Image.Image, quality: int) -> Image.Image:
    """Apply JPEG compression artifacts.

    Args:
        image: Input image
        quality: JPEG quality (1-100)

    Returns:
        Image with compression artifacts
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)


def benchmark_engine(
    engine: str,
    image: Image.Image,
    ground_truth: str,
    gpu: bool = False,
) -> BenchmarkResult:
    """Run benchmark for a single OCR engine.

    Args:
        engine: Name of OCR engine ('tesseract', 'easyocr', etc.)
        image: Image to OCR
        ground_truth: Ground truth text
        gpu: Whether to use GPU acceleration

    Returns:
        BenchmarkResult with all metrics
    """
    start_time = time.time()

    try:
        # Run OCR based on engine type
        if engine == "tesseract":
            ocr_text = pdfocr.ocr_with_tesseract(image)
        elif engine == "easyocr":
            ocr_text = pdfocr.ocr_with_easyocr(image, langs=["en"], gpu=gpu)
        else:
            # For engines not yet implemented in pdfocr, skip
            ocr_text = ""
            print(f"Warning: Engine '{engine}' not implemented yet, skipping...")

        elapsed = time.time() - start_time

        # Calculate metrics
        cer = calculate_cer(ocr_text, ground_truth)
        wer = calculate_wer(ocr_text, ground_truth)
        bleu = calculate_bleu(ocr_text, ground_truth)
        jaccard = calculate_jaccard(ocr_text, ground_truth)
        exact_match = calculate_exact_line_match(ocr_text, ground_truth)
        char_accuracy = 1.0 - cer
        errors = find_errors(ocr_text, ground_truth)

        return BenchmarkResult(
            engine=engine,
            char_accuracy=char_accuracy,
            cer=cer,
            wer=wer,
            bleu=bleu,
            exact_line_match=exact_match,
            jaccard=jaccard,
            time_seconds=elapsed,
            ocr_text=ocr_text,
            errors=errors,
        )

    except Exception as e:
        print(f"Error benchmarking {engine}: {e}")
        # Return empty result
        return BenchmarkResult(
            engine=engine,
            char_accuracy=0.0,
            cer=1.0,
            wer=1.0,
            bleu=0.0,
            exact_line_match=0.0,
            jaccard=0.0,
            time_seconds=time.time() - start_time,
            ocr_text="",
            errors=[],
        )


def run_benchmark(
    text_file: Path,
    engines: List[str],
    settings: ScanSettings,
    gpu: bool = False,
    save_images_dir: Optional[Path] = None,
) -> Tuple[List[BenchmarkResult], str]:
    """Run full benchmark across all specified engines.

    Args:
        text_file: Path to ground truth text file
        engines: List of engine names to test
        settings: Scan simulation settings
        gpu: Whether to use GPU acceleration
        save_images_dir: Optional directory to save test images

    Returns:
        Tuple of (results list, ground truth text)
    """
    # Load ground truth
    ground_truth = text_file.read_text()

    # Render text to image
    print("Rendering text to image...")
    image = render_text_to_image(ground_truth, settings)

    # Save image if requested
    if save_images_dir:
        save_images_dir.mkdir(parents=True, exist_ok=True)
        image_path = save_images_dir / "test_image.png"
        image.save(image_path)
        print(f"Saved test image to: {image_path}")

    # Run benchmarks
    results = []
    for engine in engines:
        print(f"Benchmarking {engine}...")
        result = benchmark_engine(engine, image, ground_truth, gpu=gpu)
        results.append(result)

    # Sort by character accuracy (descending)
    results.sort(key=lambda r: r.char_accuracy, reverse=True)

    return results, ground_truth


def print_results(
    results: List[BenchmarkResult], ground_truth: str, settings: ScanSettings, verbose: bool = False
) -> None:
    """Print formatted benchmark results.

    Args:
        results: List of benchmark results
        ground_truth: Ground truth text
        settings: Scan settings used
        verbose: Whether to show detailed output
    """
    # Calculate stats
    num_chars = len(ground_truth)
    num_words = len(ground_truth.split())
    num_lines = len(ground_truth.split("\n"))

    # Print header
    print("\n" + "=" * 80)
    print(" " * 20 + "OCR ENGINE BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Ground Truth: {num_chars} characters, {num_words} words, {num_lines} lines")
    print(f"Scan Settings: noise={settings.noise_level:.2f}, blur={settings.blur_radius:.1f}, "
          f"contrast={settings.contrast:.2f}")
    print("=" * 80)
    print()

    # Print results table
    print("RESULTS (ranked by Character Accuracy):")
    print()

    # Table header
    header = (
        "┌─────────────────┬───────────┬─────────┬─────────┬─────────┬─────────┬─────────┐"
    )
    print(header)
    print(
        "│ Engine          │ Char Acc  │ CER     │ WER     │ BLEU    │ Exact   │ Time    │"
    )
    print(
        "├─────────────────┼───────────┼─────────┼─────────┼─────────┼─────────┼─────────┤"
    )

    # Medals for top 3
    medals = ["🥇", "🥈", "🥉"]

    for i, result in enumerate(results):
        medal = medals[i] if i < 3 else "  "
        print(
            f"│ {medal} {result.engine:<13} │ {result.char_accuracy*100:>6.2f}%   │ "
            f"{result.cer:>7.4f} │ {result.wer:>7.4f} │ {result.bleu:>7.4f} │ "
            f"{result.exact_line_match*100:>5.1f}%  │ {result.time_seconds:>6.2f}s │"
        )

    print(
        "└─────────────────┴───────────┴─────────┴─────────┴─────────┴─────────┴─────────┘"
    )
    print()

    if verbose:
        print("DETAILED METRICS:")
        print("─" * 80)
        for result in results:
            correct_chars = int(num_chars * result.char_accuracy)
            # WER is an approximate metric; calculate estimated correct words
            # This is a rough estimate since WER normalization is complex
            estimated_correct_words = max(0, int(num_words * (1.0 - result.wer)))
            print(f"\n{result.engine}:")
            print(f"  - Characters: {correct_chars}/{num_chars} correct")
            print(f"  - Words: ~{estimated_correct_words}/{num_words} correct (estimate)")
            print(f"  - Total errors: {len(result.errors)}")
            
            # Find common error patterns
            error_counts: Dict[str, int] = {}
            for error in result.errors[:20]:  # Limit to first 20
                if error["type"] == "substitution":
                    key = f"'{error['expected']}' → '{error['got']}'"
                    error_counts[key] = error_counts.get(key, 0) + 1
            
            if error_counts:
                common = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                print("  - Common errors:", ", ".join(f"{k} ({v}x)" for k, v in common))

        print("\n" + "=" * 80)


def save_results_json(results: List[BenchmarkResult], output_path: Path) -> None:
    """Save results to JSON file.

    Args:
        results: List of benchmark results
        output_path: Path to output JSON file
    """
    data = [asdict(result) for result in results]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to: {output_path}")


def save_results_csv(results: List[BenchmarkResult], output_path: Path) -> None:
    """Save results to CSV file.

    Args:
        results: List of benchmark results
        output_path: Path to output CSV file
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Engine",
                "Char Accuracy",
                "CER",
                "WER",
                "BLEU",
                "Exact Match",
                "Jaccard",
                "Time (s)",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.engine,
                    f"{result.char_accuracy:.4f}",
                    f"{result.cer:.4f}",
                    f"{result.wer:.4f}",
                    f"{result.bleu:.4f}",
                    f"{result.exact_line_match:.4f}",
                    f"{result.jaccard:.4f}",
                    f"{result.time_seconds:.2f}",
                ]
            )
    print(f"Results saved to: {output_path}")


def main() -> int:
    """Main entry point for the benchmark tool."""
    parser = argparse.ArgumentParser(
        description="Benchmark OCR engines against ground truth text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("text_file", type=Path, help="Plain text file (ground truth)")
    parser.add_argument(
        "-e",
        "--engines",
        nargs="+",
        default=["tesseract", "easyocr"],
        help="OCR engines to benchmark (default: tesseract easyocr)",
    )
    parser.add_argument(
        "--quality",
        choices=list(QUALITY_PRESETS.keys()),
        default="average_scan",
        help="Scan quality preset (default: average_scan)",
    )
    parser.add_argument(
        "--noise", type=float, help="Custom noise level (0.0-1.0)"
    )
    parser.add_argument("--blur", type=float, help="Custom blur radius")
    parser.add_argument("--rotation", type=float, help="Custom rotation in degrees")
    parser.add_argument(
        "--gpu", action="store_true", help="Use GPU acceleration"
    )
    parser.add_argument(
        "--save-images", type=Path, help="Save test images to directory"
    )
    parser.add_argument(
        "--output-json", type=Path, help="Save results as JSON"
    )
    parser.add_argument("--output-csv", type=Path, help="Save results as CSV")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed output"
    )

    args = parser.parse_args()

    # Validate input file
    if not args.text_file.exists():
        print(f"Error: Text file not found: {args.text_file}", file=sys.stderr)
        return 1

    # Get scan settings (create a copy to avoid mutating the preset)
    from dataclasses import replace
    settings = replace(QUALITY_PRESETS[args.quality])

    # Apply custom overrides
    if args.noise is not None:
        settings = replace(settings, noise_level=args.noise)
    if args.blur is not None:
        settings = replace(settings, blur_radius=args.blur)
    if args.rotation is not None:
        settings = replace(settings, rotation_deg=args.rotation)

    # Run benchmark
    try:
        results, ground_truth = run_benchmark(
            args.text_file, args.engines, settings, args.gpu, args.save_images
        )

        # Print results
        print_results(results, ground_truth, settings, args.verbose)

        # Save to files if requested
        if args.output_json:
            save_results_json(results, args.output_json)

        if args.output_csv:
            save_results_csv(results, args.output_csv)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
