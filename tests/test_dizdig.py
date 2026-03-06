"""Comprehensive tests for dizdig.py - BBS/FTP archive organizer."""

import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import dizdig

DIZDIG_PATH = Path(__file__).parent.parent / "dizdig.py"
BBS_PATH = Path(__file__).parent / "BBS"


def make_pkg(
    base: Path,
    name: str,
    diz: str = "A test package",
    files: Optional[Dict[str, bytes]] = None,
) -> Path:
    """Create a synthetic BBS package directory with FILE_ID.DIZ."""
    pkg = base / name
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "FILE_ID.DIZ").write_text(diz, encoding="utf-8")
    if files:
        for fname, content in files.items():
            data = content if isinstance(content, bytes) else content.encode()
            (pkg / fname).write_bytes(data)
    return pkg


def run_dizdig(*args: str, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    """Run dizdig.py with the given arguments and return CompletedProcess."""
    return subprocess.run(
        [sys.executable, str(DIZDIG_PATH)] + list(args),
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
    )


# ── Unit Tests ────────────────────────────────────────────────────────────────


class TestMatchesDiz:
    """Tests for matches_diz() function."""

    def test_text_mode_simple_match(self):
        """Text mode finds plain substring."""
        assert dizdig.matches_diz("This is a tracker module", "tracker", "text") is True

    def test_text_mode_case_insensitive(self):
        """Text mode is case-insensitive."""
        assert dizdig.matches_diz("TRACKER MODULE", "tracker", "text") is True
        assert dizdig.matches_diz("tracker module", "TRACKER", "text") is True

    def test_text_mode_no_match(self):
        """Text mode returns False when pattern not found."""
        assert dizdig.matches_diz("audio player", "tracker", "text") is False

    def test_text_mode_empty_content(self):
        """Text mode with empty content returns False for non-empty pattern."""
        assert dizdig.matches_diz("", "tracker", "text") is False

    def test_text_mode_empty_pattern(self):
        """Text mode with empty pattern returns True (empty string is always contained)."""
        assert dizdig.matches_diz("any content", "", "text") is True

    def test_text_mode_special_characters_literal(self):
        """Text mode treats special regex characters literally."""
        assert dizdig.matches_diz("(C) 1995 Author", "(c) 1995", "text") is True

    def test_wildcard_mode_star_match(self):
        """Wildcard mode matches with * wildcard."""
        assert dizdig.matches_diz("tracker module player", "*tracker*", "wildcard") is True

    def test_wildcard_mode_question_mark(self):
        """Wildcard mode matches single character with ?."""
        assert dizdig.matches_diz("mod", "mo?", "wildcard") is True

    def test_wildcard_mode_no_match(self):
        """Wildcard mode returns False when pattern does not match."""
        assert dizdig.matches_diz("audio player", "*tracker*", "wildcard") is False

    def test_wildcard_mode_case_insensitive(self):
        """Wildcard mode is case-insensitive."""
        assert dizdig.matches_diz("TRACKER MODULE", "*tracker*", "wildcard") is True

    def test_wildcard_mode_exact_match(self):
        """Wildcard mode can match exact strings without wildcards."""
        assert dizdig.matches_diz("hello", "hello", "wildcard") is True
        assert dizdig.matches_diz("hello world", "hello", "wildcard") is False

    def test_regex_mode_simple(self):
        """Regex mode finds simple pattern."""
        assert dizdig.matches_diz("version 5.0 release", r"5\.[0-9]+", "regex") is True

    def test_regex_mode_no_match(self):
        """Regex mode returns False when pattern does not match."""
        assert dizdig.matches_diz("version 4.0 release", r"5\.[0-9]+", "regex") is False

    def test_regex_mode_case_insensitive(self):
        """Regex mode is case-insensitive."""
        assert dizdig.matches_diz("TRACKER MODULE", "tracker", "regex") is True

    def test_regex_mode_anchored(self):
        """Regex mode supports anchored patterns."""
        assert dizdig.matches_diz("hello world", "^hello", "regex") is True
        assert dizdig.matches_diz("say hello world", "^hello", "regex") is False

    def test_regex_mode_special_chars(self):
        """Regex mode interprets regex special characters."""
        assert dizdig.matches_diz("v2.0", r"v\d+\.\d+", "regex") is True


class TestParseSizeExpr:
    """Tests for parse_size_expr() function."""

    def test_less_than_operator(self):
        """Less-than operator parses and evaluates correctly."""
        op, val, _ = dizdig.parse_size_expr("< 100")
        assert val == 100
        assert op(50, 100) is True
        assert op(150, 100) is False

    def test_greater_than_operator(self):
        """Greater-than operator parses and evaluates correctly."""
        op, val, _ = dizdig.parse_size_expr("> 100")
        assert val == 100
        assert op(150, 100) is True
        assert op(50, 100) is False

    def test_less_than_equal_operator(self):
        """Less-than-or-equal operator includes boundary."""
        op, val, _ = dizdig.parse_size_expr("<= 100")
        assert op(100, 100) is True
        assert op(101, 100) is False

    def test_greater_than_equal_operator(self):
        """Greater-than-or-equal operator includes boundary."""
        op, val, _ = dizdig.parse_size_expr(">= 100")
        assert op(100, 100) is True
        assert op(99, 100) is False

    def test_equal_operator(self):
        """Equal operator matches exact value."""
        op, val, _ = dizdig.parse_size_expr("== 512")
        assert val == 512
        assert op(512, 512) is True
        assert op(511, 512) is False

    def test_not_equal_operator(self):
        """Not-equal operator excludes exact value."""
        op, val, _ = dizdig.parse_size_expr("!= 0")
        assert val == 0
        assert op(1, 0) is True
        assert op(0, 0) is False

    def test_unit_bytes_no_suffix(self):
        """No unit means raw bytes."""
        _, val, _ = dizdig.parse_size_expr(">= 512")
        assert val == 512

    def test_unit_kb(self):
        """KB unit multiplies by 1024."""
        _, val, _ = dizdig.parse_size_expr(">= 10KB")
        assert val == 10 * 1024

    def test_unit_mb(self):
        """MB unit multiplies by 1024^2."""
        _, val, _ = dizdig.parse_size_expr(">= 1MB")
        assert val == 1024 ** 2

    def test_unit_gb(self):
        """GB unit multiplies by 1024^3."""
        _, val, _ = dizdig.parse_size_expr(">= 1GB")
        assert val == 1024 ** 3

    def test_unit_tb(self):
        """TB unit multiplies by 1024^4."""
        _, val, _ = dizdig.parse_size_expr(">= 1TB")
        assert val == 1024 ** 4

    def test_decimal_value(self):
        """Decimal values are handled correctly."""
        _, val, _ = dizdig.parse_size_expr(">= 1.5MB")
        assert val == int(1.5 * 1024 ** 2)

    def test_whitespace_around_value(self):
        """Whitespace between operator and value is stripped."""
        _, val, _ = dizdig.parse_size_expr(">=  10KB")
        assert val == 10 * 1024

    def test_whitespace_leading_trailing(self):
        """Leading/trailing whitespace around full expression is stripped."""
        _, val, _ = dizdig.parse_size_expr("  >= 10KB  ")
        assert val == 10 * 1024

    def test_whitespace_between_number_and_unit(self):
        """Whitespace between number and unit is accepted."""
        _, val, _ = dizdig.parse_size_expr(">= 10 KB")
        assert val == 10 * 1024

    def test_unit_lowercase(self):
        """Unit matching is case-insensitive."""
        _, val1, _ = dizdig.parse_size_expr(">= 1kb")
        _, val2, _ = dizdig.parse_size_expr(">= 1KB")
        assert val1 == val2 == 1024

    def test_third_element_preserves_original(self):
        """Third tuple element is the original expression string."""
        expr = ">= 10KB"
        _, _, orig = dizdig.parse_size_expr(expr)
        assert orig == expr

    def test_invalid_no_operator_raises(self):
        """Missing operator raises ValueError."""
        with pytest.raises(ValueError, match="Invalid size expression"):
            dizdig.parse_size_expr("10KB")

    def test_invalid_empty_value_after_operator_raises(self):
        """Operator with no following value raises ValueError."""
        with pytest.raises(ValueError, match="Invalid size expression"):
            dizdig.parse_size_expr(">= ")

    def test_invalid_unknown_unit_raises(self):
        """Unknown unit raises ValueError."""
        with pytest.raises(ValueError, match="Unknown size unit"):
            dizdig.parse_size_expr(">= 10XB")

    def test_invalid_empty_string_raises(self):
        """Empty expression raises ValueError."""
        with pytest.raises(ValueError):
            dizdig.parse_size_expr("")


class TestDirSize:
    """Tests for dir_size() function."""

    def test_empty_directory(self, tmp_path: Path):
        """Empty directory has size 0."""
        assert dizdig.dir_size(tmp_path) == 0

    def test_known_file_sizes(self, tmp_path: Path):
        """Sum of direct file sizes is correct."""
        (tmp_path / "a.txt").write_bytes(b"x" * 100)
        (tmp_path / "b.txt").write_bytes(b"y" * 200)
        assert dizdig.dir_size(tmp_path) == 300

    def test_single_file(self, tmp_path: Path):
        """Single file size is reported correctly."""
        (tmp_path / "exactly.bin").write_bytes(b"\x00" * 1024)
        assert dizdig.dir_size(tmp_path) == 1024

    def test_non_recursive_ignores_subdirectory_contents(self, tmp_path: Path):
        """Only direct files are counted — subdirectory contents are ignored."""
        (tmp_path / "file.txt").write_bytes(b"x" * 100)
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_bytes(b"y" * 500)
        assert dizdig.dir_size(tmp_path) == 100

    def test_multiple_files(self, tmp_path: Path):
        """Multiple files with different sizes are summed correctly."""
        sizes = [10, 20, 30, 40, 50]
        for i, size in enumerate(sizes):
            (tmp_path / f"file{i}.bin").write_bytes(b"\x00" * size)
        assert dizdig.dir_size(tmp_path) == sum(sizes)


class TestParseExclude:
    """Tests for parse_exclude() function."""

    def test_valid_ext_exclude(self):
        """Valid ext: prefix returns (prefix, value) tuple."""
        prefix, value = dizdig.parse_exclude("ext:.exe")
        assert prefix == "ext"
        assert value == ".exe"

    def test_valid_diz_exclude(self):
        """Valid diz: prefix returns (prefix, value) tuple."""
        prefix, value = dizdig.parse_exclude("diz:broken")
        assert prefix == "diz"
        assert value == "broken"

    def test_valid_size_exclude(self):
        """Valid size: prefix returns (prefix, value) tuple."""
        prefix, value = dizdig.parse_exclude("size:>1MB")
        assert prefix == "size"
        assert value == ">1MB"

    def test_prefix_is_lowercased(self):
        """Prefix is normalised to lowercase."""
        prefix, _ = dizdig.parse_exclude("EXT:.exe")
        assert prefix == "ext"

    def test_value_with_colon_preserved(self):
        """Values that contain a colon are preserved intact."""
        prefix, value = dizdig.parse_exclude("size:>=1MB")
        assert prefix == "size"
        assert value == ">=1MB"

    def test_missing_colon_raises(self):
        """String without a colon raises ValueError."""
        with pytest.raises(ValueError, match="Invalid --exclude format"):
            dizdig.parse_exclude("ext.exe")

    def test_unknown_prefix_raises(self):
        """Unrecognised prefix raises ValueError."""
        with pytest.raises(ValueError, match="Unknown --exclude prefix"):
            dizdig.parse_exclude("name:something")

    def test_empty_value_raises(self):
        """Empty value after the colon raises ValueError."""
        with pytest.raises(ValueError, match="requires a value"):
            dizdig.parse_exclude("ext:")

    def test_all_valid_prefixes_accepted(self):
        """All three valid prefixes are accepted without error."""
        dizdig.parse_exclude("ext:.zip")
        dizdig.parse_exclude("diz:cracked")
        dizdig.parse_exclude("size:>512")


class TestCheckExcludes:
    """Tests for check_excludes() function."""

    def test_exclude_by_extension(self, tmp_path: Path):
        """Package containing the excluded extension is excluded."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "program.exe").write_bytes(b"")
        assert dizdig.check_excludes(pkg, None, [("ext", ".exe")], "text") is True

    def test_not_excluded_by_extension(self, tmp_path: Path):
        """Package without the excluded extension is not excluded."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "music.mod").write_bytes(b"")
        assert dizdig.check_excludes(pkg, None, [("ext", ".exe")], "text") is False

    def test_exclude_by_diz_content(self, tmp_path: Path):
        """Package whose DIZ content matches is excluded."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        diz = "This package is broken and should be excluded"
        assert dizdig.check_excludes(pkg, diz, [("diz", "broken")], "text") is True

    def test_not_excluded_by_diz_content(self, tmp_path: Path):
        """Package whose DIZ content does not match is not excluded."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        assert dizdig.check_excludes(pkg, "fine package", [("diz", "broken")], "text") is False

    def test_exclude_by_size(self, tmp_path: Path):
        """Package matching the size condition is excluded."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "big.bin").write_bytes(b"x" * (2 * 1024 * 1024))  # 2 MB
        assert dizdig.check_excludes(pkg, None, [("size", ">1MB")], "text") is True

    def test_not_excluded_by_size(self, tmp_path: Path):
        """Package not matching the size condition is not excluded."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "small.bin").write_bytes(b"x" * 100)
        assert dizdig.check_excludes(pkg, None, [("size", ">1MB")], "text") is False

    def test_any_matching_rule_excludes(self, tmp_path: Path):
        """Any one matching rule is sufficient to exclude the package."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "music.mod").write_bytes(b"")
        diz = "broken package"
        excludes = [("ext", ".exe"), ("diz", "broken")]
        # .exe does not match, but diz: does → package should be excluded
        assert dizdig.check_excludes(pkg, diz, excludes, "text") is True

    def test_no_excludes_never_excludes(self, tmp_path: Path):
        """Empty exclude list never excludes any package."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "program.exe").write_bytes(b"")
        assert dizdig.check_excludes(pkg, "anything", [], "text") is False

    def test_none_diz_content_skips_diz_rule(self, tmp_path: Path):
        """None DIZ content does not trigger diz: exclusion."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        assert dizdig.check_excludes(pkg, None, [("diz", "broken")], "text") is False

    def test_multiple_rules_all_miss(self, tmp_path: Path):
        """Package survives when no exclude rule fires."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "music.mod").write_bytes(b"x" * 50)
        excludes = [("ext", ".exe"), ("diz", "broken"), ("size", ">1MB")]
        assert dizdig.check_excludes(pkg, "clean package", excludes, "text") is False


class TestPresets:
    """Tests for the PRESETS dictionary."""

    def test_all_required_preset_names_exist(self):
        """All seven required preset names are present."""
        required = {"tracker", "music", "graphics", "source", "dos-exe", "document", "archive"}
        assert required.issubset(set(dizdig.PRESETS.keys()))

    def test_all_extensions_start_with_dot(self):
        """Every extension in every preset starts with a dot."""
        for preset_name, exts in dizdig.PRESETS.items():
            for ext in exts:
                assert ext.startswith("."), (
                    f"Preset '{preset_name}' extension '{ext}' does not start with '.'"
                )

    def test_no_empty_preset_lists(self):
        """No preset has an empty extension list."""
        for preset_name, exts in dizdig.PRESETS.items():
            assert len(exts) > 0, f"Preset '{preset_name}' is empty"

    def test_tracker_preset_has_common_module_formats(self):
        """Tracker preset includes the most common module tracker formats."""
        exts = dizdig.PRESETS["tracker"]
        for expected in (".mod", ".s3m", ".xm", ".it"):
            assert expected in exts

    def test_dos_exe_preset_has_executables(self):
        """dos-exe preset includes standard DOS executable formats."""
        exts = dizdig.PRESETS["dos-exe"]
        for expected in (".exe", ".com", ".bat"):
            assert expected in exts

    def test_archive_preset_has_retro_formats(self):
        """Archive preset includes common retro archive formats."""
        exts = dizdig.PRESETS["archive"]
        for expected in (".zip", ".arj", ".lzh"):
            assert expected in exts

    def test_graphics_preset_has_image_formats(self):
        """Graphics preset includes common image formats."""
        exts = dizdig.PRESETS["graphics"]
        for expected in (".gif", ".bmp", ".jpg"):
            assert expected in exts

    def test_source_preset_has_code_extensions(self):
        """Source preset includes common programming language extensions."""
        exts = dizdig.PRESETS["source"]
        for expected in (".c", ".cpp", ".py"):
            assert expected in exts

    def test_document_preset_has_text_formats(self):
        """Document preset includes common text/document formats."""
        exts = dizdig.PRESETS["document"]
        for expected in (".txt", ".nfo", ".doc"):
            assert expected in exts


# ── CLI Integration Tests ─────────────────────────────────────────────────────


class TestCLI:
    """Integration tests for the dizdig command-line interface."""

    def test_no_arguments_prints_help_and_exits_zero(self):
        """Running with no arguments prints help text and exits 0."""
        result = run_dizdig()
        assert result.returncode == 0
        assert "dizdig" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_version_long_flag(self):
        """--version prints version string and exits 0."""
        result = run_dizdig("--version")
        assert result.returncode == 0
        assert "3.0.0" in result.stdout

    def test_version_short_flag(self):
        """-v prints version string and exits 0."""
        result = run_dizdig("-v")
        assert result.returncode == 0
        assert "3.0.0" in result.stdout

    def test_missing_dirs_gives_error(self):
        """Specifying --ext without INPUT_DIR/TARGET_DIR produces a non-zero exit."""
        result = run_dizdig("--ext", ".mod")
        assert result.returncode != 0

    def test_no_ext_or_diz_or_preset_gives_error(self, tmp_path: Path):
        """INPUT_DIR TARGET_DIR without --ext, --diz, or --preset produces a non-zero exit."""
        src = tmp_path / "src"
        src.mkdir()
        out = tmp_path / "out"
        result = run_dizdig(str(src), str(out))
        assert result.returncode != 0
        combined = result.stderr + result.stdout
        assert any(kw in combined for kw in ("--ext", "--diz", "--preset", "must specify"))

    def test_dry_run_reports_matches_without_moving(self, tmp_path: Path):
        """--dry-run prints matches but leaves source intact."""
        src = tmp_path / "src"
        pkg = make_pkg(src, "modpkg", "Great tracker module", {"song.mod": b"MOD"})
        out = tmp_path / "out"
        result = run_dizdig(str(src), str(out), "--ext", ".mod", "--dry-run")
        assert result.returncode == 0
        assert "[DRY]" in result.stdout
        assert pkg.exists()
        assert not (out / "modpkg").exists()

    def test_move_mode_moves_package(self, tmp_path: Path):
        """Default move mode relocates the matching package to target."""
        src = tmp_path / "src"
        pkg = make_pkg(src, "modpkg", "Tracker module", {"song.mod": b"MOD"})
        out = tmp_path / "out"
        result = run_dizdig(str(src), str(out), "--ext", ".mod")
        assert result.returncode == 0
        assert (out / "modpkg").exists()
        assert not pkg.exists()

    def test_copy_mode_copies_package_original_remains(self, tmp_path: Path):
        """--copy mode copies the package; the original is not removed."""
        src = tmp_path / "src"
        pkg = make_pkg(src, "modpkg", "Tracker module", {"song.mod": b"MOD"})
        out = tmp_path / "out"
        result = run_dizdig(str(src), str(out), "--ext", ".mod", "--copy")
        assert result.returncode == 0
        assert (out / "modpkg").exists()
        assert pkg.exists()

    def test_preset_tracker_matches_mod_files(self, tmp_path: Path):
        """--preset tracker includes .mod extension and matches packages containing it."""
        src = tmp_path / "src"
        make_pkg(src, "tracker_pkg", "Tracker stuff", {"tune.mod": b"MOD"})
        out = tmp_path / "out"
        result = run_dizdig(str(src), str(out), "--preset", "tracker", "--dry-run")
        assert result.returncode == 0
        assert "tracker_pkg" in result.stdout

    def test_preset_combined_with_ext_union(self, tmp_path: Path):
        """--preset and --ext together form a union of extensions."""
        src = tmp_path / "src"
        make_pkg(src, "mod_pkg", "Mod file", {"tune.mod": b"MOD"})
        make_pkg(src, "cpp_pkg", "C++ source", {"main.cpp": b"int main(){}"})
        out = tmp_path / "out"
        result = run_dizdig(
            str(src), str(out), "--preset", "tracker", "--ext", ".cpp", "--dry-run"
        )
        assert result.returncode == 0
        assert "mod_pkg" in result.stdout
        assert "cpp_pkg" in result.stdout

    def test_diz_text_mode_substring_match(self, tmp_path: Path):
        """--diz with text mode matches packages by DIZ substring."""
        src = tmp_path / "src"
        make_pkg(src, "4dos_pkg", "4DOS command interpreter version 5.0")
        make_pkg(src, "other_pkg", "Unrelated utility package")
        out = tmp_path / "out"
        result = run_dizdig(
            str(src), str(out), "--diz", "4dos", "--diz-mode", "text", "--dry-run"
        )
        assert result.returncode == 0
        assert "4dos_pkg" in result.stdout
        assert "other_pkg" not in result.stdout

    def test_diz_wildcard_mode(self, tmp_path: Path):
        """--diz with wildcard mode uses fnmatch glob matching."""
        src = tmp_path / "src"
        make_pkg(src, "crack_pkg", "A cracked version of something")
        make_pkg(src, "other_pkg", "Normal package")
        out = tmp_path / "out"
        result = run_dizdig(
            str(src), str(out), "--diz", "*crack*", "--diz-mode", "wildcard", "--dry-run"
        )
        assert result.returncode == 0
        assert "crack_pkg" in result.stdout
        assert "other_pkg" not in result.stdout

    def test_diz_regex_mode(self, tmp_path: Path):
        """--diz with regex mode uses regular expression matching."""
        src = tmp_path / "src"
        make_pkg(src, "v5_pkg", "Version 5.1 release notes")
        make_pkg(src, "v4_pkg", "Version 4.0 release notes")
        out = tmp_path / "out"
        result = run_dizdig(
            str(src), str(out), "--diz", r"5\.[0-9]+", "--diz-mode", "regex", "--dry-run"
        )
        assert result.returncode == 0
        assert "v5_pkg" in result.stdout
        assert "v4_pkg" not in result.stdout

    def test_and_logic_requires_both_ext_and_diz(self, tmp_path: Path):
        """--and requires BOTH extension and DIZ match; either alone is insufficient."""
        src = tmp_path / "src"
        make_pkg(src, "cpp_no_diz", "Linux-only utilities, no match", {"main.cpp": b"code"})
        make_pkg(src, "diz_no_cpp", "windows library pack", {"readme.txt": b"text"})
        make_pkg(src, "cpp_and_diz", "windows library pack", {"main.cpp": b"code"})
        out = tmp_path / "out"
        result = run_dizdig(
            str(src), str(out), "--ext", ".cpp", "--diz", "windows", "--and", "--dry-run"
        )
        assert result.returncode == 0
        assert "cpp_and_diz" in result.stdout
        assert "cpp_no_diz" not in result.stdout
        assert "diz_no_cpp" not in result.stdout

    def test_default_or_logic_either_match_sufficient(self, tmp_path: Path):
        """Default OR logic matches if either extension or DIZ matches."""
        src = tmp_path / "src"
        make_pkg(src, "cpp_pkg", "No special diz content", {"main.cpp": b"code"})
        make_pkg(src, "diz_pkg", "windows library pack", {"readme.txt": b"text"})
        out = tmp_path / "out"
        result = run_dizdig(
            str(src), str(out), "--ext", ".cpp", "--diz", "windows", "--dry-run"
        )
        assert result.returncode == 0
        assert "cpp_pkg" in result.stdout
        assert "diz_pkg" in result.stdout

    def test_size_filter_excludes_oversized_packages(self, tmp_path: Path):
        """--size <= 10KB keeps small packages and drops large ones."""
        src = tmp_path / "src"
        make_pkg(src, "small", "Small pkg", {"a.mod": b"x" * 100})
        make_pkg(src, "big", "Big pkg", {"a.mod": b"x" * (200 * 1024)})
        out = tmp_path / "out"
        result = run_dizdig(
            str(src), str(out), "--ext", ".mod", "--size", "<= 10KB", "--dry-run"
        )
        assert result.returncode == 0
        assert "small" in result.stdout
        assert "big" not in result.stdout

    def test_size_filter_range_two_constraints(self, tmp_path: Path):
        """Two --size filters combine as an AND range."""
        src = tmp_path / "src"
        make_pkg(src, "tiny", "Tiny pkg", {"a.mod": b"x" * 50})
        make_pkg(src, "medium", "Medium pkg", {"a.mod": b"x" * (50 * 1024)})
        make_pkg(src, "huge", "Huge pkg", {"a.mod": b"x" * (2 * 1024 * 1024)})
        out = tmp_path / "out"
        result = run_dizdig(
            str(src), str(out), "--ext", ".mod",
            "--size", ">= 1KB", "--size", "<= 100KB",
            "--dry-run",
        )
        assert result.returncode == 0
        assert "medium" in result.stdout
        assert "tiny" not in result.stdout
        assert "huge" not in result.stdout

    def test_exclude_ext_skips_package_with_that_extension(self, tmp_path: Path):
        """--exclude ext: omits packages that contain the excluded extension."""
        src = tmp_path / "src"
        make_pkg(src, "exe_pkg", "Has exe", {"prog.exe": b"", "data.mod": b"MOD"})
        make_pkg(src, "clean_pkg", "No exe", {"tune.mod": b"MOD"})
        out = tmp_path / "out"
        result = run_dizdig(
            str(src), str(out), "--ext", ".mod", "--exclude", "ext:.exe", "--dry-run"
        )
        assert result.returncode == 0
        assert "clean_pkg" in result.stdout
        assert "exe_pkg" not in result.stdout

    def test_exclude_diz_skips_package_with_matching_diz(self, tmp_path: Path):
        """--exclude diz: omits packages whose FILE_ID.DIZ contains the pattern."""
        src = tmp_path / "src"
        make_pkg(src, "broken_pkg", "This package is broken", {"tune.mod": b"MOD"})
        make_pkg(src, "good_pkg", "This package is fine", {"tune.mod": b"MOD"})
        out = tmp_path / "out"
        result = run_dizdig(
            str(src), str(out), "--ext", ".mod", "--exclude", "diz:broken", "--dry-run"
        )
        assert result.returncode == 0
        assert "good_pkg" in result.stdout
        assert "broken_pkg" not in result.stdout

    def test_exclude_size_skips_oversized_package(self, tmp_path: Path):
        """--exclude size: omits packages matching the size expression."""
        src = tmp_path / "src"
        make_pkg(src, "big_pkg", "Big package", {"a.mod": b"x" * (2 * 1024 * 1024)})
        make_pkg(src, "small_pkg", "Small package", {"a.mod": b"x" * 100})
        out = tmp_path / "out"
        result = run_dizdig(
            str(src), str(out), "--ext", ".mod", "--exclude", "size:>1MB", "--dry-run"
        )
        assert result.returncode == 0
        assert "small_pkg" in result.stdout
        assert "big_pkg" not in result.stdout

    def test_invalid_exclude_format_produces_error(self, tmp_path: Path):
        """Invalid --exclude value produces a non-zero exit and error output."""
        result = run_dizdig(
            str(tmp_path), str(tmp_path / "out"), "--ext", ".mod", "--exclude", "nocolon"
        )
        assert result.returncode != 0
        combined = result.stderr + result.stdout
        assert any(kw in combined.lower() for kw in ("exclude", "error", "invalid"))

    def test_skip_already_existing_destination(self, tmp_path: Path):
        """Packages whose destination already exists are skipped with a SKIP message."""
        src = tmp_path / "src"
        make_pkg(src, "modpkg", "A module", {"tune.mod": b"MOD"})
        out = tmp_path / "out"
        out.mkdir()
        (out / "modpkg").mkdir()  # pre-create destination
        result = run_dizdig(str(src), str(out), "--ext", ".mod", "--dry-run")
        assert result.returncode == 0
        combined = result.stdout.lower()
        assert "skip" in combined or "exist" in combined

    def test_target_inside_scan_tree_is_handled(self, tmp_path: Path):
        """Target dir nested inside input tree doesn't crash and processes other packages."""
        src = tmp_path / "src"
        make_pkg(src, "modpkg", "A module", {"tune.mod": b"MOD"})
        out = src / "out"  # target is inside the input tree
        result = run_dizdig(str(src), str(out), "--ext", ".mod")
        assert result.returncode == 0
        # modpkg should have been moved into the (nested) target
        assert (out / "modpkg").exists()

    def test_stats_output_contains_all_fields(self, tmp_path: Path):
        """Output includes Scanned, Matched, Moved/Copied, Skipped, and Time lines."""
        src = tmp_path / "src"
        make_pkg(src, "pkg1", "A module", {"tune.mod": b"MOD"})
        out = tmp_path / "out"
        result = run_dizdig(str(src), str(out), "--ext", ".mod")
        assert result.returncode == 0
        for keyword in ("Scanned", "Matched", "Skipped", "Time"):
            assert keyword in result.stdout
        assert "Moved" in result.stdout or "Copied" in result.stdout

    def test_non_matching_packages_are_not_moved(self, tmp_path: Path):
        """Packages that do not match any criterion remain in place."""
        src = tmp_path / "src"
        make_pkg(src, "nomatch_pkg", "Plain text document", {"readme.txt": b"text"})
        out = tmp_path / "out"
        result = run_dizdig(str(src), str(out), "--ext", ".mod")
        assert result.returncode == 0
        assert not (out / "nomatch_pkg").exists()
        assert (src / "nomatch_pkg").exists()


# ── Edge Case Tests ───────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case and integration tests."""

    def test_uppercase_file_id_diz_is_found(self, tmp_path: Path):
        """Standard uppercase FILE_ID.DIZ is detected by the rglob scan."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "FILE_ID.DIZ").write_text("A tracker module", encoding="utf-8")
        (pkg / "song.mod").write_bytes(b"MOD_DATA")
        out = tmp_path / "out"
        result = run_dizdig(str(tmp_path), str(out), "--ext", ".mod", "--dry-run")
        assert result.returncode == 0
        assert "pkg" in result.stdout

    def test_cp437_encoded_diz_is_read(self, tmp_path: Path):
        """FILE_ID.DIZ with CP437 encoding is read without error."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        # Raw CP437 bytes: 0x84=ä and 0x94=ö in the CP437 codepage
        raw = b"Tracker module \x84\x94 special chars"
        (pkg / "FILE_ID.DIZ").write_bytes(raw)
        (pkg / "song.mod").write_bytes(b"MOD_DATA")
        content = (pkg / "FILE_ID.DIZ").read_text(encoding="cp437", errors="ignore")
        assert dizdig.matches_diz(content, "tracker", "text") is True

    def test_empty_package_only_diz_no_crash(self, tmp_path: Path):
        """Package containing only FILE_ID.DIZ (no other files) does not crash."""
        src = tmp_path / "src"
        pkg = src / "empty_pkg"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "FILE_ID.DIZ").write_text("Empty package", encoding="utf-8")
        out = tmp_path / "out"
        result = run_dizdig(str(src), str(out), "--ext", ".mod", "--dry-run")
        assert result.returncode == 0

    def test_diz_only_empty_package_matches_via_diz_flag(self, tmp_path: Path):
        """Package with only FILE_ID.DIZ can still match via --diz."""
        src = tmp_path / "src"
        pkg = src / "diz_only"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "FILE_ID.DIZ").write_text("Special tracker bundle", encoding="utf-8")
        out = tmp_path / "out"
        result = run_dizdig(
            str(src), str(out), "--diz", "tracker", "--dry-run"
        )
        assert result.returncode == 0
        assert "diz_only" in result.stdout

    # ── Real BBS sample data integration tests ──────────────────────────────

    def test_bbs_sample_2m30_dos_exe_preset(self, tmp_path: Path):
        """Integration: 2m30 package matches dos-exe preset (contains .EXE/.COM)."""
        if not BBS_PATH.exists():
            pytest.skip("BBS test data not found")
        out = tmp_path / "out"
        result = run_dizdig(str(BBS_PATH), str(out), "--preset", "dos-exe", "--dry-run")
        assert result.returncode == 0
        assert "2m30" in result.stdout

    def test_bbs_sample_2m30_diz_content_match(self, tmp_path: Path):
        """Integration: 2m30 FILE_ID.DIZ is matched via --diz text search."""
        if not BBS_PATH.exists():
            pytest.skip("BBS test data not found")
        out = tmp_path / "out"
        # '2M' and 'reliable' are words present in the 2m30 FILE_ID.DIZ
        result = run_dizdig(
            str(BBS_PATH), str(out), "--diz", "reliable", "--diz-mode", "text", "--dry-run"
        )
        assert result.returncode == 0
        assert "2m30" in result.stdout

    def test_bbs_sample_4dos595_diz_content(self, tmp_path: Path):
        """Integration: 4dos595 FILE_ID.DIZ is matched via --diz text search."""
        if not (BBS_PATH / "4dos595").exists():
            pytest.skip("4dos595 BBS sample not found")
        out = tmp_path / "out"
        result = run_dizdig(
            str(BBS_PATH), str(out), "--diz", "command.com", "--diz-mode", "text", "--dry-run"
        )
        assert result.returncode == 0
        assert "4dos595" in result.stdout

    def test_bbs_sample_rkive14_archive_preset(self, tmp_path: Path):
        """Integration: rkive14 (an archiver) matches --diz 'archiver'."""
        if not (BBS_PATH / "rkive14").exists():
            pytest.skip("rkive14 BBS sample not found")
        out = tmp_path / "out"
        result = run_dizdig(
            str(BBS_PATH), str(out), "--diz", "archiver", "--diz-mode", "text", "--dry-run"
        )
        assert result.returncode == 0
        assert "rkive14" in result.stdout

    def test_bbs_sample_list91m_dos_exe_preset(self, tmp_path: Path):
        """Integration: list91m matches dos-exe preset (contains .COM files)."""
        if not (BBS_PATH / "list91m").exists():
            pytest.skip("list91m BBS sample not found")
        out = tmp_path / "out"
        result = run_dizdig(str(BBS_PATH), str(out), "--preset", "dos-exe", "--dry-run")
        assert result.returncode == 0
        assert "list91m" in result.stdout

    def test_bbs_sample_6xopt072_diz_content(self, tmp_path: Path):
        """Integration: 6xopt072 FILE_ID.DIZ is matched via --diz wildcard."""
        if not (BBS_PATH / "6xopt072").exists():
            pytest.skip("6xopt072 BBS sample not found")
        out = tmp_path / "out"
        result = run_dizdig(
            str(BBS_PATH), str(out), "--diz", "*6x86*", "--diz-mode", "wildcard", "--dry-run"
        )
        assert result.returncode == 0
        assert "6xopt072" in result.stdout

    def test_bbs_sample_gifexe44_diz_regex(self, tmp_path: Path):
        """Integration: gifexe44 FILE_ID.DIZ is matched via --diz regex."""
        if not (BBS_PATH / "gifexe44").exists():
            pytest.skip("gifexe44 BBS sample not found")
        out = tmp_path / "out"
        result = run_dizdig(
            str(BBS_PATH), str(out), "--diz", r"gif.*exe", "--diz-mode", "regex", "--dry-run"
        )
        assert result.returncode == 0
        assert "gifexe44" in result.stdout

    def test_bbs_sample_multiple_dos_exe_packages(self, tmp_path: Path):
        """Integration: dos-exe preset matches multiple packages in BBS dir."""
        if not BBS_PATH.exists():
            pytest.skip("BBS test data not found")
        out = tmp_path / "out"
        result = run_dizdig(str(BBS_PATH), str(out), "--preset", "dos-exe", "--dry-run")
        assert result.returncode == 0
        # Multiple packages should be matched (2m30, 4dos595, list91m, qview231, etc.)
        matched = [
            name for name in ("2m30", "4dos595", "list91m", "qview231", "rkive14", "6xopt072")
            if name in result.stdout
        ]
        assert len(matched) >= 3, f"Expected ≥3 packages, got: {matched}"

    def test_multiple_packages_only_matching_ones_moved(self, tmp_path: Path):
        """Only packages matching the criteria are moved; others remain."""
        src = tmp_path / "src"
        make_pkg(src, "match1", "Tracker module", {"a.mod": b"MOD"})
        make_pkg(src, "match2", "Another module", {"b.mod": b"MOD"})
        make_pkg(src, "nomatch", "Document package", {"doc.txt": b"text"})
        out = tmp_path / "out"
        result = run_dizdig(str(src), str(out), "--ext", ".mod")
        assert result.returncode == 0
        assert (out / "match1").exists()
        assert (out / "match2").exists()
        assert not (out / "nomatch").exists()
        assert (src / "nomatch").exists()


# ── Archive helper unit tests ─────────────────────────────────────────────────


class TestGetSupportedFormats:
    """Tests for get_supported_formats() function."""

    def test_zip_always_supported(self):
        """ZIP is always available via stdlib."""
        fmt = dizdig.get_supported_formats()
        assert fmt["zip"] is True

    def test_tar_always_supported(self):
        """TAR/GZ/BZ2/XZ are always available via stdlib."""
        fmt = dizdig.get_supported_formats()
        assert fmt["tar"] is True

    def test_returns_dict_with_required_keys(self):
        """Function returns a dict containing at least zip, tar, lzh, libarchive."""
        fmt = dizdig.get_supported_formats()
        for key in ("zip", "tar", "lzh", "libarchive"):
            assert key in fmt

    def test_lzh_is_bool(self):
        """lzh value is a boolean (True if lhafile installed, False otherwise)."""
        fmt = dizdig.get_supported_formats()
        assert isinstance(fmt["lzh"], bool)

    def test_libarchive_is_bool(self):
        """libarchive value is a boolean."""
        fmt = dizdig.get_supported_formats()
        assert isinstance(fmt["libarchive"], bool)


class TestPeekArchive:
    """Unit tests for the peek_archive() function."""

    def _make_zip(self, path: Path, files: dict, diz: str | None = None) -> Path:
        """Create a ZIP at *path* containing *files* and optionally FILE_ID.DIZ."""
        import zipfile as zf
        with zf.ZipFile(path, "w") as z:
            if diz is not None:
                z.writestr("FILE_ID.DIZ", diz.encode("cp437"))
            for name, data in files.items():
                z.writestr(name, data if isinstance(data, bytes) else data.encode())
        return path

    def test_peek_zip_returns_diz_content(self, tmp_path: Path):
        """peek_archive on a ZIP with FILE_ID.DIZ returns its content."""
        z = self._make_zip(
            tmp_path / "test.zip",
            {"song.mod": b"MOD"},
            diz="Great tracker module",
        )
        result = dizdig.peek_archive(z)
        assert result is not None
        diz_content, exts, size = result
        assert diz_content is not None
        assert "great tracker module" in diz_content.lower()

    def test_peek_zip_no_diz_returns_none_diz(self, tmp_path: Path):
        """peek_archive on a ZIP without FILE_ID.DIZ returns None for diz_content."""
        z = self._make_zip(tmp_path / "nodiz.zip", {"song.mod": b"MOD"})
        result = dizdig.peek_archive(z)
        assert result is not None
        diz_content, exts, size = result
        assert diz_content is None

    def test_peek_zip_detects_extensions(self, tmp_path: Path):
        """peek_archive returns the set of extensions found inside the ZIP."""
        z = self._make_zip(
            tmp_path / "multi.zip",
            {"song.mod": b"MOD", "prog.exe": b"EXE", "readme.txt": b"text"},
        )
        result = dizdig.peek_archive(z)
        assert result is not None
        _, exts, _ = result
        assert ".mod" in exts
        assert ".exe" in exts
        assert ".txt" in exts

    def test_peek_zip_calculates_uncompressed_size(self, tmp_path: Path):
        """peek_archive total_size equals sum of uncompressed file sizes."""
        data = b"x" * 1000
        z = self._make_zip(tmp_path / "sized.zip", {"a.bin": data, "b.bin": data})
        result = dizdig.peek_archive(z)
        assert result is not None
        _, _, total_size = result
        assert total_size == 2000

    def test_peek_zip_diz_case_insensitive(self, tmp_path: Path):
        """FILE_ID.DIZ is found regardless of case in the archive."""
        import zipfile as zf
        path = tmp_path / "case.zip"
        with zf.ZipFile(path, "w") as z:
            z.writestr("file_id.diz", b"lowercase diz")
        result = dizdig.peek_archive(path)
        assert result is not None
        diz_content, _, _ = result
        assert diz_content is not None

    def test_peek_corrupt_archive_returns_none(self, tmp_path: Path):
        """peek_archive on a corrupt/unreadable file returns None."""
        bad = tmp_path / "bad.zip"
        bad.write_bytes(b"this is not a zip file at all")
        result = dizdig.peek_archive(bad)
        assert result is None

    def test_peek_real_4dos595_zip(self):
        """peek_archive on the real 4dos595.zip finds FILE_ID.DIZ and .com/.exe."""
        real_zip = BBS_PATH / "4dos595.zip"
        if not real_zip.exists():
            pytest.skip("4dos595.zip not found")
        result = dizdig.peek_archive(real_zip)
        assert result is not None
        diz_content, exts, total_size = result
        assert diz_content is not None
        assert "4dos" in diz_content.lower()
        # The archive contains DOS executables
        assert ".com" in exts or ".exe" in exts
        assert total_size > 0

    def test_peek_tar_gz_returns_diz_and_extensions(self, tmp_path: Path):
        """peek_archive works on .tar.gz archives via stdlib tarfile."""
        import tarfile, io
        path = tmp_path / "test.tar.gz"
        buf_diz = b"A tarball package"
        buf_mod = b"MOD_DATA"
        with tarfile.open(path, "w:gz") as tf:
            for name, data in [("FILE_ID.DIZ", buf_diz), ("song.mod", buf_mod)]:
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        result = dizdig.peek_archive(path)
        assert result is not None
        diz_content, exts, size = result
        assert diz_content is not None
        assert "tarball" in diz_content.lower()
        assert ".mod" in exts


class TestExtractArchive:
    """Unit tests for the extract_archive() function."""

    def _make_zip(self, path: Path, files: dict) -> Path:
        import zipfile as zf
        with zf.ZipFile(path, "w") as z:
            for name, data in files.items():
                z.writestr(name, data if isinstance(data, bytes) else data.encode())
        return path

    def test_extract_zip_files_appear_in_dest(self, tmp_path: Path):
        """Extracted ZIP files are present under the destination directory."""
        z = self._make_zip(
            tmp_path / "pkg.zip",
            {"FILE_ID.DIZ": b"hello", "song.mod": b"MOD"},
        )
        dest = tmp_path / "out"
        ok = dizdig.extract_archive(z, dest)
        assert ok is True
        assert (dest / "song.mod").exists()
        assert (dest / "FILE_ID.DIZ").exists()

    def test_extract_zip_creates_dest_if_missing(self, tmp_path: Path):
        """extract_archive creates the destination directory when absent."""
        z = self._make_zip(tmp_path / "pkg.zip", {"a.txt": b"hi"})
        dest = tmp_path / "new" / "subdir"
        assert not dest.exists()
        ok = dizdig.extract_archive(z, dest)
        assert ok is True
        assert dest.is_dir()

    def test_extract_zip_returns_false_on_corrupt(self, tmp_path: Path):
        """extract_archive returns False for a corrupt archive."""
        bad = tmp_path / "bad.zip"
        bad.write_bytes(b"not a zip")
        dest = tmp_path / "out"
        ok = dizdig.extract_archive(bad, dest)
        assert ok is False

    def test_extract_tar_gz(self, tmp_path: Path):
        """extract_archive works for .tar.gz archives."""
        import tarfile, io
        path = tmp_path / "pkg.tar.gz"
        data = b"hello from tar"
        with tarfile.open(path, "w:gz") as tf:
            info = tarfile.TarInfo(name="hello.txt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        dest = tmp_path / "out"
        ok = dizdig.extract_archive(path, dest)
        assert ok is True
        assert (dest / "hello.txt").read_bytes() == data


# ── Archive CLI integration tests ─────────────────────────────────────────────


class TestCLIArchiveScan:
    """Integration tests for archive peeking via the CLI (always enabled)."""

    def _make_zip(self, path: Path, files: dict, diz: str | None = None) -> Path:
        import zipfile as zf
        path.parent.mkdir(parents=True, exist_ok=True)
        with zf.ZipFile(path, "w") as z:
            if diz is not None:
                z.writestr("FILE_ID.DIZ", diz.encode("cp437"))
            for name, data in files.items():
                z.writestr(name, data if isinstance(data, bytes) else data.encode())
        return path

    def test_archive_with_matching_ext_extracted_to_target(self, tmp_path: Path):
        """ZIP containing a matching extension is extracted to target."""
        src = tmp_path / "src"
        self._make_zip(src / "mods.zip", {"tune.mod": b"MOD", "FILE_ID.DIZ": b"mods"})
        out = tmp_path / "out"
        result = run_dizdig(str(src), str(out), "--ext", ".mod")
        assert result.returncode == 0
        assert (out / "mods").is_dir()
        assert (out / "mods" / "tune.mod").exists()

    def test_archive_dry_run_reports_but_does_not_extract(self, tmp_path: Path):
        """--dry-run reports the archive match but leaves it and target alone."""
        src = tmp_path / "src"
        z = self._make_zip(src / "mods.zip", {"tune.mod": b"MOD", "FILE_ID.DIZ": b"mods"})
        out = tmp_path / "out"
        result = run_dizdig(str(src), str(out), "--ext", ".mod", "--dry-run")
        assert result.returncode == 0
        assert "[DRY]" in result.stdout
        assert z.exists()
        assert not (out / "mods").exists()

    def test_archive_diz_match_extracts(self, tmp_path: Path):
        """--diz matching FILE_ID.DIZ inside a ZIP triggers extraction."""
        src = tmp_path / "src"
        self._make_zip(src / "util.zip", {"prog.exe": b"EXE"}, diz="4DOS command replacement")
        out = tmp_path / "out"
        result = run_dizdig(str(src), str(out), "--diz", "4dos", "--dry-run")
        assert result.returncode == 0
        assert "util" in result.stdout

    def test_archive_no_match_not_extracted(self, tmp_path: Path):
        """Archive whose contents don't match the filter is left alone."""
        src = tmp_path / "src"
        self._make_zip(src / "docs.zip", {"readme.txt": b"text"}, diz="plain docs")
        out = tmp_path / "out"
        result = run_dizdig(str(src), str(out), "--ext", ".mod", "--dry-run")
        assert result.returncode == 0
        # "docs" should not appear as a matched/extracted item (no .mod inside)
        assert "docs" not in result.stdout

    def test_archive_size_filter_by_uncompressed_size(self, tmp_path: Path):
        """--size filter applies to uncompressed content size of archive."""
        src = tmp_path / "src"
        # Small archive: ~100 bytes uncompressed
        self._make_zip(src / "small.zip", {"a.mod": b"x" * 100})
        # Large archive: ~200 KB uncompressed
        self._make_zip(src / "big.zip", {"b.mod": b"x" * (200 * 1024)})
        out = tmp_path / "out"
        result = run_dizdig(
            str(src), str(out), "--ext", ".mod", "--size", "<= 10KB", "--dry-run"
        )
        assert result.returncode == 0
        assert "small" in result.stdout
        assert "big" not in result.stdout

    def test_archive_copy_leaves_original(self, tmp_path: Path):
        """--copy extracts the archive but leaves the original .zip in place."""
        src = tmp_path / "src"
        z = self._make_zip(src / "mods.zip", {"tune.mod": b"MOD"})
        out = tmp_path / "out"
        result = run_dizdig(str(src), str(out), "--ext", ".mod", "--copy")
        assert result.returncode == 0
        assert (out / "mods").is_dir()
        assert z.exists()

    def test_archive_move_deletes_original(self, tmp_path: Path):
        """Default (move) extracts the archive and removes the original .zip."""
        src = tmp_path / "src"
        z = self._make_zip(src / "mods.zip", {"tune.mod": b"MOD"})
        out = tmp_path / "out"
        result = run_dizdig(str(src), str(out), "--ext", ".mod")
        assert result.returncode == 0
        assert (out / "mods").is_dir()
        assert not z.exists()

    def test_archive_exclude_ext_inside_archive(self, tmp_path: Path):
        """--exclude ext: skips archives that contain the excluded extension."""
        src = tmp_path / "src"
        self._make_zip(src / "withexe.zip", {"prog.exe": b"EXE", "tune.mod": b"MOD"})
        self._make_zip(src / "clean.zip", {"tune.mod": b"MOD"})
        out = tmp_path / "out"
        result = run_dizdig(
            str(src), str(out), "--ext", ".mod", "--exclude", "ext:.exe", "--dry-run"
        )
        assert result.returncode == 0
        assert "clean" in result.stdout
        assert "withexe" not in result.stdout

    def test_archive_skip_when_dest_exists(self, tmp_path: Path):
        """Archive extraction is skipped when destination directory already exists."""
        src = tmp_path / "src"
        self._make_zip(src / "mods.zip", {"tune.mod": b"MOD"})
        out = tmp_path / "out"
        (out / "mods").mkdir(parents=True)  # pre-create destination
        result = run_dizdig(str(src), str(out), "--ext", ".mod")
        assert result.returncode == 0
        combined = result.stdout.lower()
        assert "skip" in combined or "exist" in combined

    def test_stats_include_archive_counts(self, tmp_path: Path):
        """Stats output includes 'Archives scanned' and 'Archives extracted' lines."""
        src = tmp_path / "src"
        self._make_zip(src / "mods.zip", {"tune.mod": b"MOD"})
        out = tmp_path / "out"
        result = run_dizdig(str(src), str(out), "--ext", ".mod")
        assert result.returncode == 0
        assert "Archives scanned" in result.stdout
        assert "Archives extracted" in result.stdout

    def test_real_4dos595_zip_extracted_by_ext(self, tmp_path: Path):
        """Real 4dos595.zip is matched by --preset dos-exe and extracted."""
        if not (BBS_PATH / "4dos595.zip").exists():
            pytest.skip("4dos595.zip not found")
        src = tmp_path / "src"
        src.mkdir()
        import shutil
        shutil.copy(BBS_PATH / "4dos595.zip", src / "4dos595.zip")
        out = tmp_path / "out"
        result = run_dizdig(str(src), str(out), "--preset", "dos-exe", "--dry-run")
        assert result.returncode == 0
        assert "4dos595" in result.stdout


# ── Archive edge-case tests ───────────────────────────────────────────────────


class TestArchiveEdgeCases:
    """Edge cases for archive peeking and extraction."""

    def _make_zip(self, path: Path, files: dict, diz: str | None = None) -> Path:
        import zipfile as zf
        path.parent.mkdir(parents=True, exist_ok=True)
        with zf.ZipFile(path, "w") as z:
            if diz is not None:
                z.writestr("FILE_ID.DIZ", diz.encode("cp437"))
            for name, data in files.items():
                z.writestr(name, data if isinstance(data, bytes) else data.encode())
        return path

    def test_empty_archive_no_crash(self, tmp_path: Path):
        """Archive with no files inside doesn't crash and returns empty result."""
        import zipfile as zf
        path = tmp_path / "empty.zip"
        with zf.ZipFile(path, "w"):
            pass  # empty archive
        result = dizdig.peek_archive(path)
        # Should return a valid (empty) tuple, not None
        assert result is not None
        diz_content, exts, size = result
        assert diz_content is None
        assert len(exts) == 0
        assert size == 0

    def test_archive_in_target_tree_is_skipped(self, tmp_path: Path):
        """Archives already inside the target tree are not processed again."""
        src = tmp_path / "src"
        out = tmp_path / "out"
        # Place an archive inside the target — it must be ignored
        out.mkdir(parents=True)
        self._make_zip(out / "intarget.zip", {"tune.mod": b"MOD"})
        # Also put a real package in src so the scan has something to do
        make_pkg(src, "pkg", "tracker", {"tune.mod": b"MOD"})
        result = run_dizdig(str(src), str(out), "--ext", ".mod", "--dry-run")
        assert result.returncode == 0
        # "intarget" should NOT appear as a matched archive
        assert "intarget" not in result.stdout

    def test_diz_in_subdirectory_inside_archive(self, tmp_path: Path):
        """FILE_ID.DIZ nested in a subdir inside the archive is still detected."""
        import zipfile as zf
        path = tmp_path / "nested.zip"
        with zf.ZipFile(path, "w") as z:
            z.writestr("subdir/FILE_ID.DIZ", b"nested diz content")
            z.writestr("subdir/song.mod", b"MOD")
        result = dizdig.peek_archive(path)
        assert result is not None
        diz_content, exts, _ = result
        assert diz_content is not None
        assert "nested diz" in diz_content.lower()
        assert ".mod" in exts

    def test_archive_preset_match(self, tmp_path: Path):
        """--preset tracker matches a ZIP containing tracker module files."""
        src = tmp_path / "src"
        self._make_zip(src / "tracks.zip", {"intro.mod": b"M", "main.s3m": b"S"})
        out = tmp_path / "out"
        result = run_dizdig(str(src), str(out), "--preset", "tracker", "--dry-run")
        assert result.returncode == 0
        assert "tracks" in result.stdout

    def test_archive_and_logic_requires_both(self, tmp_path: Path):
        """--and requires both extension AND DIZ match inside the archive."""
        src = tmp_path / "src"
        # Has .mod but DIZ doesn't mention 'special'
        self._make_zip(src / "nomatch.zip", {"tune.mod": b"MOD"}, diz="generic music pack")
        # Has .mod AND DIZ mentions 'special'
        self._make_zip(src / "match.zip", {"tune.mod": b"MOD"}, diz="special tracker")
        out = tmp_path / "out"
        result = run_dizdig(
            str(src), str(out), "--ext", ".mod", "--diz", "special", "--and", "--dry-run"
        )
        assert result.returncode == 0
        assert "match" in result.stdout
        assert "nomatch" not in result.stdout
