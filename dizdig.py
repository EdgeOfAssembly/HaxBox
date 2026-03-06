#!/usr/bin/env python3
"""
dizdig - Dig through archives and directories, sort by content.

Move (or copy) entire package directories based on:
- File extensions (.mod, .cpp, .exe, etc.)
- Content of FILE_ID.DIZ (text, wildcard, regex)
- Directory size constraints with human-friendly units
- Preset extension groups for common file categories
- Any combination with AND/OR logic
- Flexible --exclude system for negated matching

Version: 2.0.0
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


VERSION: str = "2.0.0"

# ── Preset extension groups ──────────────────────────────────────────────────

PRESETS: Dict[str, List[str]] = {
    "tracker":  [".mod", ".s3m", ".xm", ".it", ".stm", ".med", ".oct",
                 ".669", ".far", ".mtm", ".ult", ".wow", ".okt"],
    "music":    [".mp3", ".wav", ".mid", ".snd", ".voc", ".au", ".aif",
                 ".flac", ".ogg", ".wma", ".ra", ".cmf", ".rol"],
    "graphics": [".gif", ".bmp", ".pcx", ".tga", ".jpg", ".jpeg", ".png",
                 ".iff", ".lbm", ".ansi", ".ans", ".ico", ".wmf", ".eps"],
    "source":   [".c", ".cpp", ".h", ".hpp", ".pas", ".asm", ".bas",
                 ".py", ".pl", ".java", ".for", ".cob", ".lsp"],
    "dos-exe":  [".exe", ".com", ".bat", ".cmd", ".btm", ".pif"],
    "document": [".txt", ".doc", ".nfo", ".diz", ".1st", ".me", ".asc",
                 ".pdf", ".rtf", ".htm", ".html", ".man", ".inf"],
    "archive":  [".zip", ".arj", ".lzh", ".lha", ".rar", ".ace", ".zoo",
                 ".arc", ".gz", ".tar", ".bz2", ".xz", ".7z", ".cab"],
}

# ── Size parsing ─────────────────────────────────────────────────────────────

SIZE_UNITS: Dict[str, int] = {
    "b":  1,
    "kb": 1024,
    "mb": 1024 ** 2,
    "gb": 1024 ** 3,
    "tb": 1024 ** 4,
}

SIZE_OPERATORS = {
    "<":  lambda a, b: a < b,
    ">":  lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def parse_size_expr(expr: str) -> Tuple:
    """
    Parse a size expression like '>= 10KB' into (operator_func, byte_value).

    Args:
        expr: Size expression string e.g. '>= 10KB', '< 512', '!= 0'

    Returns:
        Tuple of (comparison_function, size_in_bytes)

    Raises:
        ValueError: If expression format is invalid
    """
    expr = expr.strip()

    # Try two-char operators first, then single-char
    op = None
    rest = None
    for candidate in ("<=", ">=", "!=", "==", "<", ">"):
        if expr.startswith(candidate):
            op = candidate
            rest = expr[len(candidate):].strip()
            break

    if op is None or not rest:
        raise ValueError(
            f"Invalid size expression: '{expr}'\n"
            f"  Expected: OPERATOR VALUE[UNIT]\n"
            f"  Examples: '>= 10KB', '< 1MB', '!= 0', '== 4096'\n"
            f"  Operators: <, >, <=, >=, ==, !=\n"
            f"  Units: KB, MB, GB, TB (default: bytes)\n"
            f"  IMPORTANT: You MUST quote this value! --size \">= 10KB\""
        )

    # Split number from unit
    match = re.match(r'^([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]*)\s*$', rest)
    if not match:
        raise ValueError(
            f"Cannot parse size value: '{rest}'\n"
            f"  Expected a number optionally followed by KB, MB, GB, or TB"
        )

    number = float(match.group(1))
    unit = match.group(2).lower() if match.group(2) else "b"

    if unit not in SIZE_UNITS:
        raise ValueError(
            f"Unknown size unit: '{match.group(2)}'\n"
            f"  Supported: KB, MB, GB, TB (or nothing for bytes)"
        )

    byte_value = int(number * SIZE_UNITS[unit])
    return (SIZE_OPERATORS[op], byte_value, expr)


def dir_size(path: Path) -> int:
    """Calculate total size of all files in a directory (non-recursive)."""
    return sum(f.stat().st_size for f in path.iterdir() if f.is_file())


# ── DIZ matching ─────────────────────────────────────────────────────────────

def matches_diz(content: str, pattern: str, mode: str) -> bool:
    """
    Check if FILE_ID.DIZ content matches the given pattern.

    Args:
        content: The text content of FILE_ID.DIZ
        pattern: Search pattern (plain text, wildcard or regex)
        mode: 'text', 'wildcard' or 'regex'

    Returns:
        True if pattern matches according to selected mode
    """
    content = content.lower()
    p = pattern.lower()
    if mode == "regex":
        return bool(re.search(p, content))
    elif mode == "wildcard":
        return fnmatch.fnmatch(content, p)
    else:
        return p in content


# ── Exclude parsing ──────────────────────────────────────────────────────────

def parse_exclude(raw: str) -> Tuple[str, str]:
    """
    Parse an --exclude argument like 'ext:.exe' or 'diz:broken'.

    Args:
        raw: Raw exclude string in 'prefix:value' format

    Returns:
        Tuple of (prefix, value)

    Raises:
        ValueError: If format is invalid or prefix is unknown
    """
    if ":" not in raw:
        raise ValueError(
            f"Invalid --exclude format: '{raw}'\n"
            f"  Expected prefix:value\n"
            f"  Examples: ext:.exe  diz:broken  size:>1MB"
        )

    prefix, _, value = raw.partition(":")
    prefix = prefix.lower().strip()
    value = value.strip()

    valid_prefixes = ("ext", "diz", "size")
    if prefix not in valid_prefixes:
        raise ValueError(
            f"Unknown --exclude prefix: '{prefix}'\n"
            f"  Supported prefixes: {', '.join(valid_prefixes)}"
        )

    if not value:
        raise ValueError(
            f"--exclude {prefix}: requires a value after the colon\n"
            f"  Example: --exclude {prefix}:{'<value>' if prefix != 'ext' else '.exe'}"
        )

    return (prefix, value)


def check_excludes(
    pkg_dir: Path,
    diz_content: Optional[str],
    excludes: List[Tuple[str, str]],
    diz_mode: str,
) -> bool:
    """
    Check if a package should be excluded based on --exclude rules.

    Args:
        pkg_dir: Path to the package directory
        diz_content: Content of FILE_ID.DIZ (or None if not read)
        excludes: List of parsed (prefix, value) exclude tuples
        diz_mode: DIZ matching mode for diz: excludes

    Returns:
        True if the package should be EXCLUDED (skipped)
    """
    for prefix, value in excludes:
        if prefix == "ext":
            ext = value.lower() if value.startswith(".") else f".{value.lower()}"
            if any(f.suffix.lower() == ext for f in pkg_dir.iterdir() if f.is_file()):
                return True

        elif prefix == "diz":
            if diz_content and matches_diz(diz_content, value, diz_mode):
                return True

        elif prefix == "size":
            try:
                op_func, byte_val, _ = parse_size_expr(value)
                if op_func(dir_size(pkg_dir), byte_val):
                    return True
            except ValueError:
                pass  # Already validated at startup

    return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    preset_list = "\n".join(
        f"    {name:12s} {', '.join(exts)}" for name, exts in PRESETS.items()
    )

    parser = argparse.ArgumentParser(
        prog="dizdig",
        description="Dig through directories, sort packages by content, extensions, and metadata",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"""
Presets:
{preset_list}

Exclude system (prefix:value):
    ext:<extension>    Exclude packages containing files with this extension
    diz:<pattern>      Exclude packages whose FILE_ID.DIZ matches this pattern
    size:<expression>  Exclude packages matching this size condition

Examples:
  %(prog)s music/tracker --ext .mod .s3m .xm .it .stm
  %(prog)s music/tracker --preset tracker
  %(prog)s music/4dos --diz "4dos" --diz-mode text
  %(prog)s programming/cpp --ext .cpp --diz "windows" --and
  %(prog)s cracks --diz "*crack*" --diz-mode wildcard --dry-run
  %(prog)s music/versions --diz "5\\.[0-9]+" --diz-mode regex
  %(prog)s small_stuff --size "<= 10KB"
  %(prog)s mid_range --size ">= 10KB" --size "<= 1MB"
  %(prog)s cleaned --preset tracker --exclude ext:.exe --exclude diz:broken
  %(prog)s sorted --ext .mod --exclude size:">1MB" --copy
        """,
    )
    parser.add_argument("target", nargs="?", metavar="TARGET_DIR",
                        help="Target folder (required unless -v/--version)")
    parser.add_argument("--ext", nargs="*", metavar="EXT",
                        help="File extensions to match (e.g. .mod .s3m .cpp)")
    parser.add_argument("--preset", metavar="NAME",
                        choices=list(PRESETS.keys()),
                        help=f"Use a preset extension group ({', '.join(PRESETS.keys())})")
    parser.add_argument("--diz", metavar="PATTERN",
                        help="Text/pattern to search inside FILE_ID.DIZ")
    parser.add_argument("--diz-mode", choices=["text", "wildcard", "regex"],
                        default="text",
                        help="Matching mode for --diz (default: text)")
    parser.add_argument("--size", action="append", metavar="EXPR",
                        help='Size filter (quoted!). E.g. --size ">= 10KB" --size "< 1MB"')
    parser.add_argument("--exclude", action="append", metavar="PREFIX:VALUE",
                        help="Exclude filter. E.g. --exclude ext:.exe --exclude diz:broken")
    parser.add_argument("--and", action="store_true", dest="and_logic",
                        help="Require BOTH extension AND DIZ match (default: OR)")
    parser.add_argument("--copy", action="store_true",
                        help="Copy packages instead of moving them")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Only show what would happen, don't move/copy anything")
    parser.add_argument("-v", "--version", action="version",
                        version=f"%(prog)s {VERSION}",
                        help="Show version and exit")

    # Print help when no arguments given
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if not args.target:
        parser.error("the following arguments are required: TARGET_DIR")

    # ── Resolve extensions (direct + preset) ──

    exts: Set[str] = set()
    ext_source_parts: List[str] = []

    if args.ext:
        exts.update(e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.ext)
        ext_source_parts.append(", ".join(sorted(exts)))

    if args.preset:
        preset_exts = PRESETS[args.preset]
        exts.update(preset_exts)
        ext_source_parts.append(f"preset:{args.preset}")

    if not exts and not args.diz:
        parser.error("You must specify at least --ext, --preset, or --diz")

    # ── Validate size expressions early ──

    size_filters = []
    if args.size:
        for expr in args.size:
            try:
                size_filters.append(parse_size_expr(expr))
            except ValueError as e:
                parser.error(str(e))

    # ── Validate excludes early ──

    excludes: List[Tuple[str, str]] = []
    if args.exclude:
        for raw in args.exclude:
            try:
                excludes.append(parse_exclude(raw))
            except ValueError as e:
                parser.error(str(e))

        # Also validate any size: excludes
        for prefix, value in excludes:
            if prefix == "size":
                try:
                    parse_size_expr(value)
                except ValueError as e:
                    parser.error(f"In --exclude size:{value} — {e}")

    # ── Setup ──

    target: Path = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    action = "Copying" if args.copy else "Moving"
    logic_str = "AND" if args.and_logic else "OR"
    ext_display = ", ".join(ext_source_parts) if ext_source_parts else "none"
    size_display = " AND ".join(s[2] for s in size_filters) if size_filters else "none"
    exclude_display = ", ".join(f"{p}:{v}" for p, v in excludes) if excludes else "none"

    print(f"dizdig v{VERSION}\n"
          f"{'─' * 50}\n"
          f"Target:       {target}\n"
          f"Action:       {action}\n"
          f"Extensions:   {ext_display}\n"
          f"DIZ pattern:  {args.diz or 'none'} ({args.diz_mode})\n"
          f"Size filter:  {size_display}\n"
          f"Excludes:     {exclude_display}\n"
          f"Logic:        {logic_str}\n"
          f"Mode:         {'DRY-RUN' if args.dry_run else 'LIVE'}\n"
          f"{'─' * 50}\n")

    # ── Scan ──

    start_time = time.time()
    scanned: int = 0
    matched: int = 0
    moved: int = 0
    skipped: int = 0

    for diz_path in Path(".").rglob("*FILE_ID.DIZ"):
        pkg_dir = diz_path.parent
        scanned += 1

        try:
            rel = pkg_dir.relative_to(Path("."))
        except ValueError:
            continue

        # Skip if already inside target tree
        try:
            pkg_resolved = pkg_dir.resolve()
            target_resolved = target.resolve()
            if pkg_resolved == target_resolved or target_resolved in pkg_resolved.parents:
                continue
        except (OSError, ValueError):
            continue

        # ── Check extensions ──

        has_ext = False
        if exts:
            has_ext = any(
                f.suffix.lower() in exts
                for f in pkg_dir.iterdir() if f.is_file()
            )

        # ── Check FILE_ID.DIZ content ──

        diz_content: Optional[str] = None
        has_diz = False
        if args.diz or any(p == "diz" for p, _ in excludes):
            try:
                diz_content = diz_path.read_text(encoding="cp437", errors="ignore")
                if args.diz:
                    has_diz = matches_diz(diz_content, args.diz, args.diz_mode)
            except Exception:
                pass

        # ── Inclusion decision ──

        if args.and_logic:
            match = (has_ext and has_diz) if (exts and args.diz) else (has_ext or has_diz)
        else:
            match = (has_ext or has_diz) if (exts or args.diz) else False

        if not match:
            continue

        # ── Check size filters (all must pass) ──

        if size_filters:
            pkg_size = dir_size(pkg_dir)
            if not all(op(pkg_size, val) for op, val, _ in size_filters):
                continue

        # ── Check excludes ──

        if excludes and check_excludes(pkg_dir, diz_content, excludes, args.diz_mode):
            continue

        matched += 1
        dest = target / rel

        if dest.exists():
            print(f"  SKIP (exists): {rel}")
            skipped += 1
            continue

        prefix = "[DRY] " if args.dry_run else ""
        print(f"  {prefix}{action}: {rel} → {dest}")

        if not args.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if args.copy:
                shutil.copytree(str(pkg_dir), str(dest))
            else:
                pkg_dir.rename(dest)
            moved += 1

    # ── Stats ──

    elapsed = time.time() - start_time
    action_past = "Copied" if args.copy else "Moved"

    print(f"\n{'─' * 50}\n"
          f"Scanned:  {scanned:,} packages\n"
          f"Matched:  {matched:,}\n"
          f"{action_past}:   {moved:,}\n"
          f"Skipped:  {skipped:,} (already existed)\n"
          f"Time:     {elapsed:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)