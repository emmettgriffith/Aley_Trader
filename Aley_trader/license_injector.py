#!/usr/bin/env python3
"""
license_injector.py

Purpose:
    Insert a lightweight license / tracking watermark comment at the top of source files
    (HTML, JS, CSS, PY) to discourage casual code theft. Each file gets a unique
    tracking ID so you can identify the origin of a leaked file.

What it does:
    1. Recursively scans from the directory this script is in (or a custom path arg)
    2. For each eligible file, if a watermark is NOT already present in the first
       few lines, it prepends one.
    3. Generates a short random ID like LIC-AB12CD34.

Non-destructive:
    - Creates an in-place backup with extension .bak before modifying.
    - Skips files over 1 MB to avoid large unintended edits.
    - Skips virtual envs, git folders, common build/output directories.

Usage:
    python3 license_injector.py            # scan current directory
    python3 license_injector.py path/to/dir

Reverting:
    Use the created .bak files (or restore from version control).

NOTE:
    Adding MANY random strings *inside* the code can hurt readability and
    maintainability. A consistent header watermark is usually enough. If you
    still want deeper watermarking (e.g., no-op strings inside functions), you
    can extend the `deep_watermark_py` function (currently off by default).
"""
from __future__ import annotations
import os
import sys
import random
import string
from pathlib import Path
from datetime import datetime

# Configuration
ELIGIBLE_EXT = {'.html', '.htm', '.css', '.js', '.py'}
SKIP_DIR_NAMES = {'.git', '.venv', 'venv', '__pycache__', 'dist', 'build', 'node_modules'}
MAX_FILE_SIZE = 1_000_000  # 1 MB safeguard
HEADER_TAG = 'Tracking ID:'  # Marker to detect existing header

LICENSE_LINES = [
    "All Rights Reserved © 2025 Company Name.",
    "Unauthorized copying, distribution, or disclosure prohibited.",
]

def make_tracking_id() -> str:
    core = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    return f"LIC-{core}"

def detect_already_watermarked(text: str) -> bool:
    head = "\n".join(text.splitlines()[:5])
    return HEADER_TAG in head

def build_header_comment(ext: str) -> str:
    tid = make_tracking_id()
    core = f"License Watermark | {LICENSE_LINES[0]} {LICENSE_LINES[1]} {HEADER_TAG} {tid}"
    if ext == '.py':
        return f"# {core}\n"
    elif ext in {'.js', '.css'}:
        return f"/* {core} */\n"
    elif ext in {'.html', '.htm'}:
        return f"<!-- {core} -->\n"
    else:
        return f"# {core}\n"

def deep_watermark_py(original: str) -> str:
    """Optionally insert harmless no-op markers inside Python code.
    Currently disabled (returns input) to avoid noise.
    """
    return original  # Extend if you really want deeper markers

def process_file(path: Path) -> bool:
    try:
        if path.suffix.lower() not in ELIGIBLE_EXT:
            return False
        if path.stat().st_size > MAX_FILE_SIZE:
            return False
        text = path.read_text(encoding='utf-8', errors='ignore')
        if detect_already_watermarked(text):
            return False
        header = build_header_comment(path.suffix.lower())
        new_body = deep_watermark_py(text) if path.suffix.lower() == '.py' else text
        path_backup = path.with_suffix(path.suffix + '.bak')
        if not path_backup.exists():
            path_backup.write_text(text, encoding='utf-8')
        path.write_text(header + new_body, encoding='utf-8')
        return True
    except Exception as e:
        print(f"[WARN] Failed {path}: {e}")
        return False

def scan(start: Path) -> None:
    modified = 0
    checked = 0
    for root, dirs, files in os.walk(start):
        # Prune skip dirs in-place
        dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES]
        for name in files:
            p = Path(root) / name
            checked += 1
            if process_file(p):
                modified += 1
    print(f"Scan complete. Checked {checked} files. Added watermarks to {modified} files.")

if __name__ == '__main__':
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    if not target.exists():
        print(f"Path does not exist: {target}")
        sys.exit(1)
    print(f"[INFO] Starting watermark injection at {target} - {datetime.utcnow().isoformat()}Z")
    scan(target)
