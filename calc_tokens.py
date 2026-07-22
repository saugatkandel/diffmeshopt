import subprocess
from collections import defaultdict
from pathlib import Path

import nbformat
import tiktoken

# =============================================================================
# Configuration
# =============================================================================

# GPT tokenizer
enc = tiktoken.encoding_for_model("gpt-5")

# Set to None to include every tracked file.
extensions = {
    ".py",
    ".ipynb",
    ".cpp",
    ".cc",
    ".c",
    ".h",
    ".hpp",
    ".cu",
    ".cuh",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
}

# Whether markdown cells in notebooks contribute to the token count.
COUNT_NOTEBOOK_MARKDOWN = True


# =============================================================================
# Helpers
# =============================================================================


def analyze_file(path: Path):
    """
    Returns
    -------
    tokens : int
        GPT token count.
    loc : int
        Number of non-empty lines in the extracted source.
    """

    if path.suffix == ".ipynb":
        with path.open("r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        pieces = []

        for cell in nb.cells:
            if cell.cell_type == "code":
                pieces.append(cell.source)
            elif COUNT_NOTEBOOK_MARKDOWN and cell.cell_type == "markdown":
                pieces.append(cell.source)

        text = "\n\n".join(pieces)

    else:
        text = path.read_text(encoding="utf-8", errors="ignore")

    tokens = len(enc.encode(text))
    loc = sum(1 for line in text.splitlines() if line.strip())

    return tokens, loc


# =============================================================================
# Gather files
# =============================================================================

# Includes:
#   - tracked files
#   - untracked files
# Excludes:
#   - ignored files (.gitignore)
files = subprocess.check_output(
    [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
    ],
    text=True,
).splitlines()

# =============================================================================
# Analyze
# =============================================================================

file_stats = []
ext_stats = defaultdict(lambda: {"tokens": 0, "loc": 0})

total_tokens = 0
total_loc = 0

for filename in files:
    path = Path(filename)

    if not path.is_file():
        continue

    if extensions is not None and path.suffix not in extensions:
        continue

    try:
        tokens, loc = analyze_file(path)
    except Exception as e:
        print(f"Skipping {path}: {e}")
        continue

    file_stats.append((tokens, loc, path))

    ext_stats[path.suffix]["tokens"] += tokens
    ext_stats[path.suffix]["loc"] += loc

    total_tokens += tokens
    total_loc += loc

file_stats.sort(reverse=True)

# =============================================================================
# Per-file report
# =============================================================================

print("=" * 130)
print(f"{'Tokens':>12} {'LOC':>10} {'Tok/LOC':>10} {'%':>7}  File")
print("=" * 130)

for tokens, loc, path in file_stats:
    pct = 100 * tokens / total_tokens if total_tokens else 0
    ratio = tokens / loc if loc else 0

    print(f"{tokens:12,d} {loc:10,d} {ratio:10.2f} {pct:6.2f}%  {path}")

# =============================================================================
# Extension summary
# =============================================================================

print()
print("=" * 130)
print(f"{'Ext':>8} {'Tokens':>12} {'LOC':>10} {'Tok/LOC':>10}")
print("=" * 130)

for ext, stats in sorted(
    ext_stats.items(),
    key=lambda x: x[1]["tokens"],
    reverse=True,
):
    ratio = stats["tokens"] / stats["loc"] if stats["loc"] else 0

    print(f"{ext:>8} {stats['tokens']:12,d} {stats['loc']:10,d} {ratio:10.2f}")

# =============================================================================
# Summary
# =============================================================================

print()
print("=" * 130)
print("Summary")
print("=" * 130)

print(f"Total tokens : {total_tokens:,}")
print(f"Total LOC    : {total_loc:,}")
print(f"Files        : {len(file_stats):,}")
print(f"Avg tokens   : {total_tokens / len(file_stats):,.1f}")
print(f"Avg LOC      : {total_loc / len(file_stats):,.1f}")
print(f"Tokens / LOC : {total_tokens / total_loc:.2f}")
