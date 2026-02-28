"""
Fallback module for analyzing system binaries when GitHub corpus is not available.
"""

import logging
from pathlib import Path
from typing import List
import sys
sys.path.append(str(Path(__file__).parent.parent))

from extraction.disassemble import disassemble_binary, save_corpus_data
from utils.helpers import ensure_output_dir

logger = logging.getLogger(__name__)

# Common system binaries organized by category
SYSTEM_BINARIES = {
    "text_processing": ["grep", "sed", "awk", "sort", "cut", "tr", "uniq", "comm", "paste", "join"],
    "compression": ["gzip", "bzip2", "xz"],
    "system": ["ls", "cp", "mv", "rm", "cat", "find", "mkdir", "chmod", "chown", "ln"],
    "networking": ["curl", "wget", "ping"],
    "shells": ["bash", "dash"],
    "editors": ["vim", "nano"],
    "compilers": ["gcc", "cc"],
    "interpreters": ["python3", "perl"],
    "version_control": ["git"],
    "archives": ["tar"],
    "utilities": ["head", "tail", "wc", "tee", "md5sum", "sha256sum", "base64", "date", "touch"]
}

SMOKE_BINARIES = [
    "ls", "cat", "grep", "sort", "head", "tail",
    "wc", "cut", "tr", "sed", "gzip", "find",
    "bash", "python3", "curl",
]


def run_smoke_test(args) -> int:
    """Run an end-to-end smoke test on a small fixed corpus loaded from config."""
    output_dir = Path(args.output_dir)
    config_path = Path(args.config)

    smoke_binaries = SMOKE_BINARIES

    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            smoke_cfg = config.get("corpora", {}).get("smoke", {})
            if "binaries" in smoke_cfg:
                smoke_binaries = smoke_cfg["binaries"]
            if "output_dir" in smoke_cfg and args.output_dir == "results/smoke":
                output_dir = Path(smoke_cfg["output_dir"])
        except Exception as e:
            logger.warning(f"Could not load config ({e}); using built-in smoke corpus")
    else:
        logger.info("No config.yaml found; using built-in smoke corpus")

    logger.info(f"Smoke corpus: {smoke_binaries}")

    from utils.helpers import filter_valid_binaries
    binary_paths = filter_valid_binaries(smoke_binaries)

    if not binary_paths:
        logger.error("No smoke corpus binaries found in /usr/bin")
        return 1

    logger.info(f"Found {len(binary_paths)}/{len(smoke_binaries)} smoke binaries")

    # Build name→category map from SYSTEM_BINARIES for metadata tagging
    name_to_category = {
        name: cat
        for cat, names in SYSTEM_BINARIES.items()
        for name in names
    }
    binaries = []
    for p in binary_paths:
        b = disassemble_binary(p, category=name_to_category.get(p.name))
        if b:
            binaries.append(b)

    if not binaries:
        logger.error("Failed to disassemble any smoke corpus binaries")
        return 1

    corpus_dir = ensure_output_dir(output_dir / "corpus")
    save_corpus_data(binaries, corpus_dir)

    from analysis.pipeline import run_analysis_pipeline
    analysis_args = type("Args", (), {
        "corpus_dir": str(corpus_dir),
        "output_dir": str(output_dir / "results"),
    })()
    return run_analysis_pipeline(analysis_args)


def find_system_binaries(limit: int = 30, max_size_mb: int = 10) -> List[tuple]:
    """Find available system binaries up to the specified limit.

    Returns a list of (Path, category) tuples, deduplicated by inode.
    Skips multi-call mega-binaries larger than max_size_mb.
    """
    bin_dirs = [Path("/usr/bin"), Path("/bin")]
    max_bytes = max_size_mb * 1024 * 1024
    found: List[tuple] = []       # (Path, category)
    seen_inodes: set = set()
    categories_found = {}

    for category, binary_names in SYSTEM_BINARIES.items():
        categories_found[category] = []

        for binary_name in binary_names:
            if len(found) >= limit:
                break

            for bin_dir in bin_dirs:
                binary_path = bin_dir / binary_name
                if not (binary_path.exists() and binary_path.is_file()):
                    continue

                real = binary_path.resolve()
                st = real.stat()
                if st.st_size > max_bytes:
                    logger.debug(f"Skipping {binary_path} — {st.st_size//1024//1024}MB exceeds {max_size_mb}MB cap")
                    break

                if st.st_ino in seen_inodes:
                    logger.debug(f"Skipping {binary_path} — duplicate inode {st.st_ino}")
                    break

                seen_inodes.add(st.st_ino)
                found.append((binary_path, category))
                categories_found[category].append(binary_name)
                logger.debug(f"Found {category} binary: {binary_path}")
                break

        if len(found) >= limit:
            break

    for category, binaries in categories_found.items():
        if binaries:
            names = ', '.join(binaries[:5]) + ('...' if len(binaries) > 5 else '')
            logger.info(f"{category}: {len(binaries)} binaries ({names})")

    logger.info(f"Found {len(found)} unique system binaries")
    return found

def analyze_system_binaries(args) -> int:
    """Analyze system binaries as fallback corpus."""
    output_dir = Path(args.output_dir)
    limit = args.limit

    logger.info(f"Searching for up to {limit} unique system binaries...")

    entries = find_system_binaries(limit)

    if not entries:
        logger.error("No system binaries found")
        return 1

    binaries = []
    for binary_path, category in entries:
        binary = disassemble_binary(binary_path, category=category)
        if binary:
            binaries.append(binary)

    if not binaries:
        logger.error("Failed to disassemble any system binaries")
        return 1

    corpus_dir = ensure_output_dir(output_dir / "corpus")
    save_corpus_data(binaries, corpus_dir)

    logger.info("Running full analysis pipeline...")
    from analysis.pipeline import run_analysis_pipeline
    analysis_args = type('Args', (), {
        'corpus_dir': str(corpus_dir),
        'output_dir': str(output_dir / "results"),
    })()
    return run_analysis_pipeline(analysis_args)

def get_binary_categories() -> dict:
    """Return the system binary categories for use in analysis."""
    return SYSTEM_BINARIES