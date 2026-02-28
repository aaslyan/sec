"""
Generic corpus runner: executes any named corpus defined in config.yaml.

Usage:
    python binary_dna.py corpus coreutils_system
    python binary_dna.py corpus mixed_system
    python binary_dna.py corpus smoke
"""

import logging
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from extraction.disassemble import disassemble_binary, save_corpus_data
from extraction.fallback import SYSTEM_BINARIES, find_system_binaries
from utils.helpers import filter_valid_binaries, ensure_output_dir

logger = logging.getLogger(__name__)

# name → category lookup built from SYSTEM_BINARIES
_NAME_TO_CATEGORY = {
    name: cat
    for cat, names in SYSTEM_BINARIES.items()
    for name in names
}


def _load_config(config_path: Path) -> dict:
    import yaml
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_named_corpus(args) -> int:
    """Run extraction + analysis for a named corpus from config.yaml."""
    config_path = Path(args.config)

    try:
        config = _load_config(config_path)
    except Exception as e:
        logger.error(f"Cannot load config: {e}")
        return 1

    corpora = config.get("corpora", {})
    corpus_name = args.name

    if corpus_name not in corpora:
        logger.error(f"Unknown corpus '{corpus_name}'. Available: {list(corpora)}")
        return 1

    cfg = corpora[corpus_name]
    output_dir = Path(cfg.get("output_dir", f"results/{corpus_name}"))

    logger.info(f"Running corpus '{corpus_name}' → {output_dir}")

    # ── Binary list corpora (smoke, mixed_system, …) ─────────────────────────
    if "binaries" in cfg:
        binary_names = cfg["binaries"]
        binary_paths = filter_valid_binaries(binary_names)
        if not binary_paths:
            logger.error("No binaries found for corpus")
            return 1
        logger.info(f"Found {len(binary_paths)}/{len(binary_names)} unique binaries")
        binaries = []
        for p in binary_paths:
            b = disassemble_binary(p, category=_NAME_TO_CATEGORY.get(p.name))
            if b:
                binaries.append(b)

    # ── Limit-based corpora (coreutils_system) ────────────────────────────────
    elif "limit" in cfg:
        limit = cfg["limit"]
        entries = find_system_binaries(limit)
        if not entries:
            logger.error("No system binaries found")
            return 1
        binaries = []
        for binary_path, category in entries:
            b = disassemble_binary(binary_path, category=category)
            if b:
                binaries.append(b)

    else:
        logger.error(f"Corpus '{corpus_name}' has neither 'binaries' nor 'limit' key")
        return 1

    if not binaries:
        logger.error("Failed to disassemble any binaries")
        return 1

    corpus_dir = ensure_output_dir(output_dir / "corpus")
    save_corpus_data(binaries, corpus_dir)

    from analysis.pipeline import run_analysis_pipeline
    analysis_args = type("Args", (), {
        "corpus_dir": str(corpus_dir),
        "output_dir": str(output_dir / "results"),
    })()
    return run_analysis_pipeline(analysis_args)
