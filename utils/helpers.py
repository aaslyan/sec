"""
Shared utilities and helper functions for the Binary DNA project.
"""

import json
import pickle
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class Instruction:
    """Represents a single disassembled instruction."""
    address: int
    mnemonic: str
    operands: str
    raw_bytes: bytes

    def __post_init__(self):
        """Clean up mnemonic to handle variations."""
        # Normalize mnemonic (remove suffixes like .plt, handle prefixes)
        self.mnemonic = self.mnemonic.split('@')[0].split('.')[0].lower()

@dataclass
class Function:
    """Represents a function with its instruction sequence."""
    name: str
    binary_name: str
    instructions: List[Instruction]
    
    @property
    def opcode_sequence(self) -> List[str]:
        """Return just the mnemonics as a sequence."""
        return [instr.mnemonic for instr in self.instructions]
    
    @property
    def size(self) -> int:
        """Return the number of instructions in this function."""
        return len(self.instructions)

@dataclass
class Binary:
    """Represents a binary with all its functions."""
    name: str
    path: str
    functions: List[Function]
    # corpus metadata
    inode: Optional[int] = None
    file_size: Optional[int] = None   # bytes on disk
    category: Optional[str] = None    # e.g. "compression", "shells"
    compiler: Optional[str] = None    # e.g. "gcc", "rustc", "clang"

    @property
    def full_opcode_sequence(self) -> List[str]:
        """Return all opcodes across all functions, concatenated."""
        return [op for func in self.functions for op in func.opcode_sequence]

    @property
    def instruction_count(self) -> int:
        """Return total number of instructions across all functions."""
        return sum(len(func.instructions) for func in self.functions)

    @property
    def function_count(self) -> int:
        """Return the number of functions in this binary."""
        return len(self.functions)

def save_json(data: Any, filepath: Path) -> None:
    """Save data as JSON, handling dataclasses."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def serialize_dataclass(obj):
        if hasattr(obj, '__dataclass_fields__'):
            return asdict(obj)
        elif isinstance(obj, bytes):
            return obj.hex()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=serialize_dataclass)

def load_json(filepath: Path) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_pickle(data: Any, filepath: Path) -> None:
    """Save data using pickle."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filepath: Path) -> Any:
    """Load data from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def build_vocabulary(binaries: List[Binary]) -> Dict[str, int]:
    """Build vocabulary mapping from all opcodes in the corpus."""
    opcodes = set()
    for binary in binaries:
        opcodes.update(binary.full_opcode_sequence)
    
    # Sort for consistent ordering
    sorted_opcodes = sorted(opcodes)
    vocab = {opcode: idx for idx, opcode in enumerate(sorted_opcodes)}
    
    logger.info(f"Built vocabulary with {len(vocab)} unique opcodes")
    return vocab

def compute_opcode_frequencies(binaries: List[Binary]) -> Dict[str, int]:
    """Compute frequency count for each opcode across all binaries."""
    frequencies = {}
    for binary in binaries:
        for opcode in binary.full_opcode_sequence:
            frequencies[opcode] = frequencies.get(opcode, 0) + 1
    return frequencies

def encode_sequence(opcodes: List[str], vocab: Dict[str, int]) -> List[int]:
    """Convert opcode sequence to integer sequence using vocabulary."""
    # Use vocab size as unknown token ID instead of -1
    unknown_id = len(vocab)
    return [vocab.get(op, unknown_id) for op in opcodes]

def ensure_output_dir(path: Path) -> Path:
    """Ensure output directory exists and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def filter_valid_binaries(binaries: List[str], corpus_dir: Optional[Path] = None) -> List[Path]:
    """Filter list of binary names to valid executable paths, deduplicating by inode."""
    if corpus_dir is None:
        corpus_dir = Path("/usr/bin")

    valid_paths = []
    seen_inodes: set = set()

    for binary_name in binaries:
        path = Path(binary_name) if '/' in binary_name else corpus_dir / binary_name

        if not (path.exists() and path.is_file()):
            logger.warning(f"Binary not found: {path}")
            continue

        # Resolve symlinks then check inode to skip hard-link duplicates
        real_path = path.resolve()
        inode = real_path.stat().st_ino
        if inode in seen_inodes:
            logger.warning(f"Skipping {path} — same inode as an already-added binary")
            continue

        seen_inodes.add(inode)
        valid_paths.append(path)
        logger.debug(f"Found binary: {path}")

    logger.info(f"Found {len(valid_paths)} unique binaries out of {len(binaries)} requested")
    return valid_paths

def get_file_size(path: Path) -> int:
    """Get file size in bytes."""
    try:
        return path.stat().st_size
    except OSError:
        return 0