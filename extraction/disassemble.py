"""
Disassembly and parsing module for extracting opcode sequences from ELF binaries.

Uses objdump to disassemble binaries and extract structured instruction data.
"""

import subprocess
import re
import logging
from pathlib import Path
from typing import List, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.helpers import Instruction, Function, Binary, save_json, save_pickle, ensure_output_dir

logger = logging.getLogger(__name__)

class DisassemblyError(Exception):
    """Exception raised when disassembly fails."""
    pass

def run_objdump(binary_path: Path) -> str:
    """Run objdump on a binary and return the output."""
    try:
        cmd = ['objdump', '-d', '-M', 'intel', str(binary_path)]
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=60,  # 1 minute timeout
            check=False
        )
        
        if result.returncode != 0:
            raise DisassemblyError(f"objdump failed with code {result.returncode}: {result.stderr}")
        
        return result.stdout
        
    except subprocess.TimeoutExpired:
        raise DisassemblyError(f"objdump timed out on {binary_path}")
    except FileNotFoundError:
        raise DisassemblyError("objdump not found - please install binutils")

def parse_objdump_output(output: str, binary_name: str) -> List[Function]:
    """Parse objdump output and extract functions with their instructions."""
    functions = []
    current_function = None
    current_instructions = []
    
    # Regular expressions for parsing
    function_pattern = re.compile(r'^([0-9a-f]+) <(.+?)>:$')
    instruction_pattern = re.compile(r'^\s*([0-9a-f]+):\s*([0-9a-f ]+)\s+([a-zA-Z][a-zA-Z0-9]*)(.*)?$')
    
    for line in output.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check for function start
        func_match = function_pattern.match(line)
        if func_match:
            # Save previous function if exists
            if current_function and current_instructions:
                functions.append(Function(
                    name=current_function,
                    binary_name=binary_name,
                    instructions=current_instructions
                ))
            
            # Start new function
            current_function = func_match.group(2)
            current_instructions = []
            continue
        
        # Check for instruction
        instr_match = instruction_pattern.match(line)
        if instr_match and current_function:
            address = int(instr_match.group(1), 16)
            raw_bytes = bytes.fromhex(instr_match.group(2).replace(' ', ''))
            mnemonic = instr_match.group(3)
            operands = instr_match.group(4).strip() if instr_match.group(4) else ""
            
            instruction = Instruction(
                address=address,
                mnemonic=mnemonic,
                operands=operands,
                raw_bytes=raw_bytes
            )
            current_instructions.append(instruction)
    
    # Save final function
    if current_function and current_instructions:
        functions.append(Function(
            name=current_function,
            binary_name=binary_name,
            instructions=current_instructions
        ))
    
    logger.info(f"Extracted {len(functions)} functions from {binary_name}")
    return functions

def disassemble_binary(binary_path: Path, category: Optional[str] = None,
                       compiler: Optional[str] = None) -> Optional[Binary]:
    """Disassemble a single binary and return structured data."""
    try:
        logger.info(f"Disassembling {binary_path}")

        # Run objdump
        output = run_objdump(binary_path)

        # Parse output
        functions = parse_objdump_output(output, binary_path.name)

        if not functions:
            logger.warning(f"No functions found in {binary_path}")
            return None

        stat = binary_path.resolve().stat()
        binary = Binary(
            name=binary_path.name,
            path=str(binary_path),
            functions=functions,
            inode=stat.st_ino,
            file_size=stat.st_size,
            category=category,
            compiler=compiler,
        )

        logger.info(f"Successfully disassembled {binary_path}: "
                    f"{binary.function_count} functions, {binary.instruction_count} instructions")

        return binary

    except DisassemblyError as e:
        logger.error(f"Failed to disassemble {binary_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error disassembling {binary_path}: {e}")
        return None

def save_corpus_data(binaries: List[Binary], output_dir: Path) -> None:
    """Save extracted corpus data to disk."""
    from utils.helpers import build_vocabulary
    corpus_dir = ensure_output_dir(output_dir)

    # corpus.json — human-readable, includes per-binary metadata
    json_data = []
    for binary in binaries:
        binary_data = {
            'name': binary.name,
            'path': binary.path,
            'inode': binary.inode,
            'file_size': binary.file_size,
            'category': binary.category,
            'compiler': binary.compiler,
            'function_count': binary.function_count,
            'instruction_count': binary.instruction_count,
            'functions': [
                {
                    'name': func.name,
                    'opcodes': func.opcode_sequence,
                    'instruction_count': len(func.instructions),
                }
                for func in binary.functions
            ],
        }
        json_data.append(binary_data)

    save_json(json_data, corpus_dir / 'corpus.json')

    # corpus.pkl — full objects for downstream analysis
    save_pickle(binaries, corpus_dir / 'corpus.pkl')

    # sequences/ — one flat opcode-per-line file per binary
    sequences_dir = ensure_output_dir(corpus_dir / 'sequences')
    for binary in binaries:
        with open(sequences_dir / f"{binary.name}.txt", 'w') as f:
            for opcode in binary.full_opcode_sequence:
                f.write(f"{opcode}\n")

    # vocab.json — shared opcode vocabulary (sorted, stable index)
    vocab = build_vocabulary(binaries)
    save_json(vocab, corpus_dir / 'vocab.json')

    # corpus_summary.json — aggregate + per-binary stats for quick inspection
    seq_lengths = [binary.instruction_count for binary in binaries]
    summary = {
        'num_binaries': len(binaries),
        'vocab_size': len(vocab),
        'total_instructions': int(sum(seq_lengths)),
        'total_functions': int(sum(b.function_count for b in binaries)),
        'seq_length': {
            'min': int(min(seq_lengths)),
            'max': int(max(seq_lengths)),
            'mean': float(sum(seq_lengths) / len(seq_lengths)),
        },
        'categories': sorted(set(b.category for b in binaries if b.category)),
        'compilers': sorted(set(b.compiler for b in binaries if b.compiler)),
        'binaries': [
            {
                'name': b.name,
                'category': b.category,
                'compiler': b.compiler,
                'file_size': b.file_size,
                'function_count': b.function_count,
                'instruction_count': b.instruction_count,
            }
            for b in binaries
        ],
    }
    save_json(summary, corpus_dir / 'corpus_summary.json')

    logger.info(f"Saved corpus data to {corpus_dir} "
                f"({len(binaries)} binaries, {len(vocab)} vocab, "
                f"{summary['total_instructions']} instructions)")

def extract_corpus(args) -> int:
    """Extract opcode sequences from specified binaries."""
    from utils.helpers import filter_valid_binaries
    
    # Parse binary list
    binary_names = [name.strip() for name in args.binaries.split(',')]
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else Path("/usr/bin")
    output_dir = Path(args.output)
    
    # Find valid binaries
    binary_paths = filter_valid_binaries(binary_names, corpus_dir)
    
    if not binary_paths:
        logger.error("No valid binaries found")
        return 1
    
    # Disassemble each binary
    binaries = []
    for binary_path in binary_paths:
        binary = disassemble_binary(binary_path)
        if binary:
            binaries.append(binary)
    
    if not binaries:
        logger.error("Failed to disassemble any binaries")
        return 1
    
    # Save results
    save_corpus_data(binaries, output_dir)
    
    logger.info(f"Successfully extracted {len(binaries)} binaries")
    return 0