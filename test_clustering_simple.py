#!/usr/bin/env python3
"""
Simple clustering test script that creates synthetic binary data for testing.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from utils.helpers import Binary, Function, Instruction, save_pickle, ensure_output_dir

def create_synthetic_binaries():
    """Create synthetic binary data for clustering test."""
    
    # Create simple instruction patterns
    instructions1 = [
        Instruction(0x1000, "mov", "rax, rbx", b'\x48\x89\xd8'),
        Instruction(0x1003, "add", "rax, 1", b'\x48\x83\xc0\x01'),
        Instruction(0x1007, "cmp", "rax, 10", b'\x48\x83\xf8\x0a'),
        Instruction(0x100b, "jl", "0x1000", b'\x7c\xf3'),
        Instruction(0x100d, "ret", "", b'\xc3')
    ]
    
    instructions2 = [
        Instruction(0x2000, "push", "rbp", b'\x55'),
        Instruction(0x2001, "mov", "rbp, rsp", b'\x48\x89\xe5'),
        Instruction(0x2004, "mov", "rax, 0", b'\x48\xc7\xc0\x00\x00\x00\x00'),
        Instruction(0x200b, "pop", "rbp", b'\x5d'),
        Instruction(0x200c, "ret", "", b'\xc3')
    ]
    
    instructions3 = [
        Instruction(0x3000, "mov", "rdi, rax", b'\x48\x89\xc7'),
        Instruction(0x3003, "call", "printf", b'\xe8\x00\x00\x00\x00'),
        Instruction(0x3008, "mov", "rax, 0", b'\x48\xc7\xc0\x00\x00\x00\x00'),
        Instruction(0x300f, "ret", "", b'\xc3')
    ]
    
    # Create functions
    func1 = Function("main", "binary1", instructions1)
    func2 = Function("helper", "binary2", instructions2)  
    func3 = Function("print_func", "binary3", instructions3)
    
    # Create binaries with different patterns
    binary1 = Binary("binary1", "/fake/path/binary1", [func1])
    binary2 = Binary("binary2", "/fake/path/binary2", [func2])
    binary3 = Binary("binary3", "/fake/path/binary3", [func3])
    
    return [binary1, binary2, binary3]

if __name__ == "__main__":
    print("Creating synthetic binaries for clustering test...")
    
    # Create synthetic data
    binaries = create_synthetic_binaries()
    
    # Create test corpus directory
    corpus_dir = ensure_output_dir(Path("test_clustering_synthetic"))
    
    # Save corpus data
    save_pickle(binaries, corpus_dir / "corpus.pkl")
    
    # Create simple corpus.json for compatibility
    import json
    json_data = []
    for binary in binaries:
        binary_data = {
            'name': binary.name,
            'path': binary.path,
            'functions': []
        }
        for func in binary.functions:
            func_data = {
                'name': func.name,
                'opcodes': func.opcode_sequence,
                'instruction_count': len(func.instructions)
            }
            binary_data['functions'].append(func_data)
        json_data.append(binary_data)
    
    with open(corpus_dir / 'corpus.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Created {len(binaries)} synthetic binaries in {corpus_dir}")
    
    # Show binary info
    for binary in binaries:
        print(f"{binary.name}: {binary.instruction_count} instructions, opcodes: {binary.full_opcode_sequence}")