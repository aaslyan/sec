#!/usr/bin/env python3
"""
Binary DNA: Statistical Analysis of Program Instruction Sequences

CLI entry point for analyzing compiled binaries using computational biology techniques.
"""

import argparse
import sys
import logging
from pathlib import Path

def setup_logging(level=logging.INFO):
    """Configure logging for the application."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze binary instruction sequences using computational biology techniques"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract opcode sequences from binaries')
    extract_parser.add_argument('--binaries', required=True, help='Comma-separated list of binary names or paths')
    extract_parser.add_argument('--output', required=True, help='Output directory for extracted data')
    extract_parser.add_argument('--corpus-dir', help='Directory containing binaries (default: /usr/bin)')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Run full statistical analysis pipeline')
    analyze_parser.add_argument('--corpus-dir', required=True, help='Directory containing extracted corpus data')
    analyze_parser.add_argument('--output-dir', required=True, help='Output directory for analysis results')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate HTML report from analysis results')
    report_parser.add_argument('--results-dir', required=True, help='Directory containing analysis results')
    report_parser.add_argument('--output', required=True, help='Output HTML file path')
    
    # System command (fallback to system binaries)
    system_parser = subparsers.add_parser('system', help='Analyze system binaries from /usr/bin')
    system_parser.add_argument('--output-dir', required=True, help='Output directory for analysis')
    system_parser.add_argument('--limit', type=int, default=30, help='Maximum number of binaries to analyze')
    
    # Fast command (performance-optimized analysis)
    fast_parser = subparsers.add_parser('fast', help='Fast analysis with parallel processing and sampling')
    fast_parser.add_argument('--corpus-dir', required=True, help='Directory containing extracted corpus data')
    fast_parser.add_argument('--output-dir', required=True, help='Output directory for analysis results')
    fast_parser.add_argument('--workers', type=int, help='Number of parallel workers (default: CPU count)')
    fast_parser.add_argument('--max-memory', type=int, default=500, help='Memory limit per worker in MB')
    fast_parser.add_argument('--sample-size', type=int, help='Maximum sequence length per binary')
    
    # Smoke test command
    smoke_parser = subparsers.add_parser('smoke', help='End-to-end smoke test on a small fixed corpus')
    smoke_parser.add_argument('--config', default='config.yaml', help='Config file (default: config.yaml)')
    smoke_parser.add_argument('--output-dir', default='results/smoke', help='Output directory (default: results/smoke)')

    # Named corpus command — runs any corpus defined in config.yaml
    corpus_parser = subparsers.add_parser('corpus', help='Run a named corpus from config.yaml')
    corpus_parser.add_argument('name', help='Corpus name (e.g. smoke, coreutils_system, mixed_system)')
    corpus_parser.add_argument('--config', default='config.yaml', help='Config file (default: config.yaml)')

    # LM command — n-gram language model + perplexity vs entropy
    lm_parser = subparsers.add_parser('lm', help='Train n-gram LMs and compare perplexity to entropy rates')
    lm_parser.add_argument('--corpus-dir', required=True, help='Corpus directory (contains sequences/)')
    lm_parser.add_argument('--results-dir', required=True, help='Analysis results directory (contains ngram_analysis.json)')
    lm_parser.add_argument('--output', required=True, help='Output JSON path for LM analysis results')

    # Clone detection command
    clones_parser = subparsers.add_parser('clones', help='Detect code clones across binary corpus')
    clones_parser.add_argument('--corpus-dir', required=True, help='Corpus directory (contains corpus.pkl)')
    clones_parser.add_argument('--output-dir', required=True, help='Output directory for clone results')

    # Compiler matrix experiment
    cm_parser = subparsers.add_parser(
        'compiler-matrix',
        help='5 C projects × {gcc,clang} × {O0,O2,O3} — compiler fingerprinting experiment'
    )
    cm_parser.add_argument('--output-dir', default='results/compiler_matrix',
                           help='Output directory (default: results/compiler_matrix)')

    # GitHub corpus builder command
    github_parser = subparsers.add_parser('github', help='Build corpus from GitHub repositories')
    github_parser.add_argument('--output-dir', required=True, help='Output directory for corpus')
    github_parser.add_argument('--languages', nargs='+', default=['C', 'C++', 'Rust', 'Go', 'Zig'], 
                               help='Programming languages to include')
    github_parser.add_argument('--repos-per-language', type=int, default=10, 
                               help='Number of repositories per language')
    github_parser.add_argument('--min-stars', type=int, default=100, 
                               help='Minimum repository stars')
    github_parser.add_argument('--github-token', help='GitHub API token for higher rate limits')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        if args.command == 'smoke':
            from extraction.fallback import run_smoke_test
            return run_smoke_test(args)
        elif args.command == 'corpus':
            from extraction.corpus_runner import run_named_corpus
            return run_named_corpus(args)
        elif args.command == 'extract':
            from extraction.disassemble import extract_corpus
            return extract_corpus(args)
        elif args.command == 'analyze':
            from analysis.pipeline import run_full_analysis
            return run_full_analysis(args)
        elif args.command == 'report':
            from visualization.report import generate_html_report
            return generate_html_report(args)
        elif args.command == 'system':
            from extraction.fallback import analyze_system_binaries
            return analyze_system_binaries(args)
        elif args.command == 'fast':
            from analysis.performance import run_fast_analysis
            return run_fast_analysis(args)
        elif args.command == 'lm':
            from analysis.lm import run_lm_command
            return run_lm_command(args)
        elif args.command == 'clones':
            from clone_detection.pipeline import run_clones_command
            return run_clones_command(args)
        elif args.command == 'compiler-matrix':
            from experiments.compiler_matrix import run_compiler_matrix
            return run_compiler_matrix(args)
        elif args.command == 'github':
            from corpus.github_builder import run_github_corpus_builder
            return run_github_corpus_builder(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
    except Exception as e:
        logger.error(f"Error executing {args.command}: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())