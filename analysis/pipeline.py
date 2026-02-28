"""
Main analysis pipeline that orchestrates all statistical analyses.
"""

import logging
from pathlib import Path
from typing import List
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.helpers import load_pickle, ensure_output_dir, Binary
from analysis.frequency import run_frequency_analysis
from analysis.ngrams import run_ngram_analysis
from analysis.compression import run_compression_analysis

logger = logging.getLogger(__name__)

def load_corpus(corpus_dir: Path) -> List[Binary]:
    """Load corpus data from the extraction phase."""
    corpus_file = corpus_dir / "corpus.pkl"
    
    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
    
    binaries = load_pickle(corpus_file)
    logger.info(f"Loaded {len(binaries)} binaries from corpus")
    
    # Log summary statistics
    total_instructions = sum(b.instruction_count for b in binaries)
    total_functions = sum(b.function_count for b in binaries)
    
    logger.info(f"Corpus summary: {total_functions} functions, {total_instructions} instructions")
    
    return binaries

def run_analysis_pipeline(args) -> int:
    """Run the complete analysis pipeline."""
    try:
        corpus_dir = Path(args.corpus_dir)
        output_dir = ensure_output_dir(Path(args.output_dir))
        
        logger.info("Starting analysis pipeline...")
        
        # Load corpus
        binaries = load_corpus(corpus_dir)
        
        if not binaries:
            logger.error("No binaries loaded from corpus")
            return 1
        
        # Run frequency analysis (Phase 2.1)
        freq_results = run_frequency_analysis(binaries, output_dir)
        
        # Run n-gram analysis (Phase 2.2)
        ngram_results = run_ngram_analysis(binaries, output_dir)
        
        # Run compression analysis (Phase 2.3)
        compression_results = run_compression_analysis(binaries, output_dir)
        
        # Run motif discovery and positional analysis (Phase 2.4 & 2.5)
        from analysis.motifs import run_motif_analysis
        motif_results = run_motif_analysis(binaries, output_dir)
        
        # Run clustering analysis (Phase 3)
        from clustering.pipeline import run_clustering_analysis
        clustering_results = run_clustering_analysis(binaries, output_dir)
        
        # Run information-theoretic analysis (Phase 4)
        from analysis.information import run_information_analysis
        information_results = run_information_analysis(binaries, output_dir)
        
        # Run compiler fingerprinting analysis (Phase 5)
        from analysis.compiler_fingerprinting_simple import run_compiler_fingerprinting
        compiler_results = run_compiler_fingerprinting(binaries, output_dir)

        # Generate all plots
        try:
            from visualization.plots import generate_all_plots
            import matplotlib
            matplotlib.use('Agg')   # non-interactive backend
            plots_dir = ensure_output_dir(output_dir / "plots")
            plot_files = generate_all_plots(output_dir, plots_dir)
            logger.info(f"Generated {len(plot_files)} plots in {plots_dir}")
        except Exception as e:
            logger.warning(f"Plot generation failed (non-fatal): {e}")

        logger.info("Analysis pipeline completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Analysis pipeline failed: {e}")
        return 1

def run_full_analysis(args) -> int:
    """Entry point for full analysis command."""
    return run_analysis_pipeline(args)