# Binary DNA API Reference

This document provides comprehensive API documentation for the Binary DNA toolkit, enabling researchers and developers to extend the toolkit and integrate it into their own research workflows.

## Core Data Structures

### Binary Class

The `Binary` class represents a compiled binary with all its functions and instruction sequences.

```python
from utils.helpers import Binary

@dataclass
class Binary:
    """Represents a binary with all its functions."""
    name: str                    # Binary name (e.g., 'grep')
    path: str                    # File path to binary
    functions: List[Function]    # List of Function objects
    metadata: Optional[Dict] = None  # Additional metadata (language, repo, etc.)
    
    @property
    def full_opcode_sequence(self) -> List[str]:
        """Return all opcodes across all functions, concatenated."""
        
    @property
    def instruction_count(self) -> int:
        """Return total number of instructions."""
        
    @property  
    def function_count(self) -> int:
        """Return the number of functions."""
```

**Usage Example:**
```python
# Load a binary corpus
from utils.helpers import load_pickle
binaries = load_pickle("corpus.pkl")

for binary in binaries:
    print(f"{binary.name}: {binary.instruction_count:,} instructions")
    print(f"Opcode sequence: {binary.full_opcode_sequence[:10]}...")  # First 10
    
    if binary.metadata:
        print(f"Source language: {binary.metadata.get('language', 'unknown')}")
```

### Function Class

The `Function` class represents a single function within a binary.

```python
@dataclass
class Function:
    """Represents a function with its instruction sequence."""
    name: str                      # Function name
    binary_name: str              # Parent binary name  
    instructions: List[Instruction]  # List of Instruction objects
    
    @property
    def opcode_sequence(self) -> List[str]:
        """Return just the mnemonics as a sequence."""
        
    @property
    def size(self) -> int:
        """Return the number of instructions."""
```

### Instruction Class

The `Instruction` class represents a single disassembled instruction.

```python
@dataclass
class Instruction:
    """Represents a single disassembled instruction."""
    address: int        # Memory address
    mnemonic: str      # Instruction mnemonic (normalized)
    operands: str      # Operand string
    raw_bytes: bytes   # Raw instruction bytes
```

## Analysis Modules

### Frequency Analysis

Module: `analysis.frequency`

#### Core Functions

```python
def analyze_opcode_frequencies(binaries: List[Binary]) -> Dict:
    """
    Analyze opcode frequency distributions across binaries.
    
    Args:
        binaries: List of Binary objects to analyze
        
    Returns:
        Dictionary with frequency analysis results:
        {
            'frequency_distribution': {
                'top_50_opcodes': [(opcode, count, rank), ...],
                'total_instructions': int,
                'unique_opcodes': int
            },
            'zipf_analysis': {
                'global_zipf': {
                    'alpha': float,      # Zipf exponent
                    'constant': float,   # Normalization constant
                    'r_squared': float   # Goodness of fit
                }
            }
        }
    """

def compute_zipf_analysis(frequency_data: List[Tuple]) -> Dict:
    """
    Fit Zipf's law to frequency data.
    
    Args:
        frequency_data: List of (opcode, frequency, rank) tuples
        
    Returns:
        Dictionary with Zipf fitting results
    """
```

**Usage Example:**
```python
from analysis.frequency import analyze_opcode_frequencies

# Analyze frequency patterns
results = analyze_opcode_frequencies(binaries)

# Extract Zipf parameters
zipf_params = results['zipf_analysis']['global_zipf']
alpha = zipf_params['alpha']
r_squared = zipf_params['r_squared']

print(f"Zipf exponent: α={alpha:.3f}")
print(f"Goodness of fit: R²={r_squared:.3f}")

# Get most common instructions
top_opcodes = results['frequency_distribution']['top_50_opcodes']
for opcode, count, rank in top_opcodes[:10]:
    print(f"{rank}. {opcode}: {count:,} occurrences")
```

### N-gram Analysis

Module: `analysis.ngrams`

#### Core Functions

```python
def extract_ngrams(sequence: List[str], n: int) -> List[Tuple]:
    """
    Extract n-grams from instruction sequence.
    
    Args:
        sequence: List of instruction mnemonics
        n: N-gram length
        
    Returns:
        List of n-gram tuples
    """

def compute_entropy_rate(ngrams: List, prev_ngrams: List) -> float:
    """
    Compute entropy rate for n-gram sequences.
    
    Args:
        ngrams: N-grams of length n
        prev_ngrams: N-grams of length n-1
        
    Returns:
        Entropy rate in bits
    """

def run_ngram_analysis(binaries: List[Binary], output_dir: Path) -> Dict:
    """
    Run comprehensive n-gram analysis.
    
    Returns:
        {
            'entropy_analysis': {
                'entropy_rates': [
                    {'n': int, 'entropy': float, 'entropy_rate': float},
                    ...
                ]
            },
            'ngram_statistics': {...}
        }
    """
```

**Usage Example:**
```python
from analysis.ngrams import extract_ngrams, compute_entropy_rate

# Extract bigrams from a binary
sequence = binary.full_opcode_sequence
bigrams = extract_ngrams(sequence, 2)
trigrams = extract_ngrams(sequence, 3)

# Compute entropy rate  
entropy_rate = compute_entropy_rate(trigrams, bigrams)
print(f"Trigram entropy rate: {entropy_rate:.3f} bits")

# Most common patterns
from collections import Counter
common_bigrams = Counter([' '.join(bg) for bg in bigrams]).most_common(10)
for pattern, count in common_bigrams:
    print(f"{pattern}: {count}")
```

### Compression Analysis

Module: `analysis.compression`

#### Core Functions

```python
def encode_sequence_for_compression(sequence: List[str], vocab: Dict[str, int]) -> bytes:
    """
    Encode instruction sequence for compression analysis.
    
    Args:
        sequence: Instruction sequence
        vocab: Vocabulary mapping opcode -> integer
        
    Returns:
        Encoded byte sequence
    """

def compute_ncd(seq1_compressed: bytes, seq2_compressed: bytes, 
                joint_compressed: bytes) -> float:
    """
    Compute Normalized Compression Distance.
    
    Args:
        seq1_compressed: Compressed sequence 1
        seq2_compressed: Compressed sequence 2  
        joint_compressed: Compressed concatenated sequences
        
    Returns:
        NCD value between 0 and 1
    """

def run_compression_analysis(binaries: List[Binary], output_dir: Path) -> Dict:
    """
    Run complete compression-based analysis.
    
    Returns:
        {
            'per_binary_results': [
                {
                    'binary_name': str,
                    'zlib_ratio': float,
                    'lzma_ratio': float,
                    'lz_complexity': float
                },
                ...
            ],
            'global_statistics': {...}
        }
    """
```

### Clustering Analysis

Module: `clustering.ncd`

#### Core Functions

```python
class NCDAnalyzer:
    """Normalized Compression Distance analyzer."""
    
    def __init__(self, compressors: List[str] = None):
        """Initialize with list of compression algorithms."""
        
    def compute_ncd_matrix(self, binaries: List[Binary]) -> np.ndarray:
        """
        Compute NCD distance matrix for binaries.
        
        Args:
            binaries: List of Binary objects
            
        Returns:
            Symmetric distance matrix
        """
        
    def analyze_ncd_statistics(self, matrix: np.ndarray, 
                             binary_names: List[str]) -> Dict:
        """Compute statistics for NCD matrix."""
```

**Usage Example:**
```python
from clustering.ncd import NCDAnalyzer

# Create NCD analyzer
ncd_analyzer = NCDAnalyzer(compressors=['zlib', 'lzma'])

# Compute distance matrix
distance_matrix = ncd_analyzer.compute_ncd_matrix(binaries)
print(f"Distance matrix shape: {distance_matrix.shape}")

# Analyze statistics
binary_names = [b.name for b in binaries]
stats = ncd_analyzer.analyze_ncd_statistics(distance_matrix, binary_names)
print(f"Mean distance: {stats['mean_distance']:.3f}")
print(f"Most similar pair: {stats['most_similar_pairs'][0]}")
```

### Compiler Fingerprinting

Module: `analysis.compiler_fingerprinting_simple`

#### Core Functions

```python
class SimpleCompilerFingerprinter:
    """Heuristic-based compiler fingerprinting."""
    
    def extract_compiler_features(self, binary: Binary) -> Dict[str, float]:
        """
        Extract 35+ compiler-specific features.
        
        Args:
            binary: Binary to analyze
            
        Returns:
            Dictionary of feature_name -> value
        """
        
    def identify_compiler_heuristic(self, binary: Binary) -> Dict:
        """
        Identify compiler using heuristic rules.
        
        Returns:
            {
                'predicted_compiler': str,      # 'gcc', 'clang', 'rustc'
                'compiler_confidence': float,   # 0-1 confidence
                'predicted_optimization': str,  # 'O0', 'O2', 'O3'
                'optimization_confidence': float,
                'features': Dict[str, float],   # All extracted features
                'analysis_details': Dict
            }
        """
        
    def analyze_corpus_compilers(self, binaries: List[Binary]) -> Dict:
        """Analyze compiler patterns across corpus."""
```

**Usage Example:**
```python
from analysis.compiler_fingerprinting_simple import SimpleCompilerFingerprinter

# Create fingerprinter
fingerprinter = SimpleCompilerFingerprinter()

# Analyze single binary
analysis = fingerprinter.identify_compiler_heuristic(binary)
print(f"Compiler: {analysis['predicted_compiler']} ({analysis['compiler_confidence']:.3f})")
print(f"Optimization: {analysis['predicted_optimization']} ({analysis['optimization_confidence']:.3f})")

# Key features
features = analysis['features']
key_features = ['vectorization_ratio', 'lea_ratio', 'cmov_ratio']
for feature in key_features:
    print(f"{feature}: {features.get(feature, 0):.4f}")

# Analyze entire corpus
corpus_results = fingerprinter.analyze_corpus_compilers(binaries)
print(f"Compiler distribution: {corpus_results['corpus_summary']['compiler_distribution']}")
```

## Visualization API

Module: `visualization.plots`

#### Core Functions

```python
def plot_zipf_distribution(frequency_data: List[tuple], zipf_params: Dict, 
                          output_path: Path) -> None:
    """Plot rank-frequency distribution with Zipf's law fit."""

def plot_entropy_rates(entropy_data: List[Dict], output_path: Path) -> None:
    """Plot entropy rate vs n-gram length."""

def plot_compiler_distribution(compiler_results: Dict, output_path: Path) -> None:
    """Plot compiler and optimization distribution."""

def plot_distance_matrix_heatmap(distance_matrix: List[List[float]], 
                                binary_names: List[str], title: str, 
                                output_path: Path) -> None:
    """Plot distance matrix as heatmap."""

def generate_all_plots(results_dir: Path, plot_dir: Path) -> List[Path]:
    """Generate all plots from analysis results."""
```

**Usage Example:**
```python
from visualization.plots import plot_compiler_distribution, generate_all_plots
from pathlib import Path

# Generate specific plot
plot_compiler_distribution(compiler_results, Path('./compiler_dist.png'))

# Generate all available plots
plot_files = generate_all_plots(
    results_dir=Path('./analysis_results'),
    plot_dir=Path('./plots')
)
print(f"Generated {len(plot_files)} plots")
```

## Utility Functions

Module: `utils.helpers`

#### Data I/O Functions

```python
def save_json(data: Any, filepath: Path) -> None:
    """Save data as JSON with proper serialization."""

def load_json(filepath: Path) -> Any:
    """Load data from JSON file."""

def save_pickle(data: Any, filepath: Path) -> None:
    """Save data using pickle."""

def load_pickle(filepath: Path) -> Any:
    """Load data from pickle file."""

def build_vocabulary(binaries: List[Binary]) -> Dict[str, int]:
    """Build instruction vocabulary from corpus."""
```

**Usage Example:**
```python
from utils.helpers import save_json, load_json, build_vocabulary

# Save analysis results
save_json(analysis_results, Path('./results.json'))

# Load previous results
previous_results = load_json(Path('./results.json'))

# Build vocabulary for compression analysis
vocab = build_vocabulary(binaries)
print(f"Vocabulary size: {len(vocab)}")
print(f"Most common instructions: {list(vocab.keys())[:10]}")
```

#### Binary Extraction

```python
def extract_binary_features(binary_path: Path) -> Binary:
    """
    Extract features from a binary file.
    
    Args:
        binary_path: Path to binary file
        
    Returns:
        Binary object with extracted features
    """
```

## Extension Points

### Adding New Analysis Methods

1. **Create Analysis Module:**
```python
# analysis/my_analysis.py
def run_my_analysis(binaries: List[Binary], output_dir: Path) -> Dict:
    """
    Custom analysis implementation.
    
    Args:
        binaries: Input binaries
        output_dir: Output directory
        
    Returns:
        Analysis results dictionary
    """
    results = {}
    
    for binary in binaries:
        # Your analysis logic here
        binary_results = analyze_single_binary(binary)
        results[binary.name] = binary_results
    
    # Save results
    from utils.helpers import save_json
    save_json(results, output_dir / "my_analysis.json")
    
    return results
```

2. **Integrate into Pipeline:**
```python
# analysis/pipeline.py - add to run_analysis_pipeline()
from analysis.my_analysis import run_my_analysis

def run_analysis_pipeline(args) -> int:
    # ... existing analyses ...
    
    # Add your analysis
    my_results = run_my_analysis(binaries, output_dir)
    
    return 0
```

### Adding New Visualization Types

```python
# visualization/plots.py
def plot_my_analysis(results: Dict, output_path: Path) -> None:
    """Custom visualization for my analysis."""
    import matplotlib.pyplot as plt
    
    # Extract data from results
    data = results.get('my_data', [])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data)
    ax.set_title('My Analysis Results')
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved my analysis plot to {output_path}")

# Add to generate_all_plots()
def generate_all_plots(results_dir: Path, plot_dir: Path) -> List[Path]:
    # ... existing plots ...
    
    try:
        my_results = load_json(results_dir / "my_analysis.json")
        my_plot = plot_dir / "my_analysis.png"
        plot_my_analysis(my_results, my_plot)
        plot_files.append(my_plot)
    except FileNotFoundError:
        logger.debug("No my_analysis results found")
    
    return plot_files
```

### Custom Feature Extractors

```python
class CustomFeatureExtractor:
    """Extract domain-specific features from binaries."""
    
    def extract_features(self, binary: Binary) -> Dict[str, float]:
        """Extract custom features."""
        features = {}
        
        sequence = binary.full_opcode_sequence
        
        # Example: Control flow complexity
        jumps = sum(1 for instr in sequence if instr.startswith('j'))
        features['jump_density'] = jumps / len(sequence)
        
        # Example: Stack usage patterns
        pushes = sum(1 for instr in sequence if instr == 'push')
        pops = sum(1 for instr in sequence if instr == 'pop')
        features['stack_balance'] = abs(pushes - pops) / max(pushes + pops, 1)
        
        return features
```

## Error Handling

The Binary DNA toolkit includes comprehensive error handling:

```python
import logging

logger = logging.getLogger(__name__)

try:
    # Analysis code
    result = analyze_binary(binary)
except Exception as e:
    logger.error(f"Analysis failed for {binary.name}: {e}")
    # Graceful degradation
    result = {'error': str(e), 'binary_name': binary.name}

# Check for errors in results
if 'error' in result:
    print(f"Analysis failed: {result['error']}")
else:
    print(f"Analysis successful: {len(result)} features extracted")
```

## Performance Considerations

### Memory Management

```python
def analyze_large_binary(binary: Binary, max_instructions: int = 50000):
    """Analyze large binary with memory constraints."""
    sequence = binary.full_opcode_sequence
    
    if len(sequence) > max_instructions:
        # Sample instructions for memory efficiency
        import random
        random.seed(42)  # Reproducible sampling
        sequence = random.sample(sequence, max_instructions)
        
    return analyze_sequence(sequence)
```

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def parallel_analysis(binaries: List[Binary], n_workers: int = None):
    """Analyze binaries in parallel."""
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
        
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(analyze_binary, binary) for binary in binaries]
        
        results = []
        for future in futures:
            try:
                result = future.result(timeout=300)  # 5 minute timeout
                results.append(result)
            except Exception as e:
                logger.error(f"Parallel analysis failed: {e}")
                
    return results
```

This API reference provides comprehensive documentation for extending and integrating the Binary DNA toolkit into custom research workflows. The modular design enables researchers to build upon existing functionality while maintaining compatibility with the core analysis pipeline.