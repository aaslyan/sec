# Binary DNA Installation Guide

This guide provides step-by-step instructions for setting up the Binary DNA toolkit on various systems.

## System Requirements

### Operating System Support
- **Linux**: Ubuntu 18.04+, Debian 10+, CentOS 7+, Fedora 30+
- **macOS**: 10.15+ (Catalina and later)
- **Windows**: Windows 10+ with WSL2 (recommended) or native Windows

### Hardware Requirements
- **CPU**: Multi-core processor recommended (2+ cores)
- **RAM**: 4GB minimum, 8GB+ recommended for large corpus analysis
- **Storage**: 2GB for toolkit, additional space for corpus data
- **Network**: Internet connection for GitHub corpus building

### Dependencies

#### Core Requirements
- **Python**: 3.8 or later
- **GNU Binutils**: objdump for disassembly
- **Git**: For repository cloning and version control

#### Python Packages
- **NumPy**: ≥1.19.0 (numerical computations)
- **Matplotlib**: ≥3.3.0 (visualization)
- **Requests**: ≥2.25.0 (GitHub API integration)
- **SciPy**: ≥1.6.0 (statistical functions, optional but recommended)

#### Optional Dependencies
- **Seaborn**: ≥0.11.0 (enhanced visualizations)
- **Scikit-learn**: ≥0.24.0 (machine learning features)
- **UMAP**: ≥0.5.0 (dimensionality reduction)

## Installation Methods

### Method 1: Standard Installation (Recommended)

#### Step 1: Clone Repository
```bash
# Clone the Binary DNA repository
git clone <repository-url> binary-dna
cd binary-dna

# Verify toolkit structure
ls -la
```

#### Step 2: Set Up Python Environment
```bash
# Create virtual environment (recommended)
python3 -m venv binary-dna-env
source binary-dna-env/bin/activate  # Linux/macOS
# binary-dna-env\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip
```

#### Step 3: Install Python Dependencies
```bash
# Install core requirements
pip install numpy>=1.19.0 matplotlib>=3.3.0 requests>=2.25.0

# Install optional dependencies for full functionality
pip install scipy>=1.6.0 seaborn>=0.11.0

# Alternative: Install from requirements file (if available)
pip install -r requirements.txt
```

#### Step 4: Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install binutils git python3-dev build-essential
```

**CentOS/RHEL:**
```bash
sudo yum update
sudo yum install binutils git python3-devel gcc
```

**macOS:**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Using Homebrew (recommended)
brew install binutils git
```

**Windows (WSL2):**
```bash
# In WSL2 Ubuntu
sudo apt-get update
sudo apt-get install binutils git python3-dev build-essential
```

#### Step 5: Verify Installation
```bash
# Test objdump availability
objdump --version

# Test Python environment
python3 -c "import numpy, matplotlib, requests; print('Dependencies OK')"

# Test Binary DNA toolkit
python3 binary_dna.py --help
```

### Method 2: Conda Installation

```bash
# Create conda environment
conda create -n binary-dna python=3.9
conda activate binary-dna

# Install dependencies
conda install numpy matplotlib requests scipy
conda install -c conda-forge seaborn

# Clone repository
git clone <repository-url> binary-dna
cd binary-dna

# Verify installation
python binary_dna.py --help
```

### Method 3: Docker Installation (Experimental)

```bash
# Build Docker image
docker build -t binary-dna .

# Run toolkit in container
docker run -it --rm -v $(pwd)/data:/data binary-dna \
  python binary_dna.py system --output-dir /data/results --limit 10

# Interactive mode
docker run -it --rm -v $(pwd)/data:/data binary-dna bash
```

## Configuration

### Environment Variables

Set these optional environment variables to customize behavior:

```bash
# GitHub API token for higher rate limits
export GITHUB_TOKEN="your_github_token_here"

# Default output directory
export BINARY_DNA_OUTPUT="/path/to/default/output"

# Parallel processing workers
export BINARY_DNA_WORKERS=8

# Memory limit per analysis (MB)
export BINARY_DNA_MEMORY_LIMIT=1000
```

### Configuration Files

Create `~/.binary_dna_config.json` for persistent settings:

```json
{
    "default_output_dir": "/home/user/binary_dna_results",
    "github_token": "your_token_here",
    "parallel_workers": 8,
    "memory_limit_mb": 1000,
    "compression_algorithms": ["zlib", "lzma"],
    "visualization_dpi": 150,
    "log_level": "INFO"
}
```

## Platform-Specific Setup

### Linux Setup

#### Ubuntu 20.04/22.04
```bash
# Complete setup script for Ubuntu
#!/bin/bash

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y \
    python3 python3-pip python3-venv \
    binutils git build-essential \
    wget curl

# Clone repository
git clone <repository-url> binary-dna
cd binary-dna

# Set up Python environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install numpy matplotlib requests scipy seaborn

# Verify installation
python binary_dna.py --help
echo "Binary DNA installation completed successfully!"
```

#### CentOS/RHEL 8
```bash
# Enable PowerTools repository for development packages
sudo dnf config-manager --set-enabled powertools
sudo dnf install -y python3 python3-pip git binutils gcc python3-devel

# Continue with standard installation...
```

### macOS Setup

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install python@3.9 binutils git

# Set up PATH (add to ~/.zshrc or ~/.bash_profile)
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc

# Continue with standard Python environment setup
python3 -m venv binary-dna-env
source binary-dna-env/bin/activate
pip install numpy matplotlib requests scipy seaborn
```

### Windows Setup

#### Windows 10/11 with WSL2 (Recommended)

1. **Enable WSL2:**
```powershell
# In PowerShell as Administrator
wsl --install
wsl --set-default-version 2
```

2. **Install Ubuntu in WSL2:**
```bash
# Install Ubuntu from Microsoft Store or
wsl --install -d Ubuntu

# In WSL2 Ubuntu terminal
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv binutils git build-essential
```

3. **Continue with Linux installation steps above**

#### Native Windows Installation

```powershell
# Install Python from python.org or Microsoft Store
# Install Git for Windows

# In Command Prompt or PowerShell
git clone <repository-url> binary-dna
cd binary-dna

# Create virtual environment
python -m venv binary-dna-env
binary-dna-env\Scripts\activate

# Install dependencies
pip install numpy matplotlib requests scipy seaborn

# Install Windows Subsystem for Linux for objdump
# Or use alternative disassemblers (advanced setup required)
```

## Troubleshooting

### Common Installation Issues

#### Issue: objdump not found
```bash
# Solution: Install binutils
# Ubuntu/Debian
sudo apt-get install binutils

# CentOS/RHEL
sudo yum install binutils

# macOS
brew install binutils
# Add to PATH: export PATH="/opt/homebrew/bin:$PATH"
```

#### Issue: Python import errors
```bash
# Check Python version
python3 --version  # Should be 3.8+

# Check package installation
python3 -c "import numpy; print(numpy.__version__)"

# Reinstall if necessary
pip install --upgrade --force-reinstall numpy matplotlib
```

#### Issue: GitHub API rate limiting
```bash
# Set GitHub token
export GITHUB_TOKEN="your_token_here"

# Or reduce corpus size
python binary_dna.py github --repos-per-language 5 --min-stars 1000
```

#### Issue: Memory errors with large binaries
```bash
# Use sampling for large binaries
python binary_dna.py fast --sample-size 10000 --max-memory 500

# Or analyze smaller corpus
python binary_dna.py system --limit 10
```

#### Issue: Permission denied errors
```bash
# Fix permissions
chmod +x binary_dna.py

# Or run with python explicitly
python3 binary_dna.py --help
```

### Performance Optimization

#### For Large-Scale Analysis
```bash
# Increase parallel workers
export BINARY_DNA_WORKERS=16

# Use fast analysis mode
python binary_dna.py fast --workers 16 --sample-size 20000

# Monitor memory usage
python binary_dna.py system --limit 50 --max-memory 2000
```

#### For Memory-Constrained Systems
```bash
# Reduce memory usage
python binary_dna.py fast --max-memory 256 --sample-size 5000

# Process smaller batches
python binary_dna.py system --limit 5
```

## Verification and Testing

### Basic Functionality Test
```bash
# Test system binary analysis (quick)
python binary_dna.py system --output-dir ./test_output --limit 3

# Verify results
ls -la ./test_output/
cat ./test_output/corpus/corpus_metadata.json
```

### Comprehensive Test Suite
```bash
# Run all test scripts
python test_github_builder.py
python test_compiler_fingerprinting.py

# Test with synthetic data
python binary_dna.py analyze --corpus-dir ./test_clustering_synthetic --output-dir ./test_results
```

### Performance Benchmark
```bash
# Benchmark analysis performance
time python binary_dna.py fast \
  --corpus-dir ./test_data \
  --output-dir ./benchmark \
  --workers 4

# Check memory usage
/usr/bin/time -v python binary_dna.py system --limit 10 --output-dir ./memory_test
```

## Next Steps

After successful installation:

1. **Quick Start**: Try the examples in `docs/RESEARCH_EXAMPLES.md`
2. **Read Documentation**: Review `docs/README.md` and `docs/API_REFERENCE.md`
3. **Build First Corpus**: Use `python binary_dna.py github` to collect data
4. **Run Analysis**: Use `python binary_dna.py analyze` for comprehensive analysis
5. **Generate Reports**: Use `python binary_dna.py report` for visualization

## Support and Community

### Getting Help
- **Documentation**: Check `docs/` directory for comprehensive guides
- **Issues**: Report bugs and feature requests via GitHub issues
- **Examples**: See `docs/RESEARCH_EXAMPLES.md` for practical applications

### Contributing
- **Code**: Submit pull requests for bug fixes and new features
- **Documentation**: Improve documentation and examples
- **Testing**: Help test on different platforms and configurations
- **Research**: Share research results and use cases

---

This installation guide ensures researchers can quickly set up the Binary DNA toolkit across different platforms and start conducting binary analysis research. For additional help, consult the API reference and research examples documentation.