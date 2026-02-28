#!/usr/bin/env python3
"""
Test script for GitHub corpus builder functionality.
"""

import logging
from pathlib import Path
from corpus.github_builder import GitHubCorpusBuilder, GitHubRepo

def test_search_functionality():
    """Test the GitHub API search functionality."""
    print("Testing GitHub repository search...")
    
    # Create builder (without token for basic testing)
    builder = GitHubCorpusBuilder(Path("./test_github_work"))
    
    try:
        # Search for a few C repositories
        repos = builder.search_repositories('C', min_stars=1000, max_repos=3)
        
        print(f"Found {len(repos)} repositories:")
        for repo in repos:
            print(f"  - {repo.owner}/{repo.name} ({repo.stars} stars)")
            print(f"    Language: {repo.language}")
            print(f"    Description: {repo.description[:100] if repo.description else 'No description'}...")
            print()
        
        return len(repos) > 0
        
    except Exception as e:
        print(f"Search test failed: {e}")
        return False

def test_binary_detection():
    """Test binary file detection."""
    print("Testing binary file detection...")
    
    builder = GitHubCorpusBuilder(Path("./test_github_work"))
    
    # Test with system binaries
    test_paths = [
        Path("/usr/bin/ls"),
        Path("/usr/bin/cat"),
        Path("/bin/bash")
    ]
    
    for path in test_paths:
        if path.exists():
            is_binary = builder.is_binary_executable(path)
            print(f"  {path}: {'✓' if is_binary else '✗'} binary executable")
    
    return True

def main():
    """Run all tests."""
    logging.basicConfig(level=logging.INFO)
    
    print("=== GitHub Corpus Builder Test ===\n")
    
    # Test 1: API search functionality
    search_success = test_search_functionality()
    print(f"GitHub search test: {'✓ PASS' if search_success else '✗ FAIL'}\n")
    
    # Test 2: Binary detection
    binary_success = test_binary_detection()
    print(f"Binary detection test: {'✓ PASS' if binary_success else '✗ FAIL'}\n")
    
    overall_success = search_success and binary_success
    print(f"Overall test result: {'✓ PASS' if overall_success else '✗ FAIL'}")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    exit(main())