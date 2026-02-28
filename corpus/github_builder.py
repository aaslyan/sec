"""
GitHub corpus builder for Binary DNA analysis.

Downloads and builds binaries from popular GitHub repositories to create
diverse datasets for analysis.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional
import json
import time
import requests
from dataclasses import dataclass
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.helpers import Binary, save_pickle

logger = logging.getLogger(__name__)

@dataclass
class GitHubRepo:
    """Repository information for corpus building."""
    owner: str
    name: str
    language: str
    stars: int
    build_command: Optional[str] = None
    binary_paths: Optional[List[str]] = None
    description: Optional[str] = None

class GitHubCorpusBuilder:
    """Builder for creating binary corpora from GitHub repositories."""
    
    def __init__(self, work_dir: Path, github_token: Optional[str] = None):
        """
        Initialize corpus builder.
        
        Args:
            work_dir: Working directory for cloning and building
            github_token: GitHub API token for higher rate limits
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.github_token = github_token or os.environ.get('GITHUB_TOKEN')
        
        # API setup
        self.session = requests.Session()
        if self.github_token:
            self.session.headers.update({'Authorization': f'token {self.github_token}'})
    
    def search_repositories(self, language: str, min_stars: int = 100, 
                          max_repos: int = 50) -> List[GitHubRepo]:
        """
        Search for popular repositories in a given language.
        
        Args:
            language: Programming language to search for
            min_stars: Minimum number of stars
            max_repos: Maximum repositories to return
            
        Returns:
            List of repository information
        """
        logger.info(f"Searching for {language} repositories with {min_stars}+ stars")
        
        repos = []
        page = 1
        per_page = min(100, max_repos)
        
        while len(repos) < max_repos:
            # GitHub Search API
            url = "https://api.github.com/search/repositories"
            params = {
                'q': f'language:{language} stars:>={min_stars}',
                'sort': 'stars',
                'order': 'desc',
                'page': page,
                'per_page': per_page
            }
            
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if not data.get('items'):
                    break
                
                for item in data['items']:
                    if len(repos) >= max_repos:
                        break
                    
                    repo = GitHubRepo(
                        owner=item['owner']['login'],
                        name=item['name'],
                        language=item['language'],
                        stars=item['stargazers_count'],
                        description=item.get('description', '')
                    )
                    repos.append(repo)
                
                page += 1
                
                # Rate limiting
                if 'X-RateLimit-Remaining' in response.headers:
                    remaining = int(response.headers['X-RateLimit-Remaining'])
                    if remaining < 10:
                        reset_time = int(response.headers['X-RateLimit-Reset'])
                        sleep_time = max(0, reset_time - int(time.time()) + 1)
                        logger.info(f"Rate limit low, sleeping {sleep_time}s")
                        time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error searching repositories: {e}")
                break
        
        logger.info(f"Found {len(repos)} repositories")
        return repos
    
    def clone_repository(self, repo: GitHubRepo) -> Optional[Path]:
        """
        Clone a repository to the working directory.
        
        Args:
            repo: Repository to clone
            
        Returns:
            Path to cloned repository or None if failed
        """
        repo_path = self.work_dir / f"{repo.owner}_{repo.name}"
        
        if repo_path.exists():
            logger.info(f"Repository {repo.owner}/{repo.name} already exists")
            return repo_path
        
        clone_url = f"https://github.com/{repo.owner}/{repo.name}.git"
        
        try:
            logger.info(f"Cloning {repo.owner}/{repo.name}")
            
            # Shallow clone to save space
            cmd = ["git", "clone", "--depth", "1", clone_url, str(repo_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"Successfully cloned {repo.owner}/{repo.name}")
                return repo_path
            else:
                logger.error(f"Failed to clone {repo.owner}/{repo.name}: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout cloning {repo.owner}/{repo.name}")
            return None
        except Exception as e:
            logger.error(f"Error cloning {repo.owner}/{repo.name}: {e}")
            return None
    
    def build_repository(self, repo: GitHubRepo, repo_path: Path) -> List[Path]:
        """
        Build a repository and find resulting binaries.
        
        Args:
            repo: Repository information
            repo_path: Path to cloned repository
            
        Returns:
            List of paths to built binaries
        """
        logger.info(f"Building {repo.owner}/{repo.name}")
        
        # Common build patterns by language
        build_commands = {
            'C': ['make', 'make all'],
            'C++': ['make', 'make all', 'cmake . && make'],
            'Rust': ['cargo build --release'],
            'Go': ['go build ./...'],
            'Zig': ['zig build'],
        }
        
        # Use custom build command if provided
        if repo.build_command:
            commands = [repo.build_command]
        else:
            commands = build_commands.get(repo.language, ['make'])
        
        built_binaries = []
        
        for cmd in commands:
            try:
                logger.debug(f"Running: {cmd}")
                
                result = subprocess.run(
                    cmd.split(),
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout
                )
                
                if result.returncode == 0:
                    logger.info(f"Build successful with: {cmd}")
                    
                    # Find built binaries
                    binaries = self.find_binaries(repo_path)
                    if binaries:
                        built_binaries.extend(binaries)
                        break
                else:
                    logger.debug(f"Build failed with {cmd}: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                logger.warning(f"Build timeout with: {cmd}")
                continue
            except Exception as e:
                logger.debug(f"Build error with {cmd}: {e}")
                continue
        
        if not built_binaries:
            logger.warning(f"No binaries built for {repo.owner}/{repo.name}")
        else:
            logger.info(f"Found {len(built_binaries)} binaries in {repo.owner}/{repo.name}")
        
        return built_binaries
    
    def find_binaries(self, repo_path: Path) -> List[Path]:
        """
        Find executable binaries in a repository.
        
        Args:
            repo_path: Path to repository
            
        Returns:
            List of binary file paths
        """
        binaries = []
        
        # Common binary locations
        search_paths = [
            repo_path,
            repo_path / "target" / "release",  # Rust
            repo_path / "bin",
            repo_path / "build",
            repo_path / "out",
            repo_path / "zig-out" / "bin",  # Zig
        ]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
                
            try:
                # Find executable files
                for file_path in search_path.rglob("*"):
                    if (file_path.is_file() and 
                        os.access(file_path, os.X_OK) and
                        self.is_binary_executable(file_path)):
                        
                        # Skip very large binaries (>100MB)
                        if file_path.stat().st_size > 100 * 1024 * 1024:
                            continue
                            
                        # Skip common non-binary executables
                        if file_path.suffix in {'.sh', '.py', '.pl', '.rb'}:
                            continue
                            
                        binaries.append(file_path)
                        
            except Exception as e:
                logger.debug(f"Error searching {search_path}: {e}")
        
        # Remove duplicates and sort by size (prefer smaller binaries)
        unique_binaries = list(set(binaries))
        unique_binaries.sort(key=lambda p: p.stat().st_size)
        
        return unique_binaries[:10]  # Limit to 10 binaries per repo
    
    def is_binary_executable(self, file_path: Path) -> bool:
        """
        Check if a file is a binary executable.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file is a binary executable
        """
        try:
            # Check ELF magic number (Linux binaries)
            with open(file_path, 'rb') as f:
                magic = f.read(4)
                return magic == b'\x7fELF'
        except:
            return False
    
    def build_corpus(self, languages: List[str], repos_per_language: int = 20,
                    min_stars: int = 100) -> List[Binary]:
        """
        Build a corpus from multiple programming languages.
        
        Args:
            languages: Programming languages to include
            repos_per_language: Number of repositories per language
            min_stars: Minimum stars for repository selection
            
        Returns:
            List of Binary objects
        """
        logger.info(f"Building corpus from {len(languages)} languages")
        
        all_binaries = []
        
        for language in languages:
            logger.info(f"Processing {language} repositories")
            
            # Search for repositories
            repos = self.search_repositories(language, min_stars, repos_per_language)
            
            for repo in repos:
                # Clone repository
                repo_path = self.clone_repository(repo)
                if not repo_path:
                    continue
                
                # Build and extract binaries
                binary_paths = self.build_repository(repo, repo_path)
                
                # Convert to Binary objects
                for binary_path in binary_paths:
                    try:
                        from extraction.disassemble import disassemble_binary

                        logger.info(f"Extracting features from {binary_path.name}")
                        binary = disassemble_binary(
                            binary_path,
                            category=repo.language,
                            compiler=repo.language,   # best proxy until we run `file`
                        )

                        if binary and binary.instruction_count > 10:
                            all_binaries.append(binary)

                    except Exception as e:
                        logger.error(f"Error extracting {binary_path}: {e}")
                
                # Clean up to save space (optional)
                # import shutil
                # shutil.rmtree(repo_path, ignore_errors=True)
        
        logger.info(f"Built corpus with {len(all_binaries)} binaries")
        return all_binaries
    
    def save_corpus(self, binaries: List[Binary], output_path: Path) -> None:
        """Save corpus to file with metadata."""
        from extraction.disassemble import save_corpus_data
        logger.info(f"Saving corpus to {output_path}")
        save_corpus_data(binaries, output_path)
        logger.info("Corpus saved successfully")

def run_github_corpus_builder(args) -> int:
    """Entry point for GitHub corpus builder command."""
    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        builder = GitHubCorpusBuilder(
            work_dir=output_dir / "github_repos",
            github_token=args.github_token
        )
        
        # Default language set
        languages = args.languages or ['C', 'C++', 'Rust', 'Go', 'Zig']
        
        # Build corpus
        binaries = builder.build_corpus(
            languages=languages,
            repos_per_language=args.repos_per_language,
            min_stars=args.min_stars
        )
        
        if not binaries:
            logger.error("No binaries extracted from repositories")
            return 1
        
        # Save corpus
        builder.save_corpus(binaries, output_dir)
        
        logger.info(f"GitHub corpus building completed: {len(binaries)} binaries")
        return 0
        
    except Exception as e:
        logger.error(f"GitHub corpus building failed: {e}")
        return 1

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build binary corpus from GitHub repositories")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--languages", nargs='+', help="Programming languages to include")
    parser.add_argument("--repos-per-language", type=int, default=10, help="Repositories per language")
    parser.add_argument("--min-stars", type=int, default=100, help="Minimum repository stars")
    parser.add_argument("--github-token", help="GitHub API token")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    exit(run_github_corpus_builder(args))