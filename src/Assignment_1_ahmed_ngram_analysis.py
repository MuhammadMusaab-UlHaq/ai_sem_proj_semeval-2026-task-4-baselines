"""
N-gram Analysis Script for Human-Annotated Data
Author: Ahmed Hassan Raza
Assignment 1: Textual Content Analysis

This script performs comprehensive n-gram analysis on the human-annotated dataset,
including extraction of unigrams, bigrams, and trigrams, vocabulary size calculation,
and visualization of word frequency distributions.
"""

import json
import re
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class NgramAnalyzer:
    """Analyzer for extracting and visualizing n-grams from story texts."""
    
    def __init__(self, data_path: str):
        """
        Initialize the analyzer with path to data.
        
        Args:
            data_path: Path to the JSONL data file
        """
        self.data_path = Path(data_path)
        self.stories = []
        self.all_tokens = []
        self.vocabulary = set()
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self) -> None:
        """Load data from JSONL file and extract all story texts."""
        print(f"Loading data from {self.data_path}...")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                # Extract all three story texts
                self.stories.append(entry['anchor_text'])
                self.stories.append(entry['text_a'])
                self.stories.append(entry['text_b'])
        
        print(f"Loaded {len(self.stories)} story texts from {len(self.stories)//3} entries")
    
    def preprocess_text(self, text: str, remove_stopwords: bool = False) -> List[str]:
        """
        Preprocess text and tokenize into words.
        
        Args:
            text: Input text string
            remove_stopwords: Whether to remove stopwords
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Keep only alphabetic tokens (remove punctuation and numbers)
        tokens = [token for token in tokens if token.isalpha()]
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        return tokens
    
    def tokenize_all_stories(self) -> None:
        """Tokenize all stories and build vocabulary."""
        print("Tokenizing all stories...")
        
        for story in self.stories:
            tokens = self.preprocess_text(story, remove_stopwords=False)
            self.all_tokens.extend(tokens)
            self.vocabulary.update(tokens)
        
        print(f"Total tokens: {len(self.all_tokens):,}")
        print(f"Vocabulary size (unique words): {len(self.vocabulary):,}")
    
    def extract_ngrams(self, n: int, top_k: int = 20, 
                       remove_stopwords: bool = True) -> List[Tuple[Tuple[str, ...], int]]:
        """
        Extract top-k most frequent n-grams.
        
        Args:
            n: N-gram size (1 for unigrams, 2 for bigrams, 3 for trigrams)
            top_k: Number of top n-grams to return
            remove_stopwords: Whether to remove stopwords before extracting n-grams
            
        Returns:
            List of (n-gram, frequency) tuples
        """
        print(f"Extracting top {top_k} {n}-grams...")
        
        # Combine all stories and tokenize
        all_text = " ".join(self.stories)
        tokens = self.preprocess_text(all_text, remove_stopwords=remove_stopwords)
        
        # Generate n-grams
        ngram_list = list(ngrams(tokens, n))
        
        # Count frequencies
        ngram_freq = Counter(ngram_list)
        
        # Get top k
        top_ngrams = ngram_freq.most_common(top_k)
        
        print(f"Found {len(ngram_freq):,} unique {n}-grams")
        
        return top_ngrams
    
    def plot_ngrams(self, ngram_results: List[Tuple[Tuple[str, ...], int]], 
                    n: int, output_dir: Path) -> None:
        """
        Create and save visualization of n-gram frequencies.
        
        Args:
            ngram_results: List of (n-gram, frequency) tuples
            n: N-gram size
            output_dir: Directory to save plots
        """
        # Prepare data for plotting
        ngram_labels = [' '.join(ng) for ng, _ in ngram_results]
        frequencies = [freq for _, freq in ngram_results]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar plot
        y_pos = range(len(ngram_labels))
        plt.barh(y_pos, frequencies, color='steelblue')
        plt.yticks(y_pos, ngram_labels)
        plt.xlabel('Frequency', fontsize=12)
        plt.ylabel(f'{n}-grams', fontsize=12)
        plt.title(f'Top {len(ngram_results)} Most Frequent {n}-grams in Human-Annotated Data', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()  # Highest at top
        
        # Add value labels on bars
        for i, v in enumerate(frequencies):
            plt.text(v + max(frequencies)*0.01, i, str(v), 
                    va='center', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        ngram_name = {1: 'unigrams', 2: 'bigrams', 3: 'trigrams'}[n]
        output_path = output_dir / f'ahmed_{ngram_name}_frequency.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {ngram_name} plot to {output_path}")
        plt.close()
    
    def generate_vocabulary_report(self, output_dir: Path) -> None:
        """
        Generate and save vocabulary statistics report.
        
        Args:
            output_dir: Directory to save report
        """
        # Calculate additional statistics
        token_freq = Counter(self.all_tokens)
        
        # Hapax legomena (words appearing only once)
        hapax = sum(1 for word, count in token_freq.items() if count == 1)
        
        # Type-token ratio (TTR)
        ttr = len(self.vocabulary) / len(self.all_tokens) if self.all_tokens else 0
        
        # Average word length
        avg_word_length = sum(len(word) for word in self.vocabulary) / len(self.vocabulary)
        
        # Prepare report
        report = f"""
# Vocabulary Analysis Report
**Author:** Ahmed Hassan Raza  
**Dataset:** Human-Annotated Data (Track A)  
**Date:** November 15, 2025

## Summary Statistics

- **Total Stories Analyzed:** {len(self.stories):,}
- **Total Tokens (Words):** {len(self.all_tokens):,}
- **Vocabulary Size (Unique Words):** {len(self.vocabulary):,}
- **Hapax Legomena (Words appearing once):** {hapax:,} ({hapax/len(self.vocabulary)*100:.2f}% of vocabulary)
- **Type-Token Ratio (TTR):** {ttr:.4f}
- **Average Word Length:** {avg_word_length:.2f} characters

## Interpretation

### Vocabulary Richness
The vocabulary size of {len(self.vocabulary):,} unique words across {len(self.all_tokens):,} total tokens indicates 
{'a rich and diverse vocabulary' if ttr > 0.1 else 'moderate vocabulary diversity'}. 
The Type-Token Ratio of {ttr:.4f} suggests that {'there is good lexical variety' if ttr > 0.1 else 'many words are repeated frequently'}.

### Rare Words
With {hapax:,} words appearing only once ({hapax/len(self.vocabulary)*100:.2f}% of vocabulary), the dataset contains 
{'a significant number of rare or unique terms' if hapax/len(self.vocabulary) > 0.5 else 'relatively few rare terms'}.
This could impact model generalization.

### Most Common Words (Top 20)
"""
        # Add top 20 most common words
        top_words = token_freq.most_common(20)
        report += "\n| Rank | Word | Frequency |\n|------|------|----------|\n"
        for i, (word, freq) in enumerate(top_words, 1):
            report += f"| {i} | {word} | {freq:,} |\n"
        
        # Save report
        report_path = output_dir / 'ahmed_vocabulary_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Saved vocabulary report to {report_path}")
    
    def run_complete_analysis(self, output_dir: str = "../plots") -> None:
        """
        Run the complete n-gram analysis pipeline.
        
        Args:
            output_dir: Directory to save output files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("Starting Complete N-gram Analysis")
        print("="*60 + "\n")
        
        # Load and tokenize data
        self.load_data()
        self.tokenize_all_stories()
        
        # Extract and plot unigrams (without stopwords for better insights)
        print("\n" + "-"*60)
        unigrams = self.extract_ngrams(n=1, top_k=20, remove_stopwords=True)
        self.plot_ngrams(unigrams, n=1, output_dir=output_path)
        
        # Extract and plot bigrams
        print("\n" + "-"*60)
        bigrams = self.extract_ngrams(n=2, top_k=20, remove_stopwords=True)
        self.plot_ngrams(bigrams, n=2, output_dir=output_path)
        
        # Extract and plot trigrams
        print("\n" + "-"*60)
        trigrams = self.extract_ngrams(n=3, top_k=20, remove_stopwords=True)
        self.plot_ngrams(trigrams, n=3, output_dir=output_path)
        
        # Generate vocabulary report
        print("\n" + "-"*60)
        self.generate_vocabulary_report(output_dir=output_path)
        
        print("\n" + "="*60)
        print("Analysis Complete!")
        print("="*60)
        print(f"\nAll outputs saved to: {output_path.absolute()}")


def main():
    """Main execution function."""
    # Path to human-annotated data
    data_path = "../data/dev_track_a.jsonl"
    
    # Initialize analyzer
    analyzer = NgramAnalyzer(data_path)
    
    # Run complete analysis
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
