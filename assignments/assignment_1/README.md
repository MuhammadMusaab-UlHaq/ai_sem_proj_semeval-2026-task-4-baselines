# Assignment 1: Problem and Data Understanding

## Overview
This assignment focuses on conducting an in-depth analysis of the SemEval 2026 Task 4 development dataset to understand its core challenges, define the problem space, and inform the strategy for subsequent assignments.

## Team Members & Contributions

### Muhammad Musaab ul Haq
- **Focus**: TF-IDF + Cosine Similarity baseline implementation and error analysis
- **Files**: 
  - `scripts/musaab_baseline.ipynb` - Baseline implementation
  - `reports/musaab_error_analysis.md` - Detailed failure case analysis
  - `reports/musaab_error_text.md` - Error text documentation

### Ahmed Hassan Raza
- **Focus**: N-gram analysis and vocabulary statistics
- **Files**:
  - `scripts/ahmed_ngram_analysis.py` - N-gram frequency analysis
  - `reports/ahmed_report.md` - Assignment report
  - `plots/ahmed_*_frequency.png` - Visualization of unigrams, bigrams, trigrams
  - `plots/ahmed_vocabulary_report.md` - Vocabulary analysis

### Abdul Mueed
- **Focus**: Statistical analysis of dataset characteristics
- **Files**:
  - `scripts/abdul_mueed_stats.py` - Statistical analysis implementation
  - `scripts/abdul_mueed_get_nlkt.py` - NLTK-based text processing
  - `plots/abdul_mueed_ab_label_balance.png` - Label distribution
  - `plots/abdul_mueed_story_length_*.png` - Story length distributions

### Usman Amjad
- **Focus**: Comparative analysis of human vs synthetic data
- **Files**:
  - `scripts/usman_compare.py` - Comparison script
  - `reports/usman_README.md` - Documentation
  - `reports/usman_amjad/` - Detailed CSV reports and plots

## Key Findings

The analysis revealed three main challenges in the dataset:

1. **High-Signal Lexical Traps**: Stories with high keyword overlap but different narratives
2. **Abstract Thematic Cores**: Similarity based on conceptual themes rather than surface words
3. **Narrative Structure**: Story outcomes and plot arcs are crucial for determining similarity

## Running the Code

Each team member's scripts can be run independently. Ensure you have the required dependencies installed:

```bash
pip install -r ../../requirements.txt
```

Then run individual scripts from the project root:

```bash
python assignments/assignment_1/scripts/ahmed_ngram_analysis.py
python assignments/assignment_1/scripts/abdul_mueed_stats.py
python assignments/assignment_1/scripts/usman_compare.py
```

For the Jupyter notebook:
```bash
jupyter notebook assignments/assignment_1/scripts/musaab_baseline.ipynb
```

## Reports

All detailed reports and analysis are available in the `reports/` subdirectory.
