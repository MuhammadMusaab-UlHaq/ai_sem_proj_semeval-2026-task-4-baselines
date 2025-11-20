# Assignment 1: Textual Content Analysis - Ahmed Hassan Raza

## Overview
This document describes the textual content analysis performed on the human-annotated dataset for SemEval-2026 Task 4.

## Tasks Completed

### 1. N-gram Extraction and Analysis
- **Unigrams**: Extracted and visualized the top 20 most frequent single words
- **Bigrams**: Extracted and visualized the top 20 most frequent two-word phrases
- **Trigrams**: Extracted and visualized the top 20 most frequent three-word phrases

### 2. Vocabulary Analysis
- Calculated total vocabulary size: **9,615 unique words**
- Analyzed 600 story texts (200 dataset entries × 3 stories per entry)
- Total tokens processed: **72,423 words**

### 3. Statistical Metrics
- **Type-Token Ratio (TTR)**: 0.1328 - indicates good lexical variety
- **Hapax Legomena**: 4,415 words (45.92% of vocabulary) appear only once
- **Average Word Length**: 7.08 characters

## Key Findings

### Vocabulary Richness
The dataset demonstrates a rich and diverse vocabulary with 9,615 unique words across 72,423 total tokens. The Type-Token Ratio of 0.1328 suggests good lexical variety, meaning the stories use a diverse range of words rather than repeating the same words frequently.

### Rare Words Distribution
Nearly half (45.92%) of the vocabulary consists of words that appear only once in the entire dataset. This high percentage of hapax legomena indicates:
- Stories cover diverse topics and themes
- Potential challenges for machine learning models with rare word generalization
- Need for robust word embedding or subword tokenization strategies

### Common N-grams Insights

**Top Unigrams (after stopword removal):**
The most frequent content words reveal that stories focus on characters and their actions, with words like "one", "man", "woman", "new", "father" appearing frequently.

**Top Bigrams:**
Common two-word phrases include narrative patterns and story elements that appear across multiple stories.

**Top Trigrams:**
Three-word phrases reveal more specific narrative structures and common story progression patterns.

## Files Generated

1. **`src/ahmed_ngram_analysis.py`**: Complete analysis script with:
   - Data loading and preprocessing
   - N-gram extraction for 1, 2, and 3-grams
   - Vocabulary statistics calculation
   - Automated visualization generation

2. **`plots/ahmed_unigrams_frequency.png`**: Visualization of top 20 unigrams
3. **`plots/ahmed_bigrams_frequency.png`**: Visualization of top 20 bigrams  
4. **`plots/ahmed_trigrams_frequency.png`**: Visualization of top 20 trigrams
5. **`plots/ahmed_vocabulary_report.md`**: Detailed vocabulary analysis report

## Implementation Details

### Technologies Used
- **Python 3.9+**
- **NLTK**: For tokenization and n-gram extraction
- **Matplotlib & Seaborn**: For visualization
- **Collections.Counter**: For frequency counting

### Data Processing Pipeline
1. Load JSONL data files
2. Extract all three story texts (anchor, text_a, text_b) from each entry
3. Tokenize text using NLTK's word_tokenize
4. Filter to keep only alphabetic tokens
5. Remove stopwords for n-gram analysis (but not for vocabulary counting)
6. Generate n-grams using NLTK's ngrams function
7. Count frequencies and extract top-k
8. Create horizontal bar chart visualizations

## Running the Analysis

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run the analysis script
python src/ahmed_ngram_analysis.py
```

The script will:
- Download required NLTK data automatically
- Process the human-annotated dataset
- Generate all visualizations in the `plots/` directory
- Create a detailed vocabulary report

## Future Improvements

1. **Comparative Analysis**: Compare n-gram distributions between similar and dissimilar story pairs
2. **Semantic Analysis**: Analyze n-grams by semantic categories (emotions, actions, settings)
3. **Temporal Patterns**: Investigate if certain n-grams appear more in story beginnings vs endings
4. **Cross-validation**: Verify findings against Track B dataset

## Assignment Completion Status

- [x] Extract and plot most frequent unigrams
- [x] Extract and plot most frequent bigrams  
- [x] Extract and plot most frequent trigrams
- [x] Calculate total vocabulary size
- [x] Generate comprehensive analysis report
- [x] Document implementation and findings

---

**Author**: Ahmed Hassan Raza  
**Date**: November 15, 2025  
**Course**: CS-272 Artificial Intelligence  
**Assignment**: 1 - Problem and Data Understanding
