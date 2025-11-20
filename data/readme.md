# Data Directory

This directory contains all datasets used in the SemEval 2026 Task 4 project.

## Structure

```
data/
├── raw/                # Original, unmodified datasets
│   ├── dev_track_a.jsonl          # Development set Track A (200 items)
│   ├── dev_track_b.jsonl          # Development set Track B
│   ├── sample_track_a.jsonl       # Sample data Track A (39 items)
│   └── sample_track_b.jsonl       # Sample data Track B
│
└── processed/          # Synthetic, augmented, or processed datasets
    ├── synthetic_data_for_classification.jsonl        # Synthetic data for     classification
    ├── synthetic_data_for_contrastive_learning.jsonl  # Synthetic data for contrastive learning
    ├── augmented_synthetic_500.jsonl                  # Augmented synthetic (500 items)
    └── combined_synthetic_for_training.jsonl          # Combined synthetic (~1900 items)
```

## Data Files

### Raw Data (`raw/`)

Original datasets from the competition. **Do not modify these files.**

- **`dev_track_a.jsonl`**: Development set for Track A (Story Similarity) - 200 items
- **`dev_track_b.jsonl`**: Development set for Track B (Story Retrieval)
- **`sample_track_a.jsonl`**: Sample subset - 39 items
- **`sample_track_b.jsonl`**: Sample subset for Track B

### Processed Data (`processed/`)

Synthetic and augmented datasets created using LLMs (~1900 items total).

- **`synthetic_data_for_classification.jsonl`**: Format optimized for classification-based learning
- **`synthetic_data_for_contrastive_learning.jsonl`**: Format optimized for contrastive learning approaches
- **`augmented_synthetic_500.jsonl`**: Augmented version with 500 samples
- **`combined_synthetic_for_training.jsonl`**: Combined synthetic dataset for model training

## Usage

Reference data using relative paths from project root:

```python
import json

# Load raw development data
with open('data/raw/dev_track_a.jsonl', 'r') as f:
    dev_data = [json.loads(line) for line in f]

# Load synthetic data
with open('data/processed/synthetic_data_for_classification.jsonl', 'r') as f:
    synthetic_data = [json.loads(line) for line in f]
```

## Adding New Data

- **Raw datasets** → `raw/` directory
- **Processed datasets** → `processed/` directory
- Update this README with file descriptions
- Add large files to `.gitignore` if needed