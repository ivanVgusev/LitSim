# Literature Similarity Analyzer

![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A machine learning project that measures similarity between literary texts to determine authorship attribution and stylistic analysis.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project analyzes literary texts using statistical methods and machine learning to:
- Determine if two texts were written by the same author
- Measure stylistic similarity between texts
- Cluster texts by author based on linguistic features

The system uses n-gram analysis combined with Jaccard and Tanimoto similarity measures, processed through a decision tree classifier.

## Features

- **Text Processing**:
  - Support for FB2, EPUB, and TXT formats
  - Text normalization (lemmatization, punctuation removal)
  - N-gram generation (2-grams, 3-grams, 4-grams)
  
- **Similarity Measures**:
  - Jaccard similarity for n-grams
  - Jaccard similarity for vocabulary
  - Tanimoto coefficient for weighted n-grams
  
- **Machine Learning**:
  - Decision Tree classifier
  - Performance metrics (accuracy, precision, recall)
  - Pre-computed feature vectors for quick analysis

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ivanVgusev/LitSim.git
   cd LitSim
   
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
3. import nltk 
    ```bash
    nltk.download('punkt')
   
## Usage
- **Basic Text Comparison**
```python
from main import compare_authors

result = compare_authors(
    "path/to/book1.txt",
    "path/to/book2.txt",
    stats_path="path/to/project/root/",
    n=3,
    lemmatise=False
)
print("Same author" if result == [0] else "Different authors")
```

- **Processing your own data**
```python
from corpus_processing import main_processing

main_processing(
    base_path="/path/to/project/root/",
    lit_folder_name="your_literature_folder")
```


## Configuration
The system can be configured through several parameters:

| Parameter     | Options    | Description                             |
|---------------|------------|-----------------------------------------|
| N-gram size   | 2, 3, 4    | The length of word sequences to analyze |
| Lemmatisation | True/False | Whether to lemmatise text               |

Example configurations are provided in the Configuration_comparison.xlsx file.

## Project Structure
```
LitSim/
├── literature/ # Input texts (organized by author)
├── literature_test/ # Test texts for evaluation
├── values/ # Precomputed features (non-normalized)
├── values_lemmatised/ # Precomputed features (normalized)
├── corpus_processing.py # Text processing pipeline
├── main.py # Main comparison interface
├── ml.py # Machine learning functions
├── n_grams.py # N-gram processing
├── readers.py # File format readers
├── statistical_methods.py # Similarity calculations
├── string_cleaner.py # Text cleaning utilities
├── plots_charts.py # Visualization functions
└── progress_monitor.py # Progress tracking
```

## Methodology

### Text Processing:
- Files are read and cleaned (punctuation, numbers removed)
- Optional normalization (words reduced to lemma forms)
- N-grams are extracted (2-4 word sequences)

### Feature Extraction:
- Jaccard similarity for unique n-grams
- Jaccard similarity for vocabulary
- Tanimoto coefficient for weighted n-gram counts

### Machine Learning:
- Decision Tree classifier trained on:
  - Positive samples (same-author comparisons)
  - Negative samples (different-author comparisons)


## Results

Performance metrics from sample runs (mean value in 100 trials):

### Corpus 1 (4 authors, 42 books)

| N | Normalised | Accuracy | Precision | Recall |
|---|------------|----------|-----------|--------|
| 2 | True       | 0.68     | 0.77      | 0.78   |
| 2 | False      | 0.69     | 0.78      | 0.78   |
| 3 | True       | 0.69     | 0.79      | 0.78   |
| 3 | False      | 0.72     | 0.81      | 0.80   |
| 4 | True       | 0.76     | 0.83      | 0.83   |
| 4 | False      | 0.66     | 0.77      | 0.75   |

### Corpus 2 (8 authors, 69 books)

| n-gram | Normalised | Accuracy | Precision | Recall |
|--------|------------|----------|-----------|--------|
| 2      | True       | 0.76     | 0.84      | 0.86   |
| 2      | False      | 0.72     | 0.81      | 0.85   |
| 3      | True       | 0.76     | 0.84      | 0.86   |
| 3      | False      | 0.72     | 0.81      | 0.85   |
| 4      | True       | 0.76     | 0.84      | 0.86   |
| 4      | False      | 0.72     | 0.81      | 0.85   |


## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
