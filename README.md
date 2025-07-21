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
    normalise=False
)
print("Same author" if result == [0] else "Different authors")
```

- **Processing your own data**
```python
from corpus_processing import corpus_processing

corpus_processing(
    main_path="/path/to/project/root/",
    literature_folder_name="your_literature_folder"
```

- **Generating Visualizations**
```python
from plots_charts import dendrogram, histogramm
# Example data
values = [[0.1, 0.2], [0.15, 0.25], [0.3, 0.4]]
names = ["Text A", "Text B", "Text C"]

dendrogram(values, names, title="Text Similarity Dendrogram")
```

## Configuration
The system can be configured through several parameters:

| Parameter          | Options          | Description |
|--------------------|------------------|-------------|
| N-gram size        | 2, 3, 4          | The length of word sequences to analyze |
| Normalization      | True/False       | Whether to lemmatize and normalize text |

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
- Evaluated using accuracy, precision, and recall metrics


## Results

Performance metrics from sample runs (see Configuration_comparison.xlsx):

### Corpus 1 (4 authors, 42 books)

| N | Normalized | Accuracy | Precision | Recall |
|---|------------|----------|-----------|--------|
| 2 | Yes        | 0.686    | 0.779     | 0.781  |
| 2 | No         | 0.698    | 0.784     | 0.781  |
| 3 | Yes        | 0.698    | 0.791     | 0.785  |
| 3 | No         | 0.729    | 0.814     | 0.803  |
| 4 | Yes        | 0.764    | 0.835     | 0.833  |
| 4 | No         | 0.667    | 0.771     | 0.757  |

### Corpus 2 (8 authors, 69 books)

| N | Normalized | Accuracy | Precision | Recall |
|---|------------|----------|-----------|--------|
| 2 | Yes        | 0.785    | 0.857     | 0.881  |
| 2 | No         | 0.783    | 0.856     | 0.879  |
| 3 | Yes        | 0.777    | 0.860     | 0.865  |
| 3 | No         | 0.798    | 0.877     | 0.872  |
| 4 | Yes        | 0.804    | 0.876     | 0.882  |
| 4 | No         | 0.780    | 0.857     | 0.873  |

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
