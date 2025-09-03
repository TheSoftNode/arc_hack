# Data Directory

This directory contains the ARC dataset files and processed data.

## Dataset Files

- `arc-agi_training_challenges.json` - Training task challenges
- `arc-agi_training_solutions.json` - Training task solutions
- `arc-agi_evaluation_challenges.json` - Evaluation task challenges
- `arc-agi_evaluation_solutions.json` - Evaluation task solutions
- `arc-agi_test_challenges.json` - Test task challenges (no solutions)
- `sample_submission.json` - Sample submission format

## Processed Data

- `processed/` - Preprocessed and cached data
- `features/` - Extracted features from grids
- `embeddings/` - Grid embeddings and representations

## Usage

The data loader (`src/data/loader.py`) automatically reads from this directory.

## Data Source

Original data from the ARC (Abstract Reasoning Corpus) dataset:

- Paper: https://arxiv.org/abs/1911.01547
- Repository: https://github.com/fchollet/ARC
