# NLI Dataset Artifact Analysis and Debiasing 🔍

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge)](https://www.python.org/)

## Overview 📊

This repository implements a novel multi-head debiasing architecture for Natural Language Inference (NLI) models, focusing on reducing dataset artifacts while maintaining high performance. Our approach combines:
- Contrastive learning
- Explicit bias prediction
- Temperature-scaled outputs
- Multi-artifact attention mechanism

## Key Features ⭐

- **Multi-Head Architecture**: Simultaneously addresses multiple types of artifacts
- **Artifact-Aware Attention**: Dynamically weighs different artifacts during inference
- **Advanced Loss Integration**: Combines task-specific and debiasing objectives
- **Comprehensive Analysis Framework**: Tools for quantifying and visualizing bias patterns

## Results 📈

<p align="center">
  <img src="research_analysis/figures/training_metrics.png" alt="Training Metrics" width="600"/>
  <br>
  <em>Training and validation metrics showing consistent improvement across epochs</em>
</p>

<p align="center">
  <img src="research_analysis/figures/confusion_matrix.png" alt="Confusion Matrix" width="600"/>
  <br>
  <em>Confusion matrix demonstrating improved prediction accuracy</em>
</p>

## Performance Highlights 🎯

```python
Bias Type Improvements:
- Length:    +3.53%
- Overlap:   +2.68%
- Subset:    +0.78%
- Negation:  +7.48%
