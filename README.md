# Empirical Analysis of Face Detection and Recognition Algorithms

## Overview
This research project provides a comprehensive framework for evaluating several classical and modern face recognition algorithms. Developed as a proof-of-concept for a doctoral-level thesis, it explores the trade-offs between computational efficiency and recognition accuracy across different mathematical approaches.

## Implemented Algorithms
1.  **LBPH (Local Binary Patterns Histograms)**: Robust against monotonic grayscale transformations and illumination changes.
2.  **EigenFaces (PCA)**: Uses Principal Component Analysis to extract features by projecting face images into a high-dimensional face space.
3.  **FisherFaces (LDA)**: Uses Linear Discriminant Analysis to maximize the ratio of between-class scatter to within-class scatter.
4.  **Deep Learning (Dlib/HOG)**: Utilizes modern 128-d face encodings mapped to a unified feature space.

## Project Structure
- `app.py`: The main research dashboard (GUI).
- `config.py`: Centralized parameter management.
- `base_engine.py`: Shared computer vision utilities.
- `training_logic.py`: Unified interface for model training.
- `recog_logic.py`: Unified interface for real-time recognition.
- `face_data/`: Dataset storage organized by individual.
- `trained_data/`: Serialized model storage (.xml).

## Getting Started

### Prerequisites
- Python 3.12+
- Dependencies: `numpy`, `opencv-contrib-python`, `face-recognition`, `setuptools`.

### Installation
```bash
# Using poetry (recommended)
poetry install
```

### Usage
1.  **Launch the Suite**:
    ```bash
    python app.py
    ```
2.  **Data Collection**: Enter a researcher/subject name and click 'Collect Dataset'. Look into the camera and move slightly to capture varied angles.
3.  **Training**: Once at least two subjects have data (especially for FisherFaces), run the training modules.
4.  **Evaluation**: Run the recognition modules to empirically test the models against live video streams.

## Research Methodology
This project implements a rigorous pipeline:
1.  **Preprocessing**: Grayscale conversion and bilateral filtering.
2.  **Detection**: Haar Cascade Classifiers for initial localization.
3.  **Feature Extraction**: Implementation of algorithm-specific mathematical transformations.
4.  **Classification**: Nearest Neighbor/Euclidean distance matching against the trained manifold.

---
*Author: Mayank*
*Thesis: Empirical Analysis of Subspace-based and Local-feature-based Face Recognition Systems*
