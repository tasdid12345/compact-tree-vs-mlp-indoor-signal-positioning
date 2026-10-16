# Small tuned LightGBM vs. MLP for indoor localization

## Motivation
The paper "EDEEPRFID-IPS: Enhanced RFID indoor positioning with deep learning for internet of things" (2023) proposes a deep learning model for indoor positioning from RFID signals. Deep models can be costly on small IoT devices, so this experiment checks whether a small, cross-validation-tuned gradient-boosted tree model can match a standard neural network while being cheaper per prediction.

## Question
Can a small CV-tuned LightGBM model match or beat a standard MLP at indoor location classification from signal-strength readings, while being cheaper to run per prediction?

## Dataset
UCI Wireless Indoor Localization (ucimlrepo id 422): 7 Wi-Fi signal-strength readings, 4 room labels. This is **not** the paper's dataset (which is not stated); it is used as a stand-in for RFID signal data.

## Baseline vs. improved
- **Baseline:** scikit-learn `MLPClassifier` with default-style settings (standing in for the paper's deep model; not the paper's own method).
- **Improved:** a small LightGBM classifier (shallow trees, few rounds) tuned by 5-fold cross-validation on the training set only.

Both use one stratified 80/20 split (seed 42), the same standardisation, the same evaluation and timing code. Metrics: test accuracy (main), macro-F1, and inference time per sample (µs). Models are run with seeds 42, 43, 44.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Reference

This project is motivated by the following paper:

> B Alsinglawi, K Rabie (2023). *EDEEPRFID-IPS: Enhanced RFID indoor positioning with deep learning for internet of things*. International Conference on Advanced Information. https://link.springer.com/chapter/10.1007/978-3-031-28451-9_13

Paper author: Prof. Belal Alsinglawi, Western Sydney University (b.alsinglawi@westernsydney.edu.au)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
