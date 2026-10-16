# Indoor localization: MLP baseline vs. CV-tuned ensemble

Dataset: UCI Wireless Indoor Localization (id 422), 2000 rows, 7 signal-strength features, 4 classes.

- Split: one stratified 80/20 split (seed 42); the scaler is fit on the training set only. Both models use it.
- Baseline (unchanged): scikit-learn `MLPClassifier` with 100 hidden units and `max_iter=500`.
- Improved: a soft-voting ensemble of LightGBM, RBF-SVM and ExtraTrees.
  - The tree models also get pairwise signal-difference features.
  - LightGBM and SVM hyper-parameters are tuned with 5-fold CV on the training set only.
  - The final model (ensemble or a single component) is chosen by training-set CV accuracy.
- The test set is used once, for final evaluation. Randomized models run with seeds 42, 43 and 44, and the means are reported.
- Outputs: `results.json` and `figures/comparison.png`.

Run: `pip install -r requirements.txt && python analysis.py`

Notes: this is a stand-in dataset, not the original paper's data. Timings are noisy CI wall-clock numbers, and the ensemble is slower per prediction than a small LightGBM.

## Reference

This project is motivated by the following paper:

> B Alsinglawi, K Rabie (2023). *EDEEPRFID-IPS: Enhanced RFID indoor positioning with deep learning for internet of things*. International Conference on Advanced Information. https://link.springer.com/chapter/10.1007/978-3-031-28451-9_13

Paper author: Prof. Belal Alsinglawi, Western Sydney University (b.alsinglawi@westernsydney.edu.au)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
