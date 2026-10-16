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

## Results (latest run)

On the UCI Wireless Indoor Localization dataset, a standard scikit-learn MLP reached 98.08% test accuracy versus 98.5% for my tuned LightGBM/SVM/ExtraTrees ensemble, a small gain that is likely within single-split noise and that came with slower per-prediction inference.

| Metric | Value |
|---|---|
| baseline_score | 0.9808 |
| baseline_score_std | 0.001179 |
| baseline_f1 | 0.9809 |
| baseline_f1_std | 0.001185 |
| baseline_us_per_sample_batch | 1.007 |
| baseline_us_per_sample_batch_std | 0.005116 |
| baseline_us_per_sample_single | 227.6 |
| baseline_us_per_sample_single_std | 0.1569 |
| improved_score | 0.985 |
| improved_score_std | 0 |
| improved_f1 | 0.9851 |
| improved_f1_std | 0 |
| improved_us_per_sample_batch | 99.1 |
| improved_us_per_sample_batch_std | 0.9702 |
| improved_us_per_sample_single | 23290 |
| improved_us_per_sample_single_std | 137.9 |
| improvement | 0.004167 |
| speedup_single_sample | 0.009771 |
| speedup_batch | 0.01016 |
| lgbm_cv_accuracy | 0.9856 |
| svc_cv_accuracy | 0.9831 |
| ensemble_cv_accuracy | 0.9856 |
| selected_model_cv_accuracy | 0.9856 |

Run on 2026-10-16. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Stand-in dataset, not the paper's RFID data; the paper's model (EDEEPRFID-IPS) is not reproduced. Selected improved model: ensemble; LightGBM params {'learning_rate': 0.05, 'n_estimators': 100, 'num_leaves': 4}; SVC params {'C': 3, 'gamma': 0.3}. Training-set CV accuracy: {'ensemble': 0.9856250000000001, 'svc': 0.9831249999999999, 'lightgbm': 0.985625}. Hyper-parameters and model choice tuned once with seed 42 using CV on the training set only; seeds only change final fits. The baseline MLP, split, scaler and evaluation code are unchanged. Timing is wall-clock on the CI CPU; the ensemble is slower per prediction than a small LightGBM, and timings are noisy and are not measurements on a real small device. Accuracy differences may be within seed/split noise (single split).

## Reference

This project is motivated by the following paper:

> B Alsinglawi, K Rabie (2023). *EDEEPRFID-IPS: Enhanced RFID indoor positioning with deep learning for internet of things*. International Conference on Advanced Information. https://link.springer.com/chapter/10.1007/978-3-031-28451-9_13

Paper author: Prof. Belal Alsinglawi, Western Sydney University (b.alsinglawi@westernsydney.edu.au)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
