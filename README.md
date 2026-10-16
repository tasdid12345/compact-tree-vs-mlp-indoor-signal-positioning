# LightGBM vs MLP on UCI Wireless Indoor Localization

Question: can a small, cross-validation-tuned LightGBM classifier match or beat a standard MLP at indoor location classification from signal-strength readings, while being cheaper per prediction?

- Dataset: UCI Wireless Indoor Localization (id 422), 2000 rows, 7 WiFi signal-strength features, 4 classes. It is downloaded directly from the UCI static file, with `ucimlrepo` as a fallback. The `ucimlrepo` import of id 422 failed in CI, which is why the direct download was added.
- Split: one stratified 80/20 split, seed 42. The scaler is fit on the training set only.
- Baseline: scikit-learn `MLPClassifier` (100 hidden units, max_iter=500).
- Improved: small LightGBM, tuned with 5-fold CV on the training set only.
- Final models are run with seeds 42, 43 and 44, and the means are reported. The test set is used once.
- Output: `results.json` and `figures/comparison.png`.

Run: `pip install -r requirements.txt && python analysis.py`

## Reference

This project is motivated by the following paper:

> B Alsinglawi, K Rabie (2023). *EDEEPRFID-IPS: Enhanced RFID indoor positioning with deep learning for internet of things*. International Conference on Advanced Information. https://link.springer.com/chapter/10.1007/978-3-031-28451-9_13

Paper author: Prof. Belal Alsinglawi, Western Sydney University (b.alsinglawi@westernsydney.edu.au)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
