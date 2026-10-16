"""LightGBM (CV-tuned, small) vs MLP on UCI Wireless Indoor Localization (id 422)."""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

SEEDS = [42, 43, 44]
np.random.seed(42)


def load():
    """Fetch the dataset with retry and exponential backoff."""
    for i in range(5):
        try:
            d = fetch_ucirepo(id=422)
            return d.data.features.to_numpy(float), d.data.targets.iloc[:, 0].to_numpy()
        except Exception as e:
            print(f"download failed ({e}); retry {i + 1}/5", flush=True)
            time.sleep(2 ** (i + 1))
    sys.exit("could not download dataset")


def us_per_sample(model, X, single):
    """Median-of-repeats inference time in microseconds per sample (same code for both models)."""
    model.predict(X[:5])  # warm-up
    times = []
    if single:  # one sample per call, as on a small device
        Xs = X[:200]
        for _ in range(3):
            t = time.perf_counter()
            for i in range(len(Xs)):
                model.predict(Xs[i:i + 1])
            times.append((time.perf_counter() - t) / len(Xs))
    else:  # whole test set in one call
        for _ in range(5):
            t = time.perf_counter()
            model.predict(X)
            times.append((time.perf_counter() - t) / len(X))
    return float(np.median(times) * 1e6)


def evaluate(model, X, y):
    """Shared evaluation: accuracy, macro-F1, batch and single-sample latency."""
    p = model.predict(X)
    return dict(acc=accuracy_score(y, p), f1=f1_score(y, p, average="macro"),
                us_batch=us_per_sample(model, X, False), us_single=us_per_sample(model, X, True))


X, y_raw = load()
y = LabelEncoder().fit_transform(y_raw)
print(f"loaded {X.shape}, classes={np.unique(y).size}", flush=True)
if len(X) < 100 or np.isnan(X).any():
    sys.exit("dataset not usable")

# single stratified split; scaler fit on train only, shared by both models
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

# --- tune LightGBM with 5-fold CV on the training set only ---
lgb_base = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, learning_rate=0.1,
                n_jobs=1, verbose=-1)
grid = {"n_estimators": [20, 50, 100], "num_leaves": [4, 8, 16], "max_depth": [3, 4]}
gs = GridSearchCV(LGBMClassifier(random_state=42, **lgb_base), grid, scoring="accuracy",
                  cv=StratifiedKFold(5, shuffle=True, random_state=42), n_jobs=1)
gs.fit(Xtr, ytr)
best = gs.best_params_
print(f"best LightGBM params {best}, cv acc {gs.best_score_:.4f}", flush=True)

# --- final fits for each seed, single test-set evaluation ---
res = {"base": [], "imp": []}
for s in SEEDS:
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=s).fit(Xtr, ytr)
    lgb = LGBMClassifier(random_state=s, **lgb_base, **best).fit(Xtr, ytr)
    res["base"].append(evaluate(mlp, Xte, yte))
    res["imp"].append(evaluate(lgb, Xte, yte))
    print(f"seed {s}: MLP acc {res['base'][-1]['acc']:.4f} | LGBM acc {res['imp'][-1]['acc']:.4f}", flush=True)


def agg(key, k):
    v = [r[k] for r in res[key]]
    return float(np.mean(v)), float(np.std(v))


metrics = {}
for key, name in [("base", "baseline"), ("imp", "improved")]:
    for k, mname in [("acc", "score"), ("f1", "f1"), ("us_batch", "us_per_sample_batch"),
                     ("us_single", "us_per_sample_single")]:
        m, sd = agg(key, k)
        metrics[f"{name}_{mname}"] = m
        metrics[f"{name}_{mname}_std"] = sd
metrics["improvement"] = metrics["improved_score"] - metrics["baseline_score"]
metrics["speedup_single_sample"] = metrics["baseline_us_per_sample_single"] / metrics["improved_us_per_sample_single"]
metrics["speedup_batch"] = metrics["baseline_us_per_sample_batch"] / metrics["improved_us_per_sample_batch"]
metrics["lgbm_cv_accuracy"] = float(gs.best_score_)
if not all(np.isfinite(v) for v in metrics.values()) or metrics["baseline_score"] < 0.3:
    sys.exit("metrics not meaningful")

# --- chart ---
os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
for a, k, t in [(ax[0], "score", "Test accuracy"), (ax[1], "us_per_sample_single", "Single-sample inference (µs)")]:
    a.bar(["MLP", "LightGBM"], [metrics[f"baseline_{k}"], metrics[f"improved_{k}"]],
          yerr=[metrics[f"baseline_{k}_std"], metrics[f"improved_{k}_std"]])
    a.set_title(t)
plt.tight_layout()
plt.savefig("figures/comparison.png", dpi=120)

out = {
    "question": "Can a small cross-validation-tuned LightGBM model match or beat a standard MLP at indoor location classification from signal-strength readings, while being cheaper to run per prediction on a small device?",
    "method": "One stratified 80/20 split (seed 42), shared standardisation fit on train. Baseline: sklearn MLPClassifier. Improved: small LightGBM tuned by 5-fold CV on the training set only. Test set used once; randomized models run with seeds 42, 43, 44 (means reported).",
    "dataset": {"name": "UCI Wireless Indoor Localization", "source_url": "https://archive.ics.uci.edu/dataset/422/wireless+indoor+localization",
                "n_rows": int(len(X)), "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
                "same_as_paper": False, "baseline_is_papers_method": False},
    "metric_name": "test accuracy", "higher_is_better": True,
    "baseline": "scikit-learn MLPClassifier, one hidden layer of 100 ReLU units, Adam, max_iter=500 (stand-in for the paper's deep model).",
    "improved": "Small LightGBM classifier (shallow trees, few boosting rounds) with n_estimators, num_leaves and max_depth chosen by 5-fold CV on the training set.",
    "metrics": metrics,
    "notes": (f"Stand-in dataset, not the paper's RFID data; the paper's model (EDEEPRFID-IPS) is not reproduced. "
              f"Selected LightGBM params: {best}. Hyper-parameters tuned once with seed 42; seeds only change final fits. "
              "Timing is wall-clock on the CI CPU: LightGBM uses n_jobs=1, but the MLP's BLAS threads are not restricted, and "
              "timings are noisy and are not measurements on a real small device. Accuracy differences may be within seed/split noise (single split)."),
}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("wrote results.json", flush=True)
