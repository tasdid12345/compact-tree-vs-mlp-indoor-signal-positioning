"""CV-tuned LightGBM / SVM / ExtraTrees ensemble vs a standard MLP on UCI Wireless Indoor Localization (id 422)."""
import io, json, os, sys, time, zipfile
from itertools import combinations
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler
from sklearn.svm import SVC

SEEDS = [42, 43, 44]
np.random.seed(42)

URLS = [
    "https://archive.ics.uci.edu/static/public/422/wireless+indoor+localization.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00422/wifi_localization.txt",
]


def _parse(content):
    """Parse raw bytes (zip archive or plain text) of wifi_localization into a DataFrame."""
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".txt")]
            if not names:
                raise ValueError("no txt file in archive")
            content = z.read(names[0])
    df = pd.read_csv(io.BytesIO(content), sep=r"\s+", header=None)
    if df.shape[1] != 8:
        raise ValueError(f"unexpected shape {df.shape}")
    return df


def load():
    """Fetch the dataset (direct UCI files, then ucimlrepo) with retry and backoff."""
    for i in range(3):
        for url in URLS:
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                df = _parse(r.content)
                return df.iloc[:, :7].to_numpy(float), df.iloc[:, 7].to_numpy()
            except Exception as e:
                print(f"download failed for {url} ({e}); attempt {i + 1}/3", flush=True)
        try:
            from ucimlrepo import fetch_ucirepo
            d = fetch_ucirepo(id=422)
            return d.data.features.to_numpy(float), d.data.targets.iloc[:, 0].to_numpy()
        except Exception as e:
            print(f"ucimlrepo failed ({e}); attempt {i + 1}/3", flush=True)
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


def add_diffs(X):
    """Feature engineering: raw signals plus all pairwise differences between the 7 signal strengths."""
    X = np.asarray(X)
    d = [X[:, i] - X[:, j] for i, j in combinations(range(X.shape[1]), 2)]
    return np.column_stack([X] + d)


def lgbm_pipe(seed, **params):
    return Pipeline([("f", FunctionTransformer(add_diffs)),
                     ("m", LGBMClassifier(random_state=seed, subsample=0.8, subsample_freq=1,
                                          colsample_bytree=0.8, n_jobs=1, verbose=-1, **params))])


def svc_pipe(seed, **params):
    return Pipeline([("m", SVC(probability=True, random_state=seed, **params))])


def et_pipe(seed):
    return Pipeline([("f", FunctionTransformer(add_diffs)),
                     ("m", ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=1))])


def make_ensemble(seed, lgb_params, svc_params):
    return VotingClassifier([("lgb", lgbm_pipe(seed, **lgb_params)),
                             ("svc", svc_pipe(seed, **svc_params)),
                             ("et", et_pipe(seed))], voting="soft")


X, y_raw = load()
y = LabelEncoder().fit_transform(y_raw)
print(f"loaded {X.shape}, classes={np.unique(y).size}", flush=True)
if len(X) < 100 or np.isnan(X).any():
    sys.exit("dataset not usable")

# single stratified split; scaler fit on train only, shared by both models
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

cv = StratifiedKFold(5, shuffle=True, random_state=42)

# --- tune components with 5-fold CV on the training set only ---
gs_l = GridSearchCV(lgbm_pipe(42), {"m__n_estimators": [50, 100, 200], "m__num_leaves": [4, 8, 16],
                                    "m__learning_rate": [0.05, 0.1]},
                    scoring="accuracy", cv=cv, n_jobs=1).fit(Xtr, ytr)
lgb_best = {k[3:]: v for k, v in gs_l.best_params_.items()}
print(f"best LightGBM params {lgb_best}, cv acc {gs_l.best_score_:.4f}", flush=True)

gs_s = GridSearchCV(svc_pipe(42), {"m__C": [1, 3, 10, 30, 100], "m__gamma": [0.01, 0.03, 0.1, 0.3, "scale"]},
                    scoring="accuracy", cv=cv, n_jobs=1).fit(Xtr, ytr)
svc_best = {k[3:]: v for k, v in gs_s.best_params_.items()}
print(f"best SVC params {svc_best}, cv acc {gs_s.best_score_:.4f}", flush=True)

ens_cv = float(cross_val_score(make_ensemble(42, lgb_best, svc_best), Xtr, ytr, scoring="accuracy",
                               cv=cv, n_jobs=1).mean())
print(f"ensemble cv acc {ens_cv:.4f}", flush=True)

# --- choose the final improved model by training-set CV accuracy only (ties favour the ensemble) ---
cands = {"ensemble": ens_cv, "svc": float(gs_s.best_score_), "lightgbm": float(gs_l.best_score_)}
choice = max(cands, key=lambda k: (cands[k], k == "ensemble"))
print(f"selected improved model: {choice} (cv acc {cands[choice]:.4f})", flush=True)


def build_improved(seed):
    if choice == "ensemble":
        return make_ensemble(seed, lgb_best, svc_best)
    if choice == "svc":
        return svc_pipe(seed, **svc_best)
    return lgbm_pipe(seed, **lgb_best)


# --- final fits for each seed, single test-set evaluation ---
res = {"base": [], "imp": []}
for s in SEEDS:
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=s).fit(Xtr, ytr)
    imp = build_improved(s).fit(Xtr, ytr)
    res["base"].append(evaluate(mlp, Xte, yte))
    res["imp"].append(evaluate(imp, Xte, yte))
    print(f"seed {s}: MLP acc {res['base'][-1]['acc']:.4f} | improved ({choice}) acc {res['imp'][-1]['acc']:.4f}", flush=True)


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
metrics["lgbm_cv_accuracy"] = float(gs_l.best_score_)
metrics["svc_cv_accuracy"] = float(gs_s.best_score_)
metrics["ensemble_cv_accuracy"] = ens_cv
metrics["selected_model_cv_accuracy"] = float(cands[choice])
if not all(np.isfinite(v) for v in metrics.values()) or metrics["baseline_score"] < 0.3:
    sys.exit("metrics not meaningful")

# --- chart ---
os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
for a, k, t in [(ax[0], "score", "Test accuracy"), (ax[1], "us_per_sample_single", "Single-sample inference (µs)")]:
    a.bar(["MLP", f"Improved ({choice})"], [metrics[f"baseline_{k}"], metrics[f"improved_{k}"]],
          yerr=[metrics[f"baseline_{k}_std"], metrics[f"improved_{k}_std"]])
    a.set_title(t)
plt.tight_layout()
plt.savefig("figures/comparison.png", dpi=120)

out = {
    "question": "Can a cross-validation-tuned model (LightGBM / SVM / ExtraTrees soft-voting ensemble with engineered pairwise-difference features) beat a standard MLP at indoor location classification from signal-strength readings?",
    "method": "One stratified 80/20 split (seed 42), shared standardisation fit on train. Baseline: sklearn MLPClassifier. Improved: LightGBM (on raw + pairwise-difference features) and RBF-SVM tuned by 5-fold CV on the training set only, combined with ExtraTrees in a soft-voting ensemble; the final improved model (ensemble vs. single components) is chosen by training-set CV accuracy. Test set used once; randomized models run with seeds 42, 43, 44 (means reported).",
    "dataset": {"name": "UCI Wireless Indoor Localization", "source_url": "https://archive.ics.uci.edu/dataset/422/wireless+indoor+localization",
                "n_rows": int(len(X)), "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
                "same_as_paper": False, "baseline_is_papers_method": False},
    "metric_name": "test accuracy", "higher_is_better": True,
    "baseline": "scikit-learn MLPClassifier, one hidden layer of 100 ReLU units, Adam, max_iter=500 (stand-in for the paper's deep model).",
    "improved": f"Selected by CV: {choice}. Candidates: soft-voting ensemble of CV-tuned LightGBM + CV-tuned RBF-SVC + ExtraTrees (300 trees), tree models use added pairwise signal-difference features.",
    "metrics": metrics,
    "notes": (f"Stand-in dataset, not the paper's RFID data; the paper's model (EDEEPRFID-IPS) is not reproduced. "
              f"Selected improved model: {choice}; LightGBM params {lgb_best}; SVC params {svc_best}. "
              f"Training-set CV accuracy: {cands}. Hyper-parameters and model choice tuned once with seed 42 using CV on the training set only; "
              "seeds only change final fits. The baseline MLP, split, scaler and evaluation code are unchanged. "
              "Timing is wall-clock on the CI CPU; the ensemble is slower per prediction than a small LightGBM, and "
              "timings are noisy and are not measurements on a real small device. Accuracy differences may be within seed/split noise (single split)."),
}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("wrote results.json", flush=True)
