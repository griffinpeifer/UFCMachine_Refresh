"""Train the UFC prediction models on point-in-time features.

Reads training_pit.csv (written by build_from_greco.py), where every fight's
features use only what was known *before* that fight. Replaces the training in
UFC_Machine_Learning_Model.ipynb for the monthly refresh. The notebook still
works for experimenting, but it computes features from career totals that
include the fight being predicted, so its accuracy numbers are inflated.

What it does:
  1. Honest evaluation: train on everything before the last 12 months, test on
     the last 12 months of fights (the model never sees them or anything after).
  2. Retrains every architecture on all fights and saves them under the same
     filenames as before, plus ufc_model_best.h5 (lowest test log-loss), which
     ufc_predictor.py loads first.
  3. Writes model_metrics.txt with the evaluation.

Usage: python scripts/train_model.py [--test-months 12]
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from tensorflow.keras import regularizers  # noqa: E402
from tensorflow.keras.callbacks import EarlyStopping  # noqa: E402
from tensorflow.keras.layers import Dense, Dropout  # noqa: E402
from tensorflow.keras.models import Sequential  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from build_from_greco import FEATURES  # noqa: E402

L2 = regularizers.L2(0.01)

# Same architectures as UFC_Machine_Learning_Model.ipynb
ARCHS = {
    "ufc_model_logistic.h5": ("Logistic regression (1 layer)",
        lambda n: [Dense(1, input_dim=n, activation="sigmoid", kernel_regularizer=L2)]),
    "ufc_prediction_model.h5": ("Original (9 layers)",
        lambda n: [Dense(16, input_dim=n, activation="relu", kernel_regularizer=L2)]
                  + [Dense(32, activation="relu", kernel_regularizer=L2) for _ in range(6)]
                  + [Dense(16, activation="relu", kernel_regularizer=L2), Dense(1, activation="sigmoid")]),
    "ufc_model_exp1_simple.h5": ("Exp 1: Simple (3 layers)",
        lambda n: [Dense(64, input_dim=n, activation="relu", kernel_regularizer=L2),
                   Dense(32, activation="relu", kernel_regularizer=L2),
                   Dense(16, activation="relu", kernel_regularizer=L2),
                   Dense(1, activation="sigmoid")]),
    "ufc_model_exp2_dropout.h5": ("Exp 2: Dropout",
        lambda n: [Dense(64, input_dim=n, activation="relu", kernel_regularizer=L2), Dropout(0.3),
                   Dense(32, activation="relu", kernel_regularizer=L2), Dropout(0.3),
                   Dense(16, activation="relu", kernel_regularizer=L2),
                   Dense(1, activation="sigmoid")]),
    "ufc_model_exp3_wide.h5": ("Exp 3: Wide",
        lambda n: [Dense(128, input_dim=n, activation="relu", kernel_regularizer=L2),
                   Dense(64, activation="relu", kernel_regularizer=L2),
                   Dense(32, activation="relu", kernel_regularizer=L2),
                   Dense(1, activation="sigmoid")]),
}


def make(fn, n):
    m = Sequential(fn(n))
    m.compile(loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"])
    return m


def fit(model, X, y, Xv=None, yv=None, epochs=200):
    cb = []
    if Xv is not None:
        cb = [EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)]
    h = model.fit(X, y, validation_data=(Xv, yv) if Xv is not None else None,
                  epochs=epochs, batch_size=64, verbose=0, callbacks=cb)
    best = int(np.argmin(h.history["val_loss"])) + 1 if Xv is not None else epochs
    return best


def logloss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_temperature(p, y):
    """Scale factor t for logits: p_cal = sigmoid(t * logit(p)). t < 1 softens
    overconfident predictions. No intercept, so P(A beats B) = 1 - P(B beats A)."""
    z = _logit(p)
    grid = np.linspace(0.05, 2.0, 196)
    losses = [logloss(y, 1 / (1 + np.exp(-t * z))) for t in grid]
    return float(grid[int(np.argmin(losses))])


def calibrate(p, t):
    return 1 / (1 + np.exp(-t * _logit(p)))


def symmetric_predict(model, X):
    """Average P(red wins) with 1 - P(blue wins) by flipping the corners.
    Removes any red/blue ordering bias; features are differences, so flipping = negating."""
    return (model.predict(X, verbose=0).ravel() + 1 - model.predict(-X, verbose=0).ravel()) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(PROJECT, "training_pit.csv"))
    ap.add_argument("--out-dir", default=PROJECT)
    ap.add_argument("--test-months", type=int, default=12)
    ap.add_argument("--only", default=None, help="Comma-separated model filenames to train (default: all)")
    ap.add_argument("--eval-only", action="store_true", help="Evaluate only; don't save models")
    a = ap.parse_args()
    archs = {k: v for k, v in ARCHS.items() if not a.only or k in a.only.split(",")}

    tf.random.set_seed(42)
    np.random.seed(42)

    df = pd.read_csv(a.data, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    X_all = df[FEATURES].fillna(0).to_numpy(dtype=float)
    y_all = df["result"].to_numpy()

    cutoff = df["date"].max() - pd.DateOffset(months=a.test_months)
    tr, te = df["date"] <= cutoff, df["date"] > cutoff
    # Validation for early stopping = most recent 15% of the training period
    tr_idx = np.where(tr)[0]
    split = tr_idx[int(len(tr_idx) * 0.85)]
    fit_idx, val_idx = tr_idx[tr_idx < split], tr_idx[tr_idx >= split]

    sc = StandardScaler(with_mean=False).fit(X_all[fit_idx])  # no centering: keeps flip(x) == -x exact
    Xf, Xv, Xt = sc.transform(X_all[fit_idx]), sc.transform(X_all[val_idx]), sc.transform(X_all[te])
    yf, yv, yt = y_all[fit_idx], y_all[val_idx], y_all[te]

    lines = [f"Point-in-time evaluation — trained on fights through {cutoff.date()}, "
             f"tested on {te.sum()} fights after it (through {df['date'].max().date()})", ""]

    # Baselines on the same test fights
    wp = df.loc[te, "Win Percentage"].to_numpy()
    tfights = df.loc[te, "Total Fights"].to_numpy()
    base_wp = np.where(wp != 0, (wp > 0).astype(int), (tfights > 0).astype(int))
    lines.append(f"{'Baseline: pick better UFC win % (ties: more fights)':<52} accuracy {np.mean(base_wp == yt):.1%}")

    results = {}
    lines.append("(log-loss: lower is better; 0.693 = no better than a coin flip)")
    for fname, (label, fn) in archs.items():
        tf.random.set_seed(42)
        m = make(fn, Xf.shape[1])
        best_ep = fit(m, Xf, yf, Xv, yv)
        # Calibrate on the validation fights: the raw nets are overconfident.
        temp = fit_temperature(symmetric_predict(m, Xv), yv)
        p = calibrate(symmetric_predict(m, Xt), temp)
        acc = float(np.mean((p > 0.5) == yt))
        ll = logloss(yt, p)
        conf = np.maximum(p, 1 - p)
        hi = conf >= 0.65
        hi_acc = float(np.mean((p[hi] > 0.5) == yt[hi])) if hi.any() else float("nan")
        results[fname] = dict(label=label, acc=acc, ll=ll, epochs=best_ep, temp=temp)
        lines.append(f"{label:<52} accuracy {acc:.1%}  log-loss {ll:.3f}  "
                     f"picks >=65%: {hi.sum()} ({hi_acc:.0%} right)  max conf {conf.max():.0%}")

    best = min(results, key=lambda k: results[k]["ll"])
    lines += ["", f"Best on held-out fights: {results[best]['label']} -> saved as ufc_model_best.h5"]
    if a.eval_only:
        print("\n".join(lines))
        return 0

    # Final models: refit on ALL fights for the number of epochs early stopping chose
    sc_all = StandardScaler(with_mean=False).fit(X_all)
    Xa = sc_all.transform(X_all)
    for fname, (label, fn) in archs.items():
        tf.random.set_seed(42)
        m = make(fn, Xa.shape[1])
        fit(m, Xa, y_all, epochs=max(results[fname]["epochs"], 5))
        m.save(os.path.join(a.out_dir, fname))
        if fname == best:
            m.save(os.path.join(a.out_dir, "ufc_model_best.h5"))
    with open(os.path.join(a.out_dir, "scaler.pkl"), "wb") as fh:
        pickle.dump(sc_all, fh)
    calib_path = os.path.join(a.out_dir, "calibration.json")
    calib = json.load(open(calib_path)) if os.path.exists(calib_path) else {}
    calib.update({k: v["temp"] for k, v in results.items()})
    calib["ufc_model_best.h5"] = results[best]["temp"]
    with open(calib_path, "w") as fh:
        json.dump(calib, fh, indent=2)
    lines.append(f"Final models retrained on all {len(df)} fights; scaler.pkl and calibration.json updated.")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(a.out_dir, "model_metrics.txt"), "w") as fh:
        fh.write(report + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
