"""Evaluate the predictor with an HONEST evaluation setup.

Two ideas keep us from fooling ourselves:

1) A locked TEST set, opened exactly once at the very end (`--test`).
2) Rolling-origin CROSS-VALIDATION for the optimization metric (`--cv`): score
   across several train/validate splits and average. Optimizing one fixed window
   hundreds of times would overfit it; averaging many windows is far harder to
   game. ALL cross-validation folds stay BEFORE the test period.

Usage:
    ./.venv/bin/python evaluate.py          # single validation window (quick look)
    ./.venv/bin/python evaluate.py --cv      # cross-validated score (optimize THIS)
    ./.venv/bin/python evaluate.py --test     # final locked test score (use ONCE)
"""

import sys

import numpy as np
import pandas as pd

from predictor import load_matches, drop_sparse_teams, train_model, match_probabilities

VAL_START = "2024-07-01"    # single-window validation: [VAL_START, TEST_START)
TEST_START = "2025-10-01"   # locked test window: [TEST_START, end]

# Rolling cross-validation: each cutoff trains on its past, validates the next
# CV_WINDOW_DAYS. All windows stay before TEST_START so the test stays untouched.
CV_CUTOFFS = ["2023-01-01", "2023-07-01", "2024-01-01", "2024-07-01", "2025-01-01"]
CV_WINDOW_DAYS = 182


def actual_outcome(home_score, away_score):
    if home_score > away_score:
        return "home"
    return "draw" if home_score == away_score else "away"


def model_probs(model, m):
    """Outcome probabilities for one match, using its real venue."""
    home_flag = 0 if m["neutral"] else 1
    p_a, draw, p_b = match_probabilities(
        model, m["home_team"], m["away_team"], home_a=home_flag, home_b=0)
    return {"home": p_a, "draw": draw, "away": p_b}


def score(get_probs, matches):
    """Average log-loss, Brier, and accuracy of get_probs over `matches`."""
    eps = 1e-15
    ll, brier, correct = 0.0, 0.0, 0
    for _, m in matches.iterrows():
        probs = get_probs(m)
        actual = actual_outcome(m["home_score"], m["away_score"])
        ll += -np.log(max(probs[actual], eps))
        brier += sum((probs[k] - (1.0 if k == actual else 0.0)) ** 2
                     for k in ("home", "draw", "away"))
        if max(probs, key=probs.get) == actual:
            correct += 1
    n = len(matches)
    return ll / n, brier / n, correct / n


def _fit_and_score(matches, eval_start, eval_end):
    """Train on matches before eval_start, score log-loss on [eval_start, eval_end)."""
    train = drop_sparse_teams(matches[matches["date"] < eval_start])
    window = matches[matches["date"] >= eval_start]
    if eval_end is not None:
        window = window[window["date"] < eval_end]
    known = set(train["home_team"]) | set(train["away_team"])
    window = window[window["home_team"].isin(known) & window["away_team"].isin(known)]
    model = train_model(train)
    ll, brier, acc = score(lambda m: model_probs(model, m), window)
    return ll, brier, acc, len(train), len(window)


def cross_val_logloss():
    """Average log-loss across the rolling CV folds. This is the metric to optimize."""
    matches = load_matches()
    print(f"{'fold (validate)':>22s} {'n':>6s} {'log-loss':>10s}")
    lls = []
    for cut in CV_CUTOFFS:
        start = pd.Timestamp(cut)
        end = start + pd.Timedelta(days=CV_WINDOW_DAYS)
        ll, _, _, _, n = _fit_and_score(matches, start, end)
        lls.append(ll)
        print(f"{start.date()!s:>14s}+{CV_WINDOW_DAYS}d {n:>6d} {ll:>10.4f}")
    mean_ll = float(np.mean(lls))
    print(f"\n>>> CV log-loss (optimize this): {mean_ll:.4f}  (std {np.std(lls):.4f})")
    return mean_ll


def evaluate(eval_start, eval_end, label):
    """Single-window evaluation: model vs baseline on one held-out window."""
    matches = load_matches()
    train = drop_sparse_teams(matches[matches["date"] < eval_start])
    base = {
        "home": (train["home_score"] > train["away_score"]).mean(),
        "draw": (train["home_score"] == train["away_score"]).mean(),
        "away": (train["home_score"] < train["away_score"]).mean(),
    }
    ll, brier, acc, n_train, n_eval = _fit_and_score(matches, eval_start, eval_end)

    window = matches[matches["date"] >= eval_start]
    if eval_end is not None:
        window = window[window["date"] < eval_end]
    known = set(train["home_team"]) | set(train["away_team"])
    window = window[window["home_team"].isin(known) & window["away_team"].isin(known)]
    base_ll, base_brier, base_acc = score(lambda m: base, window)

    print(f"\n=== {label} ===")
    print(f"Train matches: {n_train}; {label} matches: {n_eval}")
    print(f"{'':10s} {'log-loss':>10s} {'Brier':>10s} {'accuracy':>10s}")
    print(f"{'Baseline':10s} {base_ll:10.4f} {base_brier:10.4f} {base_acc:10.1%}")
    print(f"{'Model':10s} {ll:10.4f} {brier:10.4f} {acc:10.1%}")
    return ll


if __name__ == "__main__":
    if "--test" in sys.argv:
        evaluate(TEST_START, None, "TEST (LOCKED - use once)")
    elif "--cv" in sys.argv:
        cross_val_logloss()
    else:
        evaluate(VAL_START, TEST_START, "VALIDATION")
