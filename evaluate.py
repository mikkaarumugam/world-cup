"""Evaluate the predictor with an HONEST three-way time split.

  TRAIN      -> fit the model
  VALIDATION -> the score we optimize during experiments (printed by default)
  TEST       -> locked vault; only checked once at the very end (run with --test)

Optimizing against the TEST set would let us fool ourselves (overfit it). So all
experiments use VALIDATION; TEST is the final, untouched honesty check.

Usage:
    ./.venv/bin/python evaluate.py          # validation score (use during search)
    ./.venv/bin/python evaluate.py --test    # final locked test score (use ONCE)
"""

import sys

import numpy as np
import pandas as pd

from predictor import load_matches, drop_sparse_teams, train_model, match_probabilities

VAL_START = "2024-07-01"    # validation window: [VAL_START, TEST_START)
TEST_START = "2025-10-01"   # test window:       [TEST_START, end]


def actual_outcome(home_score, away_score):
    if home_score > away_score:
        return "home"
    return "draw" if home_score == away_score else "away"


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


def evaluate(eval_start, eval_end, label):
    """Train on matches before eval_start; score on [eval_start, eval_end)."""
    matches = load_matches()
    train = drop_sparse_teams(matches[matches["date"] < eval_start])
    window = matches[matches["date"] >= eval_start]
    if eval_end is not None:
        window = window[window["date"] < eval_end]

    model = train_model(train)

    # Model can only predict teams it saw in training (skip cold-start matches).
    known = set(train["home_team"]) | set(train["away_team"])
    testable = window[window["home_team"].isin(known) & window["away_team"].isin(known)]

    # Baseline: historical base rates from training, same guess for every match.
    base = {
        "home": (train["home_score"] > train["away_score"]).mean(),
        "draw": (train["home_score"] == train["away_score"]).mean(),
        "away": (train["home_score"] < train["away_score"]).mean(),
    }

    def baseline_probs(_m):
        return base

    def model_probs(m):
        home_flag = 0 if m["neutral"] else 1
        p_a, draw, p_b = match_probabilities(
            model, m["home_team"], m["away_team"], home_a=home_flag, home_b=0)
        return {"home": p_a, "draw": draw, "away": p_b}

    base_ll, base_brier, base_acc = score(baseline_probs, testable)
    model_ll, model_brier, model_acc = score(model_probs, testable)

    print(f"\n=== {label} ===")
    print(f"Train matches: {len(train)} (up to {train['date'].max().date()})")
    print(f"{label} matches: {len(testable)} "
          f"({testable['date'].min().date()} -> {testable['date'].max().date()})")
    print(f"{'':10s} {'log-loss':>10s} {'Brier':>10s} {'accuracy':>10s}")
    print(f"{'Baseline':10s} {base_ll:10.4f} {base_brier:10.4f} {base_acc:10.1%}")
    print(f"{'Model':10s} {model_ll:10.4f} {model_brier:10.4f} {model_acc:10.1%}")
    print(f"\n>>> {label} log-loss (optimize this): {model_ll:.4f}")
    return model_ll


if __name__ == "__main__":
    if "--test" in sys.argv:
        evaluate(TEST_START, None, "TEST (LOCKED - use once)")
    else:
        evaluate(VAL_START, TEST_START, "VALIDATION")
