"""Stage 2: evaluate the predictor by backtesting on held-out 'future' matches.

We split real matches by date: train on the older ones, then test on the newer
ones (whose results we already know) and grade the predictions.
"""

import numpy as np
import pandas as pd

from predictor import load_matches, train_model, match_probabilities

# Train on matches BEFORE this date; test on matches FROM this date onward.
CUTOFF = "2025-01-01"

matches = load_matches()
train = matches[matches["date"] < CUTOFF]
test = matches[matches["date"] >= CUTOFF]

print(f"Cutoff date: {CUTOFF}")
print(f"Train matches (before cutoff): {len(train):5d}  "
      f"({train['date'].min().date()} -> {train['date'].max().date()})")
print(f"Test  matches (from cutoff):   {len(test):5d}  "
      f"({test['date'].min().date()} -> {test['date'].max().date()})")

# --- The baseline: historical base rates from the TRAINING matches ----------
# The dumbest sensible predictor: ignore who's playing, just predict the overall
# frequency of home win / draw / away win. Our model must beat this to be useful.
base_home = (train["home_score"] > train["away_score"]).mean()
base_draw = (train["home_score"] == train["away_score"]).mean()
base_away = (train["home_score"] < train["away_score"]).mean()

print("\nBaseline (historical base rates from training data):")
print(f"  Home win: {base_home:.1%}")
print(f"  Draw:     {base_draw:.1%}")
print(f"  Away win: {base_away:.1%}")

# --- Train the model on ONLY the training matches ---------------------------
print("\nTraining model on the training matches...")
model = train_model(train)

# The model only knows teams it saw in training; skip test matches with a team
# it never met (it can't predict an unknown team — the 'cold start' problem).
known = set(train["home_team"]) | set(train["away_team"])
testable = test[test["home_team"].isin(known) & test["away_team"].isin(known)]
print(f"Testable matches (both teams seen in training): {len(testable)} of {len(test)}")


def actual_outcome(home_score, away_score):
    """What really happened: 'home', 'draw', or 'away'."""
    if home_score > away_score:
        return "home"
    return "draw" if home_score == away_score else "away"


def score_predictions(get_probs):
    """Average log-loss and Brier score over the testable matches.

    get_probs(match_row) must return a dict {'home':p, 'draw':p, 'away':p}.
    """
    eps = 1e-15  # floor so log(0) never happens
    log_loss_total, brier_total = 0.0, 0.0
    for _, m in testable.iterrows():
        probs = get_probs(m)
        actual = actual_outcome(m["home_score"], m["away_score"])
        # Log-loss: penalty = -log(probability we gave the TRUE outcome).
        log_loss_total += -np.log(max(probs[actual], eps))
        # Brier: squared error of each probability vs reality (1 if it happened).
        brier_total += sum((probs[k] - (1.0 if k == actual else 0.0)) ** 2
                           for k in ("home", "draw", "away"))
    n = len(testable)
    return log_loss_total / n, brier_total / n


# Baseline: same three numbers for every match.
def baseline_probs(_match):
    return {"home": base_home, "draw": base_draw, "away": base_away}


# Model: predict each match under its ACTUAL venue (home flag on unless neutral).
def model_probs(match):
    home_flag = 0 if match["neutral"] else 1
    p_home, p_draw, p_away = match_probabilities(
        model, match["home_team"], match["away_team"], home_a=home_flag, home_b=0)
    return {"home": p_home, "draw": p_draw, "away": p_away}


base_ll, base_brier = score_predictions(baseline_probs)
model_ll, model_brier = score_predictions(model_probs)

print("\n=== Results (lower is better) ===")
print(f"{'':10s} {'log-loss':>10s} {'Brier':>10s}")
print(f"{'Baseline':10s} {base_ll:10.4f} {base_brier:10.4f}")
print(f"{'Model':10s} {model_ll:10.4f} {model_brier:10.4f}")
print(f"\nModel beats baseline? "
      f"log-loss: {'YES' if model_ll < base_ll else 'NO'}, "
      f"Brier: {'YES' if model_brier < base_brier else 'NO'}")
