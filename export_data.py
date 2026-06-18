"""Generate web/data.json — everything the static frontend needs, from our model.

The frontend computes match odds client-side from each team's attack/defence
multipliers (+ baseline and Dixon-Coles rho), which reproduces the Python model
exactly. Tournament odds, validation metrics, a calibration curve, and a live
"this tournament so far" tracker are precomputed here.

    ./.venv/bin/python export_data.py
"""

import json
import os
from datetime import date

import numpy as np
import pandas as pd

from predictor import (load_matches, drop_sparse_teams, train_model,
                       match_probabilities, RHO)  # noqa
from world_cup import simulate_tournament
from evaluate import model_probs, actual_outcome, TEST_START
from flags import _ISO2, _SPECIAL  # noqa

OUT_PATH = "web/data.json"

# flagcdn uses lowercase ISO codes; England/Scotland/Wales need GB subdivisions.
ISO_OVERRIDE = {"England": "gb-eng", "Scotland": "gb-sct", "Wales": "gb-wls"}
# A few brand colours (others fall back to the accent green).
COLORS = {
    "Argentina": "#4F9BD6", "Spain": "#C8102E", "Brazil": "#1FA34A",
    "England": "#3A6DE0", "France": "#1F3A93", "Portugal": "#A8112B",
    "Belgium": "#2B2A28", "Colombia": "#F4C300", "Germany": "#3D3D3D",
    "Netherlands": "#EE7203", "Morocco": "#B81E2C", "Croatia": "#C1272D",
    "Switzerland": "#D52B1E", "Uruguay": "#5BBCD6", "Japan": "#B11A2B",
    "Mexico": "#1B6B45", "United States": "#2C2E83",
}


def iso(team):
    if team in ISO_OVERRIDE:
        return ISO_OVERRIDE[team]
    return _ISO2.get(team, "xx").lower()


def team_ratings(model):
    """Pull baseline + per-team attack/defence multipliers from the fitted model."""
    p = model.params
    baseline = float(np.exp(p["Intercept"]))
    attack, defence = {}, {}
    for name, val in p.items():
        if name.startswith("C(team)[T."):
            attack[name[len("C(team)[T."):-1]] = float(np.exp(val))
        elif name.startswith("C(opponent)[T."):
            defence[name[len("C(opponent)[T."):-1]] = float(np.exp(val))
    return baseline, attack, defence


def backtest_performance():
    """Honest metrics + reliability curve on the locked test set."""
    matches = load_matches()
    train = drop_sparse_teams(matches[matches["date"] < TEST_START])
    test = matches[matches["date"] >= TEST_START]
    known = set(train["home_team"]) | set(train["away_team"])
    test = test[test["home_team"].isin(known) & test["away_team"].isin(known)]
    model = train_model(train)

    eps = 1e-15
    ll = brier = 0.0
    correct = 0
    bins = [[] for _ in range(10)]   # reliability: predicted prob -> outcome happened?
    for _, m in test.iterrows():
        probs = model_probs(model, m)
        actual = actual_outcome(m["home_score"], m["away_score"])
        ll += -np.log(max(probs[actual], eps))
        brier += sum((probs[k] - (1.0 if k == actual else 0.0)) ** 2
                     for k in ("home", "draw", "away"))
        if max(probs, key=probs.get) == actual:
            correct += 1
        for k in ("home", "draw", "away"):
            bins[min(int(probs[k] * 10), 9)].append((probs[k], 1.0 if k == actual else 0.0))

    n = len(test)
    calibration, abs_errs = [], []
    for b in bins:
        if b:
            pred = float(np.mean([x[0] for x in b]))
            obs = float(np.mean([x[1] for x in b]))
            calibration.append({"pred": pred, "obs": obs})
            abs_errs.append(abs(pred - obs))
    return {
        "n": n,
        "brier": brier / n,
        "logloss": ll / n,
        "accuracy": correct / n,
        "calib_error": float(np.mean(abs_errs)) if abs_errs else 0.0,
        "calibration": calibration,
    }


def tournament_tracker(model):
    """How many played 2026 World Cup favourites the model called correctly."""
    raw = pd.read_csv("data/results.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    wc = raw[(raw["date"].dt.year == 2026) & (raw["tournament"] == "FIFA World Cup")
             & raw["home_score"].notna()]
    known = set(load_matches()["home_team"]) | set(load_matches()["away_team"])
    called = total = 0
    for _, m in wc.iterrows():
        if m["home_team"] not in known or m["away_team"] not in known:
            continue
        hf = 0 if m["neutral"] else 1
        a, d, b = match_probabilities(model, m["home_team"], m["away_team"], home_a=hf)
        pred = max([("home", a), ("draw", d), ("away", b)], key=lambda x: x[1])[0]
        if pred == actual_outcome(m["home_score"], m["away_score"]):
            called += 1
        total += 1
    return {"called": called, "total": total}


def main():
    matches = load_matches()
    model = train_model(matches)
    baseline, attack, defence = team_ratings(model)

    odds = simulate_tournament()   # index = team, cols r16/qf/sf/final/title
    teams = []
    for name, row in odds.iterrows():
        teams.append({
            "name": name,
            "iso": iso(name),
            "color": COLORS.get(name, "#0B6E4F"),
            "attack": round(attack.get(name, 1.0), 4),
            "defence": round(defence.get(name, 1.0), 4),
            "title": round(float(row["title"]), 4),
            "final": round(float(row["final"]), 4),
            "sf": round(float(row["sf"]), 4),
            "qf": round(float(row["qf"]), 4),
            "r16": round(float(row["r16"]), 4),
        })

    data = {
        "generated": date.today().isoformat(),
        "baseline": round(baseline, 4),
        "rho": RHO,
        "n_matches": int(len(matches)),
        "teams": teams,
        "performance": backtest_performance(),
        "tracker": tournament_tracker(model),
    }

    os.makedirs("web", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {OUT_PATH}: {len(teams)} teams, "
          f"baseline={data['baseline']}, rho={data['rho']}")
    print(f"Performance: {data['performance']['n']} test matches, "
          f"acc={data['performance']['accuracy']:.1%}, "
          f"brier={data['performance']['brier']:.3f}")
    print(f"Tracker: {data['tracker']['called']}/{data['tracker']['total']} favourites called")


if __name__ == "__main__":
    main()
