"""Generate web/data.json - everything the static frontend needs, from our model.

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
                       match_probabilities, ensure_fresh_data, RHO)  # noqa
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
    correct = decisive = decisive_correct = top2 = draws = 0
    bins = [[] for _ in range(10)]   # reliability: predicted prob -> outcome happened?
    for _, m in test.iterrows():
        probs = model_probs(model, m)
        actual = actual_outcome(m["home_score"], m["away_score"])
        ll += -np.log(max(probs[actual], eps))
        brier += sum((probs[k] - (1.0 if k == actual else 0.0)) ** 2
                     for k in ("home", "draw", "away"))
        if max(probs, key=probs.get) == actual:
            correct += 1
        if actual == "draw":
            draws += 1
        else:
            decisive += 1
            if max(probs, key=probs.get) == actual:
                decisive_correct += 1
        if actual in sorted(probs, key=probs.get, reverse=True)[:2]:
            top2 += 1
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
        "decisive_acc": decisive_correct / decisive if decisive else 0.0,
        "top2": top2 / n,
        "draw_share": draws / n,
        "calib_error": float(np.mean(abs_errs)) if abs_errs else 0.0,
        "calibration": calibration,
    }


def tournament_tracker(model):
    """How the model is doing on played 2026 World Cup matches so far.

    Top-pick accuracy is harsh in the group stage (draws are near coin-flips), so
    we also report decisive-match accuracy and top-2 coverage - fairer, fuller.
    """
    raw = pd.read_csv("data/results.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    wc = raw[(raw["date"].dt.year == 2026) & (raw["tournament"] == "FIFA World Cup")
             & raw["home_score"].notna()]
    known = set(load_matches()["home_team"]) | set(load_matches()["away_team"])
    total = top = decisive = decisive_called = top2 = draws = 0
    for _, m in wc.iterrows():
        if m["home_team"] not in known or m["away_team"] not in known:
            continue
        hf = 0 if m["neutral"] else 1
        a, d, b = match_probabilities(model, m["home_team"], m["away_team"], home_a=hf)
        probs = {"home": a, "draw": d, "away": b}
        act = actual_outcome(m["home_score"], m["away_score"])
        total += 1
        if max(probs, key=probs.get) == act:
            top += 1
        if act == "draw":
            draws += 1
        else:
            decisive += 1
            if max(probs, key=probs.get) == act:
                decisive_called += 1
        if act in sorted(probs, key=probs.get, reverse=True)[:2]:
            top2 += 1
    return {"total": total, "top": top, "decisive": decisive,
            "decisive_called": decisive_called, "top2": top2, "draws": draws}


def fixtures_and_recent(model, n_up=10, n_recent=6):
    """Predictions for upcoming WC fixtures + how recent calls turned out."""
    raw = pd.read_csv("data/results.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    wc = raw[(raw["date"].dt.year == 2026) & (raw["tournament"] == "FIFA World Cup")]
    known = set(load_matches()["home_team"]) | set(load_matches()["away_team"])

    def ok(m):
        return m["home_team"] in known and m["away_team"] in known

    def fmt(d):
        return f"{d.day} {d.strftime('%b')}"

    upcoming = []
    for _, m in wc[wc["home_score"].isna()].sort_values("date").iterrows():
        if not ok(m) or len(upcoming) >= n_up:
            continue
        a, d, b = match_probabilities(model, m["home_team"], m["away_team"],
                                      home_a=0 if m["neutral"] else 1)
        upcoming.append({"date": fmt(m["date"]), "home": m["home_team"], "away": m["away_team"],
                         "home_iso": iso(m["home_team"]), "away_iso": iso(m["away_team"]),
                         "ph": round(float(a), 3), "pd": round(float(d), 3), "pa": round(float(b), 3)})

    recent = []
    for _, m in wc[wc["home_score"].notna()].sort_values("date").iloc[::-1].iterrows():
        if not ok(m) or len(recent) >= n_recent:
            continue
        a, d, b = match_probabilities(model, m["home_team"], m["away_team"],
                                      home_a=0 if m["neutral"] else 1)
        probs = {"home": a, "draw": d, "away": b}
        correct = max(probs, key=probs.get) == actual_outcome(m["home_score"], m["away_score"])
        recent.append({"date": fmt(m["date"]), "home": m["home_team"], "away": m["away_team"],
                       "home_iso": iso(m["home_team"]), "away_iso": iso(m["away_team"]),
                       "hs": int(m["home_score"]), "as": int(m["away_score"]), "correct": bool(correct)})
    return upcoming, recent


def main():
    ensure_fresh_data(max_age_hours=0)   # always pull the latest results before exporting
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

    upcoming, recent = fixtures_and_recent(model)
    data = {
        "generated": date.today().isoformat(),
        "baseline": round(baseline, 4),
        "rho": RHO,
        "n_matches": int(len(matches)),
        "teams": teams,
        "performance": backtest_performance(),
        "tracker": tournament_tracker(model),
        "upcoming": upcoming,
        "recent": recent,
    }

    os.makedirs("web", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {OUT_PATH}: {len(teams)} teams, "
          f"baseline={data['baseline']}, rho={data['rho']}")
    print(f"Performance: {data['performance']['n']} test matches, "
          f"acc={data['performance']['accuracy']:.1%}, "
          f"brier={data['performance']['brier']:.3f}")
    t = data["tracker"]
    print(f"Tracker: {t['decisive_called']}/{t['decisive']} decisive called, "
          f"{t['top2']}/{t['total']} in top-2")


if __name__ == "__main__":
    main()
