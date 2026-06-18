"""Simulate the 2026 World Cup with our Poisson model (Monte Carlo title odds).

We can't directly compute who wins a 48-team tournament, so we simulate it
thousands of times and count how often each team wins. Real group results are
used where they exist; everything unplayed is sampled from the model.

    ./.venv/bin/python world_cup.py
"""

import numpy as np
import pandas as pd

from predictor import load_matches, train_model, expected_goals, match_probabilities

N_SIMS = 3000
RNG = np.random.default_rng(0)   # fixed seed -> reproducible results


def build():
    """Train the model and prepare the 2026 World Cup structure from the data."""
    model = train_model(load_matches())

    raw = pd.read_csv("data/results.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    wc = raw[(raw["date"].dt.year == 2026) & (raw["tournament"] == "FIFA World Cup")]
    teams = sorted(set(wc["home_team"]) | set(wc["away_team"]))

    # Recover the 12 groups: each team's group-stage opponents form a group of 4.
    opp = {t: set() for t in teams}
    for _, r in wc.iterrows():
        opp[r["home_team"]].add(r["away_team"])
        opp[r["away_team"]].add(r["home_team"])
    seen, groups = set(), []
    for t in teams:
        if t in seen:
            continue
        g = sorted({t} | opp[t])
        seen |= set(g)
        groups.append(g)

    # Group fixtures: keep real results; precompute expected goals for unplayed.
    fixtures = []
    for _, r in wc.iterrows():
        if pd.notna(r["home_score"]):
            fixtures.append((r["home_team"], r["away_team"], True,
                             int(r["home_score"]), int(r["away_score"]), None, None))
        else:
            home_flag = 0 if r["neutral"] else 1
            xgh, xga = expected_goals(model, r["home_team"], r["away_team"],
                                      home_a=home_flag, home_b=0)
            fixtures.append((r["home_team"], r["away_team"], False, None, None, xgh, xga))

    # Knockout helper: P(a beats b) at a neutral venue (draw -> 50/50 shootout).
    adv = {}
    for i, a in enumerate(teams):
        for b in teams[i + 1:]:
            pa, d, pb = match_probabilities(model, a, b)
            adv[(a, b)] = pa + d / 2
            adv[(b, a)] = pb + d / 2
    return teams, groups, fixtures, adv


# Knockout rounds, and the level a team must reach to "make" each one.
# Levels: 0=reached knockout (R32), 1=R16, 2=QF, 3=SF, 4=Final, 5=Champion.
ROUND_LEVELS = {"r16": 1, "qf": 2, "sf": 3, "final": 4, "title": 5}


def simulate_once(teams, groups, fixtures, adv):
    """Play the whole tournament once; return {team: deepest knockout level}."""
    pts = {t: 0 for t in teams}
    gd = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}

    # Group stage: real result if played, else sample goals from the model.
    for home, away, played, hs, as_, xgh, xga in fixtures:
        if played:
            hg, ag = hs, as_
        else:
            hg, ag = RNG.poisson(xgh), RNG.poisson(xga)
        gf[home] += hg; gf[away] += ag
        gd[home] += hg - ag; gd[away] += ag - hg
        if hg > ag:
            pts[home] += 3
        elif hg < ag:
            pts[away] += 3
        else:
            pts[home] += 1; pts[away] += 1

    # Rank each group by points, then goal difference, then goals (random tiebreak).
    def rank_key(t):
        return (pts[t], gd[t], gf[t], RNG.random())

    winners, runners, thirds = [], [], []
    for g in groups:
        order = sorted(g, key=rank_key, reverse=True)
        winners.append(order[0]); runners.append(order[1]); thirds.append(order[2])

    # Best 8 of the 12 third-placed teams also advance -> 32 teams.
    best_thirds = sorted(thirds, key=rank_key, reverse=True)[:8]
    bracket = winners + runners + best_thirds
    RNG.shuffle(bracket)   # randomized draw (we don't model the official bracket)

    reached = {t: 0 for t in bracket}   # everyone here reached the knockout (R32)
    level = 0
    while len(bracket) > 1:
        level += 1
        nxt = []
        for i in range(0, len(bracket), 2):
            a, b = bracket[i], bracket[i + 1]
            nxt.append(a if RNG.random() < adv[(a, b)] else b)
        for t in nxt:
            reached[t] = level
        bracket = nxt
    return reached


def simulate_tournament(n_sims=N_SIMS):
    """Return a table of each team's chance to reach R16/QF/SF/Final/Win."""
    teams, groups, fixtures, adv = build()
    counts = {name: {t: 0 for t in teams} for name in ROUND_LEVELS}
    for _ in range(n_sims):
        reached = simulate_once(teams, groups, fixtures, adv)
        for team, lvl in reached.items():
            for name, need in ROUND_LEVELS.items():
                if lvl >= need:
                    counts[name][team] += 1
    table = pd.DataFrame({name: pd.Series(c) / n_sims for name, c in counts.items()})
    return table.sort_values("title", ascending=False)


if __name__ == "__main__":
    print(f"Simulating the 2026 World Cup {N_SIMS} times...\n")
    table = simulate_tournament()
    print(f"{'Team':<22}{'R16':>8}{'QF':>8}{'SF':>8}{'Final':>8}{'Win':>8}")
    for team, r in table.head(16).iterrows():
        print(f"{team:<22}{r['r16']:>8.1%}{r['qf']:>8.1%}{r['sf']:>8.1%}"
              f"{r['final']:>8.1%}{r['title']:>8.1%}")
