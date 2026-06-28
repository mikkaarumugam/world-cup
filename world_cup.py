"""Simulate the 2026 World Cup with our Poisson model (Monte Carlo title odds).

We can't directly compute who wins a 48-team tournament, so we simulate it
thousands of times and count how often each team wins. Real group results are
used where they exist; everything unplayed is sampled from the model.

    ./.venv/bin/python world_cup.py
"""

import os
import time
import urllib.request

import numpy as np
import pandas as pd

from predictor import load_matches, train_model, expected_goals, match_probabilities

N_SIMS = 3000
RNG = np.random.default_rng(0)   # fixed seed -> reproducible results

# results.csv records a penalty shootout as a draw; this companion file says who
# actually advanced, so we can lock in the real winner of a knockout shootout.
SHOOTOUTS_URL = ("https://raw.githubusercontent.com/martj42/"
                 "international_results/master/shootouts.csv")


def shootout_winners(path="data/shootouts.csv", max_age_hours=12):
    """Real winner of each penalty shootout: {(iso_date, frozenset(teams)): winner}."""
    stale = (not os.path.exists(path)
             or time.time() - os.path.getmtime(path) > max_age_hours * 3600)
    if stale:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve(SHOOTOUTS_URL, path)
    s = pd.read_csv(path)
    return {(str(pd.to_datetime(d).date()), frozenset((h, a))): w
            for d, h, a, w in zip(s["date"], s["home_team"],
                                  s["away_team"], s["winner"])}


def build():
    """Train the model and prepare the 2026 World Cup structure from the data."""
    model = train_model(load_matches())

    raw = pd.read_csv("data/results.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    wc = raw[(raw["date"].dt.year == 2026)
             & (raw["tournament"] == "FIFA World Cup")].sort_values("date")
    teams = sorted(set(wc["home_team"]) | set(wc["away_team"]))

    # Recover the 12 groups. A team's first 3 distinct opponents are its group
    # rivals: every team plays its 3 group games before any knockout match, so
    # taking opponents in date order cleanly separates the group stage from the
    # knockout rounds (the results feed has no round/stage column to rely on).
    first_opp = {}
    for _, r in wc.iterrows():
        for x, y in ((r["home_team"], r["away_team"]),
                     (r["away_team"], r["home_team"])):
            rivals = first_opp.setdefault(x, [])
            if y not in rivals and len(rivals) < 3:
                rivals.append(y)
    seen, groups = set(), []
    for t in teams:
        if t in seen:
            continue
        g = sorted({t} | set(first_opp[t]))
        seen |= set(g)
        groups.append(g)
    group_of = {t: i for i, g in enumerate(groups) for t in g}

    # Split the fixtures by whether the two teams share a group. Same-group rows
    # are group games (keep real results, precompute expected goals for unplayed);
    # cross-group rows are knockout matches we replay in simulate_once.
    shootouts = shootout_winners()
    fixtures, knockout = [], []
    for _, r in wc.iterrows():
        home, away = r["home_team"], r["away_team"]
        if group_of[home] != group_of[away]:
            if pd.notna(r["home_score"]):
                hs, as_ = int(r["home_score"]), int(r["away_score"])
                if hs > as_:
                    winner = home
                elif as_ > hs:
                    winner = away
                else:   # drawn after extra time -> penalty shootout decided it
                    winner = shootouts.get((str(r["date"].date()),
                                            frozenset((home, away))))
                knockout.append((home, away, True, winner))   # None -> 50/50 fallback
            else:
                knockout.append((home, away, False, None))
        elif pd.notna(r["home_score"]):
            fixtures.append((home, away, True,
                             int(r["home_score"]), int(r["away_score"]), None, None))
        else:
            home_flag = 0 if r["neutral"] else 1
            xgh, xga = expected_goals(model, home, away, home_a=home_flag, home_b=0)
            fixtures.append((home, away, False, None, None, xgh, xga))

    # Tag each knockout match with its bracket round (0=R32, 1=R16, ...). A team's
    # n-th knockout game is round n, so both sides of a real bracket match share a
    # round. (knockout is already date-ordered.) This lets simulate_once pick out
    # exactly one round at a time even after later rounds appear in the data.
    played_ko, tagged = {}, []
    for home, away, played, winner in knockout:
        rnd = max(played_ko.get(home, 0), played_ko.get(away, 0))
        tagged.append((home, away, played, winner, rnd))
        played_ko[home] = played_ko.get(home, 0) + 1
        played_ko[away] = played_ko.get(away, 0) + 1
    knockout = tagged

    # Knockout helper: P(a beats b) at a neutral venue (draw -> 50/50 shootout).
    adv = {}
    for i, a in enumerate(teams):
        for b in teams[i + 1:]:
            pa, d, pb = match_probabilities(model, a, b)
            adv[(a, b)] = pa + d / 2
            adv[(b, a)] = pb + d / 2
    return teams, groups, fixtures, adv, knockout


# Knockout rounds, and the level a team must reach to "make" each one.
# Levels: 0=reached knockout (R32), 1=R16, 2=QF, 3=SF, 4=Final, 5=Champion.
ROUND_LEVELS = {"r16": 1, "qf": 2, "sf": 3, "final": 4, "title": 5}


def simulate_once(teams, groups, fixtures, adv, knockout):
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
    qualifiers = winners + runners + best_thirds

    # Knockout: replay the bracket round by round. Where the data has this round's
    # matches (between teams still alive), use the real result, or the real pairing
    # if it hasn't been played yet. Once we reach a round the data hasn't determined,
    # fall back to a random draw of who's left (we don't model the official bracket).
    reached = {t: 0 for t in qualifiers}   # everyone here reached the knockout (R32)
    current, rnd, level = list(qualifiers), 0, 0
    while len(current) > 1:
        level += 1
        cur = set(current)
        rows = [k for k in knockout if k[4] == rnd and k[0] in cur and k[1] in cur]
        if {t for k in rows for t in k[:2]} == cur:   # full real round for this field
            nxt = []
            for home, away, played, winner, _ in rows:
                if played and winner is not None:
                    nxt.append(winner)                                  # real result
                elif played:
                    nxt.append(home if RNG.random() < 0.5 else away)    # shootout
                else:
                    nxt.append(home if RNG.random() < adv[(home, away)] else away)
        else:                                          # undetermined -> random draw
            RNG.shuffle(current)
            nxt = [current[i] if RNG.random() < adv[(current[i], current[i + 1])]
                   else current[i + 1] for i in range(0, len(current), 2)]
        for t in nxt:
            reached[t] = level
        current, rnd = nxt, rnd + 1
    return reached


def simulate_tournament(n_sims=N_SIMS):
    """Return a table of each team's chance to reach R16/QF/SF/Final/Win."""
    teams, groups, fixtures, adv, knockout = build()
    counts = {name: {t: 0 for t in teams} for name in ROUND_LEVELS}
    for _ in range(n_sims):
        reached = simulate_once(teams, groups, fixtures, adv, knockout)
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
