"""World Cup 2026 match predictor (Poisson model).

The reusable engine: load data, train, and predict. Same logic we built in
Stage 1, now wrapped in functions so we can reuse it (evaluation, CLI, web app).

Run a prediction from the terminal:
    ./.venv/bin/python predictor.py Brazil Argentina
"""

import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import poisson

EARLIEST_YEAR = 2010   # ignore matches older than this. With decay on, older data
                       # is auto-discounted, so more history helps (autoresearch).
MIN_MATCHES = 10       # drop teams with fewer games than this (too little data)
HALF_LIFE_DAYS = 1095  # recency weighting: a match this many days old counts half
                       # as much (~3 years, chosen by autoresearch). None = off.


def drop_sparse_teams(matches, min_matches=MIN_MATCHES):
    """Keep only matches where BOTH teams have >= min_matches games.

    Dropping a match can push another team below the bar, so we repeat until the
    set stops changing (a fixed-point loop). Prevents tiny too-perfect records
    that make the maths run off to infinity (separation).
    """
    m = matches
    while True:
        counts = pd.concat([m["home_team"], m["away_team"]]).value_counts()
        eligible = counts[counts >= min_matches].index
        filtered = m[m["home_team"].isin(eligible) & m["away_team"].isin(eligible)]
        if len(filtered) == len(m):
            return filtered
        m = filtered


def load_matches(path="data/results.csv", earliest_year=EARLIEST_YEAR,
                 min_matches=MIN_MATCHES):
    """Load results and apply our three filters: played, recent, well-sampled."""
    m = pd.read_csv(path).dropna(subset=["home_score", "away_score"])
    m["date"] = pd.to_datetime(m["date"])
    m = m[m["date"].dt.year >= earliest_year]
    return drop_sparse_teams(m, min_matches)


def to_long(matches):
    """Reshape wide matches -> long: one row per 'team scored N vs opponent'."""
    home = pd.DataFrame({"team": matches["home_team"], "opponent": matches["away_team"],
                         "goals": matches["home_score"], "home": 1, "date": matches["date"]})
    away = pd.DataFrame({"team": matches["away_team"], "opponent": matches["home_team"],
                         "goals": matches["away_score"], "home": 0, "date": matches["date"]})
    return pd.concat([home, away], ignore_index=True)


def train_model(matches, half_life_days=HALF_LIFE_DAYS):
    """Train the Poisson regression on the given matches -> returns the model.

    If half_life_days is set, weight recent matches more (recency weighting):
    a match `half_life_days` old counts half as much as today's.
    """
    long = to_long(matches)
    weights = None
    if half_life_days is not None:
        age_days = (long["date"].max() - long["date"]).dt.days
        weights = 0.5 ** (age_days / half_life_days)
    return smf.glm("goals ~ C(team) + C(opponent) + home", data=long,
                   family=sm.families.Poisson(), freq_weights=weights).fit(maxiter=300)


def expected_goals(model, team_a, team_b, home_a=0, home_b=0):
    """Expected goals for each team. home_a/home_b are the 1/0 home flags.

    Defaults (0, 0) = neutral venue. For a real match, set the home team's flag
    to 1 so it gets the learned home-advantage boost.
    """
    rows = pd.DataFrame({"team": [team_a, team_b],
                         "opponent": [team_b, team_a],
                         "home": [home_a, home_b]})
    xg = model.predict(rows)
    return xg.iloc[0], xg.iloc[1]


def scoreline_grid(model, team_a, team_b, home_a=0, home_b=0, max_goals=10):
    """Return (xg_a, xg_b, grid) where grid[i, j] = P(team_a scores i, team_b j)."""
    xg_a, xg_b = expected_goals(model, team_a, team_b, home_a, home_b)
    a_scores = poisson.pmf(np.arange(max_goals + 1), xg_a)
    b_scores = poisson.pmf(np.arange(max_goals + 1), xg_b)
    grid = np.outer(a_scores, b_scores)         # grid[i,j] = P(a=i and b=j)
    return xg_a, xg_b, grid


def match_probabilities(model, team_a, team_b, home_a=0, home_b=0, max_goals=10):
    """Return (team_a win, draw, team_b win) probabilities. Neutral by default."""
    _, _, grid = scoreline_grid(model, team_a, team_b, home_a, home_b, max_goals)
    a_win = np.tril(grid, -1).sum()             # a scored more
    draw = np.trace(grid)                       # equal score
    b_win = np.triu(grid, 1).sum()              # b scored more
    return a_win, draw, b_win


def top_scorelines(model, team_a, team_b, home_a=0, home_b=0, n=5):
    """Return the n most likely exact scorelines as [((a_goals, b_goals), prob), ...]."""
    _, _, grid = scoreline_grid(model, team_a, team_b, home_a, home_b)
    flat_order = np.argsort(grid, axis=None)[::-1][:n]   # indices of largest probs
    results = []
    for idx in flat_order:
        i, j = np.unravel_index(idx, grid.shape)         # back to (a_goals, b_goals)
        results.append(((int(i), int(j)), float(grid[i, j])))
    return results


if __name__ == "__main__":
    # Read two team names from the command line; fall back to a default matchup.
    team_a = sys.argv[1] if len(sys.argv) > 1 else "France"
    team_b = sys.argv[2] if len(sys.argv) > 2 else "England"

    model = train_model(load_matches())
    a_win, draw, b_win = match_probabilities(model, team_a, team_b)

    print(f"\n{team_a} vs {team_b} (neutral venue):")
    print(f"  {team_a} win: {a_win:6.1%}")
    print(f"  Draw:        {draw:6.1%}")
    print(f"  {team_b} win: {b_win:6.1%}")
