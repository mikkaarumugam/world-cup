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

EARLIEST_YEAR = 2018   # ignore matches older than this (staler squads)
MIN_MATCHES = 10       # drop teams with fewer games than this (too little data)


def load_matches(path="data/results.csv", earliest_year=EARLIEST_YEAR,
                 min_matches=MIN_MATCHES):
    """Load results and apply our three filters: played, recent, well-sampled."""
    m = pd.read_csv(path).dropna(subset=["home_score", "away_score"])
    m["date"] = pd.to_datetime(m["date"])
    m = m[m["date"].dt.year >= earliest_year]

    # Keep matches where BOTH teams clear the bar; repeat until stable (fixed point).
    while True:
        counts = pd.concat([m["home_team"], m["away_team"]]).value_counts()
        eligible = counts[counts >= min_matches].index
        filtered = m[m["home_team"].isin(eligible) & m["away_team"].isin(eligible)]
        if len(filtered) == len(m):
            break
        m = filtered
    return m


def to_long(matches):
    """Reshape wide matches -> long: one row per 'team scored N vs opponent'."""
    home = pd.DataFrame({"team": matches["home_team"], "opponent": matches["away_team"],
                         "goals": matches["home_score"], "home": 1})
    away = pd.DataFrame({"team": matches["away_team"], "opponent": matches["home_team"],
                         "goals": matches["away_score"], "home": 0})
    return pd.concat([home, away], ignore_index=True)


def train_model(matches):
    """Train the Poisson regression on the given matches -> returns the model."""
    long = to_long(matches)
    return smf.glm("goals ~ C(team) + C(opponent) + home", data=long,
                   family=sm.families.Poisson()).fit(maxiter=300)


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


def match_probabilities(model, team_a, team_b, home_a=0, home_b=0, max_goals=10):
    """Return (team_a win, draw, team_b win) probabilities. Neutral by default."""
    xg_a, xg_b = expected_goals(model, team_a, team_b, home_a, home_b)
    a_scores = poisson.pmf(np.arange(max_goals + 1), xg_a)
    b_scores = poisson.pmf(np.arange(max_goals + 1), xg_b)
    grid = np.outer(a_scores, b_scores)         # grid[i,j] = P(a=i and b=j)
    a_win = np.tril(grid, -1).sum()             # a scored more
    draw = np.trace(grid)                       # equal score
    b_win = np.triu(grid, 1).sum()              # b scored more
    return a_win, draw, b_win


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
