"""World Cup 2026 match predictor — built step by step.

Step 3: load the data and filter it down to clean, recent, played matches.
Step 4: reshape wide -> long (one row per team's goals).
Step 5: train a Poisson regression to learn attack/defence + home advantage.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import poisson

# How far back to look. Earlier = more data but staler; later = fresher but thinner.
# 2018 onward covers two World Cup cycles — a sensible middle ground we can tune later.
EARLIEST_YEAR = 2018

# Drop teams with fewer than this many matches — too little data to estimate a
# strength from, and they make the maths blow up. World Cup nations far exceed it.
MIN_MATCHES = 10

# --- Load -------------------------------------------------------------------
matches = pd.read_csv("data/results.csv")
print("Raw matches loaded:", len(matches))

# --- Filter 1: drop unplayed matches (future fixtures have empty score cells) -
# .dropna() removes rows where the named columns are NaN (missing).
matches = matches.dropna(subset=["home_score", "away_score"])
print("After dropping unplayed matches:", len(matches))

# --- Filter 2: keep only recent years ---------------------------------------
# Turn the text date into a real date so we can compare by year.
matches["date"] = pd.to_datetime(matches["date"])
matches = matches[matches["date"].dt.year >= EARLIEST_YEAR]
print(f"After keeping {EARLIEST_YEAR} onward:", len(matches))

# --- Filter 3: drop teams with too few matches ------------------------------
# Keep only matches where BOTH teams have >= MIN_MATCHES games. Dropping a match
# can push another team below the bar, so we repeat until the set stops changing
# (a "fixed-point" loop). This avoids teams left with too-perfect tiny records,
# which make the maths run off to infinity (separation).
while True:
    team_counts = pd.concat([matches["home_team"], matches["away_team"]]).value_counts()
    eligible = team_counts[team_counts >= MIN_MATCHES].index
    filtered = matches[matches["home_team"].isin(eligible) & matches["away_team"].isin(eligible)]
    if len(filtered) == len(matches):  # nothing dropped this pass -> stable
        break
    matches = filtered
print(f"After dropping teams with < {MIN_MATCHES} matches:", len(matches))

# --- Quick sanity peek ------------------------------------------------------
print("\nDate range now:", matches["date"].min().date(), "to", matches["date"].max().date())
print("Number of distinct teams (home side):", matches["home_team"].nunique())

# --- Step 4: reshape wide -> long ------------------------------------------
# Goal: one row per "this team scored N goals vs that opponent", so the model
# can pool all of a team's goals into one attack number (and goals-against into
# one defence number). We build two views and stack them.

# View from the HOME team's perspective: they are the attacker, and home = 1.
home_view = pd.DataFrame({
    "team":     matches["home_team"],
    "opponent": matches["away_team"],
    "goals":    matches["home_score"],
    "home":     1,   # the scoring team was at home
})

# View from the AWAY team's perspective: they are the attacker, and home = 0.
away_view = pd.DataFrame({
    "team":     matches["away_team"],
    "opponent": matches["home_team"],
    "goals":    matches["away_score"],
    "home":     0,   # the scoring team was away
})

# Stack the two views into one long table (one "card" per row).
long = pd.concat([home_view, away_view], ignore_index=True)

print("\n--- After reshaping to long format ---")
print("Long-format rows (should be ~2x the matches):", len(long))
print("\nFirst 3 rows (home perspective):")
print(long.head(3))
print("\nLast 3 rows (away perspective):")
print(long.tail(3))

# --- Step 5: train the Poisson regression -----------------------------------
# The formula: explain `goals` using the team (its attack), the opponent
# (their defence), and the home flag (home advantage). family=Poisson tells it
# the target is counts, which makes the effects multiply (log space).
print("\nTraining the model (this is the actual machine learning)...")
model = smf.glm(
    formula="goals ~ C(team) + C(opponent) + home",
    data=long,
    family=sm.families.Poisson(),
).fit(maxiter=300)  # give the guess->adjust loop room to fully settle

# Did it work? Confirm it converged and count what it learned.
print("Converged:", model.converged)
print("Parameters (numbers) learned:", len(model.params))

# A clean, robust sanity check: how big is the home-advantage boost it learned?
# The coefficient lives in log space, so exp() turns it into a goal multiplier.
home_coef = model.params["home"]
print(f"\nHome-advantage coefficient (log space): {home_coef:.3f}")
print(f"Home-advantage as a goal multiplier:   x{np.exp(home_coef):.3f}")
print("(i.e. playing at home multiplies a team's expected goals by this much)")

# --- Step 6a: read out the learned team strengths ---------------------------
# Coefficients are named like "C(team)[T.Brazil]" (attack) and
# "C(opponent)[T.Brazil]" (defence). exp() turns each from log space into a
# multiplier relative to the reference team. Attack: higher = scores more.
# Defence: it's goals the OPPONENT scores, so lower = concedes fewer = better.
attack, defence = {}, {}
for name, value in model.params.items():
    if name.startswith("C(team)[T."):
        attack[name[len("C(team)[T."):-1]] = np.exp(value)
    elif name.startswith("C(opponent)[T."):
        defence[name[len("C(opponent)[T."):-1]] = np.exp(value)

attack = pd.Series(attack).sort_values(ascending=False)
defence = pd.Series(defence).sort_values()  # lowest = best defence first

print("\n=== Strongest attacks (top 10) ===")
print(attack.head(10).round(2))
print("\n=== Best defences (top 10 — lower = fewer goals conceded) ===")
print(defence.head(10).round(2))

# --- Step 6b: predict expected goals for a matchup --------------------------
def expected_goals(team_a, team_b):
    """Expected goals for each team in a NEUTRAL-venue match (home=0 for both).

    We build a 2-row 'flashcard' and let the trained model do the
    baseline x attack x defence multiplication for us.
    """
    rows = pd.DataFrame({
        "team":     [team_a, team_b],
        "opponent": [team_b, team_a],
        "home":     [0, 0],   # neutral venue: no home boost for either side
    })
    xg = model.predict(rows)
    return xg.iloc[0], xg.iloc[1]

team_a, team_b = "France", "England"   # <- change these to try other matchups
xg_a, xg_b = expected_goals(team_a, team_b)
print(f"\n=== Expected goals (neutral venue): {team_a} vs {team_b} ===")
print(f"  {team_a}: {xg_a:.2f}")
print(f"  {team_b}: {xg_b:.2f}")

# --- Step 7: turn expected goals into win/draw/loss probabilities -----------
def match_probabilities(team_a, team_b, max_goals=10):
    """Return (team_a win, draw, team_b win) probabilities for a neutral match."""
    xg_a, xg_b = expected_goals(team_a, team_b)

    # pmf = chance of scoring exactly 0,1,2,...,max_goals for each team.
    a_scores = poisson.pmf(np.arange(max_goals + 1), xg_a)
    b_scores = poisson.pmf(np.arange(max_goals + 1), xg_b)

    # Grid of every scoreline: grid[i, j] = P(team_a scores i AND team_b scores j),
    # assuming the two are independent (so we just multiply the chances).
    grid = np.outer(a_scores, b_scores)

    a_win = np.tril(grid, -1).sum()   # below diagonal: team_a scored more
    draw  = np.trace(grid)            # diagonal: equal score
    b_win = np.triu(grid,  1).sum()   # above diagonal: team_b scored more
    return a_win, draw, b_win

a_win, draw, b_win = match_probabilities(team_a, team_b)
print(f"\n=== Prediction (neutral venue): {team_a} vs {team_b} ===")
print(f"  {team_a} win: {a_win:5.1%}")
print(f"  Draw:        {draw:5.1%}")
print(f"  {team_b} win: {b_win:5.1%}")
print(f"  (sanity: total = {a_win + draw + b_win:.3f})")
