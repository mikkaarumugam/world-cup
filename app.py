"""Streamlit web app for the World Cup 2026 match predictor.

Run it with:   ./.venv/bin/streamlit run app.py
This is a web UI wrapped around the engine in predictor.py.
"""

import pandas as pd
import streamlit as st

from predictor import (load_matches, train_model, expected_goals,
                       match_probabilities, top_scorelines)


# @st.cache_resource = "train once, reuse forever". Streamlit re-runs this script
# on every click, so without caching we'd retrain (~7s) each time. The cache
# trains the model once and hands the same one to every visitor.
@st.cache_resource
def get_model_and_teams():
    matches = load_matches()
    model = train_model(matches)
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    return model, teams, len(matches)


model, teams, n_matches = get_model_and_teams()


def default_index(name):
    return teams.index(name) if name in teams else 0


st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽")
st.title("⚽ World Cup 2026 Match Predictor")
st.caption(f"A Poisson model trained on {n_matches:,} international matches since 2010 "
           "(recent games weighted more). Predictions are for a neutral venue.")

col1, col2 = st.columns(2)
team_a = col1.selectbox("Team A", teams, index=default_index("Brazil"))
team_b = col2.selectbox("Team B", teams, index=default_index("Argentina"))

if team_a == team_b:
    st.warning("Pick two different teams.")
    st.stop()

# --- Outcome probabilities --------------------------------------------------
p_a, draw, p_b = match_probabilities(model, team_a, team_b)
m1, m2, m3 = st.columns(3)
m1.metric(f"{team_a} win", f"{p_a:.0%}")
m2.metric("Draw", f"{draw:.0%}")
m3.metric(f"{team_b} win", f"{p_b:.0%}")

chart = pd.Series({f"{team_a} win": p_a, "Draw": draw, f"{team_b} win": p_b},
                  name="probability")
st.bar_chart(chart)

# --- Expected goals + most likely scorelines --------------------------------
st.divider()
xg_a, xg_b = expected_goals(model, team_a, team_b)
st.subheader("Expected goals")
st.write(f"**{team_a} {xg_a:.2f} – {xg_b:.2f} {team_b}** "
         "(the average goals the model expects each side to score)")

scorelines = top_scorelines(model, team_a, team_b, n=5)
(top_i, top_j), top_p = scorelines[0]
st.subheader("Most likely scoreline")
st.write(f"**{team_a} {top_i}–{top_j} {team_b}**  ({top_p:.0%} chance)")

st.caption("Top 5 most likely exact scores:")
st.table(pd.DataFrame(
    [{"Scoreline": f"{team_a} {i}–{j} {team_b}", "Probability": f"{p:.1%}"}
     for (i, j), p in scorelines]
))

# --- Credibility: how good is this model? -----------------------------------
with st.expander("ℹ️ How good is this model?"):
    st.markdown(
        """
        Tested by **backtesting** on a locked test set never used during
        development: trained on matches up to Sept 2025, then asked to predict
        **745 real matches (Oct 2025 – Jun 2026) it had never seen**, and graded
        against what actually happened.

        | Metric | Baseline* | This model |
        |---|---|---|
        | Top-pick accuracy (higher better) | 48.5% | **60.4%** |
        | Log-loss (lower better) | 1.048 | **0.846** |
        | Brier score (lower better) | 0.631 | **0.497** |

        \\*Baseline = always predict the historical average (home 47% / draw 23% /
        away 30%), ignoring who's playing.

        The model beats that baseline by **~19%** on the probability metrics.
        *Honest caveat:* accuracy is flattered by easy mismatches and capped by
        hard-to-predict draws — judge it mainly on log-loss/Brier.
        """
    )

with st.expander("🔧 How it works"):
    st.markdown(
        """
        Each team has a learned **attack** and **defence** strength (plus a home
        bonus), fitted with a **Poisson regression** on historical goals. For a
        matchup we combine those into expected goals for each side, then use the
        **Poisson distribution** to get the chance of every scoreline — summed
        into win / draw / loss. Simple and fully transparent on purpose.
        """
    )
