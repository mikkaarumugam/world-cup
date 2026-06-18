"""Streamlit web app for the World Cup 2026 match predictor.

Run it with:   ./.venv/bin/streamlit run app.py
A web UI wrapped around the engine in predictor.py + the simulator in world_cup.py.
"""

import altair as alt
import pandas as pd
import streamlit as st

from predictor import (load_matches, train_model, expected_goals,
                       match_probabilities, top_scorelines)
from world_cup import simulate_tournament


@st.cache_resource
def get_model_and_teams():
    matches = load_matches()
    model = train_model(matches)
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    return model, teams, len(matches)


@st.cache_data
def get_title_odds():
    return simulate_tournament()


model, teams, n_matches = get_model_and_teams()


def default_index(name):
    return teams.index(name) if name in teams else 0


st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽")
st.title("⚽ World Cup 2026 Predictor")
st.caption(f"A Poisson model trained on {n_matches:,} international matches since 2010 "
           "(recent games weighted more).")

tab_match, tab_cup = st.tabs(["Match predictor", "🏆 World Cup odds"])

# --- Tab 1: head-to-head match predictor ------------------------------------
with tab_match:
    st.caption("Pick any two teams (neutral venue).")
    col1, col2 = st.columns(2)
    team_a = col1.selectbox("Team A", teams, index=default_index("Brazil"))
    team_b = col2.selectbox("Team B", teams, index=default_index("Argentina"))

    if team_a == team_b:
        st.warning("Pick two different teams.")
    else:
        p_a, draw, p_b = match_probabilities(model, team_a, team_b)
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{team_a} win", f"{p_a:.0%}")
        m2.metric("Draw", f"{draw:.0%}")
        m3.metric(f"{team_b} win", f"{p_b:.0%}")

        # Explicit order (sort=None) so the bars match the cards above.
        chart_df = pd.DataFrame({"outcome": [f"{team_a} win", "Draw", f"{team_b} win"],
                                 "probability": [p_a, draw, p_b]})
        st.altair_chart(
            alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("outcome:N", sort=None, title=None),
                y=alt.Y("probability:Q", axis=alt.Axis(format="%"), title=None),
                color=alt.Color("outcome:N", sort=None, legend=None)),
            use_container_width=True)

        st.divider()
        xg_a, xg_b = expected_goals(model, team_a, team_b)
        st.write(f"**Expected goals:** {team_a} {xg_a:.2f} – {xg_b:.2f} {team_b}")
        scorelines = top_scorelines(model, team_a, team_b, n=5)
        (ti, tj), tp = scorelines[0]
        st.write(f"**Most likely scoreline:** {team_a} {ti}–{tj} {team_b} ({tp:.0%})")
        st.table(pd.DataFrame(
            [{"Scoreline": f"{team_a} {i}–{j} {team_b}", "Probability": f"{p:.1%}"}
             for (i, j), p in scorelines]))

# --- Tab 2: full tournament title odds (Monte Carlo) ------------------------
with tab_cup:
    st.caption("Title odds from simulating the rest of the tournament 3,000 times "
               "(real groups & results so far; the model plays out the rest).")
    odds = get_title_odds().head(16).reset_index(names="Team")

    st.altair_chart(
        alt.Chart(odds).mark_bar().encode(
            x=alt.X("title_pct:Q", axis=alt.Axis(format="%"), title="Chance of winning"),
            y=alt.Y("Team:N", sort="-x", title=None),
            color=alt.value("#4c78a8")),
        use_container_width=True)

    st.table(pd.DataFrame({
        "Team": odds["Team"],
        "Win title": odds["title_pct"].map("{:.1%}".format),
        "Reach final": odds["final_pct"].map("{:.1%}".format),
    }))

# --- Shared expanders -------------------------------------------------------
with st.expander("ℹ️ How good is this model?"):
    st.markdown(
        """
        Backtested on a **locked test set never used during development**: trained
        up to Sept 2025, then predicting **745 real matches (Oct 2025 – Jun 2026)**.

        | Metric | Baseline* | This model |
        |---|---|---|
        | Top-pick accuracy (higher better) | 48.5% | **60.4%** |
        | Log-loss (lower better) | 1.048 | **0.846** |
        | Brier score (lower better) | 0.631 | **0.497** |

        \\*Baseline = always predict the historical average, ignoring who's playing.
        The model beats it by **~19%** on the probability metrics.
        """
    )

with st.expander("🔧 How it works"):
    st.markdown(
        """
        Each team has a learned **attack** and **defence** strength (plus a home
        bonus), fitted with a **Poisson regression** on historical goals, weighting
        recent matches more and with a Dixon–Coles low-score correction. For a
        matchup we get expected goals for each side, then the **Poisson
        distribution** gives every scoreline's chance → win / draw / loss. The cup
        odds come from simulating the whole tournament thousands of times.
        """
    )
