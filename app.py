"""Streamlit web app for the World Cup 2026 predictor.

Run it with:   ./.venv/bin/streamlit run app.py
A web UI over the engine in predictor.py + the simulator in world_cup.py.
"""

import altair as alt
import pandas as pd
import streamlit as st

from predictor import (load_matches, train_model, expected_goals,
                       match_probabilities, top_scorelines, ensure_fresh_data)
from world_cup import simulate_tournament
from flags import with_flag


# ttl="12h" -> the cache expires twice a day, so the next visit re-pulls fresh
# results and retrains. New matches flow in automatically with no servers/cron.
@st.cache_resource(ttl="12h")
def get_model_and_teams():
    ensure_fresh_data()   # re-download the CSV if it's stale
    matches = load_matches()
    model = train_model(matches)
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    return model, teams, len(matches)


@st.cache_data(ttl="12h")
def get_title_odds():
    return simulate_tournament()


model, teams, n_matches = get_model_and_teams()


def default_index(name):
    return teams.index(name) if name in teams else 0


def whole_percents(values):
    """Round fractions to whole percents that still sum to 100 (largest remainder)."""
    scaled = [v * 100 for v in values]
    floors = [int(x) for x in scaled]
    leftover = round(sum(scaled)) - sum(floors)
    order = sorted(range(len(values)), key=lambda i: scaled[i] - floors[i], reverse=True)
    for i in range(leftover):
        floors[order[i]] += 1
    return floors


st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽")
st.title("⚽ World Cup 2026 Predictor")
st.caption(f"Win/draw/loss odds and tournament simulations from a transparent "
           f"Poisson model — trained on {n_matches:,} international matches since 2010.")

tab_match, tab_cup = st.tabs(["⚔️ Match predictor", "🏆 World Cup odds"])

# --- Tab 1: head-to-head match predictor ------------------------------------
with tab_match:
    col1, col2 = st.columns(2)
    team_a = col1.selectbox("Team A", teams, index=default_index("Brazil"),
                            format_func=with_flag)
    team_b = col2.selectbox("Team B", teams, index=default_index("Argentina"),
                            format_func=with_flag)

    if team_a == team_b:
        st.warning("Pick two different teams.")
    else:
        p_a, draw, p_b = match_probabilities(model, team_a, team_b)
        wa, wd, wb = whole_percents([p_a, draw, p_b])

        # Favourite callout.
        if wd >= wa and wd >= wb:
            st.info(f"🤝 Too close to call — a draw is the single most likely result ({wd}%).")
        elif wa >= wb:
            st.success(f"⭐ {with_flag(team_a)} favoured — {wa}% to win.")
        else:
            st.success(f"⭐ {with_flag(team_b)} favoured — {wb}% to win.")

        m1, m2, m3 = st.columns(3)
        m1.metric(f"{with_flag(team_a)} win", f"{wa}%")
        m2.metric("Draw", f"{wd}%")
        m3.metric(f"{with_flag(team_b)} win", f"{wb}%")

        # Bars in explicit order (sort=None) with intuitive colours.
        labels = [f"{team_a} win", "Draw", f"{team_b} win"]
        chart_df = pd.DataFrame({"outcome": labels, "probability": [p_a, draw, p_b]})
        st.altair_chart(
            alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("outcome:N", sort=None, title=None),
                y=alt.Y("probability:Q", axis=alt.Axis(format="%"), title=None),
                color=alt.Color("outcome:N", sort=None, legend=None,
                                scale=alt.Scale(domain=labels,
                                                range=["#2563eb", "#94a3b8", "#ef4444"]))),
            use_container_width=True)

        st.divider()
        xg_a, xg_b = expected_goals(model, team_a, team_b)
        st.write(f"**Expected goals:** {with_flag(team_a)} {xg_a:.2f} "
                 f"– {xg_b:.2f} {with_flag(team_b)}")
        scorelines = top_scorelines(model, team_a, team_b, n=5)
        (ti, tj), tp = scorelines[0]
        st.write(f"**Most likely scoreline:** {team_a} {ti}–{tj} {team_b} ({tp:.0%})")
        st.table(pd.DataFrame(
            [{"Scoreline": f"{team_a} {i}–{j} {team_b}", "Probability": f"{p:.1%}"}
             for (i, j), p in scorelines]))

# --- Tab 2: full tournament odds (Monte Carlo) ------------------------------
with tab_cup:
    st.caption("From simulating the rest of the tournament 3,000 times — real groups "
               "& results so far, the model plays out the rest. All 48 teams are "
               "simulated; choose how many to show.")
    n_show = st.slider("Teams to show", min_value=8, max_value=48, value=16)
    odds = get_title_odds().head(n_show).reset_index(names="Team")
    odds["Flag"] = odds["Team"].map(with_flag)

    st.altair_chart(
        alt.Chart(odds).mark_bar(color="#16a34a").encode(
            x=alt.X("title:Q", axis=alt.Axis(format="%"), title="Chance of winning"),
            y=alt.Y("Flag:N", sort="-x", title=None)),
        use_container_width=True)

    st.dataframe(
        pd.DataFrame({
            "Team": odds["Flag"],
            "Reach R16": odds["r16"], "Reach QF": odds["qf"], "Reach SF": odds["sf"],
            "Reach final": odds["final"], "Win title": odds["title"],
        }),
        hide_index=True, use_container_width=True,
        column_config={c: st.column_config.ProgressColumn(
            c, format="percent", min_value=0, max_value=1)
            for c in ["Reach R16", "Reach QF", "Reach SF", "Reach final", "Win title"]},
    )

# --- Shared expanders + footer ----------------------------------------------
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
        bonus), fitted with a **Poisson regression** on historical goals — weighting
        recent matches more, with a Dixon–Coles low-score correction. For a matchup
        we get expected goals for each side, then the **Poisson distribution** gives
        every scoreline's chance → win / draw / loss. Tournament odds come from
        simulating the whole bracket thousands of times.
        """
    )

st.divider()
st.caption("Built with a simple, transparent Poisson model · "
           "[Source on GitHub](https://github.com/mikkaarumugam/world-cup)")
