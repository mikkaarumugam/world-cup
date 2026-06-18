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

# Outcome colours, reused by the metric/bar styling.
COLOR_A, COLOR_DRAW, COLOR_B = "#2563eb", "#64748b", "#ef4444"


@st.cache_resource(ttl="12h")
def get_model_and_teams():
    ensure_fresh_data()
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


def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;800&display=swap');
    html, body, .stApp, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stHeader"] {{ display:none; }}
    .block-container {{ padding-top: 1.5rem; max-width: 840px; }}

    .hero {{
        background: linear-gradient(135deg, #14532d 0%, #16a34a 60%, #22c55e 100%);
        color: #fff; padding: 1.6rem 1.9rem; border-radius: 18px; margin-bottom: 1.3rem;
        box-shadow: 0 10px 30px rgba(22,163,74,.25);
    }}
    .hero h1 {{ margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -.5px; }}
    .hero p {{ margin: .45rem 0 0; opacity: .92; font-size: .95rem; max-width: 90%; }}

    [data-testid="stMetric"] {{
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 14px;
        padding: .9rem .6rem; text-align: center;
    }}
    [data-testid="stMetricValue"] {{ font-weight: 800; }}

    .matchup {{ text-align:center; font-size:1.5rem; font-weight:700; margin:.3rem 0 1rem; }}
    .matchup .vs {{ color:#94a3b8; font-weight:400; margin:0 .6rem; font-size:1.1rem; }}

    .probbar {{ display:flex; height:48px; border-radius:12px; overflow:hidden;
        font-weight:700; color:#fff; box-shadow: inset 0 0 0 1px rgba(0,0,0,.04); }}
    .probbar .seg {{ display:flex; align-items:center; justify-content:center; font-size:.95rem; }}
    .probbar .a {{ background:{COLOR_A}; }}
    .probbar .d {{ background:{COLOR_DRAW}; }}
    .probbar .b {{ background:{COLOR_B}; }}
    .legend {{ display:flex; justify-content:space-between; margin:.5rem .1rem 0;
        color:#475569; font-size:.85rem; }}
    .legend i {{ display:inline-block; width:11px; height:11px; border-radius:3px;
        margin-right:6px; vertical-align:middle; }}

    .footer {{ text-align:center; color:#94a3b8; font-size:.82rem; margin-top:1.5rem; }}
    </style>
    """, unsafe_allow_html=True)


def prob_bar(team_a, team_b, wa, wd, wb):
    """A single stacked win/draw/loss bar (the iconic sports-prediction viz)."""
    def seg(width, cls):
        label = f"{width}%" if width >= 8 else ""
        return f'<div class="seg {cls}" style="width:{width}%">{label}</div>'
    bar = f'<div class="probbar">{seg(wa,"a")}{seg(wd,"d")}{seg(wb,"b")}</div>'
    legend = (f'<div class="legend">'
              f'<span><i style="background:{COLOR_A}"></i>{with_flag(team_a)} win</span>'
              f'<span><i style="background:{COLOR_DRAW}"></i>Draw</span>'
              f'<span><i style="background:{COLOR_B}"></i>{with_flag(team_b)} win</span></div>')
    st.markdown(bar + legend, unsafe_allow_html=True)


st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽")
inject_css()

st.markdown(f"""
<div class="hero">
  <h1>⚽ World Cup 2026 Predictor</h1>
  <p>Win/draw/loss odds and full-tournament simulations from a transparent Poisson
     model — trained on {n_matches:,} international matches and auto-updating with new results.</p>
</div>
""", unsafe_allow_html=True)

if st.button("🔄 Refresh with latest results",
             help="Re-download today's results and retrain (~25s)"):
    ensure_fresh_data(max_age_hours=0)
    get_model_and_teams.clear()
    get_title_odds.clear()
    st.rerun()

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

        st.markdown(f'<div class="matchup">{with_flag(team_a)}'
                    f'<span class="vs">vs</span>{with_flag(team_b)}</div>',
                    unsafe_allow_html=True)

        if wd >= wa and wd >= wb:
            st.info(f"🤝 Too close to call — a draw is the most likely single result ({wd}%).")
        elif wa >= wb:
            st.success(f"⭐ {with_flag(team_a)} favoured — {wa}% to win.")
        else:
            st.success(f"⭐ {with_flag(team_b)} favoured — {wb}% to win.")

        prob_bar(team_a, team_b, wa, wd, wb)

        st.divider()
        xg_a, xg_b = expected_goals(model, team_a, team_b)
        c1, c2 = st.columns(2)
        c1.metric(f"{with_flag(team_a)} — expected goals", f"{xg_a:.2f}")
        c2.metric(f"{with_flag(team_b)} — expected goals", f"{xg_b:.2f}")

        scorelines = top_scorelines(model, team_a, team_b, n=5)
        (ti, tj), tp = scorelines[0]
        st.markdown(f"**Most likely scoreline:** {team_a} {ti}–{tj} {team_b} ({tp:.0%})")
        st.table(pd.DataFrame(
            [{"Scoreline": f"{team_a} {i}–{j} {team_b}", "Probability": f"{p:.1%}"}
             for (i, j), p in scorelines]))

# --- Tab 2: full tournament odds (Monte Carlo) ------------------------------
with tab_cup:
    st.caption("From simulating the rest of the tournament 3,000 times — real groups "
               "& results so far, the model plays out the rest. All 48 teams are simulated.")
    n_show = st.slider("Teams to show", min_value=8, max_value=48, value=16)
    odds = get_title_odds().head(n_show).reset_index(names="Team")
    odds["Flag"] = odds["Team"].map(with_flag)

    bars = alt.Chart(odds).mark_bar(color="#16a34a", cornerRadiusEnd=5).encode(
        x=alt.X("title:Q", axis=alt.Axis(format="%", grid=False), title="Chance of winning"),
        y=alt.Y("Flag:N", sort="-x", title=None))
    labels = alt.Chart(odds).mark_text(align="left", dx=4, color="#334155",
                                       fontWeight="bold").encode(
        x="title:Q", y=alt.Y("Flag:N", sort="-x"), text=alt.Text("title:Q", format=".1%"))
    st.altair_chart((bars + labels).configure_view(strokeWidth=0), use_container_width=True)

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

st.markdown('<div class="footer">Built with a simple, transparent Poisson model · '
            '<a href="https://github.com/mikkaarumugam/world-cup">Source on GitHub</a></div>',
            unsafe_allow_html=True)
