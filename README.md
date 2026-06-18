# ⚽ World Cup 2026 Predictor

**Live site → https://mikkaarumugam.github.io/world-cup/**

A transparent, calibrated football forecaster. It learns each national team's
**attack** and **defence** strength from ~15,500 real international results, then
turns any matchup into **win / draw / loss** odds, expected goals, and the most
likely scorelines. On top of that it runs a Monte Carlo simulation of the whole
2026 World Cup to estimate each team's chance of reaching every round, all the way
to lifting the trophy.

The site rebuilds itself **every day**, so as real results come in, the
predictions update on their own.

> This is my **first machine-learning project**, built to learn while building.
> The README below is the tour. The full, plain-English, step-by-step journal of
> how it was built (including the bugs and the dead ends) lives in
> **[`LEARNING_LOG.md`](LEARNING_LOG.md)** - that's the real gold mine.

---

## What it does

The site has three tabs, all driven by the same model:

- **Match predictor** - pick any two teams and get win/draw/loss odds, expected
  goals, and a heatmap of every scoreline's probability. Also shows the model's
  call on upcoming fixtures and how its recent calls actually turned out.
- **World Cup odds** - each team's simulated chance of reaching the Round of 16,
  quarters, semis, final, and winning it. Group results already played are locked
  in; everything unplayed is simulated.
- **Model performance** - the honest scorecard: how accurate and how
  *well-calibrated* the forecasts are on matches the model never trained on, plus
  a live "this tournament so far" tracker.

---

## How the model works

At its heart is a **Poisson regression** - a classic statistical model (no neural
nets, no LLM), which is exactly what makes it interpretable.

### 1. Learn team strengths

Every match is reshaped so each team's goals are one row, then a single model is
fit:

```
goals ~ C(team) + C(opponent) + home      (family = Poisson)
```

In plain English: *explain how many goals a team scores using who they are
(attack), who they're playing (the opponent's defence), and whether they're at
home.* The fit learns, with nothing hand-coded:

- an **attack** multiplier for every team,
- a **defence** multiplier for every team, and
- a single **home-advantage** term.

Because Poisson regression works in log space, these effects **multiply**. The
expected goals for one side are:

```
expected goals = baseline × attack(team) × defence(opponent) × home-boost
```

The strengths it discovers are credible on their own - top attacks land on Spain,
Brazil, France, Portugal; top defences on Argentina, Brazil, England. The model
even rates Morocco's defence highly, matching their real 2022 run, something
nobody told it.

Two refinements were added after measuring that they help (see the process
section):

- **Time-decay weighting** - a match ~3 years old counts half as much as a recent
  one, so the ratings track current form without throwing away history.
- Trained on matches back to **2010** (decay safely lets us use more data).

### 2. Turn strengths into a result

For a matchup we combine the relevant strengths into expected goals for each side,
then use the **Poisson distribution** to get the probability of each team scoring
0, 1, 2, ... goals. Multiplying those gives a grid of every scoreline; summing the
cells gives win / draw / loss:

```
below the diagonal → team A wins   |   diagonal → draw   |   above → team B wins
```

A small **Dixon-Coles correction** (rho = -0.05) nudges the low-scoring cells
(0-0, 1-0, 0-1, 1-1), because real football has slightly more tight draws than
pure independence would predict.

### 3. Simulate the tournament

We can't directly compute who wins a 48-team tournament, so we play it out
**3,000 times** (`world_cup.py`). Each simulation samples goals for every unplayed
fixture from the model, runs the group stage and knockouts, and records how deep
each team got. Counting across all 3,000 runs gives each team's odds for every
round. Real group results are used wherever they already exist.

---

## How good is it?

Backtested **honestly** on a locked test set of **749 international matches
(Oct 2025 - Jun 2026) the model never saw** during training or tuning:

| Metric | This model | Baseline |
|---|---|---|
| Top-pick accuracy *(higher better)* | **60.5%** | 48.5% *(always pick the home team)* |
| Log-loss *(lower better)* | **0.846** | 1.048 *(base rates)* · 1.099 *(blind 33/33/33)* |
| Brier score *(lower better)* | **0.497** | 0.631 *(base rates)* |
| Calibration error *(lower better)* | **2.0 pts** | - |

A few things worth knowing about these numbers:

- **Judge it mainly by log-loss / Brier, not accuracy.** Accuracy is flattered by
  easy mismatches and capped by genuinely unpredictable draws. The probability
  scores are the honest measure, and the model beats a sensible baseline by ~20%
  on both.
- **It's well-calibrated.** When the model says "30% chance," that outcome really
  happens about 30% of the time - its stated odds are off by only ~2 points on
  average. Calibration is what makes the probabilities trustworthy, not just the
  rank order.
- **On decisive games it's right ~78%** of the time, and the true result lands in
  its **top two outcomes ~85%** of the time. (Draws are near coin-flips no model
  predicts well, so the headline accuracy is naturally dragged down by them.)

The "this tournament so far" tracker on the site reports the same idea on live
2026 games (a small sample), which is why those figures move around and don't
match the backtest exactly.

---

## How it was built (the rigorous bit)

The fun of this project wasn't getting *a* number, it was learning how to not fool
myself. The process, in order:

1. **Backtest, don't guess.** Train on older matches, test on newer ones whose
   results are already known. Split by **time, not randomly**, to avoid data
   leakage (using the future to predict the past).
2. **Beat a baseline.** If the model can't beat "just predict the historical
   base rates," then learning team strengths added nothing. It does, comfortably.
3. **Score probabilities properly** with log-loss and Brier, which punish being
   confident *and* wrong, rather than crude right/wrong accuracy.
4. **A 3-way split to tune safely.** Train / validation / a locked test vault
   opened exactly once. Tuning against the test set repeatedly would overfit it;
   validation absorbs that risk.
5. **Autoresearch (Karpathy-style).** A measure → keep-if-better → revert loop
   over the model, scored by **cross-validated** log-loss (see
   [`program.md`](program.md)). Real wins: time-decay and using more history.
   A correctly-*rejected* idea: downweighting friendlies made things worse, so it
   was dropped. Negative results are results.
6. **Knowing when to stop.** I measured the model's information floor (the score a
   perfect version of *this* model could get) and found we were already sitting on
   it. Football is partly random, so even a flawless forecaster scores well above
   zero. The takeaway: more tuning is low-value here; a real jump needs new
   *information* (xG, lineups), not more knobs.

The full narrative, with every experiment and lesson, is in
[`LEARNING_LOG.md`](LEARNING_LOG.md).

---

## Repo map

| File | What it is |
|---|---|
| `predictor.py` | The model engine: load data, train, predict, plus a CLI. |
| `world_cup.py` | Monte Carlo simulation of the 2026 tournament (title odds). |
| `evaluate.py` | The honest scorer: backtest, cross-validation, and locked test. |
| `export_data.py` | Generates `web/data.json` (everything the site needs) from the model. |
| `web/` | The static site: `index.html` + the generated `data.json`. |
| `app.py` | The original Streamlit app (an alternative, server-side interface). |
| `flags.py` | Team → country-flag lookup for the UI. |
| `LEARNING_LOG.md` | The full build journal (zero-to-model, in plain English). |
| `program.md` | The autoresearch brief and rules. |

---

## Run it locally

```bash
python -m venv .venv
./.venv/bin/pip install -r requirements.txt

# 1) Regenerate the site's data from the latest live results (~30-40s: it
#    downloads results, trains the model, and simulates the tournament).
./.venv/bin/python export_data.py

# 2) Preview the static site.
cd web && ../.venv/bin/python -m http.server 8502
#    → open http://localhost:8502

# Other entry points:
./.venv/bin/python predictor.py Brazil Argentina   # one-off CLI prediction
./.venv/bin/python world_cup.py                    # print tournament odds
./.venv/bin/python evaluate.py --test              # the locked-test scorecard
./.venv/bin/streamlit run app.py                   # the Streamlit version
```

The international results dataset downloads automatically on first run (it's
gitignored, not committed).

---

## How it stays current

The site is hosted on **GitHub Pages** and rebuilt by a **GitHub Actions**
workflow ([`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)) that:

1. installs the dependencies,
2. runs `export_data.py` (which force-downloads the latest international results,
   retrains, and re-simulates), and
3. publishes the `web/` folder.

It runs on **every push to `main`**, on a **daily 06:00 UTC schedule**, and
**on demand** from the Actions tab. "Live" means as fresh as the upstream dataset,
which updates roughly daily.

---

## Data & credits

International results from
**[martj42/international_results](https://github.com/martj42/international_results)**
(every men's international since 1872). Built with pandas, statsmodels, and scipy.

Possible next steps: shrink small-sample team ratings toward the mean, try a
Negative Binomial family for over-dispersion, or bring in richer features like
expected goals (xG) and lineups - which, per the information-floor finding, is
where the real accuracy gains would come from.
