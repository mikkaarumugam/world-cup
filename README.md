# ⚽ World Cup 2026 Match Predictor

A simple, transparent football match predictor. It learns each national team's
**attack** and **defence** strength from historical international results, then
predicts the chance of a **win / draw / loss** for any matchup — plus expected
goals and the most likely scorelines.

Built as a learn-while-building first ML project. See `LEARNING_LOG.md` for the
plain-English journey from zero ML knowledge.

## How it works

A **Poisson regression** learns, from ~17,000 matches since 2010 (with recent
matches weighted more heavily):
- an attack strength and a defence strength for each team, and
- a home-advantage effect.

For a matchup we combine the relevant strengths into **expected goals** for each
side, then use the **Poisson distribution** to turn those into the probability of
every scoreline, summed into win / draw / loss.

## How good is it?

Backtested honestly on a **locked test set never used during development**: trained
on matches up to Sept 2025, then tested on **745 real matches (Oct 2025 – Jun 2026)
it had never seen**.

| Metric | Baseline* | This model |
|---|---|---|
| Top-pick accuracy (higher better) | 48.5% | **60.4%** |
| Log-loss (lower better) | 1.048 | **0.846** |
| Brier score (lower better) | 0.631 | **0.497** |

\*Baseline = always guess the historical average, ignoring who's playing.
The model beats it by ~19% on the probability metrics. (Accuracy is flattered by
easy mismatches and capped by hard-to-predict draws, so judge mainly on
log-loss/Brier.)

## Run it

```bash
python -m venv .venv
./.venv/bin/pip install -r requirements.txt

# Download the data (international results since 1872):
mkdir -p data
curl -sSL -o data/results.csv \
  https://raw.githubusercontent.com/martj42/international_results/master/results.csv

# Command line:
./.venv/bin/python predictor.py Brazil Argentina

# Web app:
./.venv/bin/streamlit run app.py

# Evaluate the model:
./.venv/bin/python evaluate.py
```

## Files

- `predictor.py` — the model engine (load, train, predict) + a CLI.
- `app.py` — the Streamlit web app.
- `evaluate.py` — the backtest that grades the model vs a baseline.
- `LEARNING_LOG.md` — the build journal.

## Possible improvements

Time-decay weighting (recent matches matter more), tournament weighting,
Dixon–Coles low-score correction, and shrinking small-sample team ratings.
Data: [martj42/international_results](https://github.com/martj42/international_results).
