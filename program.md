# Autoresearch brief — World Cup match predictor

Goal: improve the model's predictions, measured honestly.

## The metric (optimize this)
Run `./.venv/bin/python evaluate.py` and minimize **VALIDATION log-loss** (lower
is better). Current baseline: **0.8678**.

## The loop
1. Make ONE change at a time to `predictor.py` (the model/features).
2. Run `./.venv/bin/python evaluate.py` and read the validation log-loss.
3. **Keep** the change only if validation log-loss improves; otherwise **revert**.
4. Log the experiment (what, result, kept/reverted) in `LEARNING_LOG.md`.

## Rules (do not break)
- Only edit `predictor.py`. **Never edit `evaluate.py`** (the scorer must stay fixed).
- **Never run `--test` or look at the test split** until the very end. The test set
  is a one-time honesty check, not an optimization target.
- No data leakage: never use a match's own result (or any future match) to predict it.
- The model must still **converge** and stay simple/interpretable.

## Ideas to explore (roughly best-first)
1. **Time-decay weighting** — weight recent matches more (try several half-lives).
2. **Tournament weighting** — downweight friendlies vs competitive games.
3. **Dixon-Coles low-score correction** — fix the independence assumption for
   0-0 / 1-0 / 1-1 scorelines.
4. **Shrinkage** — pull small-sample team ratings toward the average.
5. **Tune dials** — sweep `EARLIEST_YEAR`, `MIN_MATCHES`, `max_goals`.

## Final step (once, at the end)
Run `./.venv/bin/python evaluate.py --test` exactly once to report the honest
out-of-sample score of the final model.
