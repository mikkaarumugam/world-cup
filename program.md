# Autoresearch brief — World Cup match predictor

Goal: find the best-generalizing model we can honestly defend, with every
experiment logged. Modest gains expected — the model is already near its
information floor (see LEARNING_LOG), so big jumps would need new features/data,
not tuning. The value is the rigorous, documented process.

## The metric (optimize this)
Run `./.venv/bin/python evaluate.py --cv` and minimize **CV log-loss** (lower is
better) — the average across rolling cross-validation folds. Current best: **0.8567**.
Do NOT optimize the single-window or test scores.

## The loop
1. Make ONE change at a time to `predictor.py` (the model/features).
2. Run `./.venv/bin/python evaluate.py --cv` and read the CV log-loss.
3. **Keep** the change only if CV log-loss improves; otherwise **revert** it fully.
4. Log every experiment (idea, CV result, kept/reverted, one-line lesson) in
   `LEARNING_LOG.md`.

## Stop when
100 experiments done, OR ~15 consecutive experiments with no improvement —
whichever comes first. Then report a short summary.

## Rules (do not break)
- Only edit `predictor.py`. NEVER edit `evaluate.py` (the scorer must stay fixed).
- NEVER run `--test` or look at the test split until the very end (one-time check).
- No data leakage: never use a match's own result (or any future match) to predict it.
- The model must still CONVERGE (check it) and stay reasonably interpretable.
- Keep changes small and reversible; one idea per experiment.

## Ideas to explore (the easy dials are already done: 3y decay + history to 2010)
- **Dixon-Coles low-score correction** (fix independence for 0-0/1-0/1-1).
- **Shrinkage / regularization** of team ratings toward the mean.
- **Negative Binomial** family instead of Poisson (handles over-dispersion).
- **Tune dials**: `MIN_MATCHES`, `max_goals`, the decay `HALF_LIFE_DAYS`,
  `EARLIEST_YEAR` (re-check jointly).
- **Opponent-specific or competition-specific** effects (carefully — watch overfit).
- Combinations of the above that each individually helped.

## Final step (once, at the very end)
Run `./.venv/bin/python evaluate.py --test` exactly once to report the honest
out-of-sample score of the final model, and compare to the starting model.
