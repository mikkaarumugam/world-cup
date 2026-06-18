# My ML Learning Log — World Cup 2026 Predictor

A plain-English journal of building a football match predictor from zero ML knowledge.

---

## Step 1 — Setting up the toolbox

**What we built:** A "virtual environment" (a sealed toolbox just for this project) and installed
three libraries into it.

**New words I learned:**
- **Virtual environment (venv):** a private folder holding this project's own Python + libraries, so
  it can't clash with anything else on my computer. Good hygiene; pros do it per project.
- **Library:** pre-written code someone else made that I can reuse instead of writing from scratch.
- **pandas:** a "spreadsheet for code" — loads data into a table I can filter and reshape.
- **statsmodels:** the statistics library that will *learn* team strengths for me.
- **scipy:** scientific-maths library; I'll use its `poisson` piece to turn expected goals into
  scoreline probabilities.

**Versions installed:** pandas 3.0.3, statsmodels 0.14.6, scipy 1.17.1 (Python 3.13.3).

**The big idea I'm building toward:** every team gets two numbers — an attack strength and a defence
strength — learned from past results. Combine them to estimate expected goals, then use the Poisson
distribution to turn that into win/draw/loss percentages.

---

## Step 2 — Getting the data + first look

**What we built:** downloaded a famous free dataset (every international match 1872→2026, ~49.5k rows)
into `data/results.csv`, and wrote `peek.py` to look at it.

**New words:**
- **DataFrame:** pandas' word for a *table* (rows = records, columns = fields). Like an Excel sheet
  driven by code.
- **CSV:** a plain-text spreadsheet file (comma-separated values).

**What I saw:** 49,477 matches × 9 columns (date, home_team, away_team, home_score, away_score,
tournament, city, country, neutral). The `neutral` column flags neutral-venue games — exactly what
lets us handle home advantage properly. The file even contains *future* 2026 fixtures (no scores yet).

**Gotcha learned:** pandas hides middle columns with `...` to fit the terminal; fix with
`pd.set_option("display.max_columns", None)`.

---

## Step 3 — Filtering the data

**What we built:** started the real script `predictor.py`. It loads the CSV and applies two filters,
printing the row count after each so I can watch the data shrink.

**The two filters & why:**
1. **Drop unplayed matches** — future fixtures have empty score cells (pandas calls missing data
   **NaN** = "Not a Number"). Can't learn from a score that doesn't exist. (`.dropna`)
2. **Keep recent years (2018+)** — a trade-off: ALL history = stable but stale; very recent only =
   fresh but too thin/noisy. The middle window balances both.

**Result:** 49,477 → 49,425 (52 future games removed) → 8,128 (kept 2018→2026). 278 distinct teams.

**New ideas:**
- **NaN** = missing value.
- **Modelling is full of judgement calls** (e.g. how far back to look) — no single "correct" answer.
- A team with very few matches gets a **noisy** strength estimate — too little data. We care mostly
  about the ~48 World Cup nations, who all play plenty.

---

## Step 4 — Reshaping wide -> long

**What we built:** turned each match (one wide row, two scores) into TWO rows, one per team's point
of view: `team`, `opponent`, `goals`, `home`. 8,128 matches -> 16,256 long rows.

**Why long, not wide (key insight):**
1. A model predicts **one number per row** (the *target*). Two scores on one row = two answers; the
   model can't learn from that. One row per team's goals = one clean target.
2. **Pooling:** in wide format a team's goals are split between `home_score`/`away_score` depending on
   venue. Long format puts every goal a team scored into rows where `team`=them, so the model gathers
   ALL their attacking evidence into one number (and goals-against into one defence number).

**New words:**
- **Wide vs long format**, **target/response** (the one column we predict), **features/predictors**
  (the describing columns).
- **Dummy variable:** a yes/no written as 1/0 (here `home`: 1 = team was at home, 0 = away). Models
  need numbers, not text. The `home` flag describes the `team` in that row, not the opponent.
- **Shape follows the model:** there's no single "right" data shape — you reshape to fit the model
  you intend to train.

---

## Step 5 — Training the Poisson model (the actual machine learning!)

**What we built:** one `statsmodels` call that LEARNED every team's attack & defence + home advantage:
`goals ~ C(team) + C(opponent) + home`, family = Poisson. Result: 432 parameters learned, and home
advantage = **x1.271** (playing at home boosts expected goals ~27% — realistic!).

**Where the "machine learning" actually is:** NOT loading/reshaping data (that's prep). It's the
`.fit()` call, where an optimizer runs the *guess -> score -> adjust* loop (maximum likelihood) to
FIND the parameters. No LLM involved — this is a classic statistical ML model. statsmodels just
packages ~200 years of maths (Poisson 1830s, Fisher's maximum likelihood 1920s, GLMs 1972).

**New words:**
- **Parameter:** a number the model *learns from data* (vs a human hand-coding it). LLMs learn
  billions of these the same way in spirit — just vastly bigger.
- **Formula `goals ~ C(team) + C(opponent) + home`:** "explain goals using team (attack), opponent
  (defence), home boost." `C()` = treat as categories.
- **Log space / log link:** Poisson regression works in logs, which makes effects *multiply*
  (baseline x attack x defence x home). `exp()` converts a coefficient back to a goal multiplier.
- **Converged:** the trust flag. `True` = the guess->adjust loop settled into a stable best answer.
  `False` = numbers never settled (don't trust). Check it every time.

**Debugging saga (the realest ML lesson):**
- First fit CRASHED: "estimation infeasible" — caused by **separation** (teams with too-perfect or
  too-tiny records push a coefficient to +/- infinity, breaking the maths).
- Added Filter 3 (drop teams with < 10 matches) -> stopped crashing but `Converged: False`.
- Diagnosed (didn't guess!): Abkhazia & Ynys Môn scored in EVERY game -> infinite attack. Subtle:
  Abkhazia passed the >=10 filter but dropped to 3 matches once tiny opponents were removed.
- Fix: make Filter 3 a **fixed-point loop** (re-apply until the team set stops changing). Converged!
- Takeaways: always check `Converged`; separation is real; filters interact (re-apply till stable);
  diagnose the cause before changing things; sometimes the cure is better DATA, not more iterations.

---

## Step 6 — Reading the strengths + predicting expected goals

**Part A — read out what it learned.** Pulled each team's attack & defence coefficient out of the
model, `exp()`'d them into multipliers relative to a reference team, and ranked them. Result (with
NOTHING hand-coded): top attacks = Spain, Germany, Brazil, Portugal, France...; best defences =
Argentina, Brazil, England, Spain, Morocco... Credible! Proof the learning worked. (Morocco's strong
defence matches their real 2022 World Cup run — model found something true I never told it.)

**Part B — expected goals for a matchup.** Built `expected_goals(a, b)`: make a 2-row "flashcard"
(team, opponent, home=0 for neutral) and let `model.predict()` do baseline x attack x defence.
France vs England -> France 0.97, England 1.12. Key idea: a prediction is the CLASH of two teams
(my attack vs your defence), not either team alone. Elite defences pull expected goals down.

**New words:**
- **Reference team / baseline:** statsmodels measures every team relative to one yardstick team.
  Absolute scale is arbitrary; only the ordering/ratios matter.
- **dtype (float64, int64, object, bool, datetime64):** the kind of data in a column. float64 =
  decimals. Wrong dtypes (numbers stored as text) are a common real-world bug.

---

## Step 7 — Win / Draw / Loss probabilities (FINISH LINE 🎉)

**What we built:** `match_probabilities(a, b)` — the final piece.
1. Poisson **pmf** = chance each team scores exactly 0,1,2,... goals.
2. **Grid** of every scoreline = multiply the two (assuming **independence** — a simplification;
   the Dixon-Coles tweak fixes it later).
3. Sum cells: below diagonal = team_a win, diagonal = draw, above = team_b win.

**Result:** France 31.1% / Draw 29.9% / England 39.0% (totals to 1.000 — sanity check passed).
Sanity across the spectrum: Brazil 94% vs Vietnam (lopsided ✓), Spain ~36% vs Brazil (coin-flip ✓).

**New words:**
- **pmf (probability mass function):** the chance of each exact whole-number outcome.
- **Independence assumption:** treating the two teams' goals as unrelated so we can multiply their
  chances. Not perfectly true in football, but keeps the model simple and see-through.

**THE FULL PIPELINE I BUILT:** raw scores -> learned attack/defence -> expected goals for a matchup
-> Poisson scoreline grid -> win/draw/loss %. A complete end-to-end ML model, from zero.

---
# STAGE 2 — Is the model actually any good? (Evaluation)

## Step 1 — Refactor into functions + a CLI

**What we built:** reorganised `predictor.py` from a top-to-bottom script into reusable functions
(`load_matches`, `to_long`, `train_model`, `expected_goals`, `match_probabilities`) so we can train on
ANY subset of data (needed for evaluation) and reuse the engine everywhere. Added a command line:
`./.venv/bin/python predictor.py Brazil Argentina`.

**New ideas:**
- **Refactoring** = change the structure, NOT the behaviour. Proof it was safe: the default
  France/England prediction stayed identical (31.1/29.9/39.0).
- **`sys.argv`:** the list of command-line arguments (`sys.argv[1]` = first word after the script).
- **`if __name__ == "__main__":`** = code that runs only when the file is run directly, NOT when it's
  imported. Makes a file both a reusable library AND a runnable program. (Proof: importing predictor
  into evaluate.py did NOT fire its demo prediction.)

## Steps 2-4 — Backtesting: split, baseline, score (in `evaluate.py`)

**The core idea — backtesting:** you don't invent test cases. Take REAL past matches, train on the
older ones, and test on the newer ones (whose results you already know). Replay history as if it were
the future; reality already graded the exam.

- **Train/test split by TIME (not random):** train 2018-2024 (6,584 matches), test 2025-2026 (1,307).
  Splitting by time avoids **data leakage** (using the future to predict the past) — a top way people
  fool themselves into thinking a model is good.
- **Baseline to beat:** predict the historical **base rates** for every match, ignoring who's playing
  (Home 47.4% / Draw 23.4% / Away 29.2%). If the model can't beat this, learning team strengths added
  nothing.
- **Fairness fix:** grade each test match under its ACTUAL venue (home team gets home=1 unless
  neutral). Generalised `expected_goals`/`match_probabilities` with `home_a`/`home_b` (default neutral,
  so the CLI is unchanged).
- **Scoring probabilities (can't use right/wrong):**
  - **Log-loss** = -log(probability you gave the TRUE outcome), averaged. Savagely punishes being
    confident AND wrong. Lower = better.
  - **Brier score** = mean squared error of the probabilities vs reality (1/0). Lower = better.
- **Cold start:** the model can't predict a team it never saw in training — skip those test matches
  (here: 0 skipped, all 1,307 testable).

**RESULT (out-of-sample, leak-free):**
| | log-loss | Brier |
|---|---|---|
| Baseline | 1.0468 | 0.6307 |
| Model | 0.8309 | 0.4886 |

Model beats baseline by ~20% on BOTH. Reference: a clueless 33/33/33 guess = 1.099 log-loss.

**Honest caveat (= credibility):** the absolute 0.831 is flattered by easy mismatch games in the test
set. The trustworthy claim is the RELATIVE, leak-free one: "beats a sensible baseline by ~20%
out-of-sample." Knowing this caveat matters more than the score.

---
# STAGE 4 — Autoresearch (Karpathy-style automated experiments)

**The idea (from Andrej Karpathy's `autoresearch`):** an AI agent loops — change the model, score it
on ONE metric, keep if better / revert if worse, repeat. You write a plain-English brief (`program.md`)
and never hand-edit the experiments. We realised we'd already invented the manual version ("measure,
don't guess"), and our project is a great fit: a clean engine file + a fast single-metric scorer, ~7s
per run on CPU (his needs a GPU).

**The critical guardrail — a 3-way time split** (rebuilt `evaluate.py`):
- TRAIN -> fit the model
- VALIDATION -> the score we optimize during experiments
- TEST -> a locked vault, opened exactly ONCE at the very end.
Why: optimizing against the test set hundreds of times would let us **overfit it** (find tweaks that
fit that one set by luck). Validation absorbs that risk; test stays an honest final check.

**The loop we ran (me as the agent):** edit `predictor.py` -> run `evaluate.py` -> keep/revert by
VALIDATION log-loss (start = 0.8678). `program.md` holds the brief + rules (only edit predictor.py,
never touch the scorer or the test split, no leakage).

**Experiments:**
1. **Time-decay weighting** (recent matches count more, via a half-life). Swept half-lives.
   KEPT: 3-year half-life. 0.8678 -> 0.8640. Lesson: gentle decay helps; AGGRESSIVE recency HURTS
   badly (3-month = 0.97) because international teams play rarely — discount old games too fast and
   you starve the model.
2. **Tournament weighting** (downweight friendlies). REVERTED: every setting was worse. Lesson: a
   plausible idea was WRONG — friendlies (30% of games) still carry signal. The harness caught my
   bad intuition.
3. **More history + decay together.** With decay auto-discounting old data, including back to 2010
   (5,913 -> 13,525 train matches) helped. KEPT: from 2010. 0.8640 -> 0.8605. Lesson: decay and
   history are partners — decay lets you safely use more data.

**Final HONEST check (locked test set, opened once — 745 matches, Oct 2025 -> Jun 2026):**
| config | test log-loss | accuracy |
|---|---|---|
| Original (from 2018, no decay) | 0.8561 | 60.2% |
| Autoresearched (from 2010, 3y decay) | **0.8464** | **61.1%** |
Baseline on this test set: 1.0478 log-loss / 48.5% accuracy.

**Takeaways:**
- The validation gain (~0.8%) TRANSFERRED to the test (~1.1%) -> we improved for real, didn't just
  overfit validation. The 3-way split is what let us trust this.
- Gains are MODEST (~1%) — realistic for a simple model + small search space. The value is the
  rigorous, self-correcting PROCESS (2 real wins, 1 correctly-rejected idea), not the 1%.
- Negative results are results: rejecting friendly-downweighting is as valuable as keeping decay.
