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
