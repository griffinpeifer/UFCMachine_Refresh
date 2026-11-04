# UFCMachine

A UFC fight outcome predictor built on a deep learning neural network. Give it two fighters' names and it returns the probability that each one wins, based on how their career statistics compare.

To create the model, we found the historical fight data, scraped fighter stats from the web, cleaned and joined the two datasets, and trained a Keras neural network that reached **71.35% accuracy** on a held-out test set. This repo (`UFCMachine_Refresh`) is a copy of that project, which I am modernizing and retraining on current data.

## How it works, at a glance

```
ufcstats.com fighter pages ──scrape──► raw fighter stats ──clean──► fighters_clean.csv ─┐
                                                                                         ├─► stat differences ─► neural network ─► P(win)
Kaggle UFC fight history (1993–2019) ──────────────────────────────► matches_clean.csv ─┘
```

The core idea: a fight is described by the **difference** between the two fighters' stats (Red minus Blue), and the model learns which differences tend to predict a win.

---

## 1. Finding the data

We needed two kinds of data:

| Need | Source | What it gave us |
|---|---|---|
| Who fought whom, and who won | Kaggle: [UFC-Fight historical data 1993–2019](https://www.kaggle.com/rajeevw/ufcdata) | Every UFC fight since 1993, with the Red and Blue corner fighters, the winner and the weight class |
| What each fighter is like | [ufcstats.com](http://ufcstats.com/statistics/fighters) (the UFC's official stats site) | Career stats for every fighter on record |

The Kaggle dataset is what got the project started: once we had every UFC fight result since 1993, the question became which fighter statistics best separate winners from losers. We reduced it to the four columns we needed (`Red`, `Blue`, `Winner`, `Weight Class`) and saved it as `matches_clean.csv`.

To let someone predict a fight by typing in fighter **names** rather than raw numbers, we also needed a lookup table of stats for every fighter, so we built our own scraper for ufcstats.com.

## 2. Scraping fighter stats (`UFC_scraping.ipynb`)

- Used **Splinter + Selenium (ChromeDriver)** to load the ufcstats.com fighter listing and **BeautifulSoup** to parse it.
- Collected each fighter's profile link from the listing table and de-duplicated them (each row links the same fighter several times).
- Visited every profile with a `requests` session and pulled:
  - **Name, nickname and record** (W-L-D) from the page header
  - **Physical attributes:** height, weight, reach, stance, date of birth
  - **Career stats:** significant strikes landed per minute (SLpM), striking accuracy, strikes absorbed per minute (SApM), striking defense, takedown average, takedown accuracy, takedown defense, submission average
- Stored each fighter as a dictionary keyed by the stat labels on the page, then saved the full roster to CSV (roughly 3,400 fighters).

![df_fighters](https://github.com/griffinpeifer/UFCMachine/blob/master/df_fighters.PNG?raw=true)

## 3. Cleaning (`UFC_Cleaning.ipynb`)

Everything scraped from the site came back as text (`6' 3"`, `235 lbs.`, `58%`, `Sep 02, 1981`), so cleaning was mostly turning it into numbers the model could use:

| Field | Raw value | Cleaned value |
|---|---|---|
| Record | `20-5-0` | Split into `Wins`, `Losses`, `Draws`, plus a new **Win Percentage** = wins ÷ total fights |
| Height | `6' 3"` | Split on feet/inches and converted to decimal feet → `6.25` |
| Weight | `235 lbs.` | `235` |
| Reach | `76"` | `76` |
| Date of birth | `Sep 02, 1981` | Parsed to a date and converted to **Age** in years (with a fix for two-digit years that parsed into the future) |
| Str. Acc., Str. Def., TD Acc., TD Def. | `58%` | `0.58` |
| Nickname | blank | `No Name` |

We also dropped columns the model wouldn't use (stance, the raw record string, the CSV index).

**Handling missing data:** many fighters had no height, reach, date of birth or striking stats on file (shown as `--` on the site). We removed every fighter with a missing field, which left **1,672 fighters**. That was an acceptable trade-off: the fighters with missing data were mostly older, retired fighters with short careers, so they matter little for predicting future fights.

The cleaned table is saved as `fighters_clean.csv`.

## 4. Building the training set (`UFC_Machine_Learning_Model.ipynb`)

1. **Feature set:** 14 stats per fighter: height, weight, reach, age, SLpM, striking accuracy, SApM, striking defense, takedown average, takedown accuracy, takedown defense, submission average, win percentage and total fights (a feature we added as wins + losses + draws, to capture experience).
2. **Label:** for each fight, `result = 1` if the Red-corner fighter won, `0` if Blue won.
3. **Joining the datasets:** kept only fights where *both* fighters were in our cleaned fighter table, so every fight had complete stats.
4. **Stat differential:** replaced each fight's two fighter names with the difference between their stats (Red − Blue) for all 14 features. This turns every fight into one row of 14 numbers plus the result.

The final training set had **4,386 fights** (2,763 Red wins, 1,623 Blue wins), saved as `training_model.csv`.

![df_matches with names](https://github.com/griffinpeifer/UFCMachine/blob/master/df_matches_names.PNG?raw=true)

![df_matches with stat differential](https://github.com/griffinpeifer/UFCMachine/blob/master/df_matches_differential.PNG?raw=true)

## 5. Training the model

- **Split:** 70% train / 30% test (`train_test_split`, `random_state=0`).
- **Scaling:** `StandardScaler` fit on the training set only, then applied to the test set, so every feature is on the same scale.
- **Architecture:** a Keras `Sequential` network with **eight hidden layers**:

  | Layer | Units | Activation |
  |---|---|---|
  | Hidden 1 | 16 | ReLU |
  | Hidden 2–7 | 32 each | ReLU |
  | Hidden 8 | 16 | ReLU |
  | Output | 1 | Sigmoid (probability Red wins) |

  Every hidden layer uses **L2 regularization (0.01)** to limit overfitting.
- **Training:** binary cross-entropy loss, Adam optimizer, 200 epochs, batch size 64.
- **Result:** **71.35% accuracy on the held-out test set.**

## 6. Making predictions

To predict a fight, the model looks up both fighters in `fighters_clean.csv`, takes the difference in their 14 stats, scales it with the same scaler used in training, and outputs the probability that the first fighter wins.

```python
print(predict(model, "Khabib Nurmagomedov", "Tony Ferguson"))
# [[0.698]]  -> about 70% Khabib
```

**Removing corner bias:** because the model was trained on Red-minus-Blue differences, swapping the order of the two fighters doesn't always give exactly mirrored probabilities. To get one consistent answer, we ran the matchup both ways and averaged:

```python
fight1 = predict(model, "Gregor Gillespie", "Kevin Lee")   # 0.638
fight2 = predict(model, "Kevin Lee", "Gregor Gillespie")   # 0.445
fighter1_win = (fight1 + (1 - fight2)) / 2                  # 0.596 -> Gillespie ~60%
```

The original model powered a small **Flask** website where users could choose any two rostered fighters and see the prediction.

---

## 7. The 2026 refresh: data through 2026

The original data stopped in 2019, so I rebuilt the pipeline to bring in every UFC fight through 2026 and keep it current. This happened in two stages.

### Stage 1: scraping ufcstats.com directly (early 2026)

- **Fighter scraper:** rewrote `UFC_scraping.ipynb` to use `requests` + BeautifulSoup in place of Selenium/ChromeDriver.
- **Fight-history scraper (`scrape_ufc_matches.py`):** replaced the 2019 Kaggle file by scraping every completed event and bout from ufcstats.com myself, producing an up-to-date `matches_clean.csv` (`Red`, `Blue`, `Winner`, `Weight Class`).
- **Cleaning fixes:** corrected record parsing, height/reach/weight conversion and the age calculation.
- **Model code:** moved the notebook to TensorFlow 2.x and pulled the prediction logic into a reusable module, `ufc_predictor.py`, with handling for fighters who share a name.

### Stage 2: switching data sources and rebuilding the training set (fall 2026)

In September 2026 ufcstats.com started serving a browser check that blocks scripts, so the scrapers could no longer pull data. I switched to the public [Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats) dataset, which mirrors ufcstats.com and is updated almost daily, and wrote `scripts/build_from_greco.py` to turn it into the files the model uses:

- **Career stats from round-by-round data:** SLpM, striking accuracy, SApM, striking defense, takedown average/accuracy/defense and submission average are computed from per-round fight totals using ufcstats.com's own definitions, instead of being read off fighter pages.
- **Name matching:** some fighters are spelled differently on their profile than on fight cards (`Waldo Cortes-Acosta` vs `Waldo Cortes Acosta`, `Levi Rodrigues` vs `Levi Rodrigues Jr.`). The build normalizes accents, punctuation and suffixes and maps profiles to the fight-card spelling, so predictions work with the names on the card.
- **Missing data:** fighters who appear in fights but have no profile still get a row, so their fights aren't dropped; missing weight or reach stays blank rather than 0 (a 0 created fake 70-inch reach gaps); a small `height_overrides.csv` fills in heights the site lists as `--`.
- **Point-in-time features:** the original approach used each fighter's *current* career stats, which already include the fight being predicted. The rebuilt training set (`training_pit.csv`) computes every fight's features from **only the UFC fights before it**: record, win percentage, total fights, striking and grappling rates, and age on fight night.
- **Corner order:** draws and no-contests are excluded, and Red/Blue order is randomized per fight (seeded, so it's reproducible) so the model can't learn which corner tends to win. The red-corner win rate drops to about 51%.
- **Freshness check:** the build stops if the newest event in the dataset is more than 45 days old, so a stalled source never silently produces a stale model.

The result covers **8,768 decided fights from March 1994 through October 2026**, with **4,635 fighters** in the lookup table.

### Retraining and evaluation (`scripts/train_model.py`)

- **Time-based test set:** train on everything up to 12 months ago and test on the most recent 12 months of fights, which the model never sees. This mirrors how the model is actually used: predicting fights that haven't happened yet.
- **Model comparison:** the original eight-hidden-layer network is trained alongside a logistic regression and three smaller networks (3 layers, 3 layers with dropout, and a wider 3-layer net). Each uses early stopping on the most recent 15% of the training period.
- **Calibration:** raw network probabilities were overconfident, so each model gets a temperature scale fit on the validation fights, and predictions are averaged across both corner orders (the same trick as the original).
- **Model selection:** the model with the lowest test log-loss is saved as `ufc_model_best.h5`; then every model is retrained on all fights for live use.

Latest results (trained through Oct 2025, tested on 553 fights from Oct 2025 to Oct 2026):

| Model | Test accuracy | Log-loss |
|---|---|---|
| Baseline: pick the fighter with the better UFC win % | 56.2% | n/a |
| Logistic regression (selected as best) | 61.3% | 0.644 |
| 3-layer network | 61.8% | 0.646 |
| 3-layer network + dropout | 60.9% | 0.649 |
| Wide 3-layer network | 62.0% | 0.647 |
| Original 8-hidden-layer network | 47.7% | 0.693 |

On this cleaner, time-ordered data the deep original network collapses to predicting about 50% for everything, while simpler models do best. When the selected model is at least 65% confident, it is right about 74% of the time.

### Keeping it current

- **Monthly refresh (`scripts/monthly_refresh.sh`):** a macOS `launchd` job pulls the latest dataset, rebuilds the data, retrains every model and writes `model_metrics.txt`. It backs up the previous data and models first and restores them if any step fails.
- **Live tracking (`scripts/track_predictions.py`):** before each card I log the model's picks, then score them against the real results afterwards. The log and running tally are kept locally in `tracking/` (not committed).

## Tools used

**Original:** Python, pandas, NumPy, BeautifulSoup, Splinter/Selenium, requests, scikit-learn, Keras (TensorFlow backend), Jupyter, Flask.

**Refresh:** TensorFlow 2.x / Keras, scikit-learn, pandas, requests, Bash, macOS launchd.

## Repository contents

| File | Purpose |
|---|---|
| `UFC_scraping.ipynb` | Scrapes fighter profiles from ufcstats.com |
| `UFC_Cleaning.ipynb` | Cleans and converts the scraped fighter stats |
| `UFC_Machine_Learning_Model.ipynb` | Builds the stat-differential training set, trains and evaluates the network, makes predictions |
| `scrape_ufc_matches.py` | Refresh stage 1: scrapes every UFC fight result from ufcstats.com |
| `ufc_predictor.py` | Reusable prediction module (loads the best model, scaler and calibration) |
| `scripts/build_from_greco.py` | Refresh stage 2: builds fighter stats and point-in-time training data from the Greco1899 dataset |
| `scripts/train_model.py` | Time-based evaluation, model comparison, calibration and retraining |
| `scripts/monthly_refresh.sh` | Monthly download → build → retrain job, with backups |
| `scripts/track_predictions.py` | Logs pre-fight picks and scores them against results |
| `requirements.txt` | Dependencies |
| `archive/` | Helper and debug scripts from stage 1 of the refresh |

Generated data and model files (`*.csv`, `*.h5`, `scaler.pkl`) are not committed. To rebuild them, clone the Greco1899 dataset into `data/scrape_ufc_stats/` and run `scripts/build_from_greco.py`, then `scripts/train_model.py`.
