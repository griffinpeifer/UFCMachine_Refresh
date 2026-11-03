# UFCMachine

A UFC fight outcome predictor built on a deep learning neural network. Give it two fighters' names and it returns the probability that each one wins, based on how their career statistics compare.

We built the original version in fall 2019 ([original repo](https://github.com/griffinpeifer/UFCMachine)): we found the historical fight data, scraped fighter stats from the web, cleaned and joined the two datasets, and trained a Keras neural network that reached **71.35% accuracy** on a held-out test set. This repo (`UFCMachine_Refresh`) is my own copy of that project, which I am modernizing and retraining on current data.

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

## Tools used

Python, pandas, NumPy, BeautifulSoup, Splinter/Selenium, requests, scikit-learn, Keras (TensorFlow backend), Jupyter, Flask.

## Repository contents

| File | Purpose |
|---|---|
| `UFC_scraping.ipynb` | Scrapes fighter profiles from ufcstats.com |
| `UFC_Cleaning.ipynb` | Cleans and converts the scraped fighter stats |
| `UFC_Machine_Learning_Model.ipynb` | Builds the stat-differential training set, trains and evaluates the network, makes predictions |
| `scrape_ufc_matches.py`, `ufc_predictor.py`, `requirements.txt` | Added in my refresh: updated scraper, reusable prediction module and dependencies |
| `archive/` | Helper and debug scripts from my refresh |

Generated data and model files (`*.csv`, `*.h5`, `scaler.pkl`) are not committed; running the notebooks in order (scraping → cleaning → model) recreates them.
