# UFCMachine_Refresh
Creating a new copy of UFC Machine to modernize and retrain the project with current data through 2026.


# UFCMachine

UFCMachine is a site which offers UFC fight predictions based on a deep learning model. The model is designed to predict the outcome of future fights based on the historical data from all UFC fights between 1993-2026, and allows users to input two fighters of their choosing using a backend dataset of UFC rostered fighters and their statistics.

The model behind UFCMachine utilizes a Keras driven neural network with eight layers, and was trained using the difference in fight statistics between the winner and loser of each fight.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Scrape Fighter Data

Run `UFC_scraping.ipynb` to scrape current fighter statistics from ufcstats.com. This will create `ufc_fighters_raw.csv`.

### 3. Clean Fighter Data

Run `UFC_Cleaning.ipynb` to clean and process the scraped fighter data. This will create `fighters_clean.csv`.

### 4. Download Match Data

You can create the match data by scraping from ufcstats.com:

**Option A: Using the helper script (recommended)**
```bash
python download_match_data.py
```
or directly:
```bash
python scrape_ufc_matches.py
```

This will scrape all UFC fights from ufcstats.com and create `matches_clean.csv` with columns: `Red`, `Blue`, `Winner`, `Weight Class`.

**Note:** This scraping process may take some time (30-60 minutes) depending on your internet connection, as it scrapes all UFC fights since 1993. The script will show progress and you can limit the number of fights for testing purposes.

**Option B: Using the original repository (advanced)**
If you want to use the original [UFC-Predictions repository](https://github.com/WarrierRajeev/UFC-Predictions) which has more detailed preprocessing:
1. Clone the repository: `git clone https://github.com/WarrierRajeev/UFC-Predictions.git`
2. Follow their instructions: `python -m src.create_ufc_data`
3. Process the output to create `matches_clean.csv` with the required columns

### 5. Train the Model

Run `UFC_Machine_Learning_Model.ipynb` to train the neural network model. The trained model will be saved as `ufc_prediction_model.h5`.

## Approach

After finding the dataset https://www.kaggle.com/rajeevw/ufcdata containing all UFC fights since 1993, we created a predictive model which provides the most probable winner based on which statistics were most highly correlated to winning past matches.

### Scraping
-------------
In order to predict future fights based on searching fighter names instead of statistics, we scrape the career stats for every fighter on the UFC roster from http://ufcstats.com/statistics/fighters. These stats include career statistics such as fight record, height, weight, reach, significant strikes landed per minute, and more for each fighter's name. This scraping method can be seen in `UFC_scraping.ipynb`.

![df_fighters](https://github.com/griffinpeifer/UFCMachine/blob/master/df_fighters.PNG)

Cleaning
------------
Once all fighters and their career stats were input into a dataframe, the data was cleaned and all fighters who had missing data fields were eliminated from the dataset, yielding 1672 fighters. This was an acceptable refinement because the fighters who were missing data tended to be older, retired fighters, and as a result they won't have much applicability to a future fight predictor. 

![df_matches with names](https://github.com/griffinpeifer/UFCMachine/blob/master/df_matches_names.PNG)

A value of 1 in the 'result' column indicates a win by the 'Red' fighter.

![df_matches with stat differential](https://github.com/griffinpeifer/UFCMachine/blob/master/df_matches_differential.PNG)

### Model Development
-------------
Once the dataset was scraped and cleaned, we use it to enhance the df_matches dataframe, which contains the winning and losing fighters from all matches since 1993. By replacing the fighters' names in this dataframe with their career stats, we calculate a stat differential between the two fighters in each match, and train our model to learn which statistical categories are most highly correlated with winning. This can be seen in the `UFC_Machine_Learning_Model.ipynb` file.

Using the updated df_matches, an eight-layer neural network was developed which achieved 71.35% test accuracy and is used as a viable prediction model for future fights.

## Project Structure

```
UFCMachine_Refresh/
├── UFC_scraping.ipynb              # Scrape fighter stats from ufcstats.com
├── UFC_Cleaning.ipynb              # Clean and process fighter data
├── UFC_Machine_Learning_Model.ipynb # Train the neural network model
├── scrape_ufc_matches.py           # Script to scrape fight data from ufcstats.com
├── download_match_data.py          # Helper script to create match data (uses scraper)
├── predict_fight.py                # Helper script to predict fight outcomes
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore file for generated data
└── README.md                       # This file
```

## Data Files (Generated)

- `ufc_fighters_raw.csv` - Raw scraped fighter data
- `fighters_clean.csv` - Cleaned fighter statistics
- `matches_clean.csv` - Historical UFC match data (from Kaggle)
- `ufc_prediction_model.h5` - Trained model file
- `scaler.pkl` - Scaler used for feature normalization (saved during training)

## Usage

After training the model, you can use it to predict fight outcomes:

### Using the Prediction Script

The easiest way to make predictions is using the provided script:

```bash
python predict_fight.py "Fighter 1 Name" "Fighter 2 Name"
```

Example:
```bash
python predict_fight.py "Khabib Nurmagomedov" "Tony Ferguson"
```

### Using Python Code

You can also use the model programmatically:

```python
from tensorflow.keras.models import load_model
import pandas as pd
import pickle

# Load the model, scaler, and fighter data
model = load_model('ufc_prediction_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
df_fighters = pd.read_csv('fighters_clean.csv', index_col='Name')

# Predict a fight
def predict_fight(fighter1, fighter2):
    stats = [col for col in df_fighters.columns 
            if df_fighters[col].dtype in ['int64', 'float64']]
    data1 = df_fighters.loc[fighter1, stats]
    data2 = df_fighters.loc[fighter2, stats]
    data_diff = (data1 - data2).values.reshape(1, -1)
    data_diff_scaled = scaler.transform(data_diff)
    prediction = model.predict(data_diff_scaled, verbose=0)
    return prediction[0][0]

# Example
probability = predict_fight("Khabib Nurmagomedov", "Tony Ferguson")
print(f"Probability of fighter 1 winning: {probability:.2%}")
```

## Notes

- The scraping process may take some time as it respects rate limits
- Make sure fighter names in match data match exactly with names in the fighter database
- The model accuracy may vary based on the quality and recency of the training data
