# How to Use Your UFC Prediction Model

## 🎯 Best Method: Jupyter Notebook (Recommended)

This is the easiest and most flexible way to use your model.

### Step 1: Run the Prediction Cell

If you haven't already, add and run the `PREDICTION_CELL.py` code in a new notebook cell. This will:
- Load your best model (Experiment 1)
- Load the scaler
- Create the `predict_fight()` function

### Step 2: Make Predictions

Simply call the function with two fighter names:

```python
# Single prediction
prob = predict_fight("Khabib Nurmagomedov", "Tony Ferguson")
print(f"Khabib win probability: {prob:.2%}")
```

### Step 3: Get Formatted Results

For a nicer output:

```python
def predict_and_display(fighter1, fighter2):
    """Predict and display formatted results"""
    prob = predict_fight(fighter1, fighter2)
    if prob is not None:
        winner = fighter1 if prob > 0.5 else fighter2
        confidence = prob if prob > 0.5 else (1-prob)
        
        print("="*60)
        print(f"{fighter1} vs {fighter2}")
        print("="*60)
        print(f"{fighter1:30} {prob:>6.2%}")
        print(f"{fighter2:30} {(1-prob):>6.2%}")
        print("-"*60)
        print(f"🏆 Predicted Winner: {winner}")
        print(f"   Confidence: {confidence:.1%}")
        print("="*60)
        return winner, confidence
    return None, None

# Use it:
predict_and_display("Jon Jones", "Daniel Cormier")
```

---

## 📊 Method 2: Batch Predictions

Predict multiple fights at once:

```python
# List of fights to predict
upcoming_fights = [
    ("Israel Adesanya", "Alex Pereira"),
    ("Jon Jones", "Stipe Miocic"),
    ("Amanda Nunes", "Valentina Shevchenko"),
    ("Khabib Nurmagomedov", "Tony Ferguson"),
]

print("="*70)
print("UPCOMING FIGHTS PREDICTIONS")
print("="*70)
print(f"{'Fighter 1':<30} {'Fighter 2':<30} {'Predicted Winner':<30} {'Confidence':<10}")
print("-"*70)

for f1, f2 in upcoming_fights:
    prob = predict_fight(f1, f2)
    if prob is not None:
        winner = f1 if prob > 0.5 else f2
        confidence = prob if prob > 0.5 else (1-prob)
        print(f"{f1:<30} {f2:<30} {winner:<30} {confidence:>8.1%}")
```

---

## 💻 Method 3: Command Line Script

For quick predictions from terminal:

```bash
cd /Users/griffinp/UFCMachine_Refresh
source venv/bin/activate
python3 predict_fight.py "Khabib Nurmagomedov" "Tony Ferguson"
```

**Output:**
```
Predicting: Khabib Nurmagomedov vs Tony Ferguson
==================================================
Loaded model from ufc_model_exp1_simple.h5
Loaded scaler from scaler.pkl
Loaded 4444 fighters from fighters_clean.csv

==================================================
Prediction Result:
  Khabib Nurmagomedov win probability: 65.23%
  Tony Ferguson win probability: 34.77%
==================================================

Predicted Winner: Khabib Nurmagomedov (65.23% confidence)
```

---

## 🔍 Method 4: Interactive Search

Find fighters and predict:

```python
# Search for fighters
def find_fighters(search_term):
    """Find fighters matching search term"""
    matches = [f for f in df_fighters.index if search_term.lower() in f.lower()]
    return matches

# Example: Find fighters with "Jones" in name
jones_fighters = find_fighters("Jones")
print("Fighters with 'Jones':")
for f in jones_fighters:
    print(f"  - {f}")

# Predict a fight
if len(jones_fighters) > 0:
    prob = predict_fight(jones_fighters[0], "Daniel Cormier")
    print(f"\n{jones_fighters[0]} vs Daniel Cormier: {prob:.2%}")
```

---

## 📈 Method 5: Compare Multiple Models

If you want to see predictions from different models:

```python
from tensorflow.keras.models import load_model

models_to_compare = {
    'Experiment 1 (Simple)': 'ufc_model_exp1_simple.h5',
    'Experiment 2 (Dropout)': 'ufc_model_exp2_dropout.h5',
    'Experiment 3 (Wide)': 'ufc_model_exp3_wide.h5',
}

fighter1 = "Khabib Nurmagomedov"
fighter2 = "Tony Ferguson"

print(f"Comparing predictions: {fighter1} vs {fighter2}")
print("="*70)

for model_name, model_file in models_to_compare.items():
    try:
        model = load_model(model_file)
        stats = [col for col in df_fighters.columns if df_fighters[col].dtype in ['int64', 'float64']]
        
        # Handle Height: if needed
        if 'Height:' in df_fighters.columns and 'Height:_inches' not in df_fighters.columns:
            def height_to_inches(h):
                if pd.isna(h): return np.nan
                try:
                    parts = str(h).replace('"', '').split("'")
                    if len(parts) == 2:
                        return float(parts[0].strip()) * 12 + float(parts[1].strip())
                except: pass
                return np.nan
            df_fighters['Height:_inches'] = df_fighters['Height:'].apply(height_to_inches)
            if 'Height:_inches' not in stats:
                stats.append('Height:_inches')
            if 'Height:' in stats:
                stats.remove('Height:')
        
        data_diff = (df_fighters.loc[fighter1, stats] - df_fighters.loc[fighter2, stats]).values.reshape(1, -1)
        prob = model.predict(scaler.transform(data_diff), verbose=0)[0][0]
        
        winner = fighter1 if prob > 0.5 else fighter2
        print(f"{model_name:30} → {winner:30} ({prob:.1%})")
    except Exception as e:
        print(f"{model_name:30} → Error: {e}")
```

---

## 🎨 Method 6: Create a Simple Prediction Interface

For repeated use, create a helper function:

```python
class UFCPredictor:
    """Simple class to make predictions easier"""
    
    def __init__(self):
        self.model = best_model  # From prediction cell
        self.scaler = scaler
        self.df_fighters = df_fighters
        
    def predict(self, fighter1, fighter2, verbose=True):
        """Predict fight outcome"""
        prob = predict_fight(fighter1, fighter2)
        if prob is None:
            return None
        
        winner = fighter1 if prob > 0.5 else fighter2
        confidence = prob if prob > 0.5 else (1-prob)
        
        if verbose:
            print(f"{fighter1} vs {fighter2}")
            print(f"  Winner: {winner} ({confidence:.1%} confidence)")
        
        return {
            'fighter1': fighter1,
            'fighter2': fighter2,
            'fighter1_prob': prob,
            'fighter2_prob': 1-prob,
            'predicted_winner': winner,
            'confidence': confidence
        }
    
    def predict_batch(self, fights):
        """Predict multiple fights"""
        results = []
        for f1, f2 in fights:
            result = self.predict(f1, f2, verbose=False)
            if result:
                results.append(result)
        return results

# Use it:
predictor = UFCPredictor()

# Single prediction
result = predictor.predict("Khabib Nurmagomedov", "Tony Ferguson")

# Batch predictions
fights = [("Fighter A", "Fighter B"), ("Fighter C", "Fighter D")]
results = predictor.predict_batch(fights)
```

---

## 💡 Pro Tips

### 1. **Check Fighter Names**
Fighter names must match exactly as they appear in your database:
```python
# Check if fighter exists
if "Khabib Nurmagomedov" in df_fighters.index:
    print("Fighter found!")
else:
    print("Fighter not found. Similar names:")
    # Find similar names
    similar = [f for f in df_fighters.index if "Khabib" in f or "Nurmagomedov" in f]
    print(similar)
```

### 2. **Handle Missing Fighters**
```python
def safe_predict(fighter1, fighter2):
    """Predict with error handling"""
    if fighter1 not in df_fighters.index:
        print(f"❌ {fighter1} not found")
        return None
    if fighter2 not in df_fighters.index:
        print(f"❌ {fighter2} not found")
        return None
    return predict_fight(fighter1, fighter2)
```

### 3. **Save Predictions**
```python
import pandas as pd
from datetime import datetime

predictions = []

fights = [
    ("Fighter 1", "Fighter 2"),
    ("Fighter 3", "Fighter 4"),
]

for f1, f2 in fights:
    prob = predict_fight(f1, f2)
    if prob:
        predictions.append({
            'Fighter 1': f1,
            'Fighter 2': f2,
            'Fighter 1 Win Prob': prob,
            'Fighter 2 Win Prob': 1-prob,
            'Predicted Winner': f1 if prob > 0.5 else f2,
            'Confidence': prob if prob > 0.5 else (1-prob),
            'Date': datetime.now().strftime('%Y-%m-%d')
        })

df_predictions = pd.DataFrame(predictions)
df_predictions.to_csv('fight_predictions.csv', index=False)
print("Predictions saved to fight_predictions.csv")
```

---

## 🚀 Quick Start (Copy-Paste Ready)

Here's everything you need in one cell:

```python
# Quick prediction - just change the fighter names!
fighter1 = "Khabib Nurmagomedov"
fighter2 = "Tony Ferguson"

prob = predict_fight(fighter1, fighter2)
if prob:
    winner = fighter1 if prob > 0.5 else fighter2
    confidence = prob if prob > 0.5 else (1-prob)
    
    print(f"\n{'='*60}")
    print(f"{fighter1} vs {fighter2}")
    print(f"{'='*60}")
    print(f"{fighter1}: {prob:.1%}")
    print(f"{fighter2}: {(1-prob):.1%}")
    print(f"{'='*60}")
    print(f"🏆 Predicted Winner: {winner}")
    print(f"   Confidence: {confidence:.1%}")
    print(f"{'='*60}\n")
```

---

## 📝 Summary

**Best for daily use:** Jupyter Notebook with `predict_fight()` function  
**Best for quick checks:** Command line script  
**Best for multiple fights:** Batch prediction function  
**Best for sharing:** Save predictions to CSV

The model is ready to use - just call `predict_fight("Fighter 1", "Fighter 2")`!
