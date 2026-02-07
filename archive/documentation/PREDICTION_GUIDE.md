# Using Your Best Model for Predictions

## Your Best Model: Experiment 3 (Wide Layers)
- **Filename:** `ufc_model_exp3_wide.h5`
- **Architecture:** 128→64→32 neurons
- **Test Accuracy:** (your best result)

---

## Method 1: In Jupyter Notebook (Recommended)

Add this cell to your notebook:

```python
# Load the best model (Experiment 3)
from tensorflow.keras.models import load_model

# Load the best model
best_model = load_model('ufc_model_exp3_wide.h5')
print("✓ Loaded best model: ufc_model_exp3_wide.h5")

# Prediction function
def predict_fight(fighter1, fighter2):
    """
    Predict fight outcome between two fighters.
    
    Args:
        fighter1: Name of first fighter (Red corner)
        fighter2: Name of second fighter (Blue corner)
    
    Returns:
        Probability of fighter1 winning (0-1)
    """
    # Get numeric stats (same as training)
    stats = [col for col in df_fighters.columns if df_fighters[col].dtype in ['int64', 'float64']]
    
    # Handle Height: conversion if needed
    if 'Height:' in df_fighters.columns:
        if 'Height:_inches' not in df_fighters.columns:
            def height_to_inches(height_str):
                if pd.isna(height_str) or height_str == '':
                    return np.nan
                try:
                    parts = str(height_str).replace('"', '').split("'")
                    if len(parts) == 2:
                        feet = float(parts[0].strip())
                        inches = float(parts[1].strip())
                        return feet * 12 + inches
                    return np.nan
                except:
                    return np.nan
            df_fighters['Height:_inches'] = df_fighters['Height:'].apply(height_to_inches)
        if 'Height:_inches' in df_fighters.columns and 'Height:_inches' not in stats:
            stats.append('Height:_inches')
        if 'Height:' in stats:
            stats.remove('Height:')
    
    # Get fighter stats
    data1 = df_fighters.loc[fighter1, stats]
    data2 = df_fighters.loc[fighter2, stats]
    
    # Calculate difference (fighter1 - fighter2)
    data_diff = (data1 - data2).values.reshape(1, -1)
    
    # Scale using the same scaler from training
    data_diff_scaled = scaler.transform(data_diff)
    
    # Predict
    prediction = best_model.predict(data_diff_scaled, verbose=0)
    probability = prediction[0][0]
    
    return probability

# Example predictions
print("\n" + "="*60)
print("EXAMPLE PREDICTIONS")
print("="*60)

# Example 1
fighter1 = "Khabib Nurmagomedov"
fighter2 = "Tony Ferguson"
prob = predict_fight(fighter1, fighter2)
print(f"\n{fighter1} vs {fighter2}")
print(f"  {fighter1} win probability: {prob:.2%}")
print(f"  {fighter2} win probability: {(1-prob):.2%}")
if prob > 0.5:
    print(f"  🏆 Predicted Winner: {fighter1}")
else:
    print(f"  🏆 Predicted Winner: {fighter2}")

# Example 2
fighter1 = "Jon Jones"
fighter2 = "Daniel Cormier"
prob = predict_fight(fighter1, fighter2)
print(f"\n{fighter1} vs {fighter2}")
print(f"  {fighter1} win probability: {prob:.2%}")
print(f"  {fighter2} win probability: {(1-prob):.2%}")
if prob > 0.5:
    print(f"  🏆 Predicted Winner: {fighter1}")
else:
    print(f"  🏆 Predicted Winner: {fighter2}")
```

---

## Method 2: Command Line Script

I've updated `predict_fight.py` to use your best model by default.

### Usage:
```bash
cd /Users/griffinp/UFCMachine_Refresh
source venv/bin/activate
python3 predict_fight.py "Khabib Nurmagomedov" "Tony Ferguson"
```

### To use a different model:
```python
# In the script, change the default:
predict_fight("Fighter 1", "Fighter 2", model_path='ufc_model_exp3_wide.h5')
```

---

## Method 3: Quick Prediction Cell

For quick one-off predictions, add this simple cell:

```python
# Quick prediction helper
def quick_predict(fighter1, fighter2):
    """Quick prediction using the best model"""
    stats = [col for col in df_fighters.columns if df_fighters[col].dtype in ['int64', 'float64']]
    
    # Handle Height: if needed (same as before)
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
    prob = best_model.predict(scaler.transform(data_diff), verbose=0)[0][0]
    
    print(f"{fighter1} vs {fighter2}")
    print(f"  {fighter1}: {prob:.1%}")
    print(f"  {fighter2}: {(1-prob):.1%}")
    print(f"  Winner: {fighter1 if prob > 0.5 else fighter2}")

# Use it:
quick_predict("Khabib Nurmagomedov", "Tony Ferguson")
```

---

## Important Notes

1. **Model File:** Make sure `ufc_model_exp3_wide.h5` exists (it should from your experiment)
2. **Scaler:** You need `scaler.pkl` from your training (should already exist)
3. **Fighter Data:** You need `fighters_clean.csv` (should already exist)
4. **Height Conversion:** The function handles Height: conversion automatically

---

## Troubleshooting

### "Model file not found"
- Make sure you ran the experimentation cell
- Check that `ufc_model_exp3_wide.h5` exists in your directory

### "Fighter not found"
- Check the exact spelling of fighter names
- Names must match exactly as they appear in `fighters_clean.csv`
- Case-sensitive!

### "Shape mismatch" error
- Make sure you're using the same scaler from training
- The model expects the same number of features as training

---

## Pro Tip: Batch Predictions

To predict multiple fights at once:

```python
fights = [
    ("Khabib Nurmagomedov", "Tony Ferguson"),
    ("Jon Jones", "Daniel Cormier"),
    ("Conor McGregor", "Nate Diaz"),
]

print("="*60)
print("BATCH PREDICTIONS")
print("="*60)
for f1, f2 in fights:
    prob = predict_fight(f1, f2)
    winner = f1 if prob > 0.5 else f2
    confidence = prob if prob > 0.5 else (1-prob)
    print(f"{f1:25} vs {f2:25} → {winner:25} ({confidence:.1%})")
```
