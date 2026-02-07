# Clean Notebook Usage - Keep Your Notebook Tidy!

## 🎯 Best Method: Import the Module

Instead of pasting code into your notebook, use the clean module:

### Step 1: Import (One Line!)

Add this single cell at the top of your notebook (or in a setup cell):

```python
from ufc_predictor import predict_fight, predict_batch
```

That's it! Everything is loaded automatically.

### Step 2: Use It

```python
# Single prediction
predict_fight("Khabib Nurmagomedov", "Tony Ferguson")
```

**Output:**
```
Khabib Nurmagomedov vs Tony Ferguson
============================================================
Khabib Nurmagomedov                   65.23%
Tony Ferguson                         34.77%
------------------------------------------------------------
🏆 Predicted Winner: Khabib Nurmagomedov
   Confidence: 65.23%
============================================================
```

### Step 3: Batch Predictions

```python
fights = [
    ("Jon Jones", "Daniel Cormier"),
    ("Israel Adesanya", "Alex Pereira"),
    ("Amanda Nunes", "Valentina Shevchenko"),
]

predict_batch(fights)
```

---

## 📝 Minimal Notebook Setup

Here's all you need in your notebook:

**Cell 1: Setup (run once)**
```python
from ufc_predictor import predict_fight, predict_batch
```

**Cell 2: Make Predictions**
```python
# Predict any fight
predict_fight("Fighter 1", "Fighter 2")
```

That's it! Your notebook stays clean.

---

## 🚀 Alternative: Command Line Only

If you want to keep your notebook completely clean, use the command line:

```bash
python3 ufc_predictor.py "Khabib Nurmagomedov" "Tony Ferguson"
```

No notebook code needed!

---

## 💡 Pro Tips

### Silent Mode (Just Get Probability)

```python
prob = predict_fight("Fighter 1", "Fighter 2", verbose=False)
print(f"Win probability: {prob:.2%}")
```

### Check Model Info

```python
from ufc_predictor import get_model_info
info = get_model_info()
print(f"Using model: {info['model_file']}")
print(f"Fighters in database: {info['num_fighters']}")
```

### Save Predictions to CSV

```python
import pandas as pd
from ufc_predictor import predict_batch

fights = [
    ("Fighter 1", "Fighter 2"),
    ("Fighter 3", "Fighter 4"),
]

results = predict_batch(fights, verbose=False)
df = pd.DataFrame(results)
df.to_csv('predictions.csv', index=False)
```

---

## 📋 Summary

**Cleanest approach:**
1. Import: `from ufc_predictor import predict_fight`
2. Use: `predict_fight("Fighter 1", "Fighter 2")`
3. Done!

**No need to:**
- ❌ Load models manually
- ❌ Load scalers manually  
- ❌ Handle Height conversion
- ❌ Write prediction logic
- ❌ Clutter your notebook

Everything is handled automatically by the module!
