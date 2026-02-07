# Safe Model Experimentation in Jupyter Notebook

## Quick Start: Copy-Paste Cell

I've created a complete experimentation cell in `EXPERIMENTATION_CELL.py`. Here's how to use it:

### Option 1: Copy the Cell Code
1. Open `EXPERIMENTATION_CELL.py`
2. Copy all the code
3. Paste into a **new cell** in your notebook (after your training cell)
4. Run it!

### Option 2: Step-by-Step in Separate Cells

Add these cells one at a time to experiment safely:

---

## Cell 1: Load Original Model (for comparison)

```python
# Load your original model to compare
from tensorflow.keras.models import load_model

try:
    original_model = load_model('ufc_prediction_model.h5')
    original_test_results = original_model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"✓ Original Model - Test Accuracy: {original_test_results[1]:.4f} ({original_test_results[1]*100:.2f}%)")
    print(f"  Original model loaded and ready for comparison")
except:
    print("⚠ Original model not found - that's okay!")
    original_model = None
```

---

## Cell 2: Experiment with New Architecture

```python
# EXPERIMENT: Simpler model (3 layers with dropout)
# This creates a NEW model - your original is safe!

from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Create new model with different architecture
model_experiment = Sequential([
    Dense(64, input_dim=X_train_scaled.shape[1], activation='relu', kernel_regularizer=regularizers.L2(0.01)),
    Dropout(0.3),  # Add dropout
    Dense(32, activation='relu', kernel_regularizer=regularizers.L2(0.01)),
    Dropout(0.3),
    Dense(16, activation='relu', kernel_regularizer=regularizers.L2(0.01)),
    Dense(1, activation='sigmoid')
])

model_experiment.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

# Train the experimental model
X_train_fit, X_val, y_train_fit, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history_exp = model_experiment.fit(
    X_train_fit, y_train_fit,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=64,
    verbose=1,
    callbacks=[early_stopping]
)

# Evaluate experimental model
test_results_exp = model_experiment.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\n✓ Experimental Model - Test Accuracy: {test_results_exp[1]:.4f} ({test_results_exp[1]*100:.2f}%)")
print(f"  Test Loss: {test_results_exp[0]:.4f}")

# Save with DIFFERENT filename (so original is safe!)
model_experiment.save('ufc_model_experiment.h5')
print(f"  Saved as: ufc_model_experiment.h5 (original model unchanged)")
```

---

## Cell 3: Compare Models

```python
# Compare original vs experimental
print("="*60)
print("MODEL COMPARISON")
print("="*60)

if original_model:
    print(f"Original Model:     {original_test_results[1]*100:.2f}% accuracy")
    print(f"Experimental Model:  {test_results_exp[1]*100:.2f}% accuracy")
    print()
    
    if test_results_exp[1] > original_test_results[1]:
        print("🏆 Experimental model is BETTER!")
        print("   You can use it by loading: model = load_model('ufc_model_experiment.h5')")
    else:
        print("✓ Original model is still better")
        print("   Keep using: ufc_prediction_model.h5")
else:
    print(f"Experimental Model:  {test_results_exp[1]*100:.2f}% accuracy")
```

---

## Key Safety Practices

### ✅ DO:
1. **Use different variable names** - `model_experiment` instead of `model`
2. **Save with different filenames** - `ufc_model_experiment.h5` instead of `ufc_prediction_model.h5`
3. **Load original first** - Compare before deciding
4. **Test in new cells** - Don't modify your original training cell

### ❌ DON'T:
1. **Don't overwrite** `ufc_prediction_model.h5` - Keep it safe!
2. **Don't reuse variable names** - Use `model_exp1`, `model_exp2`, etc.
3. **Don't modify original cell** - Create new cells for experiments

---

## Multiple Experiments Template

If you want to test many architectures:

```python
# Store all results
results = []

# Experiment 1
model_exp1 = Sequential([...])
# ... train and evaluate ...
results.append({'name': 'Exp 1', 'model': model_exp1, 'accuracy': acc1, 'filename': 'exp1.h5'})

# Experiment 2
model_exp2 = Sequential([...])
# ... train and evaluate ...
results.append({'name': 'Exp 2', 'model': model_exp2, 'accuracy': acc2, 'filename': 'exp2.h5'})

# Compare all
results_sorted = sorted(results, key=lambda x: x['accuracy'], reverse=True)
for i, r in enumerate(results_sorted, 1):
    print(f"{i}. {r['name']}: {r['accuracy']*100:.2f}% - {r['filename']}")
```

---

## Quick Reference: Model Files

After experimenting, you'll have:
- `ufc_prediction_model.h5` - Your original (safe!)
- `ufc_model_experiment.h5` - Experimental model
- `ufc_model_exp1_simple.h5` - If you run the full script
- `ufc_model_exp2_dropout.h5` - If you run the full script
- etc.

To use a specific model later:
```python
from tensorflow.keras.models import load_model
model = load_model('ufc_model_experiment.h5')  # or whichever performed best
```

---

## Pro Tip: Create a Comparison Function

Add this helper function to easily compare models:

```python
def compare_models(model1, model1_name, model2, model2_name, X_test, y_test):
    """Compare two models side-by-side"""
    results1 = model1.evaluate(X_test, y_test, verbose=0)
    results2 = model2.evaluate(X_test, y_test, verbose=0)
    
    print(f"{'Model':<30} {'Accuracy':<15} {'Loss':<12}")
    print("-" * 60)
    print(f"{model1_name:<30} {results1[1]*100:>6.2f}%        {results1[0]:>6.4f}")
    print(f"{model2_name:<30} {results2[1]*100:>6.2f}%        {results2[0]:>6.4f}")
    
    if results1[1] > results2[1]:
        print(f"\n🏆 {model1_name} is better!")
    elif results2[1] > results1[1]:
        print(f"\n🏆 {model2_name} is better!")
    else:
        print(f"\n✓ Models are similar")

# Usage:
# compare_models(original_model, "Original", model_experiment, "Experimental", X_test_scaled, y_test)
```
