# Neural Network Architecture Experimentation Guide

## Current Model Analysis

Your current model has:
- **9 layers** (16→32→32→32→32→32→32→16→1)
- **Test Accuracy: ~67.8%**
- **L2 Regularization: 0.01**
- **No Dropout**

This is quite deep! You might be able to simplify and get similar or better results.

## Key Concepts

### 1. Number of Layers

**Shallow Networks (2-3 layers)**
- ✅ Faster to train
- ✅ Less prone to overfitting
- ✅ Easier to interpret
- ❌ May underfit complex patterns

**Deep Networks (4+ layers)**
- ✅ Can learn complex patterns
- ✅ More expressive
- ❌ Slower to train
- ❌ More prone to overfitting
- ❌ Harder to tune

**For your dataset (~8K samples):** 3-4 layers is usually optimal

### 2. Layer Sizes

**Wide Layers (64-128 neurons)**
- More capacity to learn
- Can memorize training data
- Risk of overfitting

**Narrow Layers (16-32 neurons)**
- Forces generalization
- Less prone to overfitting
- May underfit

**Common Pattern:** Start wide, get narrower
- Example: `[64, 32, 16]` or `[128, 64, 32]`

### 3. Regularization Techniques

**L2 Regularization**
- Penalizes large weights
- Prevents overfitting
- Typical values: 0.001 to 0.1
- Your current: 0.01 (good!)

**Dropout**
- Randomly disables neurons during training
- Forces model to be robust
- Typical values: 0.2 to 0.5
- Your current: None (try adding it!)

**Both Together**
- Often works best
- L2: 0.01, Dropout: 0.3 is a good starting point

## How to Decide on Architecture

### Step 1: Start Simple
Begin with a simple architecture:
```python
layers = [64, 32, 16]  # 3 layers
dropout = None
l2_reg = 0.01
```

### Step 2: Check for Underfitting
**Signs:**
- Low training accuracy (< 60%)
- Low test accuracy
- Model can't learn patterns

**Solutions:**
- Add more layers
- Increase layer sizes
- Reduce regularization

### Step 3: Check for Overfitting
**Signs:**
- High training accuracy (> 80%)
- Lower test accuracy
- Large gap between train/test

**Solutions:**
- Add dropout (0.3)
- Increase L2 regularization
- Reduce layer sizes
- Add more training data

### Step 4: Use Validation Set
- Monitor `val_loss` during training
- Early stopping will help automatically
- If validation loss increases, you're overfitting

## Experimentation Script

I've created `experiment_models.py` that tests 8 different architectures:

1. **Small (2 layers)** - [32, 16]
2. **Medium (3 layers)** - [64, 32, 16]
3. **Large (4 layers)** - [128, 64, 32, 16]
4. **Deep (5 layers)** - [64, 64, 64, 32, 16]
5. **With Dropout** - [64, 32, 16] + 0.3 dropout
6. **No Regularization** - [64, 32, 16] no L2
7. **Wide Layers** - [128, 128, 64]
8. **Narrow Layers** - [16, 8, 4]

### To Run:
```bash
cd /Users/griffinp/UFCMachine_Refresh
source venv/bin/activate
python3 experiment_models.py
```

This will:
- Train all 8 models
- Compare their test accuracies
- Save the best one as `best_ufc_model.h5`
- Show you a ranked comparison

## Quick Experiments in Jupyter

You can also experiment directly in your notebook. Here are some quick tests:

### Experiment 1: Simpler Model
```python
model = Sequential([
    Dense(64, input_dim=X_train_scaled.shape[1], activation='relu', kernel_regularizer=regularizers.L2(0.01)),
    Dense(32, activation='relu', kernel_regularizer=regularizers.L2(0.01)),
    Dense(16, activation='relu', kernel_regularizer=regularizers.L2(0.01)),
    Dense(1, activation='sigmoid')
])
```

### Experiment 2: Add Dropout
```python
model = Sequential([
    Dense(64, input_dim=X_train_scaled.shape[1], activation='relu', kernel_regularizer=regularizers.L2(0.01)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=regularizers.L2(0.01)),
    Dropout(0.3),
    Dense(16, activation='relu', kernel_regularizer=regularizers.L2(0.01)),
    Dense(1, activation='sigmoid')
])
```

### Experiment 3: Different Optimizer
```python
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Lower learning rate
    metrics=['binary_accuracy']
)
```

### Experiment 4: Different Batch Size
```python
history = model.fit(
    X_train_fit, y_train_fit,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,  # Smaller batch size (default was 64)
    verbose=1,
    callbacks=[early_stopping]
)
```

## What to Look For

### Good Model:
- ✅ Test accuracy > 65% (you're at 67.8% - good!)
- ✅ Training and test accuracy are close (< 5% gap)
- ✅ Validation loss decreases and stabilizes
- ✅ Model trains in reasonable time

### Overfitting:
- ❌ Training accuracy >> test accuracy (e.g., 85% vs 65%)
- ❌ Validation loss increases after initial decrease
- **Fix:** Add dropout, increase L2, reduce layers

### Underfitting:
- ❌ Both training and test accuracy are low (< 60%)
- ❌ Model can't learn patterns
- **Fix:** Add layers, increase layer sizes, reduce regularization

## Recommended Starting Points

### For Your Dataset (~8K samples, 14 features):

**Option 1: Simple & Fast**
```python
layers = [64, 32, 16]
dropout = 0.3
l2_reg = 0.01
```

**Option 2: Balanced**
```python
layers = [128, 64, 32, 16]
dropout = 0.2
l2_reg = 0.01
```

**Option 3: Conservative (prevent overfitting)**
```python
layers = [32, 16]
dropout = 0.4
l2_reg = 0.05
```

## Tips

1. **Change one thing at a time** - Understand what each change does
2. **Use early stopping** - Already implemented, prevents overfitting
3. **Monitor validation loss** - Best indicator of model health
4. **Save good models** - Compare them later
5. **Document experiments** - Keep track of what works

## Expected Results

For UFC fight prediction:
- **Baseline (random):** 50%
- **Good model:** 65-70%
- **Excellent model:** 70-75%
- **Your current:** 67.8% ✅

You're already doing well! Experimentation might get you to 70%+.
