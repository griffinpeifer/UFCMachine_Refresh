"""
SAFE MODEL EXPERIMENTATION CELL
Copy this into a new Jupyter notebook cell to experiment without affecting your original model.

This cell:
- Loads your existing model for comparison
- Creates new experimental models with different names
- Saves experimental models with unique filenames
- Compares all models side-by-side
- Never overwrites your original model
"""

# ============================================================================
# SAFE EXPERIMENTATION - Your original model is safe!
# ============================================================================

# First, load your existing model for comparison (optional - comment out if you haven't trained it yet)
try:
    from tensorflow.keras.models import load_model
    original_model = load_model('ufc_prediction_model.h5')
    print("✓ Loaded original model: ufc_prediction_model.h5")
    # Evaluate original model for comparison
    original_test_results = original_model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"  Original Model - Test Accuracy: {original_test_results[1]:.4f} ({original_test_results[1]*100:.2f}%)")
except:
    print("⚠ Original model not found (that's okay if you haven't trained it yet)")
    original_model = None

print("\n" + "="*80)
print("EXPERIMENTING WITH NEW ARCHITECTURES")
print("="*80)
print()

# Store results for comparison
experiment_results = []

# ============================================================================
# EXPERIMENT 1: Simpler Model (3 layers)
# ============================================================================
print("Experiment 1: Simpler Model (3 layers)")
print("-" * 80)

model_exp1 = Sequential([
    Dense(64, input_dim=X_train_scaled.shape[1], activation='relu', kernel_regularizer=regularizers.L2(0.01)),
    Dense(32, activation='relu', kernel_regularizer=regularizers.L2(0.01)),
    Dense(16, activation='relu', kernel_regularizer=regularizers.L2(0.01)),
    Dense(1, activation='sigmoid')
])

model_exp1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

# Train with validation split
X_train_fit, X_val, y_train_fit, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=0
)

history_exp1 = model_exp1.fit(
    X_train_fit, y_train_fit,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=64,
    verbose=1,  # Set to 0 to hide training progress
    callbacks=[early_stopping]
)

test_results_exp1 = model_exp1.evaluate(X_test_scaled, y_test, verbose=0)
print(f"✓ Test Accuracy: {test_results_exp1[1]:.4f} ({test_results_exp1[1]*100:.2f}%)")
print(f"  Test Loss: {test_results_exp1[0]:.4f}")
print(f"  Epochs Trained: {len(history_exp1.history['loss'])}")

# Save with unique name
model_exp1.save('ufc_model_exp1_simple.h5')
print(f"  Saved as: ufc_model_exp1_simple.h5\n")

experiment_results.append({
    'name': 'Experiment 1: Simple (3 layers)',
    'model': model_exp1,
    'accuracy': test_results_exp1[1],
    'loss': test_results_exp1[0],
    'filename': 'ufc_model_exp1_simple.h5'
})

# ============================================================================
# EXPERIMENT 2: With Dropout
# ============================================================================
print("Experiment 2: With Dropout (3 layers + dropout)")
print("-" * 80)

from tensorflow.keras.layers import Dropout

model_exp2 = Sequential([
    Dense(64, input_dim=X_train_scaled.shape[1], activation='relu', kernel_regularizer=regularizers.L2(0.01)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=regularizers.L2(0.01)),
    Dropout(0.3),
    Dense(16, activation='relu', kernel_regularizer=regularizers.L2(0.01)),
    Dense(1, activation='sigmoid')
])

model_exp2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

history_exp2 = model_exp2.fit(
    X_train_fit, y_train_fit,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=64,
    verbose=1,
    callbacks=[early_stopping]
)

test_results_exp2 = model_exp2.evaluate(X_test_scaled, y_test, verbose=0)
print(f"✓ Test Accuracy: {test_results_exp2[1]:.4f} ({test_results_exp2[1]*100:.2f}%)")
print(f"  Test Loss: {test_results_exp2[0]:.4f}")
print(f"  Epochs Trained: {len(history_exp2.history['loss'])}")

model_exp2.save('ufc_model_exp2_dropout.h5')
print(f"  Saved as: ufc_model_exp2_dropout.h5\n")

experiment_results.append({
    'name': 'Experiment 2: With Dropout',
    'model': model_exp2,
    'accuracy': test_results_exp2[1],
    'loss': test_results_exp2[0],
    'filename': 'ufc_model_exp2_dropout.h5'
})

# ============================================================================
# EXPERIMENT 3: Wider Layers
# ============================================================================
print("Experiment 3: Wider Layers (128→64→32)")
print("-" * 80)

model_exp3 = Sequential([
    Dense(128, input_dim=X_train_scaled.shape[1], activation='relu', kernel_regularizer=regularizers.L2(0.01)),
    Dense(64, activation='relu', kernel_regularizer=regularizers.L2(0.01)),
    Dense(32, activation='relu', kernel_regularizer=regularizers.L2(0.01)),
    Dense(1, activation='sigmoid')
])

model_exp3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

history_exp3 = model_exp3.fit(
    X_train_fit, y_train_fit,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=64,
    verbose=1,
    callbacks=[early_stopping]
)

test_results_exp3 = model_exp3.evaluate(X_test_scaled, y_test, verbose=0)
print(f"✓ Test Accuracy: {test_results_exp3[1]:.4f} ({test_results_exp3[1]*100:.2f}%)")
print(f"  Test Loss: {test_results_exp3[0]:.4f}")
print(f"  Epochs Trained: {len(history_exp3.history['loss'])}")

model_exp3.save('ufc_model_exp3_wide.h5')
print(f"  Saved as: ufc_model_exp3_wide.h5\n")

experiment_results.append({
    'name': 'Experiment 3: Wide Layers',
    'model': model_exp3,
    'accuracy': test_results_exp3[1],
    'loss': test_results_exp3[0],
    'filename': 'ufc_model_exp3_wide.h5'
})

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
print("="*80)
print("COMPARISON SUMMARY")
print("="*80)
print()

# Add original model if available
if original_model:
    experiment_results.insert(0, {
        'name': 'Original Model (9 layers)',
        'model': original_model,
        'accuracy': original_test_results[1],
        'loss': original_test_results[0],
        'filename': 'ufc_prediction_model.h5'
    })

# Sort by accuracy
experiment_results_sorted = sorted(experiment_results, key=lambda x: x['accuracy'], reverse=True)

print(f"{'Rank':<6} {'Model Name':<35} {'Test Accuracy':<15} {'Test Loss':<12} {'Filename':<30}")
print("-" * 100)
for i, result in enumerate(experiment_results_sorted, 1):
    marker = "🏆" if i == 1 else "  "
    print(f"{marker} {i:<4} {result['name']:<35} {result['accuracy']*100:>6.2f}%        {result['loss']:>6.4f}     {result['filename']:<30}")

print()
print("="*80)
print(f"🏆 BEST MODEL: {experiment_results_sorted[0]['name']}")
print(f"   Accuracy: {experiment_results_sorted[0]['accuracy']*100:.2f}%")
print(f"   Saved as: {experiment_results_sorted[0]['filename']}")
print("="*80)
print()
print("💡 TIP: All models are saved with unique names. Your original model is safe!")
print("   To use a different model, just load it:")
print(f"   model = load_model('{experiment_results_sorted[0]['filename']}')")

# Store the best model in a variable for easy access
best_model = experiment_results_sorted[0]['model']
best_model_name = experiment_results_sorted[0]['filename']

print(f"\n✓ Best model stored in variable: best_model")
print(f"✓ Best model filename: {best_model_name}")
