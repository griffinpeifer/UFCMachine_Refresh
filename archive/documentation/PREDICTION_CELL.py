"""
PREDICTION CELL - Copy this into a new Jupyter notebook cell
Automatically finds and uses your best model to predict fight outcomes
"""

# ============================================================================
# AUTO-DETECT BEST MODEL AND SETUP PREDICTION FUNCTION
# ============================================================================

from tensorflow.keras.models import load_model
import os
import glob

# Method 1: Check if best_model variable exists from experimentation cell
try:
    if 'best_model' in globals() and best_model is not None:
        print("✓ Using best_model from experimentation cell")
        model_to_use = best_model
        model_name = best_model_name if 'best_model_name' in globals() else "best_model (from experiments)"
    else:
        raise NameError("best_model not found")
except:
    # Method 2: Find all experiment models and test them
    print("Searching for available models...")
    
    # List of possible model files (in order of preference)
    # Experiment 1 (Simple) is the best, so prioritize it
    model_files = [
        'ufc_model_exp1_simple.h5',  # Experiment 1 - Best model!
        'ufc_model_exp3_wide.h5',
        'ufc_model_exp2_dropout.h5',
        'ufc_prediction_model.h5',  # Original model
        'best_ufc_model.h5',  # From experiment_models.py script
    ]
    
    # Also check for any experiment models
    experiment_models = glob.glob('ufc_model_exp*.h5')
    model_files.extend(experiment_models)
    
    # Remove duplicates while preserving order
    seen = set()
    model_files = [f for f in model_files if f not in seen and not seen.add(f)]
    
    # Find existing models
    existing_models = [f for f in model_files if os.path.exists(f)]
    
    if not existing_models:
        print("❌ Error: No model files found!")
        print("   Make sure you've trained at least one model.")
        raise FileNotFoundError("No model files found")
    
    print(f"Found {len(existing_models)} model(s):")
    for m in existing_models:
        print(f"  - {m}")
    
    # If we have X_test_scaled and y_test, evaluate all and pick best
    try:
        if 'X_test_scaled' in globals() and 'y_test' in globals():
            print("\nEvaluating models to find the best one...")
            best_accuracy = 0
            best_model_file = None
            
            for model_file in existing_models:
                try:
                    test_model = load_model(model_file)
                    results = test_model.evaluate(X_test_scaled, y_test, verbose=0)
                    accuracy = results[1]
                    print(f"  {model_file}: {accuracy*100:.2f}% accuracy")
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model_file = model_file
                        model_to_use = test_model
                except Exception as e:
                    print(f"  ⚠ Could not evaluate {model_file}: {e}")
                    continue
            
            if best_model_file:
                print(f"\n✓ Best model: {best_model_file} ({best_accuracy*100:.2f}% accuracy)")
                model_name = best_model_file
            else:
                # Fallback to first available model
                model_to_use = load_model(existing_models[0])
                model_name = existing_models[0]
                print(f"\n✓ Using: {model_name}")
        else:
            # No test data available, prioritize Experiment 1 (best model)
            if 'ufc_model_exp1_simple.h5' in existing_models:
                model_to_use = load_model('ufc_model_exp1_simple.h5')
                model_name = 'ufc_model_exp1_simple.h5'
                print(f"\n✓ Using Experiment 1 (Simple) - Best model!")
            elif 'ufc_model_exp3_wide.h5' in existing_models:
                model_to_use = load_model('ufc_model_exp3_wide.h5')
                model_name = 'ufc_model_exp3_wide.h5'
            elif 'ufc_model_exp2_dropout.h5' in existing_models:
                model_to_use = load_model('ufc_model_exp2_dropout.h5')
                model_name = 'ufc_model_exp2_dropout.h5'
            else:
                model_to_use = load_model(existing_models[0])
                model_name = existing_models[0]
            print(f"\n✓ Using: {model_name}")
    except Exception as e:
        print(f"⚠ Could not auto-detect best model: {e}")
        print(f"   Using first available: {existing_models[0]}")
        model_to_use = load_model(existing_models[0])
        model_name = existing_models[0]

# Store as best_model for consistency
best_model = model_to_use
print(f"✓ Model loaded and ready: {model_name}\n")

# Load scaler (should already exist from training)
import pickle
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("✓ Loaded scaler")

# Make sure df_fighters is loaded (should be from earlier cells)
# If not, uncomment:
# df_fighters = pd.read_csv('fighters_clean.csv', index_col='Name')

print("✓ Ready for predictions!\n")

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_fight(fighter1, fighter2):
    """
    Predict fight outcome between two fighters.
    
    Args:
        fighter1: Name of first fighter (Red corner)
        fighter2: Name of second fighter (Blue corner)
    
    Returns:
        Probability of fighter1 winning (0-1)
    """
    # Get stats - MUST match exactly what was used in training (14 columns)
    # Training used these columns in this exact order:
    training_columns = [
        'Weight:', 'Age', 'SLpM:', 'Str. Acc.:', 'SApM:', 'Str. Def:', 
        'TD Avg.:', 'TD Acc.:', 'TD Def.:', 'Sub. Avg.:', 'Reach:', 
        'Win Percentage', 'Total Fights', 'Height:_inches'
    ]
    
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
    
    # Calculate Total Fights if needed (Wins + Losses + Draws)
    if 'Total Fights' not in df_fighters.columns:
        if all(col in df_fighters.columns for col in ['Wins', 'Losses', 'Draws']):
            df_fighters['Total Fights'] = df_fighters['Wins'] + df_fighters['Losses'] + df_fighters['Draws']
        else:
            raise ValueError("Cannot calculate 'Total Fights': Missing Wins, Losses, or Draws columns")
    
    # Use only the exact columns from training, in the same order
    stats = [col for col in training_columns if col in df_fighters.columns]
    
    # Verify we have all 14 columns
    if len(stats) != 14:
        missing = [col for col in training_columns if col not in stats]
        raise ValueError(f"Expected 14 features, got {len(stats)}. Missing: {missing}")
    
    # Check if fighters exist
    if fighter1 not in df_fighters.index:
        print(f"❌ Error: '{fighter1}' not found in database")
        return None
    if fighter2 not in df_fighters.index:
        print(f"❌ Error: '{fighter2}' not found in database")
        return None
    
    # Get fighter stats
    data1 = df_fighters.loc[fighter1, stats]
    data2 = df_fighters.loc[fighter2, stats]
    
    # Handle duplicate indices - if loc returns DataFrame, take first row
    if isinstance(data1, pd.DataFrame):
        data1 = data1.iloc[0]
    if isinstance(data2, pd.DataFrame):
        data2 = data2.iloc[0]
    
    # Calculate difference (fighter1 - fighter2)
    data_diff = (data1 - data2).values.reshape(1, -1)
    
    # Scale using the same scaler from training
    data_diff_scaled = scaler.transform(data_diff)
    
    # Predict
    prediction = best_model.predict(data_diff_scaled, verbose=0)
    probability = prediction[0][0]
    
    return probability

# ============================================================================
# EXAMPLE PREDICTIONS
# ============================================================================

print("="*70)
print("EXAMPLE PREDICTIONS")
print("="*70)

# Example 1
fighter1 = "Khabib Nurmagomedov"
fighter2 = "Tony Ferguson"
prob = predict_fight(fighter1, fighter2)
if prob is not None:
    print(f"\n{fighter1} vs {fighter2}")
    print(f"  {fighter1} win probability: {prob:.2%}")
    print(f"  {fighter2} win probability: {(1-prob):.2%}")
    winner = fighter1 if prob > 0.5 else fighter2
    confidence = prob if prob > 0.5 else (1-prob)
    print(f"  🏆 Predicted Winner: {winner} ({confidence:.2%} confidence)")

# Example 2
fighter1 = "Jon Jones"
fighter2 = "Daniel Cormier"
prob = predict_fight(fighter1, fighter2)
if prob is not None:
    print(f"\n{fighter1} vs {fighter2}")
    print(f"  {fighter1} win probability: {prob:.2%}")
    print(f"  {fighter2} win probability: {(1-prob):.2%}")
    winner = fighter1 if prob > 0.5 else fighter2
    confidence = prob if prob > 0.5 else (1-prob)
    print(f"  🏆 Predicted Winner: {winner} ({confidence:.2%} confidence)")

print("\n" + "="*70)
print("💡 TIP: Use predict_fight('Fighter 1', 'Fighter 2') to predict any fight!")
print(f"💡 Currently using model: {model_name}")
print("="*70)
