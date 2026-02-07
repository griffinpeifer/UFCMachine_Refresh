"""
UFC Fight Predictor Module
Clean, reusable module for predicting UFC fights.

Usage in Jupyter:
    from ufc_predictor import predict_fight
    predict_fight("Fighter 1", "Fighter 2")

Usage from command line:
    python3 ufc_predictor.py "Fighter 1" "Fighter 2"
"""

import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model

# Global variables (loaded on first use)
_model = None
_scaler = None
_df_fighters = None
_model_name = None

def _load_resources():
    """Load model, scaler, and fighter data (lazy loading)"""
    global _model, _scaler, _df_fighters, _model_name
    
    if _model is None:
        # Find best model (prioritize Experiment 1)
        model_files = [
            'ufc_model_exp1_simple.h5',
            'ufc_model_exp3_wide.h5',
            'ufc_model_exp2_dropout.h5',
            'ufc_prediction_model.h5',
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                _model = load_model(model_file)
                _model_name = model_file
                break
        
        if _model is None:
            raise FileNotFoundError(
                "No model file found. Please train a model first.\n"
                "Expected files: ufc_model_exp1_simple.h5, ufc_model_exp3_wide.h5, etc."
            )
    
    if _scaler is None:
        if not os.path.exists('scaler.pkl'):
            raise FileNotFoundError("scaler.pkl not found. Please train a model first.")
        with open('scaler.pkl', 'rb') as f:
            _scaler = pickle.load(f)
    
    if _df_fighters is None:
        if not os.path.exists('fighters_clean.csv'):
            raise FileNotFoundError("fighters_clean.csv not found. Please run UFC_Cleaning.ipynb first.")
        _df_fighters = pd.read_csv('fighters_clean.csv', index_col='Name')
        
        # Handle Height: conversion if needed
        if 'Height:' in _df_fighters.columns and 'Height:_inches' not in _df_fighters.columns:
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
            _df_fighters['Height:_inches'] = _df_fighters['Height:'].apply(height_to_inches)

def predict_fight(fighter1, fighter2, verbose=True):
    """
    Predict the outcome of a fight between two fighters.
    
    Args:
        fighter1: Name of first fighter (Red corner)
        fighter2: Name of second fighter (Blue corner)
        verbose: If True, print formatted results. If False, return only probability.
    
    Returns:
        float: Probability of fighter1 winning (0-1)
        None: If fighters not found or error occurs
    
    Example:
        >>> predict_fight("Khabib Nurmagomedov", "Tony Ferguson")
        Khabib Nurmagomedov vs Tony Ferguson
        ============================================================
        Khabib Nurmagomedov                   65.23%
        Tony Ferguson                         34.77%
        ------------------------------------------------------------
        🏆 Predicted Winner: Khabib Nurmagomedov
           Confidence: 65.23%
        ============================================================
        0.6523
    """
    _load_resources()
    
    # Check if fighters exist
    if fighter1 not in _df_fighters.index:
        print(f"❌ Error: '{fighter1}' not found in fighter database.")
        return None
    
    if fighter2 not in _df_fighters.index:
        print(f"❌ Error: '{fighter2}' not found in fighter database.")
        return None
    
    # Get stats - MUST match exactly what was used in training (14 columns)
    # Training used these columns in this exact order:
    training_columns = [
        'Weight:', 'Age', 'SLpM:', 'Str. Acc.:', 'SApM:', 'Str. Def:', 
        'TD Avg.:', 'TD Acc.:', 'TD Def.:', 'Sub. Avg.:', 'Reach:', 
        'Win Percentage', 'Total Fights', 'Height:_inches'
    ]
    
    # Handle Height: conversion if needed
    if 'Height:' in _df_fighters.columns:
        if 'Height:_inches' not in _df_fighters.columns:
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
            _df_fighters['Height:_inches'] = _df_fighters['Height:'].apply(height_to_inches)
    
    # Calculate Total Fights if needed (Wins + Losses + Draws)
    if 'Total Fights' not in _df_fighters.columns:
        if all(col in _df_fighters.columns for col in ['Wins', 'Losses', 'Draws']):
            _df_fighters['Total Fights'] = _df_fighters['Wins'] + _df_fighters['Losses'] + _df_fighters['Draws']
        else:
            raise ValueError("Cannot calculate 'Total Fights': Missing Wins, Losses, or Draws columns")
    
    # Use only the exact columns from training, in the same order
    stats = [col for col in training_columns if col in _df_fighters.columns]
    
    # Check if we have all required columns
    missing = [col for col in training_columns if col not in stats]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    if len(stats) != 14:
        raise ValueError(f"Expected 14 features, got {len(stats)}. Columns: {stats}")
    
    # Calculate stat difference
    # Handle duplicate indices by taking first match
    data1 = _df_fighters.loc[fighter1, stats]
    data2 = _df_fighters.loc[fighter2, stats]
    
    # If loc returns DataFrame (duplicate indices), take first row
    if isinstance(data1, pd.DataFrame):
        data1 = data1.iloc[0]
    if isinstance(data2, pd.DataFrame):
        data2 = data2.iloc[0]
    
    # Ensure we have Series, then calculate difference
    data_diff = (data1 - data2).values.reshape(1, -1)
    
    # Scale and predict
    data_diff_scaled = _scaler.transform(data_diff)
    prediction = _model.predict(data_diff_scaled, verbose=0)
    probability = float(prediction[0][0])
    
    # Display results if verbose
    if verbose:
        winner = fighter1 if probability > 0.5 else fighter2
        confidence = probability if probability > 0.5 else (1 - probability)
        
        print(f"\n{fighter1} vs {fighter2}")
        print("="*60)
        print(f"{fighter1:30} {probability:>6.1%}")
        print(f"{fighter2:30} {(1-probability):>6.1%}")
        print("-"*60)
        print(f"🏆 Predicted Winner: {winner}")
        print(f"   Confidence: {confidence:.1%}")
        print("="*60)
    
    return probability

def predict_batch(fights, verbose=True):
    """
    Predict multiple fights at once.
    
    Args:
        fights: List of tuples [(fighter1, fighter2), ...]
        verbose: If True, print results. If False, return only probabilities.
    
    Returns:
        list: List of dictionaries with prediction results
    
    Example:
        >>> fights = [("Fighter A", "Fighter B"), ("Fighter C", "Fighter D")]
        >>> results = predict_batch(fights)
    """
    _load_resources()
    
    results = []
    
    if verbose:
        print("="*70)
        print("BATCH PREDICTIONS")
        print("="*70)
        print(f"{'Fighter 1':<30} {'Fighter 2':<30} {'Winner':<30} {'Confidence':<10}")
        print("-"*70)
    
    for fighter1, fighter2 in fights:
        prob = predict_fight(fighter1, fighter2, verbose=False)
        if prob is not None:
            winner = fighter1 if prob > 0.5 else fighter2
            confidence = prob if prob > 0.5 else (1-prob)
            
            result = {
                'fighter1': fighter1,
                'fighter2': fighter2,
                'fighter1_prob': prob,
                'fighter2_prob': 1-prob,
                'predicted_winner': winner,
                'confidence': confidence
            }
            results.append(result)
            
            if verbose:
                print(f"{fighter1:<30} {fighter2:<30} {winner:<30} {confidence:>8.1%}")
    
    if verbose:
        print("="*70)
    
    return results

def get_model_info():
    """Get information about the loaded model"""
    _load_resources()
    return {
        'model_file': _model_name,
        'num_fighters': len(_df_fighters),
        'model_loaded': _model is not None,
        'scaler_loaded': _scaler is not None
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python3 ufc_predictor.py 'Fighter 1' 'Fighter 2'")
        print("\nExample:")
        print("  python3 ufc_predictor.py 'Khabib Nurmagomedov' 'Tony Ferguson'")
        sys.exit(1)
    
    predict_fight(sys.argv[1], sys.argv[2])
