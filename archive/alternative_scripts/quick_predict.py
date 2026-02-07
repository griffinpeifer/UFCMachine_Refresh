#!/usr/bin/env python3
"""
Quick prediction script - simplest way to predict fights
Usage: python3 quick_predict.py "Fighter 1" "Fighter 2"
"""

import sys
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

def quick_predict(fighter1, fighter2):
    """Quick and simple prediction"""
    # Load model (prioritize Experiment 1)
    try:
        model = load_model('ufc_model_exp1_simple.h5')
    except:
        try:
            model = load_model('ufc_prediction_model.h5')
        except:
            print("❌ Error: No model found. Please train a model first.")
            return None
    
    # Load scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load fighters
    df_fighters = pd.read_csv('fighters_clean.csv', index_col='Name')
    
    # Check fighters exist
    if fighter1 not in df_fighters.index:
        print(f"❌ '{fighter1}' not found")
        return None
    if fighter2 not in df_fighters.index:
        print(f"❌ '{fighter2}' not found")
        return None
    
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
            def height_to_inches(h):
                if pd.isna(h): return np.nan
                try:
                    parts = str(h).replace('"', '').split("'")
                    if len(parts) == 2:
                        return float(parts[0].strip()) * 12 + float(parts[1].strip())
                except: pass
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
    
    if len(stats) != 14:
        missing = [col for col in training_columns if col not in stats]
        raise ValueError(f"Expected 14 features, got {len(stats)}. Missing: {missing}")
    
    # Calculate difference and predict
    data1 = df_fighters.loc[fighter1, stats]
    data2 = df_fighters.loc[fighter2, stats]
    
    # Handle duplicate indices - if loc returns DataFrame, take first row
    if isinstance(data1, pd.DataFrame):
        data1 = data1.iloc[0]
    if isinstance(data2, pd.DataFrame):
        data2 = data2.iloc[0]
    
    data_diff = (data1 - data2).values.reshape(1, -1)
    prob = model.predict(scaler.transform(data_diff), verbose=0)[0][0]
    
    # Display result
    print("\n" + "="*60)
    print(f"{fighter1} vs {fighter2}")
    print("="*60)
    print(f"{fighter1:30} {prob:>6.1%}")
    print(f"{fighter2:30} {(1-prob):>6.1%}")
    print("-"*60)
    winner = fighter1 if prob > 0.5 else fighter2
    confidence = prob if prob > 0.5 else (1-prob)
    print(f"🏆 Predicted Winner: {winner}")
    print(f"   Confidence: {confidence:.1%}")
    print("="*60)
    
    return prob

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 quick_predict.py 'Fighter 1' 'Fighter 2'")
        print("\nExample:")
        print("  python3 quick_predict.py 'Khabib Nurmagomedov' 'Tony Ferguson'")
        sys.exit(1)
    
    quick_predict(sys.argv[1], sys.argv[2])
