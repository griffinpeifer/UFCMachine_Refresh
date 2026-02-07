"""
Helper script to predict UFC fight outcomes using the trained model.

Usage:
    python predict_fight.py "Fighter 1 Name" "Fighter 2 Name"
    
Example:
    python predict_fight.py "Khabib Nurmagomedov" "Tony Ferguson"
"""

import sys
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.models import load_model

def predict_fight(fighter1, fighter2, model_path=None, 
                  scaler_path='scaler.pkl', fighters_path='fighters_clean.csv'):
    """
    Predict the outcome of a fight between two fighters.
    
    Args:
        fighter1: Name of the first fighter (Red corner)
        fighter2: Name of the second fighter (Blue corner)
        model_path: Path to the trained model file
        scaler_path: Path to the saved scaler
        fighters_path: Path to the cleaned fighters CSV
    
    Returns:
        Probability of fighter1 winning (0-1)
    """
    try:
        # Auto-detect best model if not specified
        if model_path is None:
            import os
            import glob
            
            # Prioritize Experiment 1 (Simple) - the best model
            if os.path.exists('ufc_model_exp1_simple.h5'):
                model_path = 'ufc_model_exp1_simple.h5'
                print(f"Using best model: {model_path} (Experiment 1 - Simple)")
            # Check for other experiment models
            elif glob.glob('ufc_model_exp*.h5'):
                experiment_models = sorted(glob.glob('ufc_model_exp*.h5'))
                model_path = experiment_models[0]
                print(f"Auto-selected model: {model_path}")
            elif os.path.exists('ufc_prediction_model.h5'):
                model_path = 'ufc_prediction_model.h5'
                print(f"Using original model: {model_path}")
            else:
                raise FileNotFoundError("No model files found. Please train a model first.")
        
        # Load the model
        model = load_model(model_path)
        print(f"Loaded model from {model_path}")
        
        # Load the scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Loaded scaler from {scaler_path}")
        
        # Load fighter data
        df_fighters = pd.read_csv(fighters_path, index_col='Name')
        print(f"Loaded {len(df_fighters)} fighters from {fighters_path}")
        
        # Check if fighters exist
        if fighter1 not in df_fighters.index:
            print(f"Error: '{fighter1}' not found in fighter database.")
            print("Available fighters (first 10):")
            print(df_fighters.index.tolist()[:10])
            return None
            
        if fighter2 not in df_fighters.index:
            print(f"Error: '{fighter2}' not found in fighter database.")
            print("Available fighters (first 10):")
            print(df_fighters.index.tolist()[:10])
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
        
        if len(stats) != 14:
            missing = [col for col in training_columns if col not in stats]
            raise ValueError(f"Expected 14 features, got {len(stats)}. Missing: {missing}")
        
        # Calculate stat differential
        data1 = df_fighters.loc[fighter1, stats]
        data2 = df_fighters.loc[fighter2, stats]
        
        # Handle duplicate indices - if loc returns DataFrame, take first row
        if isinstance(data1, pd.DataFrame):
            data1 = data1.iloc[0]
        if isinstance(data2, pd.DataFrame):
            data2 = data2.iloc[0]
        
        data_diff = (data1 - data2).values.reshape(1, -1)
        
        # Scale and predict
        data_diff_scaled = scaler.transform(data_diff)
        prediction = model.predict(data_diff_scaled, verbose=0)
        probability = prediction[0][0]
        
        return probability
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Make sure you have:")
        print("1. Trained the model (UFC_Machine_Learning_Model.ipynb)")
        print("2. Generated fighters_clean.csv (UFC_Cleaning.ipynb)")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict_fight.py 'Fighter 1 Name' 'Fighter 2 Name'")
        print("\nExample:")
        print("  python predict_fight.py 'Khabib Nurmagomedov' 'Tony Ferguson'")
        sys.exit(1)
    
    fighter1 = sys.argv[1]
    fighter2 = sys.argv[2]
    
    print(f"Predicting: {fighter1} vs {fighter2}")
    print("=" * 50)
    
    prob = predict_fight(fighter1, fighter2)
    
    if prob is not None:
        print("\n" + "=" * 50)
        print(f"Prediction Result:")
        print(f"  {fighter1} win probability: {prob:.2%}")
        print(f"  {fighter2} win probability: {(1-prob):.2%}")
        print("=" * 50)
        
        if prob > 0.5:
            print(f"\nPredicted Winner: {fighter1} ({prob:.2%} confidence)")
        elif prob < 0.5:
            print(f"\nPredicted Winner: {fighter2} ({(1-prob):.2%} confidence)")
        else:
            print("\nPrediction: Too close to call (50/50)")
