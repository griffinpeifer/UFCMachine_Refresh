"""
Python script version of UFC_Cleaning.ipynb
Run this to clean the scraped fighter data
"""

import pandas as pd
import numpy as np
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("UFC Fighter Data Cleaner")
print("=" * 60)

# Read the raw scraped data
try:
    df = pd.read_csv("ufc_fighters_raw.csv")
    print(f"\n✅ Loaded {len(df)} fighters from ufc_fighters_raw.csv")
except FileNotFoundError:
    print("\n❌ Error: ufc_fighters_raw.csv not found.")
    print("Please run run_scraping.py first!")
    exit(1)

print("\nStep 1: Parsing win/loss records...")
try:
    df[['Wins','Losses','Draws']] = df.Record.str.split("-",expand=True)
    df['Wins'] = df['Wins'].astype(int)
    df['Losses'] = df['Losses'].astype(int)
    df['Draws'] = df['Draws'].astype(int)
    df['Win Percentage'] = round(df['Wins']/(df['Wins'] + df['Losses'] + df['Draws']),2)
    print("✅ Record parsing complete")
except Exception as e:
    print(f"⚠️  Warning in record parsing: {e}")

print("\nStep 2: Filtering out rows with missing critical data...")
initial_count = len(df)
if 'Height:' in df.columns:
    df = df[~df['Height:'].str.contains("--", na=False)]
if 'Reach:' in df.columns:
    df = df[~df['Reach:'].str.contains("--", na=False)]
if 'DOB:' in df.columns:
    df = df[~df['DOB:'].str.contains("--", na=False)]
if 'Str. Acc.:' in df.columns:
    df = df[~df['Str. Acc.:'].str.contains(" ", na=False)]

df['Nickname'] = df['Nickname'].fillna('No Name')
print(f"✅ Filtered from {initial_count} to {len(df)} fighters with complete data")

print("\nStep 3: Parsing height, reach, and weight...")
try:
    # Parse height (format: "5' 11\"")
    if 'Height:' in df.columns:
        df[['Feet','Inches']] = df['Height:'].str.split("'",expand=True)
    
    # Parse reach (format: "72\"")
    if 'Reach:' in df.columns:
        df['Reach:'] = df['Reach:'].str.slice(0,-1).astype(int)
    
    # Parse weight (format: "170 lbs.")
    if 'Weight:' in df.columns:
        df['Weight:'] = df['Weight:'].str.slice(0,-5).astype(int)
    
    # Convert height to total inches
    if 'Inches' in df.columns:
        df['Inches'] = df['Inches'].str.slice(0,-1)
        df['Feet'] = df['Feet'].astype(int)
        df['Inches'] = df['Inches'].astype(int)
        df['Inches'] = df['Inches']/12
        df['Height:'] = round(df['Feet'] + df['Inches'],2)
    
    print("✅ Height, reach, and weight parsed")
except Exception as e:
    print(f"⚠️  Warning in parsing: {e}")

print("\nStep 4: Dropping unnecessary columns...")
cols_to_drop = ['Feet', 'Inches', 'Record', 'Unnamed: 0', 'STANCE:']
cols_to_drop = [col for col in cols_to_drop if col in df.columns]
df = df.drop(cols_to_drop, axis=1)

# Rename DOB to Age
if 'DOB:' in df.columns:
    df = df.rename(columns={'DOB:': "Age"})
print("✅ Columns cleaned")

print("\nStep 5: Converting date of birth to age...")
try:
    df['Age'] = pd.to_datetime(df['Age'],format='%b %d, %Y')
    now = dt.today()
    df['Age'] = df['Age'].where(df['Age'] < now, df['Age'] -  np.timedelta64(100, 'Y'))   
    df['Age'] = (now - df['Age']).astype('<m8[Y]')
    print("✅ Age conversion complete")
except Exception as e:
    print(f"⚠️  Warning in age conversion: {e}")

print("\nStep 6: Converting percentages to decimals...")
try:
    percentage_cols = ["Str. Acc.:", "Str. Def:", "TD Acc.:", "TD Def.:"]
    for col in percentage_cols:
        if col in df.columns:
            df[col] = df[col].replace({'%': ''}, regex=True)
    
    df['Str. Acc.:'] = df['Str. Acc.:'].astype(float)
    df['Str. Acc.:'] = df['Str. Acc.:']/100
    df['Str. Def:'] = df['Str. Def:'].astype(float)
    df['Str. Def:'] = df['Str. Def:']/100
    df['TD Acc.:'] = df['TD Acc.:'].astype(float)
    df['TD Acc.:'] = df['TD Acc.:']/100
    df['TD Def.:'] = df['TD Def.:'].astype(float)
    df['TD Def.:'] = df['TD Def.:']/100
    print("✅ Percentage conversion complete")
except Exception as e:
    print(f"⚠️  Warning in percentage conversion: {e}")

print("\nStep 7: Saving cleaned data...")
df.to_csv('fighters_clean.csv', index=False)
print(f"✅ Saved {len(df)} cleaned fighters to fighters_clean.csv")
print(f"\nFinal columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

print("\n" + "=" * 60)
print("Cleaning complete!")
print("=" * 60)
print("\nNext step: Train the model by opening UFC_Machine_Learning_Model.ipynb in Jupyter")
print("Or run: jupyter notebook UFC_Machine_Learning_Model.ipynb")
