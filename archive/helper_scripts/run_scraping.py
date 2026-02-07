"""
Python script version of UFC_scraping.ipynb
Run this to scrape fighter data from ufcstats.com
"""

import pandas as pd
import requests
import json
import time
from bs4 import BeautifulSoup
from tqdm import tqdm
import re

print("=" * 60)
print("UFC Fighter Data Scraper")
print("=" * 60)

# Step 1: Setup
ufc_url = "http://ufcstats.com/statistics/fighters"
base_url = "http://ufcstats.com"

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})

print("\nStep 1: Fetching main fighters page...")
response = session.get(ufc_url)
response.raise_for_status()

# Step 2: Parse HTML and find fighter links
print("Step 2: Parsing HTML and finding fighter links...")
soup = BeautifulSoup(response.content, 'html.parser')

results = soup.find_all("td", class_="b-statistics__table-col")
print(f"Found {len(results)} table cells")

# Step 3: Extract unique fighter links
print("Step 3: Extracting unique fighter links...")
link_list = []
for result in results:
    link2 = result.find("a", class_="b-link b-link_style_black")
    if link2 is not None:
        link3 = link2['href']
        link_list.append(link3)

# Remove duplicates while preserving order
output = []
for x in link_list:
    if x not in output:
        output.append(x)

print(f"Found {len(output)} unique fighter links")
print(f"First few links: {output[:3]}")

# Step 4: Scrape fighter data
print("\nStep 4: Scraping fighter data from each link...")
print("This will take a while... Please be patient.")
fighters_data = []

for url in tqdm(output, desc="Scraping fighters"):
    try:
        react = session.get(url, timeout=10)
        react.raise_for_status()
        soup = BeautifulSoup(react.content, 'lxml')

        # Extract name
        name_elem = soup.select_one('.b-content__title-highlight')
        if not name_elem:
            continue
        name = name_elem.text.strip()
        
        # Extract record
        record_elem = soup.select_one('.b-content__title-record')
        if not record_elem:
            continue
        record = record_elem.text.strip().split()[-1]
        
        # Extract nickname
        nick_elem = soup.select_one('.b-content__Nickname')
        nick = nick_elem.text.strip() if nick_elem else ""

        fighter = {"Name": name, "Record": record, "Nickname": nick}

        # Extract stats - improved parsing logic
        list_items = soup.select('li.b-list__box-item')
        for item in list_items:
            title_elem = item.find(class_='b-list__box-item-title')
            if title_elem:
                stat = title_elem.text.strip()
                value = None
                
                # Method 1: Look for value element
                value_elem = item.find(class_='b-list__box-item-value')
                if value_elem:
                    value = value_elem.text.strip()
                else:
                    # Method 2: Get all text and remove the title
                    full_text = item.get_text(strip=True)
                    stat_text = title_elem.get_text(strip=True)
                    if full_text.startswith(stat_text):
                        value = full_text[len(stat_text):].strip()
                    else:
                        # Method 3: Try next sibling
                        next_sibling = title_elem.find_next_sibling()
                        if next_sibling:
                            value = next_sibling.get_text(strip=True)
                
                if stat and value and value != stat:
                    fighter[stat] = value
        
        fighters_data.append(fighter)
        
        # Be polite - add a small delay to avoid overwhelming the server
        time.sleep(0.5)
        
    except Exception as e:
        print(f"\nError scraping {url}: {e}")
        continue

print(f"\nStep 5: Saving data...")
# Convert to DataFrame and save
if fighters_data:
    df_fighters = pd.DataFrame(fighters_data)
    df_fighters.to_csv('ufc_fighters_raw.csv', index=False)
    print(f"✅ Saved {len(df_fighters)} fighters to ufc_fighters_raw.csv")
    print(f"\nColumns: {df_fighters.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df_fighters.head())
else:
    print("❌ No fighter data collected!")

print("\n" + "=" * 60)
print("Scraping complete!")
print("=" * 60)
print("\nNext step: Run the cleaning script:")
print("  python3 run_cleaning.py")
print("Or open UFC_Cleaning.ipynb in Jupyter")
