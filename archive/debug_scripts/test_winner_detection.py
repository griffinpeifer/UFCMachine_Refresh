"""
Quick test script to check winner detection on a sample fight page
"""
import requests
from bs4 import BeautifulSoup
import re

# Test with a known completed fight URL
# You can get a fight URL from ufcstats.com
test_url = input("Enter a UFC fight URL from ufcstats.com to test (or press Enter to skip): ").strip()

if not test_url:
    print("No URL provided. To test, run this script and paste a fight URL.")
    exit()

print(f"\nTesting winner detection on: {test_url}\n")

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
})

try:
    response = session.get(test_url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Get fighters
    fighters = []
    person_divs = soup.findAll("div", {"class": "b-fight-details__person"})
    print(f"Found {len(person_divs)} person divs")
    
    for i, div in enumerate(person_divs):
        fighter_name = div.find("h3", {"class": "b-fight-details__person-name"})
        if fighter_name:
            name = fighter_name.text.strip()
            fighters.append(name)
            print(f"\nFighter {i+1}: {name}")
            
            # Check for status indicators
            print("  Status elements found:")
            status_elements = div.findAll(["i", "span", "div"])
            for elem in status_elements:
                elem_class = elem.get("class", [])
                elem_text = elem.get_text().strip()
                if elem_class or elem_text:
                    print(f"    - Class: {elem_class}, Text: '{elem_text}'")
    
    # Check page for winner/draw indicators
    print("\n\nPage text snippets containing 'win', 'draw', 'green', 'gray':")
    page_text = soup.get_text()
    for match in re.finditer(r'.{0,50}(win|draw|green|gray).{0,50}', page_text, re.I):
        print(f"  ...{match.group()}...")
        
except Exception as e:
    print(f"Error: {e}")
