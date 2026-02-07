"""
Debug script to inspect HTML structure and find how winners are indicated
"""
import requests
from bs4 import BeautifulSoup
import re
import sys

def inspect_fight_page(url):
    """Inspect a fight page to see how winners are indicated."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        print("=" * 80)
        print(f"INSPECTING: {url}")
        print("=" * 80)
        
        # Get fighters
        fighters = []
        person_divs = soup.findAll("div", {"class": "b-fight-details__person"})
        print(f"\n1. PERSON DIVS: Found {len(person_divs)}")
        for i, div in enumerate(person_divs):
            fighter_name = div.find("h3", {"class": "b-fight-details__person-name"})
            if fighter_name:
                name = fighter_name.text.strip()
                fighters.append(name)
                print(f"   Fighter {i+1}: {name}")
                
                # Check ALL elements in this div
                print(f"   All elements in div {i+1}:")
                for elem in div.findAll(True):  # Find all elements
                    tag = elem.name
                    classes = elem.get("class", [])
                    text = elem.get_text().strip()[:50]  # First 50 chars
                    if classes or text:
                        print(f"      <{tag}> class={classes} text='{text}'")
        
        # Check for result/method section
        print(f"\n2. RESULT/METHOD SECTIONS:")
        method_elements = soup.findAll("i", {"class": re.compile(r"fight|result|method", re.I)})
        for elem in method_elements:
            print(f"   Found: {elem.get('class')} = {elem.get_text().strip()}")
        
        # Check fight details text items
        print(f"\n3. FIGHT DETAILS TEXT ITEMS:")
        text_items = soup.findAll("i", {"class": "b-fight-details__text-item"})
        for item in text_items:
            print(f"   {item.get_text().strip()}")
        
        # Check for any status indicators
        print(f"\n4. STATUS INDICATORS:")
        all_i_tags = soup.findAll("i")
        for i_tag in all_i_tags:
            classes = i_tag.get("class", [])
            text = i_tag.get_text().strip()
            if any(keyword in str(classes).lower() for keyword in ["status", "green", "gray", "win", "draw"]):
                print(f"   <i> class={classes} text='{text}'")
        
        # Check tables for result information
        print(f"\n5. TABLE DATA:")
        tables = soup.findAll("table")
        for i, table in enumerate(tables[:3]):  # First 3 tables
            print(f"   Table {i+1}:")
            rows = table.findAll("tr")[:3]  # First 3 rows
            for row in rows:
                cells = row.findAll(["td", "th"])
                row_data = [cell.get_text().strip()[:30] for cell in cells]
                print(f"      {row_data}")
        
        # Check page for winner keywords
        print(f"\n6. WINNER KEYWORDS IN PAGE TEXT:")
        page_text = soup.get_text()
        winner_patterns = [
            r'def\.',
            r'defeated',
            r'winner',
            r'won by',
            r'via',
        ]
        for pattern in winner_patterns:
            matches = re.finditer(pattern, page_text, re.I)
            for match in list(matches)[:3]:  # First 3 matches
                start = max(0, match.start() - 30)
                end = min(len(page_text), match.end() + 30)
                context = page_text[start:end]
                print(f"   '{pattern}': ...{context}...")
        
        return soup, fighters
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, []

if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        # Get a sample fight URL - you can find one from ufcstats.com
        url = input("Enter a UFC fight URL from ufcstats.com: ").strip()
    
    if url:
        inspect_fight_page(url)
    else:
        print("No URL provided")
