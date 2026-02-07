"""
Script to inspect a fight page and save HTML for analysis
"""
import requests
from bs4 import BeautifulSoup
import sys

def inspect_and_save(url, output_file="fight_page_inspection.html"):
    """Inspect a fight page and save key sections."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Save the full HTML
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(soup.prettify())
        print(f"Saved full HTML to {output_file}")
        
        # Also print key sections
        print("\n" + "="*80)
        print("PERSON DIVS:")
        print("="*80)
        person_divs = soup.findAll("div", {"class": "b-fight-details__person"})
        for i, div in enumerate(person_divs):
            print(f"\n--- Fighter {i+1} Div ---")
            print(div.prettify()[:2000])  # First 2000 chars
        
        print("\n" + "="*80)
        print("TEXT ITEMS (Result/Method):")
        print("="*80)
        text_items = soup.findAll("i", {"class": "b-fight-details__text-item"})
        for item in text_items:
            print(f"  {item.get_text().strip()}")
            print(f"    HTML: {str(item)[:200]}")
        
        print("\n" + "="*80)
        print("ALL I TAGS (Status Indicators):")
        print("="*80)
        all_i_tags = soup.findAll("i")
        for i_tag in all_i_tags[:20]:  # First 20
            classes = i_tag.get("class", [])
            text = i_tag.get_text().strip()
            if classes or text:
                print(f"  class={classes}, text='{text[:50]}'")
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Enter a UFC fight URL from ufcstats.com: ").strip()
    
    if url:
        inspect_and_save(url)
        print(f"\n✓ Inspection complete. Check {output_file} for full HTML")
    else:
        print("No URL provided")
