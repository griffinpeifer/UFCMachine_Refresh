"""
Scrape UFC fight results from ufcstats.com and create matches_clean.csv

This script is based on the UFC-Predictions repository by WarrierRajeev:
https://github.com/WarrierRajeev/UFC-Predictions

Usage:
    python scrape_ufc_matches.py
"""

import time
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from typing import Dict, List
import pickle
from pathlib import Path
from datetime import datetime
from dateutil import parser

class UFCMatchScraper:
    def __init__(self, min_year=2000):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.all_events_url = "http://ufcstats.com/statistics/events/completed?page=all"
        self.base_url = "http://ufcstats.com"
        self.min_year = min_year  # Minimum year to scrape (default: 2000)
        
    def make_soup(self, url: str) -> BeautifulSoup:
        """Create BeautifulSoup object from URL."""
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')
    
    def get_event_date(self, event_url: str) -> tuple:
        """Extract the date from an event page. Returns (year, month, day) or None if future/invalid."""
        try:
            soup = self.make_soup(event_url)
            event_info = ""
            for info in soup.findAll("li", {"class": "b-list__box-list-item"}):
                if "Date:" in info.text:
                    event_info = info.text
                    break
            
            if event_info:
                # Extract date - format is typically "Date: Mon. Jun. 12, 2023" or similar
                date_text = event_info.replace("Date:", "").strip()
                
                # Try to parse the full date
                try:
                    # Common formats: "Mon. Jun. 12, 2023" or "Mon, Jun 12, 2023"
                    date_text_clean = date_text.replace(".", "").replace(",", "")
                    # Try parsing with dateutil which handles various formats
                    event_date = parser.parse(date_text_clean, fuzzy=True)
                    
                    # Check if event is in the future - skip future events
                    if event_date.date() > datetime.now().date():
                        return None  # Future event, skip it
                    
                    return (event_date.year, event_date.month, event_date.day)
                except:
                    # Fallback: just extract year
                    year_match = re.search(r'\b(20\d{2})\b', date_text)
                    if year_match:
                        year = int(year_match.group(1))
                        # If year is in the future, skip it
                        if year > datetime.now().year:
                            return None
                        return (year, None, None)
            
            return None
        except Exception as e:
            print(f"Error getting date from {event_url}: {e}")
            return None
    
    def get_event_links(self) -> List[str]:
        """Get all event links from the completed events page, filtered by year."""
        print(f"Fetching event links (filtering for events from {self.min_year} onwards)...")
        soup = self.make_soup(self.all_events_url)
        event_links = []
        
        for link in soup.findAll("td", {"class": "b-statistics__table-col"}):
            for href in link.findAll("a"):
                event_link = href.get("href")
                if event_link and event_link.startswith("http"):
                    event_links.append(event_link)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in event_links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)
        
        print(f"Found {len(unique_links)} total events")
        print(f"Filtering events by date (year >= {self.min_year})...")
        
        # Filter events by date
        filtered_links = []
        current_year = datetime.now().year
        
        for event_url in tqdm(unique_links, desc="Checking event dates"):
            event_date = self.get_event_date(event_url)
            if event_date:
                # event_date is now (year, month, day) or (year, None, None)
                event_year = event_date[0]
                if event_year >= self.min_year and event_year <= current_year:
                    filtered_links.append(event_url)
            elif event_date is None:
                # None means future event or couldn't parse - skip it
                pass
            # Small delay to avoid overwhelming the server
            time.sleep(0.2)
        
        print(f"Found {len(filtered_links)} events from {self.min_year} onwards")
        return filtered_links
    
    def get_fight_links_from_event(self, event_url: str) -> List[str]:
        """Get all fight links from an event page."""
        try:
            soup = self.make_soup(event_url)
            fight_links = []
            
            for row in soup.findAll(
                "tr",
                {"class": "b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click"}
            ):
                fight_link = row.get("data-link")
                if fight_link:
                    fight_links.append(fight_link)
            
            return fight_links
        except Exception as e:
            print(f"Error getting fight links from {event_url}: {e}")
            return []
    
    def get_fight_data(self, fight_url: str) -> Dict:
        """Extract fight data from a fight page."""
        try:
            soup = self.make_soup(fight_url)
            
            # Use person divs method first (more reliable for fighter names)
            fighters = []
            person_divs = soup.findAll("div", {"class": "b-fight-details__person"})
            
            for div in person_divs:
                fighter_name = div.find("h3", {"class": "b-fight-details__person-name"})
                if fighter_name:
                    name = fighter_name.text.strip().replace("\n", "").replace("  ", " ")
                    # Clean up the name - remove extra spaces
                    name = " ".join(name.split())
                    # Check if name looks like it contains two names (unusually long or has many words)
                    # Most fighter names are 2-3 words (First Last or First Middle Last)
                    # If we see 4+ words, it might be concatenated
                    if len(name.split()) <= 4:  # Reasonable name length
                        fighters.append(name)
            
            if len(fighters) < 2:
                # Try alternative: look for links with fighter names
                fighter_links = soup.findAll("a", {"class": "b-link b-link_style_black"})
                if len(fighter_links) >= 2:
                    fighters = []
                    for link in fighter_links[:2]:
                        name = link.text.strip().replace("\n", "").replace("  ", " ")
                        name = " ".join(name.split())
                        if name and len(name.split()) <= 4:
                            fighters.append(name)
            
            if len(fighters) < 2:
                return None
            
            red_fighter = fighters[0]
            blue_fighter = fighters[1]
            
            # Validate that we have two different fighters
            if red_fighter == blue_fighter:
                return None
            
            # Get winner - check multiple methods
            winner = None
            is_draw = False
            has_result = False
            
            # Method 1: Check the "Fight Details" section which often shows the result
            # Look for sections that contain result information
            fight_details_sections = soup.findAll(["div", "section"], {"class": re.compile(r"fight-details|result|outcome", re.I)})
            for section in fight_details_sections:
                section_text = section.get_text()
                section_text_lower = section_text.lower()
                
                # Check for draw/no contest first
                if re.search(r'\b(draw|no contest|nc)\b', section_text_lower):
                    is_draw = True
                    has_result = True
                    break
                
                # Check for winner patterns: "Fighter1 def. Fighter2" or "Fighter1 wins"
                # Try red fighter first
                if red_fighter in section_text:
                    # Check if there's a result keyword near the name
                    red_index = section_text.lower().find(red_fighter.lower())
                    context = section_text[max(0, red_index-50):min(len(section_text), red_index+len(red_fighter)+50)].lower()
                    if any(word in context for word in ["def.", "defeated", "wins", "won", "ko", "tko", "submission", "decision"]):
                        winner = red_fighter
                        has_result = True
                        break
                
                # Try blue fighter
                if blue_fighter in section_text:
                    blue_index = section_text.lower().find(blue_fighter.lower())
                    context = section_text[max(0, blue_index-50):min(len(section_text), blue_index+len(blue_fighter)+50)].lower()
                    if any(word in context for word in ["def.", "defeated", "wins", "won", "ko", "tko", "submission", "decision"]):
                        winner = blue_fighter
                        has_result = True
                        break
            
            # Method 2: Check for status indicators in person divs
            if not has_result:
                person_divs = soup.findAll("div", {"class": "b-fight-details__person"})
                
                for div_idx, div in enumerate(person_divs):
                    fighter_name_elem = div.find("h3", {"class": "b-fight-details__person-name"})
                    if not fighter_name_elem:
                        continue
                    
                    current_fighter = fighter_name_elem.text.strip().replace("\n", "").replace("  ", " ")
                    
                    # Check for status icon with green style (winner)
                    status_icon = div.find("i", {"class": re.compile(r"status.*green|green.*status", re.I)})
                    if status_icon:
                        winner = current_fighter
                        has_result = True
                        break
                    
                    # Check for gray style (draw/loser)
                    status_icon_gray = div.find("i", {"class": re.compile(r"status.*gray|gray.*status", re.I)})
                    if status_icon_gray:
                        # If both are gray, it's a draw. If only one is gray, the other won.
                        # We'll check both divs to determine
                        other_div = person_divs[1 - div_idx] if len(person_divs) > 1 else None
                        if other_div:
                            other_status = other_div.find("i", {"class": re.compile(r"status.*green|green.*status", re.I)})
                            if other_status:
                                # Other fighter won
                                other_name = other_div.find("h3", {"class": "b-fight-details__person-name"})
                                if other_name:
                                    winner = other_name.text.strip().replace("\n", "").replace("  ", " ")
                                    has_result = True
                                    break
                            else:
                                # Both gray = draw
                                is_draw = True
                                has_result = True
                                break
            
            # Method 2: Check fight result/method section (very reliable)
            if not has_result:
                # Look for all text-item elements that might contain result
                text_items = soup.findAll("i", {"class": "b-fight-details__text-item"})
                for item in text_items:
                    item_text = item.get_text().strip()
                    item_text_lower = item_text.lower()
                    
                    # Check if this item contains result information
                    if any(keyword in item_text_lower for keyword in ["method", "result", "round", "time"]):
                        # Check if it mentions one of our fighters
                        if red_fighter in item_text:
                            winner = red_fighter
                            has_result = True
                            break
                        elif blue_fighter in item_text:
                            winner = blue_fighter
                            has_result = True
                            break
                        # Check for draw
                        elif "draw" in item_text_lower or "no contest" in item_text_lower:
                            is_draw = True
                            has_result = True
                            break
                
                # Also check parent elements of text-items for context
                if not has_result:
                    for item in text_items:
                        parent = item.parent
                        if parent:
                            parent_text = parent.get_text()
                            # Check if parent contains fighter name and result
                            if red_fighter in parent_text and any(word in parent_text.lower() for word in ["def", "win", "ko", "sub", "dec"]):
                                winner = red_fighter
                                has_result = True
                                break
                            elif blue_fighter in parent_text and any(word in parent_text.lower() for word in ["def", "win", "ko", "sub", "dec"]):
                                winner = blue_fighter
                                has_result = True
                                break
            
            # Method 3: Check fight statistics tables - winner often has better stats or is listed first
            if not has_result:
                # Look for fight statistics tables
                stats_tables = soup.findAll("table", {"class": "b-fight-details__table"})
                
                # Check if tables have actual fight data (not just matchup/preview)
                for table in stats_tables:
                    # Check table headers - sometimes winner is in first column
                    headers = table.findAll("th", {"class": "b-fight-details__table-col"})
                    header_links = [h.find("a") for h in headers if h.find("a")]
                    
                    # Check if this is a round-by-round stats table (indicates completed fight)
                    table_text = table.get_text().lower()
                    if any(keyword in table_text for keyword in ["round", "rnd", "sig. str", "total str", "takedown"]):
                        # This is a stats table from a completed fight
                        # Check table headers for fighter order
                        if len(header_links) >= 2:
                            first_fighter = header_links[0].get_text().strip() if header_links[0] else ""
                            second_fighter = header_links[1].get_text().strip() if header_links[1] else ""
                            
                            # In many cases, winner is listed first in stats tables
                            # But this isn't always reliable, so we'll use it as a last resort
                            # Actually, let's not rely on order - instead check for result indicators
                            pass
                
                # Check entire page text for result patterns
                page_text = soup.get_text()
                page_text_lower = page_text.lower()
                
                # Look for explicit result patterns
                result_patterns = [
                    rf"{re.escape(red_fighter)}.*?def\.?.*?{re.escape(blue_fighter)}",
                    rf"{re.escape(blue_fighter)}.*?def\.?.*?{re.escape(red_fighter)}",
                    rf"{re.escape(red_fighter)}.*?wins?",
                    rf"{re.escape(blue_fighter)}.*?wins?",
                ]
                
                for pattern in result_patterns:
                    match = re.search(pattern, page_text, re.I)
                    if match:
                        matched_text = match.group(0).lower()
                        if red_fighter.lower() in matched_text and any(word in matched_text for word in ["def", "win", "won"]):
                            winner = red_fighter
                            has_result = True
                            break
                        elif blue_fighter.lower() in matched_text and any(word in matched_text for word in ["def", "win", "won"]):
                            winner = blue_fighter
                            has_result = True
                            break
                
                # Check for draw in page text
                if not has_result and re.search(r'\bdraw\b', page_text_lower):
                    # Verify it's actually about the fight result, not just the word "draw" in another context
                    draw_context = re.search(r'\bdraw\b', page_text_lower)
                    if draw_context:
                        context_start = max(0, draw_context.start() - 50)
                        context_end = min(len(page_text), draw_context.end() + 50)
                        context = page_text[context_start:context_end].lower()
                        # If draw is near fight-related keywords, it's likely a draw result
                        if any(word in context for word in ["fight", "match", "result", "decision", "majority", "split"]):
                            is_draw = True
                            has_result = True
            
            # Method 4: Check if fight has round-by-round statistics (indicates completed fight)
            # If it doesn't have stats, it's likely a future/uncompleted fight
            has_round_stats = False
            stats_tables = soup.findAll("table", {"class": "b-fight-details__table"})
            for table in stats_tables:
                table_text = table.get_text().lower()
                # Look for round-by-round indicators
                if any(keyword in table_text for keyword in ["round 1", "rnd 1", "round 2", "rnd 2", "r1", "r2"]):
                    has_round_stats = True
                    break
            
            # If no round stats found and no result, this is likely a future/uncompleted fight - skip it
            if not has_round_stats and not has_result:
                return None
            
            # Method 5: Check for explicit draw/no contest indicators (only if we have stats)
            if not has_result and has_round_stats:
                # Look for explicit draw indicators in result sections
                result_sections = soup.findAll(["div", "span", "td", "i"], {"class": re.compile(r'result|outcome|method|text-item', re.I)})
                for section in result_sections[:10]:
                    section_text = section.get_text().strip().lower()
                    if re.search(r'\b(draw|no contest|nc)\b', section_text):
                        # Verify it's about the fight result
                        if any(word in section_text for word in ["fight", "match", "decision", "majority", "split"]):
                            is_draw = True
                            has_result = True
                            break
            
            # If we still don't have a result after all checks, skip this fight
            if not has_result:
                return None
            
            # Get fight type (weight class)
            fight_type_elem = soup.find("i", {"class": "b-fight-details__fight-title"})
            fight_type = ""
            if fight_type_elem:
                fight_type = fight_type_elem.text.strip().replace("\n", "").replace("  ", "")
            
            # Extract weight class from fight type
            weight_class = self._extract_weight_class(fight_type)
            
            # Determine winner string
            if is_draw:
                winner_str = "Draw"
            elif winner:
                winner_str = winner
            else:
                # No result found - skip this fight (likely future)
                return None
            
            return {
                "Red": red_fighter,
                "Blue": blue_fighter,
                "Winner": winner_str,
                "Weight Class": weight_class,
                "Fight Type": fight_type
            }
        except Exception as e:
            print(f"Error scraping fight {fight_url}: {e}")
            return None
    
    def _get_fight_data_from_divs(self, soup: BeautifulSoup, fight_url: str) -> Dict:
        """Fallback method to get fight data from person divs."""
        fighters = []
        for div in soup.findAll("div", {"class": "b-fight-details__person"}):
            fighter_name = div.find("h3", {"class": "b-fight-details__person-name"})
            if fighter_name:
                name = fighter_name.text.strip().replace("\n", "").replace("  ", " ")
                # Clean up the name - remove extra spaces
                name = " ".join(name.split())
                fighters.append(name)
        
        if len(fighters) < 2:
            return None
        
        red_fighter = fighters[0]
        blue_fighter = fighters[1]
        
        # Validate that we have two different fighters
        if red_fighter == blue_fighter:
            return None
        
        # Get winner - check for both win (green) and draw indicators
        winner = None
        is_draw = False
        has_result = False
        
        for div in soup.findAll("div", {"class": "b-fight-details__person"}):
            # Check for winner (green status)
            if div.find("i", {"class": "b-fight-details__person-status b-fight-details__person-status_style_green"}):
                winner_name = div.find("h3", {"class": "b-fight-details__person-name"})
                if winner_name:
                    winner = winner_name.text.strip().replace("\n", "").replace("  ", "")
                    has_result = True
            # Check for draw indicator
            if div.find("i", {"class": "b-fight-details__person-status b-fight-details__person-status_style_gray"}):
                is_draw = True
                has_result = True
        
        # If there's no winner and no draw indicator, this is likely a future fight - skip it
        if not has_result:
            return None
        
        # Get fight type
        fight_type_elem = soup.find("i", {"class": "b-fight-details__fight-title"})
        fight_type = ""
        if fight_type_elem:
            fight_type = fight_type_elem.text.strip().replace("\n", "").replace("  ", "")
        
        weight_class = self._extract_weight_class(fight_type)
        
        # Determine winner string
        if is_draw:
            winner_str = "Draw"
        elif winner:
            winner_str = winner
        else:
            # No result found - skip this fight (likely future)
            return None
        
        return {
            "Red": red_fighter,
            "Blue": blue_fighter,
            "Winner": winner_str,
            "Weight Class": weight_class,
            "Fight Type": fight_type
        }
    
    def _extract_weight_class(self, fight_type: str) -> str:
        """Extract weight class from fight type string."""
        weight_classes = [
            "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight",
            "Strawweight", "Flyweight", "Bantamweight", "Featherweight",
            "Lightweight", "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight"
        ]
        
        for wc in weight_classes:
            if wc in fight_type:
                return wc
        
        return "Unknown"
    
    def scrape_all_fights(self, max_fights: int = None) -> List[Dict]:
        """Scrape all fights from all events."""
        print("=" * 60)
        print("Starting UFC match data scraping")
        print("=" * 60)
        
        # Get all event links
        event_links = self.get_event_links()
        
        # Limit events if needed (for testing)
        if max_fights:
            print(f"Note: Limiting to approximately {max_fights} fights for testing")
        
        all_fights = []
        total_fights = 0
        
        print("\nScraping fights from events...")
        for event_idx, event_url in enumerate(tqdm(event_links, desc="Processing events")):
            fight_links = self.get_fight_links_from_event(event_url)
            
            for fight_url in fight_links:
                if max_fights and total_fights >= max_fights:
                    break
                
                fight_data = self.get_fight_data(fight_url)
                if fight_data:
                    all_fights.append(fight_data)
                    total_fights += 1
                
                # Be polite - small delay between requests
                time.sleep(0.3)
            
            if max_fights and total_fights >= max_fights:
                break
            
            # Small delay between events
            time.sleep(0.5)
        
        print(f"\nScraped {len(all_fights)} fights")
        return all_fights
    
    def save_matches_csv(self, fights: List[Dict], filename: str = "matches_clean.csv"):
        """Save fights data to CSV file."""
        if not fights:
            print("No fights to save!")
            return
        
        df = pd.DataFrame(fights)
        
        # Ensure we have the required columns
        required_cols = ["Red", "Blue", "Winner", "Weight Class"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = ""
        
        # Select and reorder columns
        df = df[required_cols]
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"\nSaved {len(df)} matches to {filename}")
        print(f"\nFirst few rows:")
        print(df.head(10))
        print(f"\nDataFrame info:")
        print(df.info())

def main():
    """Main function to run the scraper."""
    # Ask user for minimum year (default: 2000)
    print("UFC Match Data Scraper")
    print("=" * 60)
    print("This will scrape UFC fights from ufcstats.com")
    print(f"Default: Scrapes fights from 2000 onwards")
    
    min_year_input = input("\nEnter minimum year to scrape (default: 2000, press Enter for default): ").strip()
    min_year = 2000
    if min_year_input:
        try:
            min_year = int(min_year_input)
            if min_year < 1993 or min_year > 2030:
                print(f"Invalid year {min_year}, using default 2000")
                min_year = 2000
        except ValueError:
            print("Invalid input, using default year 2000")
    
    scraper = UFCMatchScraper(min_year=min_year)
    
    # Ask user if they want to limit the number of fights (for testing)
    print(f"\nScraping fights from {min_year} onwards")
    print("This may take a while depending on the number of fights...")
    
    response = input("\nLimit number of fights for testing? (y/n): ").strip().lower()
    max_fights = None
    if response == 'y':
        try:
            max_fights = int(input("Enter maximum number of fights to scrape: "))
        except ValueError:
            print("Invalid input, scraping all fights...")
    
    # Scrape all fights
    fights = scraper.scrape_all_fights(max_fights=max_fights)
    
    # Save to CSV
    if fights:
        scraper.save_matches_csv(fights)
        print("\n" + "=" * 60)
        print("Scraping complete!")
        print("=" * 60)
    else:
        print("No fights were scraped. Please check for errors.")

if __name__ == "__main__":
    main()
