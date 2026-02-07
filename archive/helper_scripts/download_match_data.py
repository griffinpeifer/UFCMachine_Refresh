"""
Script to create UFC match data by scraping from ufcstats.com.
This script helps create an updated UFC dataset for training the model.

Based on the UFC-Predictions repository by WarrierRajeev:
https://github.com/WarrierRajeev/UFC-Predictions

Usage:
    python download_match_data.py
    or
    python scrape_ufc_matches.py
"""

import os
import sys

def main():
    """Main function - uses the scraper to create match data."""
    print("UFC Match Data Creator")
    print("=" * 60)
    print("This script will scrape UFC fight data from ufcstats.com")
    print("and create matches_clean.csv for model training.")
    print("=" * 60)
    
    # Check if data already exists
    if os.path.exists('matches_clean.csv'):
        response = input("\nmatches_clean.csv already exists. Create new file? (y/n): ")
        if response.lower() != 'y':
            print("Skipping. Using existing matches_clean.csv")
            return
    
    # Import and run the scraper
    try:
        from scrape_ufc_matches import UFCMatchScraper
        
        # Ask user for minimum year (default: 2000)
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
            print("Match data creation complete!")
            print("You can now use matches_clean.csv for model training.")
            print("=" * 60)
        else:
            print("\nNo fights were scraped. Please check for errors.")
    
    except ImportError as e:
        print(f"\nError: Could not import scraper module: {e}")
        print("Make sure scrape_ufc_matches.py is in the same directory.")
        print("\nAlternative: Run directly with:")
        print("  python scrape_ufc_matches.py")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
