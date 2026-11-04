"""Diagnose why UFC_scraping.ipynb finds 0 fighter links.

Fetches one ufcstats.com letter page and reports what the fighter table
looks like now, so the scraper's selectors can be updated.
Writes the raw HTML to logs/fighter_page_sample.html for inspection.
"""
import os
import requests
from bs4 import BeautifulSoup

URL = "http://ufcstats.com/statistics/fighters?char=a&page=all"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "logs", "fighter_page_sample.html")

s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"})
r = s.get(URL, timeout=20)
print("Status:", r.status_code, "| final URL:", r.url, "| bytes:", len(r.content))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "wb") as f:
    f.write(r.content)

soup = BeautifulSoup(r.content, "html.parser")
print("Title:", soup.title.string.strip() if soup.title and soup.title.string else None)

tbody = soup.find("tbody")
print("Has <tbody>:", tbody is not None)

old = (tbody or soup).find_all("a", {"class": "b-link b-link_style_black"}, href=True)
print("Links matching old selector:", len(old))

detail = [a["href"] for a in soup.find_all("a", href=True) if "fighter-details" in a["href"]]
print("Any links containing 'fighter-details':", len(detail), "| unique:", len(set(detail)))
for h in list(dict.fromkeys(detail))[:3]:
    print("   e.g.", h)

if detail:
    a = next(a for a in soup.find_all("a", href=True) if "fighter-details" in a["href"])
    print("Sample link tag:", str(a)[:300])
else:
    text = soup.get_text(" ", strip=True)
    print("No fighter links. First 400 chars of page text:")
    print("  ", text[:400])
