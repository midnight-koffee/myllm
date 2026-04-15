import requests
from bs4 import BeautifulSoup
import re
import time
import os

OUTPUT_FILE = os.path.join("data", "raw", "micro_internet.txt")

def get_random_wikipedia_article():
    """Fetches the HTML of a random Wikipedia article with a proper User-Agent."""
    url = "https://en.wikipedia.org/wiki/Special:Random"
    
    # Polite identification: tell Wikipedia who we are
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    print(f"Fetching: {url}")
    response = requests.get(url, headers=headers)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch article. Status code: {response.status_code}")
        return None

def clean_html(html_content):
    """Strips HTML tags and citation brackets to return pure text."""
    soup = BeautifulSoup(html_content, "html.parser")
    content = soup.find(id="mw-content-text")
    if not content:
        return ""
    paragraphs = content.find_all('p')
    text = " ".join([p.get_text() for p in paragraphs])
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_dataset(num_articles=5):
    """Runs the pipeline to gather and save clean text."""
    print(f"Starting the scrape of {num_articles} articles...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
        for i in range(num_articles):
            html = get_random_wikipedia_article()
            if html:
                clean_text = clean_html(html)
                text_length = len(clean_text)
                print(f"Extracted length: {text_length} characters")
                
                if text_length > 500:
                    file.write(clean_text + "\n<|endoftext|>\n")
                    print(f"[{i+1}/{num_articles}] Added article to dataset.")
                else:
                    print(f"[{i+1}/{num_articles}] Skipped (too short).")
            time.sleep(1)  # Be respectful to the server
    print(f"\nDone! Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    build_dataset(num_articles=5)