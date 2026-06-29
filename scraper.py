import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_berlin_events():
    # Target URL (Berlin.de events page)
    url = "https://www.berlin.de/en/events/"
    
    # User-Agent header (Taake website humein bot samajh kar block na kare)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    print("Fetching data from Berlin.de...")
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Error: Failed to retrieve data. Status Code: {response.status_code}")
        return
        
    print("Data fetched! Parsing HTML...")
    soup = BeautifulSoup(response.text, 'html.parser')
    events_data = []
    
    # Hum website ke articles (event blocks) ko dhoondh rahe hain
    event_blocks = soup.find_all('article')
    
    for block in event_blocks:
        # Title nikalna
        title_tag = block.find('h3')
        title = title_tag.text.strip() if title_tag else None
        
        # Description nikalna
        p_tags = block.find_all('p')
        description = " ".join([p.text.strip() for p in p_tags]) if p_tags else None
        
        # Agar title majood hai tabhi data save karo
        if title:
            events_data.append({
                "Title": title,
                "Description": description,
                "Location": "Berlin" # Metadata placeholder
            })
            
    # Pandas DataFrame bana kar CSV mein save karna
    if events_data:
        df = pd.DataFrame(events_data)
        df.to_csv("raw_events.csv", index=False, encoding='utf-8')
        print(f"Success! {len(df)} events scraped and saved to 'raw_events.csv'")
    else:
        print("No events found. The website layout might have changed.")

if __name__ == "__main__":
    scrape_berlin_events()