import pandas as pd
import time
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

def scrape_with_selenium():
    print("[1/3] Starting Precision Web Scraper...")
    chrome_options = Options()
    chrome_options.add_argument("--headless=new") 
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    events = []
    
    bad_keywords = ["navigation", "menu", "choose language", "homepage", "newsletter", "you are here"]
    urls_to_scrape = [
        "https://www.berlin.de/en/events/concerts/",
        "https://www.berlin.de/en/events/exhibitions/",
        "https://www.berlin.de/en/events/comedy/",
        "https://www.berlin.de/en/events/party/",
        "https://www.gratis-in-berlin.de/heute",
        "https://www.gratis-in-berlin.de/morgen"
    ]
    
    for url in urls_to_scrape:
        print(f" -> Fetching data from: {url}") 
        try:
            driver.get(url)
            time.sleep(3) 
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            for heading in soup.find_all(['h2', 'h3']):
                title = heading.get_text(strip=True)
                title_lower = title.lower()
                
                if 10 < len(title) < 90 and not any(bad in title_lower for bad in bad_keywords):
                    desc_tag = heading.find_next(['p', 'div'])
                    desc = desc_tag.get_text(strip=True) if desc_tag else ""
                    desc_lower = desc.lower()
                    
                    if any(bad in desc_lower for bad in bad_keywords):
                        continue 
                    
                    desc = desc.replace(".mehr", "").replace("mehr", "").strip()
                    if len(desc) < 15:
                        desc = f"Experience '{title}' live in Berlin's diverse cultural scene."
                    
                    # FIX 1: Description ko 2 se 3 line (150 chars) tak limit kar diya
                    if len(desc) > 150:
                        desc = desc[:147] + "..."
                    
                    real_date = "Upcoming"
                    parent = heading.find_parent()
                    real_address = "Berlin, Germany"
                    
                    if parent:
                        parent_text = parent.get_text(" ", strip=True)
                        date_match = re.search(r'\b\d{1,2}\.\s?\w*\.?\s?\d{0,4}\b|\b\d{1,2}\.\d{1,2}\.\d{2,4}\b', parent_text)
                        if date_match:
                            real_date = date_match.group(0)
                            
                        # FIX 2: Proper Venue aur Location fetch karne ki strict logic
                        loc_tag = parent.find(['span', 'div', 'p', 'a'], class_=lambda x: x and any(word in str(x).lower() for word in ['ort', 'location', 'venue', 'address', 'locality']))
                        if loc_tag:
                            ext_loc = loc_tag.get_text(strip=True)
                            if 3 < len(ext_loc) < 60: 
                                real_address = ext_loc + ", Berlin"

                    real_price = "Free / Kostenlos" if "gratis" in url else "See Official Site"
                        
                    events.append({
                        "title": title,
                        "description": desc,
                        "date": real_date,            
                        "source": url.split('/')[2], 
                        "url": url,                   
                        "price": real_price,     
                        "address": real_address  
                    })
        except Exception as e:
            pass
            
    driver.quit() 
    return pd.DataFrame(events)

def enforce_data_quality(df):
    if len(df) == 0: return df
    df = df.dropna(subset=['title']).drop_duplicates(subset=['title'])
    df['description_clean'] = df['description'].str.lower().str.strip()
    return df

if __name__ == "__main__":
    raw_df = scrape_with_selenium()
    clean_df = enforce_data_quality(raw_df)
    if len(clean_df) > 0:
        clean_df.to_csv("clean_events.csv", index=False)
        print(f"[3/3] ✅ Success! {len(clean_df)} perfectly formatted events saved.")