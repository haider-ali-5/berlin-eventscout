import pandas as pd
import pickle
import os
from sentence_transformers import SentenceTransformer

def generate_incremental_embeddings():
    print("⏳ Loading Data...")
    df = pd.read_csv("clean_events.csv")
    model = SentenceTransformer('all-mpnet-base-v2')
    
    embeddings = []
    start_index = 0
    
    # Check if previous embeddings exist (Incremental Logic)
    if os.path.exists("embeddings.pkl"):
        with open("embeddings.pkl", "rb") as f:
            embeddings = list(pickle.load(f))
            start_index = len(embeddings)
            print(f"✅ Found {start_index} existing embeddings.")
            
    # Sirf naye events ko encode karo
    if start_index < len(df):
        print(f"🚀 Encoding {len(df) - start_index} NEW events...")
        new_events = df.iloc[start_index:]
        new_texts = (new_events['title'].astype(str) + " " + new_events['description'].astype(str)).tolist()
        
        new_embs = model.encode(new_texts, show_progress_bar=True)
        embeddings.extend(new_embs)
        
        # Save updated embeddings back to the file
        with open("embeddings.pkl", "wb") as f:
            pickle.dump(embeddings, f)
        print("✅ New embeddings successfully added and saved!")
    else:
        print("⚡ No new events found. Embeddings are already up to date.")

if __name__ == "__main__":
    generate_incremental_embeddings()