import streamlit as st
import pandas as pd
import pickle
import pydeck as pdk
import hashlib
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Berlin EventScout", layout="wide")

# Map ke dots ko alag alag jagah set karne ka function
def get_fast_coordinates(address_text, title_text):
    combined_text = str(address_text) + str(title_text)
    hash_object = hashlib.md5(combined_text.encode())
    hash_hex = hash_object.hexdigest()
    offset_lat = (int(hash_hex[:4], 16) % 1000) / 18000 - 0.025
    offset_lon = (int(hash_hex[4:8], 16) % 1000) / 12000 - 0.045
    return 52.5200 + offset_lat, 13.4050 + offset_lon

@st.cache_resource
def load_system():
    df = pd.read_csv("clean_events.csv")
    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    return df, embeddings, SentenceTransformer('all-mpnet-base-v2')

df, embs, model = load_system()

st.title("🇩🇪 Berlin EventScout AI Dashboard")
st.markdown("Discover unique local events across Berlin using Semantic AI.")
st.divider()

# Search Bar
query = st.text_input("🔍 What kind of event are you looking for? (e.g., FIFA, Tech, Party, Comedy):")

if not query:
    st.info("💡 Type a keyword above to search, or explore today's featured events below:")
    display_indices = list(range(min(5, len(df))))
    
    # Default State: Bina search ke top 5 events dikhana
    for idx in display_indices:
        event = df.iloc[idx]
        col1, col2 = st.columns([2.5, 1.5])
        with col1:
            st.markdown(f"### {event['title']}")
            st.write(f"**💰 Price:** {event.get('price')} | **📅 Date:** {event.get('date')}")
            st.write(f"**📍 Location:** {event.get('address')} | Source: {event.get('source')}")
            st.write(f"*{event['description']}*")
            st.link_button("🌐 View Official Event Page", event.get('url', 'https://berlin.de'))
        with col2:
            lat, lon = get_fast_coordinates(event.get('address'), event.get('title'))
            st.pydeck_chart(pdk.Deck(
                map_style=None, 
                initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=12, pitch=0),
                layers=[pdk.Layer('ScatterplotLayer', data=[{'lat': lat, 'lon': lon}], get_position='[lon, lat]', get_color=[220, 50, 50, 200], get_radius=180)],
            ), height=250)
        st.divider()
else:
    # Search State: Jab user kuch search kare
    q_vec = model.encode(query)
    hits = util.semantic_search(q_vec, embs, top_k=8)[0]
    
    # STRICT THRESHOLD FILTER: Sirf woh events uthao jo sach mein relevant hon (score >= 0.25)
    relevant_hits = [hit for hit in hits if hit['score'] >= 0.25]
    
    if not relevant_hits:
        # Agar fuzool search ho ya data na ho, toh warning do
        st.warning(f"⚠️ No highly relevant events found for '{query}'. Please try a different term.")
    else:
        st.subheader("🎯 Top Relevant Events")
        for hit in relevant_hits:
            event = df.iloc[hit['corpus_id']]
            col1, col2 = st.columns([2.5, 1.5])
            with col1:
                st.markdown(f"### {event['title']}")
                st.write(f"**Match Score:** {hit['score']:.2f} | **💰 Price:** {event.get('price')} | **📅 Date:** {event.get('date')}")
                st.write(f"**📍 Location:** {event.get('address')} | Source: {event.get('source')}")
                st.write(f"*{event['description']}*")
                st.link_button("🌐 View Official Event Page", event.get('url', 'https://berlin.de'))
            with col2:
                lat, lon = get_fast_coordinates(event.get('address'), event.get('title'))
                st.pydeck_chart(pdk.Deck(
                    map_style=None, 
                    initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=12, pitch=0),
                    layers=[pdk.Layer('ScatterplotLayer', data=[{'lat': lat, 'lon': lon}], get_position='[lon, lat]', get_color=[220, 50, 50, 200], get_radius=180)],
                ), height=250)
            st.divider()