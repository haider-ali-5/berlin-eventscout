import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer, util

def run_perturbation_test():
    print("⏳ [Phase 6] Starting Perturbation Analysis (Robustness Test)...")

    # Wahi same data aur naya best model load kar rahe hain
    df = pd.read_csv("clean_events.csv")
    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
        
    model = SentenceTransformer("all-mpnet-base-v2")

    # Original Query vs Perturbed (Spelling mistakes / Ajeeb formatting)
    test_cases = {
        "1. Clean Query (Perfect)": "weekend club party",
        "2. Typo (Spelling Error)": "wekend clb prty",
        "3. Semantic Noise (Ajeeb input)": "party but like on a weekend idk"
    }

    print("\n🔍 Testing AI Robustness against Typos and Noise:\n")

    for label, query in test_cases.items():
        q_vec = model.encode(query)
        # Hum sirf top 3 results check karenge test ke liye
        hits = util.semantic_search(q_vec, embeddings, top_k=3)[0]
        
        print(f"{label} -> '{query}'")
        for i, hit in enumerate(hits):
            event = df.iloc[hit['corpus_id']]
            # Threshold 0.25 ke mutabiq check kar rahe hain (Jo Optuna ne bataya tha)
            status = "✅ Pass" if hit['score'] >= 0.25 else "❌ Fail"
            print(f"  {i+1}. [Score: {hit['score']:.2f}] {event['title'][:55]}... {status}")
        print("-" * 65)

    print("\n💡 Conclusion: Agar Typo wali query bhi sahi events la rahi hai, toh aapka AI System 100% Robust hai!")

if __name__ == "__main__":
    run_perturbation_test()