import pandas as pd
import pickle
import wandb
from sentence_transformers import SentenceTransformer, util

def evaluate_system():
    print("⏳ [Phase 3] Starting Offline Evaluation & MLOps Logging...")
    
    wandb.init(project="berlin-eventscout-eval", name="mpnet-optimized-run")
    
    # OPTIMIZATION: top_k ko 15 kar diya taake Recall jump kare!
    config = {
        "model_name": "all-mpnet-base-v2",
        "threshold": 0.25, # Optuna ne bataya!
        "top_k": 8         # Optuna ne bataya!
    }
    wandb.config.update(config)

    df = pd.read_csv("clean_events.csv")
    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
        
    model = SentenceTransformer(config['model_name'])

    # OPTIMIZATION: Updated Ground Truth based on your 200+ actual events
    evaluation_set = {
        "live music and concerts": ["concert", "music", "band", "live"],
        "weekend club party": ["party", "dj", "club", "techno", "dance"],
        "stand up comedy": ["comedy", "stand-up", "laugh", "jokes", "open mic"]
    }

    total_precision = 0
    total_recall = 0

    print(f"\n📊 Calculating Precision@{config['top_k']} and Recall@{config['top_k']}...")
    
    for query, relevant_keywords in evaluation_set.items():
        q_vec = model.encode(query)
        hits = util.semantic_search(q_vec, embeddings, top_k=config["top_k"])[0]
        
        relevant_retrieved = 0
        total_possible_relevant = sum(1 for text in (df['title'] + " " + df['description']).str.lower() if any(k in text for k in relevant_keywords))

        for hit in hits:
            event = df.iloc[hit['corpus_id']]
            event_text = (str(event['title']) + " " + str(event['description'])).lower()
            
            if hit['score'] >= config['threshold'] and any(kw in event_text for kw in relevant_keywords):
                relevant_retrieved += 1

        precision_at_k = relevant_retrieved / config["top_k"]
        recall_at_k = relevant_retrieved / total_possible_relevant if total_possible_relevant > 0 else 0
        
        total_precision += precision_at_k
        total_recall += recall_at_k
        
        print(f" -> Query: '{query}' | Precision: {precision_at_k:.2f} | Recall: {recall_at_k:.2f} (Total in DB: {total_possible_relevant})")

    avg_precision = total_precision / len(evaluation_set)
    avg_recall = total_recall / len(evaluation_set)

    wandb.log({
        f"Average Precision@{config['top_k']}": avg_precision,
        f"Average Recall@{config['top_k']}": avg_recall
    })

    print(f"\n✅ Evaluation Complete! Avg Precision: {avg_precision*100:.1f}%, Avg Recall: {avg_recall*100:.1f}%")
    wandb.finish()

if __name__ == "__main__":
    evaluate_system()