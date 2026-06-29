import pandas as pd
import pickle
import optuna
from sentence_transformers import SentenceTransformer, util

# Yeh function Optuna ko sikhayega ke "Best Score" kisy kehte hain
def objective(trial, df, embeddings, evaluation_set, encoded_queries):
    # 1. AI ko suggest karne do ke best combinations kya ho sakte hain
    threshold = trial.suggest_float("threshold", 0.15, 0.45)
    top_k = trial.suggest_int("top_k", 5, 20)
    
    total_precision = 0
    total_recall = 0
    
    # 2. Evaluation Logic (Fast Forward Mode)
    for query, relevant_keywords in evaluation_set.items():
        q_vec = encoded_queries[query]
        hits = util.semantic_search(q_vec, embeddings, top_k=top_k)[0]
        
        relevant_retrieved = 0
        total_possible_relevant = sum(1 for text in (df['title'] + " " + df['description']).str.lower() if any(k in text for k in relevant_keywords))

        for hit in hits:
            event = df.iloc[hit['corpus_id']]
            event_text = (str(event['title']) + " " + str(event['description'])).lower()
            
            if hit['score'] >= threshold and any(kw in event_text for kw in relevant_keywords):
                relevant_retrieved += 1

        precision_at_k = relevant_retrieved / top_k
        recall_at_k = relevant_retrieved / total_possible_relevant if total_possible_relevant > 0 else 0
        
        total_precision += precision_at_k
        total_recall += recall_at_k
        
    avg_precision = total_precision / len(evaluation_set)
    avg_recall = total_recall / len(evaluation_set)
    
    # 3. F1-Score: Precision aur Recall dono ko balance karne ka mathematical formula
    if avg_precision + avg_recall == 0:
        return 0
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    
    return f1_score # Optuna is F1-score ko highest karne ki koshish karega

if __name__ == "__main__":
    print("⏳ Loading Data & Model for Optuna Tuning...")
    df = pd.read_csv("clean_events.csv")
    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
        
    model = SentenceTransformer("all-mpnet-base-v2")
    
    # Humari wahi purani solid evaluation set
    evaluation_set = {
        "live music and concerts": ["concert", "music", "band", "live"],
        "weekend club party": ["party", "dj", "club", "techno", "dance"],
        "stand up comedy": ["comedy", "stand-up", "laugh", "jokes", "open mic"]
    }
    
    # Pre-encoding: Taake Optuna ko har dafa model run na karna pare (Is se time bachega)
    encoded_queries = {query: model.encode(query) for query in evaluation_set.keys()}
    
    print("🚀 Starting Optuna Hyperparameter Optimization (50 Trials)...")
    study = optuna.create_study(direction="maximize")
    
    # Lambda function ke zariye fast data pass kar rahe hain
    study.optimize(lambda trial: objective(trial, df, embeddings, evaluation_set, encoded_queries), n_trials=50)
    
    print("\n✅ Tuning Complete!")
    print(f"🏆 Best F1 Score Achieved: {study.best_value:.4f}")
    print("\n🎯 OPTIMAL HYPERPARAMETERS FOUND:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f" -> {key}: {value:.2f}")
        else:
            print(f" -> {key}: {value}")
    
    print("\n💡 Action: Ab in optimal values ko apni '3_evaluate_wb.py' aur '4_app.py' mein update karein!")