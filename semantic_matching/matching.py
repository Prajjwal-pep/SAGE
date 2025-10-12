import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Load Data ----------
with open("../SBERT_embeddings/Infosys_embeddings/social_Infosys_embeddings.json", "r", encoding="utf-8") as f:
    disclosures_data = json.load(f)

with open("../SBERT_embeddings/BRSR_embeddings/social_BRSR_embeddings.json", "r", encoding="utf-8") as f:
    regulations_data = json.load(f)

# Extract text and embeddings
disclosure_clauses = disclosures_data["clauses"]
regulation_clauses = regulations_data["clauses"]

disclosure_embeddings = np.array([c["embedding"] for c in disclosure_clauses])
regulation_embeddings = np.array([c["embedding"] for c in regulation_clauses])

# ---------- Compute Similarity ----------
similarity_matrix = cosine_similarity(disclosure_embeddings, regulation_embeddings)

# ---------- Matching ----------
matches = []
for i, disclosure in enumerate(disclosure_clauses):
    # Find the most similar regulation clause
    best_idx = np.argmax(similarity_matrix[i])
    best_score = similarity_matrix[i][best_idx]
    best_regulation = regulation_clauses[best_idx]

    matches.append({
        "disclosure_id": disclosure["clause_id"],
        "disclosure_text": disclosure["text"],
        "matched_regulation_id": best_regulation["clause_id"],
        "matched_regulation_text": best_regulation["text"],
        "similarity_score": round(float(best_score), 4)
    })

# ---------- Gap Analysis (Optional) ----------
# Identify regulation clauses not matched above a threshold (say 0.75)
threshold = 0.75
covered_regulations = set(m["matched_regulation_id"] for m in matches if m["similarity_score"] >= threshold)
uncovered_regulations = [
    r for r in regulation_clauses if r["clause_id"] not in covered_regulations
]

# ---------- Save Results ----------
output = {
    "category": disclosures_data["category"],
    "company_name": disclosures_data["company_name"],
    "report_year": disclosures_data["report_year"],
    "matches": matches,
    "uncovered_regulations": uncovered_regulations
}

with open("matching_output/matching_social.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✅ Matching complete for category: {disclosures_data['category']}")
print(f"📊 Total matches: {len(matches)} | Uncovered regulations: {len(uncovered_regulations)}")
