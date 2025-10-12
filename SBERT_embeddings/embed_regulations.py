"""
Generate SBERT embeddings for regulation clauses.
Input: environmental.json, social.json, governance.json (from classification)
Output: environmental_reg.json, social_reg.json, governance_reg.json
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Load SBERT model
print("Loading Sentence-BERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!\n")


def load_and_embed_category(category_file, category_name, regulation_id):
    """Load clauses, generate embeddings, and structure output."""
    
    # Load classified clauses
    with open(category_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    clauses_text = data["clauses"]
    print(f"Processing {len(clauses_text)} {category_name} clauses...")
    
    # Generate embeddings
    embeddings = model.encode(clauses_text, show_progress_bar=True, convert_to_numpy=True)
    
    # Structure output
    clauses = []
    category_code = category_name[0]  # E, S, or G
    
    for idx, (text, embedding) in enumerate(zip(clauses_text, embeddings), 1):
        clauses.append({
            "clause_id": f"{regulation_id}_{category_code}_{idx:03d}",
            "text": text,
            "embedding": embedding.tolist()
        })
    
    output = {
        "regulation_id": regulation_id,
        "category": category_name,
        "clauses": clauses
    }
    
    return output


def main():
    # Configuration - MODIFY THESE
    REGULATION_ID = "BRSR_2024"  # Change to your regulation name
    INPUT_DIR = "../extraction_classification/BRSR_ESG_buckets"  # Where classification outputs are
    OUTPUT_DIR = "BRSR_embeddings"  # Where to save embeddings
    
    print("=" * 70)
    print("EMBED REGULATION CLAUSES")
    print("=" * 70)
    print(f"Regulation ID: {REGULATION_ID}\n")
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    categories = {
        "environmental": "Environmental",
        "social": "Social",
        "governance": "Governance"
    }
    
    for file_name, category_name in categories.items():
        input_file = Path(INPUT_DIR) / f"{file_name}.json"
        
        if not input_file.exists():
            print(f"⚠️  Skipping {category_name} - file not found: {input_file}\n")
            continue
        
        # Process category
        output_data = load_and_embed_category(input_file, category_name, REGULATION_ID)
        
        # Save output
        output_file = Path(OUTPUT_DIR) / f"{file_name}_{OUTPUT_DIR}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        file_size = output_file.stat().st_size / (1024 * 1024)
        print(f"✓ Saved: {output_file} ({file_size:.2f} MB)\n")
    
    print("=" * 70)
    print(f"✓ Complete! Outputs saved to ./{OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()