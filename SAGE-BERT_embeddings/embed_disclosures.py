"""
Generate SAGE-BERT embeddings for company disclosure clauses.
Uses fine-tuned ESG-BERT model for better semantic similarity.
Input: environmental.json, social.json, governance.json (from classification)
Output: environmental_Infosys_embeddings.json, social_Infosys_embeddings.json, governance_Infosys_embeddings.json
"""
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm


class SAGEBERTEmbedder:
    """Wrapper for SAGE-BERT (fine-tuned ESG-BERT) to generate sentence embeddings."""
    
    def __init__(self, model_path='../SAGE-BERT'):
        """
        Initialize SAGE-BERT model.
        
        Args:
            model_path: Path to fine-tuned SAGE-BERT model
                       Use '../SAGE-BERT' for fine-tuned model
                       Use 'nbroad/ESG-BERT' for base model
        """
        print(f"Loading model from: {model_path}")
        
        # Check if fine-tuned model exists
        if not Path(model_path).exists() and model_path == '../SAGE-BERT':
            print(f"⚠️  Fine-tuned model not found at {model_path}")
            print("   Falling back to base ESG-BERT model...")
            model_path = 'nbroad/ESG-BERT'
        
        self.model = SentenceTransformer(model_path)
        self.model_name = model_path
        
        print(f"✓ Model loaded!")
        print(f"  Embedding dimension: {self.model.get_sentence_embedding_dimension()}\n")
    
    def encode(self, sentences, batch_size=16, show_progress=True):
        """
        Generate embeddings for a list of sentences.
        
        Args:
            sentences: List of strings
            batch_size: Number of sentences to process at once
            show_progress: Show progress bar
            
        Returns:
            numpy array of embeddings (n_sentences, embedding_dim)
        """
        # SentenceTransformer handles batching and progress bar internally
        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for better similarity
        )
        
        return embeddings


def load_and_embed_category(embedder, category_file, category_name, company_name, report_year):
    """Load clauses, generate embeddings, and structure output."""
    
    # Load classified clauses
    with open(category_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    clauses_text = data["clauses"]
    print(f"Processing {len(clauses_text)} {category_name} clauses...")
    
    # Generate embeddings
    embeddings = embedder.encode(clauses_text, show_progress=True)
    
    # Structure output
    clauses = []
    category_code = category_name[0]  # E, S, or G
    company_code = company_name.replace(" ", "").upper()[:4]  # First 4 letters
    
    for idx, (text, embedding) in enumerate(zip(clauses_text, embeddings), 1):
        clauses.append({
            "clause_id": f"{company_code}_{report_year}_{category_code}_{idx:03d}",
            "text": text,
            "embedding": embedding.tolist()
        })
    
    output = {
        "company_name": company_name,
        "report_year": report_year,
        "category": category_name,
        "embedding_model": embedder.model_name,
        "embedding_dim": embeddings.shape[1],
        "total_clauses": len(clauses),
        "clauses": clauses
    }
    
    return output


def main():
    # Configuration - MODIFY THESE
    COMPANY_NAME = "Infosys"  # Change to your company name
    REPORT_YEAR = 2024  # Change to report year
    INPUT_DIR = "../extraction_classification/Infosys_ESG_buckets"  # Where classification outputs are
    OUTPUT_DIR = "Infosys_embeddings"  # Where to save embeddings
    
    # Model selection
    USE_FINE_TUNED = True  # Set to False to use base ESG-BERT
    
    print("=" * 70)
    print("EMBED COMPANY DISCLOSURE CLAUSES WITH SAGE-BERT")
    print("=" * 70)
    print(f"Company: {COMPANY_NAME}")
    print(f"Report Year: {REPORT_YEAR}")
    print(f"Using fine-tuned model: {USE_FINE_TUNED}\n")
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Initialize embedder once
    model_path = '../SAGE-BERT' if USE_FINE_TUNED else 'nbroad/ESG-BERT'
    embedder = SAGEBERTEmbedder(model_path=model_path)
    
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
        output_data = load_and_embed_category(
            embedder, input_file, category_name, COMPANY_NAME, REPORT_YEAR
        )
        
        # Save output with company name in filename
        output_file = Path(OUTPUT_DIR) / f"{file_name}_{COMPANY_NAME}_embeddings.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        file_size = output_file.stat().st_size / (1024 * 1024)
        print(f"✓ Saved: {output_file} ({file_size:.2f} MB)\n")
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Company: {COMPANY_NAME}")
    print(f"Report Year: {REPORT_YEAR}")
    print(f"Model used: {embedder.model_name}")
    print(f"Embedding dimension: {embedder.model.get_sentence_embedding_dimension()}")
    print(f"Output directory: ./{OUTPUT_DIR}/")
    print("=" * 70)
    print("\n✓ Complete! Ready for semantic matching.\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()