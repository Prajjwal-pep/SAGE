"""
Generate ESG-BERT embeddings for company disclosure clauses.
Input: environmental.json, social.json, governance.json (from classification)
Output: environmental_disc.json, social_disc.json, governance_disc.json
"""
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm


class ESGBERTEmbedder:
    """Wrapper for ESG-BERT to generate sentence embeddings."""
    
    def __init__(self, model_name='nbroad/ESG-BERT', device=None):
        """Initialize ESG-BERT model and tokenizer."""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading ESG-BERT model on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        print("✓ Model loaded!\n")
    
    def mean_pooling(self, token_embeddings, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
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
        all_embeddings = []
        
        iterator = range(0, len(sentences), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings")
        
        with torch.no_grad():
            for i in iterator:
                batch = sentences[i:i + batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get embeddings
                outputs = self.model(**encoded)
                
                # Mean pooling
                embeddings = self.mean_pooling(
                    outputs.last_hidden_state,
                    encoded['attention_mask']
                )
                
                # Normalize (for better cosine similarity)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)


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
        "embedding_model": "nbroad/ESG-BERT",
        "embedding_dim": embeddings.shape[1],
        "clauses": clauses
    }
    
    return output


def main():
    # Configuration - MODIFY THESE
    COMPANY_NAME = "Infosys"  # Change to your company name
    REPORT_YEAR = 2024  # Change to report year
    INPUT_DIR = "../extraction_classification/Infosys_ESG_buckets"  # Where classification outputs are
    OUTPUT_DIR = "Infosys_embeddings"  # Where to save embeddings
    
    print("=" * 70)
    print("EMBED COMPANY DISCLOSURE CLAUSES WITH ESG-BERT")
    print("=" * 70)
    print(f"Company: {COMPANY_NAME}")
    print(f"Report Year: {REPORT_YEAR}\n")
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Initialize embedder once
    embedder = ESGBERTEmbedder()
    
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
        
        # Save output
        output_file = Path(OUTPUT_DIR) / f"{file_name}_disc.json"
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