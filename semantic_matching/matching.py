"""
Enhanced Semantic Matching & Gap Analysis with ESG-BERT Embeddings
- Finds exact disclosure paragraphs that match each regulation
- Provides comprehensive gap analysis
- Supports multiple matching strategies
"""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


class ESGMatcher:
    """Semantic matching engine for ESG regulations and disclosures."""
    
    def __init__(self, disclosure_file, regulation_file):
        """Load disclosure and regulation embeddings."""
        print(f"Loading data from:\n  - {disclosure_file}\n  - {regulation_file}")
        
        with open(disclosure_file, "r", encoding="utf-8") as f:
            self.disclosures_data = json.load(f)
        
        with open(regulation_file, "r", encoding="utf-8") as f:
            self.regulations_data = json.load(f)
        
        # Extract clauses and embeddings
        self.disclosure_clauses = self.disclosures_data["clauses"]
        self.regulation_clauses = self.regulations_data["clauses"]
        
        self.disclosure_embeddings = np.array([c["embedding"] for c in self.disclosure_clauses])
        self.regulation_embeddings = np.array([c["embedding"] for c in self.regulation_clauses])
        
        print(f"✓ Loaded {len(self.disclosure_clauses)} disclosure clauses")
        print(f"✓ Loaded {len(self.regulation_clauses)} regulation clauses\n")
    
    def compute_similarity_matrix(self):
        """Compute cosine similarity between all disclosure-regulation pairs."""
        print("Computing similarity matrix...")
        self.similarity_matrix = cosine_similarity(
            self.disclosure_embeddings, 
            self.regulation_embeddings
        )
        print(f"✓ Similarity matrix computed: {self.similarity_matrix.shape}\n")
        return self.similarity_matrix
    
    def match_regulations_to_disclosures(self, top_k=3, min_threshold=0.5):
        """
        For each regulation, find the top-k most similar disclosure paragraphs.
        This shows EXACTLY which paragraphs from the report address each regulation.
        
        Args:
            top_k: Number of best matching disclosures to return per regulation
            min_threshold: Minimum similarity score to consider a match
        
        Returns:
            List of regulation matches with their top disclosure paragraphs
        """
        print(f"Matching regulations → disclosures (top-{top_k}, threshold={min_threshold})...")
        
        matches = []
        
        for reg_idx, regulation in enumerate(self.regulation_clauses):
            # Get similarity scores for this regulation across all disclosures
            scores = self.similarity_matrix[:, reg_idx]
            
            # Get top-k matches
            top_indices = np.argsort(scores)[-top_k:][::-1]  # Descending order
            
            top_matches = []
            for disc_idx in top_indices:
                score = float(scores[disc_idx])
                if score >= min_threshold:
                    top_matches.append({
                        "disclosure_id": self.disclosure_clauses[disc_idx]["clause_id"],
                        "disclosure_text": self.disclosure_clauses[disc_idx]["text"],
                        "similarity_score": round(score, 4)
                    })
            
            match_entry = {
                "regulation_id": regulation["clause_id"],
                "regulation_text": regulation["text"],
                "matched_disclosures": top_matches,
                "best_match_score": round(float(scores[top_indices[0]]), 4),
                "is_covered": len(top_matches) > 0 and top_matches[0]["similarity_score"] >= min_threshold
            }
            
            matches.append(match_entry)
        
        print(f"✓ Matched {len(matches)} regulations\n")
        return matches
    
    def match_disclosures_to_regulations(self, threshold=0.75):
        """
        For each disclosure, find the best matching regulation.
        This shows how company paragraphs map to regulatory requirements.
        
        Args:
            threshold: Minimum similarity for valid match
        
        Returns:
            List of disclosure matches
        """
        print(f"Matching disclosures → regulations (threshold={threshold})...")
        
        matches = []
        
        for disc_idx, disclosure in enumerate(self.disclosure_clauses):
            # Find best regulation match
            best_reg_idx = np.argmax(self.similarity_matrix[disc_idx])
            best_score = float(self.similarity_matrix[disc_idx][best_reg_idx])
            best_regulation = self.regulation_clauses[best_reg_idx]
            
            matches.append({
                "disclosure_id": disclosure["clause_id"],
                "disclosure_text": disclosure["text"],
                "matched_regulation_id": best_regulation["clause_id"],
                "matched_regulation_text": best_regulation["text"],
                "similarity_score": round(best_score, 4),
                "is_strong_match": best_score >= threshold
            })
        
        print(f"✓ Matched {len(matches)} disclosures\n")
        return matches
    
    def gap_analysis(self, coverage_threshold=0.75):
        """
        Identify regulations that are NOT adequately covered by disclosures.
        
        Args:
            coverage_threshold: Minimum similarity to consider regulation covered
        
        Returns:
            Dictionary with coverage statistics and uncovered regulations
        """
        print(f"Performing gap analysis (threshold={coverage_threshold})...")
        
        # Find max similarity for each regulation
        max_similarities = np.max(self.similarity_matrix, axis=0)
        
        covered_regulations = []
        uncovered_regulations = []
        partially_covered = []
        
        for reg_idx, regulation in enumerate(self.regulation_clauses):
            max_score = float(max_similarities[reg_idx])
            best_disc_idx = int(np.argmax(self.similarity_matrix[:, reg_idx]))
            
            reg_info = {
                "regulation_id": regulation["clause_id"],
                "regulation_text": regulation["text"],
                "best_match_score": round(max_score, 4),
                "best_matching_disclosure_id": self.disclosure_clauses[best_disc_idx]["clause_id"],
                "best_matching_disclosure_text": self.disclosure_clauses[best_disc_idx]["text"]
            }
            
            if max_score >= coverage_threshold:
                covered_regulations.append(reg_info)
            elif max_score >= coverage_threshold * 0.7:  # 70% of threshold
                partially_covered.append(reg_info)
            else:
                uncovered_regulations.append(reg_info)
        
        coverage_stats = {
            "total_regulations": len(self.regulation_clauses),
            "fully_covered": len(covered_regulations),
            "partially_covered": len(partially_covered),
            "uncovered": len(uncovered_regulations),
            "coverage_percentage": round(100 * len(covered_regulations) / len(self.regulation_clauses), 2)
        }
        
        print(f"✓ Gap analysis complete:")
        print(f"  - Fully covered: {coverage_stats['fully_covered']} ({coverage_stats['coverage_percentage']}%)")
        print(f"  - Partially covered: {coverage_stats['partially_covered']}")
        print(f"  - Uncovered: {coverage_stats['uncovered']}\n")
        
        return {
            "coverage_statistics": coverage_stats,
            "covered_regulations": covered_regulations,
            "partially_covered_regulations": partially_covered,
            "uncovered_regulations": uncovered_regulations
        }


def main():
    # Run matching for all three categories
    categories = ["environmental", "social", "governance"]
    OUTPUT_DIR = "matching_output"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for CATEGORY in categories:
        DISCLOSURE_FILE = f"../SBERT_embeddings/Infosys_embeddings/{CATEGORY}_disc.json"
        REGULATION_FILE = f"../SBERT_embeddings/BRSR_embeddings/{CATEGORY}_reg.json"

        print("=" * 70)
        print("ESG SEMANTIC MATCHING & GAP ANALYSIS")
        print("=" * 70)
        print(f"Category: {CATEGORY.upper()}\n")

        try:
            matcher = ESGMatcher(DISCLOSURE_FILE, REGULATION_FILE)
            matcher.compute_similarity_matrix()
            regulation_matches = matcher.match_regulations_to_disclosures(top_k=3, min_threshold=0.5)
            disclosure_matches = matcher.match_disclosures_to_regulations(threshold=0.75)
            gap_analysis = matcher.gap_analysis(coverage_threshold=0.75)

            output = {
                "metadata": {
                    "category": matcher.disclosures_data["category"],
                    "company_name": matcher.disclosures_data["company_name"],
                    "report_year": matcher.disclosures_data["report_year"],
                    "regulation_id": matcher.regulations_data["regulation_id"],
                    "embedding_model": matcher.disclosures_data.get("embedding_model", "unknown"),
                    "analysis_date": str(np.datetime64('today'))
                },
                "regulation_to_disclosure_matches": regulation_matches,
                "disclosure_to_regulation_matches": disclosure_matches,
                "gap_analysis": gap_analysis
            }

            output_file = Path(OUTPUT_DIR) / f"matching_{CATEGORY}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            file_size = output_file.stat().st_size / (1024 * 1024)
            print(f"✓ Saved analysis to: {output_file} ({file_size:.2f} MB)")

            print("\n" + "=" * 70)
            print("SUMMARY")
            print("=" * 70)
            print(f"Total Regulations: {gap_analysis['coverage_statistics']['total_regulations']}")
            print(f"Coverage: {gap_analysis['coverage_statistics']['coverage_percentage']}%")
            print(f"Uncovered Regulations: {gap_analysis['coverage_statistics']['uncovered']}")
            print(f"Disclosure-Regulation Matches: {len(disclosure_matches)}")
            print("=" * 70)
        except FileNotFoundError as e:
            print(f"\n✗ File not found for category '{CATEGORY}': {e}")
            print("Please check your file paths in the configuration section.")
        except Exception as e:
            print(f"\n✗ Error for category '{CATEGORY}': {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n✗ File not found: {e}")
        print("Please check your file paths in the configuration section.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()