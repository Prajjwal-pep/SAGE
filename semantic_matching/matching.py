"""
Semantic matching between company disclosures and regulations.
Performs clause-wise similarity matching and gap analysis.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


def load_embeddings(file_path):
    """Load embeddings JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def compute_best_matches(disclosure_clauses, regulation_clauses):
    """
    For each disclosure clause, find the best matching regulation clause.
    Returns: list of (disclosure, best_regulation, similarity_score)
    """
    matches = []
    
    # Extract embeddings
    disc_embeddings = np.array([c["embedding"] for c in disclosure_clauses])
    reg_embeddings = np.array([c["embedding"] for c in regulation_clauses])
    
    # Compute similarity matrix: rows=disclosures, cols=regulations
    similarity_matrix = cosine_similarity(disc_embeddings, reg_embeddings)
    
    # For each disclosure, find best matching regulation
    for disc_idx, disc_clause in enumerate(disclosure_clauses):
        similarities = similarity_matrix[disc_idx]
        best_reg_idx = np.argmax(similarities)
        best_similarity = similarities[best_reg_idx]
        
        matches.append({
            "disclosure": disc_clause,
            "regulation": regulation_clauses[best_reg_idx],
            "similarity": float(best_similarity)
        })
    
    return matches, similarity_matrix


def find_uncovered_regulations(regulation_clauses, similarity_matrix, threshold=0.5):
    """
    Find regulations that are not well-covered by any disclosure.
    A regulation is uncovered if its max similarity across all disclosures < threshold.
    """
    uncovered = []
    
    # Max similarity for each regulation (across all disclosures)
    max_similarities = similarity_matrix.max(axis=0)
    
    for reg_idx, reg_clause in enumerate(regulation_clauses):
        max_sim = max_similarities[reg_idx]
        if max_sim < threshold:
            uncovered.append({
                "regulation_id": reg_clause["clause_id"],
                "regulation_text": reg_clause["text"],
                "best_similarity": float(max_sim)
            })
    
    return uncovered


def save_matches(matches, output_file, match_threshold=0.6):
    """Save matches above threshold to JSON file."""
    filtered_matches = []
    
    for match in matches:
        if match["similarity"] >= match_threshold:
            filtered_matches.append({
                "disclosure_id": match["disclosure"]["clause_id"],
                "disclosure_text": match["disclosure"]["text"],
                "matched_regulation_id": match["regulation"]["clause_id"],
                "matched_regulation_text": match["regulation"]["text"],
                "similarity_score": round(match["similarity"], 4)
            })
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_matches": len(filtered_matches),
            "match_threshold": match_threshold,
            "matches": filtered_matches
        }, f, indent=2, ensure_ascii=False)
    
    return len(filtered_matches)


def save_gaps(uncovered, output_file, gap_threshold=0.5):
    """Save uncovered regulations to JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_gaps": len(uncovered),
            "gap_threshold": gap_threshold,
            "description": f"Regulations with similarity score below {gap_threshold}",
            "uncovered_regulations": uncovered
        }, f, indent=2, ensure_ascii=False)
    
    return len(uncovered)


def process_category(regulation_file, disclosure_file, category_name, output_dir, 
                     match_threshold=0.6, gap_threshold=0.5):
    """Process a single category (Environmental, Social, or Governance)."""
    
    print(f"\n{'='*70}")
    print(f"Processing {category_name.upper()}")
    print(f"{'='*70}")
    
    # Load data
    print(f"Loading regulation: {regulation_file.name}")
    reg_data = load_embeddings(regulation_file)
    reg_clauses = reg_data["clauses"]
    
    print(f"Loading disclosure: {disclosure_file.name}")
    disc_data = load_embeddings(disclosure_file)
    disc_clauses = disc_data["clauses"]
    
    print(f"\nRegulations: {len(reg_clauses)} clauses")
    print(f"Disclosures: {len(disc_clauses)} clauses")
    
    # Compute matches
    print("\nComputing semantic similarities...")
    matches, similarity_matrix = compute_best_matches(disc_clauses, reg_clauses)
    
    # Find gaps
    print("Identifying gaps...")
    uncovered = find_uncovered_regulations(reg_clauses, similarity_matrix, gap_threshold)
    
    # Save results
    match_file = output_dir / f"{category_name}_matches.json"
    gap_file = output_dir / f"{category_name}_gaps.json"
    
    num_matches = save_matches(matches, match_file, match_threshold)
    num_gaps = save_gaps(uncovered, gap_file, gap_threshold)
    
    # Print summary
    print(f"\n✓ Matches (≥{match_threshold}): {num_matches}/{len(matches)} → {match_file.name}")
    print(f"✓ Gaps (<{gap_threshold}): {num_gaps}/{len(reg_clauses)} → {gap_file.name}")
    
    # Calculate coverage
    coverage = ((len(reg_clauses) - num_gaps) / len(reg_clauses)) * 100 if reg_clauses else 0
    print(f"\n📊 Coverage: {coverage:.1f}% of regulations are addressed")
    
    return {
        "category": category_name,
        "total_regulations": len(reg_clauses),
        "total_disclosures": len(disc_clauses),
        "matches_above_threshold": num_matches,
        "gaps": num_gaps,
        "coverage_percent": round(coverage, 2)
    }


def main():
    # Configuration - MODIFY THESE
    REGULATION_DIR = "../SBERT_embeddings/BRSR_embeddings"
    DISCLOSURE_DIR = "../SBERT_embeddings/Infosys_embeddings"
    OUTPUT_DIR = "matching_output"
    
    # Thresholds
    MATCH_THRESHOLD = 0.6  # Minimum similarity to consider a match
    GAP_THRESHOLD = 0.5  # Below this = uncovered regulation
    
    print("="*70)
    print("ESG SEMANTIC MATCHING & GAP ANALYSIS")
    print("="*70)
    print(f"\nMatch Threshold: {MATCH_THRESHOLD}")
    print(f"Gap Threshold: {GAP_THRESHOLD}")
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each category
    categories = ["environmental", "social", "governance"]
    results = []
    
    for category in categories:
        reg_file = Path(REGULATION_DIR) / f"{category}_BRSR_embeddings.json"
        disc_file = Path(DISCLOSURE_DIR) / f"{category}_Infosys_embeddings.json"
        
        # Check if files exist
        if not reg_file.exists():
            print(f"\n⚠️  Skipping {category} - regulation file not found: {reg_file}")
            continue
        if not disc_file.exists():
            print(f"\n⚠️  Skipping {category} - disclosure file not found: {disc_file}")
            continue
        
        # Process category
        result = process_category(
            reg_file, disc_file, category, output_dir,
            MATCH_THRESHOLD, GAP_THRESHOLD
        )
        results.append(result)
    
    # Overall summary
    if results:
        print(f"\n{'='*70}")
        print("OVERALL SUMMARY")
        print(f"{'='*70}")
        
        total_regs = sum(r["total_regulations"] for r in results)
        total_gaps = sum(r["gaps"] for r in results)
        overall_coverage = ((total_regs - total_gaps) / total_regs * 100) if total_regs > 0 else 0
        
        for result in results:
            print(f"\n{result['category'].upper()}:")
            print(f"  Regulations: {result['total_regulations']}")
            print(f"  Disclosures: {result['total_disclosures']}")
            print(f"  Strong Matches: {result['matches_above_threshold']}")
            print(f"  Gaps: {result['gaps']}")
            print(f"  Coverage: {result['coverage_percent']}%")
        
        print(f"\n{'='*70}")
        print(f"OVERALL COVERAGE: {overall_coverage:.1f}%")
        print(f"{'='*70}")
        
        # Save summary
        summary_file = output_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "match_threshold": MATCH_THRESHOLD,
                "gap_threshold": GAP_THRESHOLD,
                "categories": results,
                "overall_coverage_percent": round(overall_coverage, 2),
                "total_regulations": total_regs,
                "total_gaps": total_gaps
            }, f, indent=2)
        
        print(f"\n✓ Summary saved to: {summary_file}")
    
    print(f"\n✓ Complete! All results saved to ./{OUTPUT_DIR}/\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()