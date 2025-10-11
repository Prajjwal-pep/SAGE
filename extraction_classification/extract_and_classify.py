"""
Extract clauses from a PDF and classify each as Environmental, Social, or Governance using ESG-BERT.
Outputs three JSON files: environmental.json, social.json, governance.json.
"""

import fitz
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import time

# ESG-BERT model configuration
MODEL_NAME = "nbroad/ESG-BERT"

# Map 26 detailed ESG-BERT categories to E/S/G
LABELS_DETAILED = {
    0: "Social", 1: "Governance", 2: "Social", 3: "Governance",
    4: "Governance", 5: "Governance", 6: "Social", 7: "Governance",
    8: "Social", 9: "Social", 10: "Social", 11: "Social",
    12: "Governance", 13: "Environmental", 14: "Social", 15: "Environmental",
    16: "Social", 17: "Social", 18: "Governance", 19: "Environmental",
    20: "Environmental", 21: "Environmental", 22: "Governance", 23: "Environmental",
    24: "Environmental", 25: "Environmental"
}

print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded on device: {device}\n")


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    doc.close()
    return text


def extract_clauses(text):
    """Extract clauses from text with proper splitting."""
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    clauses = []
    enum_pattern = r'\s*(\([ivxlcdm]+\)|\([a-z]\)|\(\d+\))\s*'
    parts = re.split(enum_pattern, text, flags=re.IGNORECASE)
    
    current_clause = ""
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        if re.match(r'^\([ivxlcdm]+\)$|^\([a-z]\)$|^\(\d+\)$', part, re.IGNORECASE):
            if current_clause:
                processed = process_clause(current_clause)
                clauses.extend(processed)
            current_clause = ""
        else:
            current_clause += " " + part if current_clause else part
    
    if current_clause:
        processed = process_clause(current_clause)
        clauses.extend(processed)
    
    final_clauses = []
    seen = set()
    
    for clause in clauses:
        clause = re.sub(r'\s+', ' ', clause)
        clause = clause.strip(" .:;,\n\t")
        
        if not clause or len(clause) < 25:
            continue
        
        if not clause.endswith(('.', '!', '?')):
            clause += '.'
        
        clause = clause[0].upper() + clause[1:] if clause else clause
        
        if clause not in seen:
            final_clauses.append(clause)
            seen.add(clause)
    
    return final_clauses


def process_clause(text):
    """Process a single clause."""
    text = text.strip()
    if len(text) < 25:
        return [text] if text else []
    
    clauses = []
    sentences = re.split(r'\.\s+(?=[A-Z])', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 25:
            continue
        
        sub_clauses = split_on_keywords(sentence)
        clauses.extend(sub_clauses)
    
    return clauses


def split_on_keywords(text):
    """Split clause on list keywords."""
    text = text.strip()
    if len(text) < 25:
        return [text] if text else []
    
    clauses = []
    list_keywords = ['including', 'such as', 'comprising', 'namely', 'for example']
    
    has_list_keyword = any(keyword in text.lower() for keyword in list_keywords)
    
    if has_list_keyword:
        for keyword in list_keywords:
            pattern = rf'(?:,\s+)?{keyword}\s+(.+)$'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                prefix = text[:match.start()].strip()
                list_part = match.group(1).strip()
                
                items = re.split(r',\s*(?:and\s+)?|;\s*|\s+and\s+', list_part)
                
                for item in items:
                    item = item.strip(' .,;')
                    if len(item) > 10:
                        verb_match = re.search(r'(shall\s+\w+(?:\s+\w+){0,3})', prefix, re.IGNORECASE)
                        if verb_match:
                            verb_phrase = verb_match.group(1)
                            clause = f"The company {verb_phrase} {item}"
                        else:
                            clause = f"{prefix} {item}" if prefix else item
                        
                        if len(clause) > 25:
                            clauses.append(clause)
                
                return clauses if clauses else [text]
    
    if ';' in text:
        parts = text.split(';')
        for part in parts:
            part = part.strip()
            if len(part) > 25:
                clauses.append(part)
        return clauses if clauses else [text]
    
    return [text]


def classify_clauses(clauses, batch_size=16):
    """Classify clauses using ESG-BERT."""
    results = {"Environmental": [], "Social": [], "Governance": []}
    
    total = len(clauses)
    print(f"Classifying {total} clauses in batches of {batch_size}...\n")
    
    for i in range(0, total, batch_size):
        batch = clauses[i:i + batch_size]
        
        try:
            inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
            
            for clause, pred_idx in zip(batch, predictions):
                pred_idx_int = pred_idx.item()
                label = LABELS_DETAILED.get(pred_idx_int, "Social")
                results[label].append(clause)
        
        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {e}")
            continue
        
        processed = min(i + batch_size, total)
        percentage = (processed / total) * 100
        print(f"Progress: {processed}/{total} ({percentage:.1f}%)")
    
    return results


def save_to_json(classified, output_dir="output"):
    """Save results to JSON files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for label, data in classified.items():
        out_path = Path(output_dir) / f"{label.lower()}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"category": label, "count": len(data), "clauses": data}, f, indent=2, ensure_ascii=False)
        print(f"Saved {label}: {len(data)} clauses -> {out_path}")


def save_summary(classified, output_dir="output"):
    """Save summary statistics."""
    total = sum(len(clauses) for clauses in classified.values())
    summary = {
        "total_clauses": total,
        "categories": {
            label: {
                "count": len(clauses),
                "percentage": round(len(clauses) / total * 100, 2) if total > 0 else 0
            }
            for label, clauses in classified.items()
        }
    }
    
    summary_path = Path(output_dir) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    input_pdf = "regulations/BRSR.pdf"
    
    print("=" * 70)
    print("ESG-BERT CLAUSE CLASSIFIER")
    print("=" * 70)
    print(f"\nReading PDF: {input_pdf}\n")
    
    try:
        start_time = time.time()
        
        text = extract_text_from_pdf(input_pdf)
        print(f"Extracted {len(text):,} characters from PDF.")
        
        clauses = extract_clauses(text)
        print(f"Identified {len(clauses)} clauses.\n")
        
        classified = classify_clauses(clauses, batch_size=16)
        
        processing_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("CLASSIFICATION SUMMARY")
        print("=" * 70)
        total = sum(len(items) for items in classified.values())
        for category, items in classified.items():
            percentage = (len(items) / total * 100) if total > 0 else 0
            print(f"{category:15} : {len(items):3} clauses ({percentage:5.1f}%)")
        print(f"\nTotal: {total} clauses")
        print(f"Time: {processing_time:.1f}s")
        print(f"Speed: {total/processing_time:.1f} clauses/sec")
        print("=" * 70)
        
        print("\nSaving results...")
        save_to_json(classified, output_dir="output")
        save_summary(classified, output_dir="output")
        
        print("\nComplete! Output saved to ./output/\n")
        
    except FileNotFoundError:
        print(f"Error: File '{input_pdf}' not found.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()