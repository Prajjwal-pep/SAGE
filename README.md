# SAGE: Sustainability Alignment and Governance Engine

SAGE is a semantic analysis system designed to automate ESG (Environmental, Social, and Governance) compliance evaluation by aligning company sustainability disclosures with regulatory frameworks such as BRSR, GRI, IFRS, and CSRD.

The system leverages transformer-based models and similarity techniques to perform clause-level matching and identify compliance gaps efficiently.

---

## 🚀 Features

- Automated clause-to-disclosure alignment
- ESG (Environmental, Social, Governance) categorization
- Semantic similarity using transformer embeddings
- Gap detection (Complete / Partial / Missing)
- Fine-tuned model (SAGE-BERT) for improved accuracy
- Scalable and framework-agnostic design

---

## 🧠 Model Overview

### SAGE-BERT

SAGE-BERT is a fine-tuned version of ESG-BERT optimized for semantic similarity tasks.

- Base Model: ESG-BERT
- Fine-tuning: Triplet Loss
- Embedding Size: 768
- Pooling: Mean Pooling
- Similarity Metric: Cosine Similarity

### Why Fine-tuning?

Pre-trained ESG-BERT is designed for classification tasks, not similarity.  
SAGE-BERT improves this by learning:

- Anchor (Regulation)
- Positive (Matching Disclosure)
- Negative (Non-matching Disclosure)

This ensures:
- Matching clauses → closer in embedding space  
- Non-matching clauses → pushed farther apart  

---

## ⚙️ Tech Stack

- Python 3.x
- FastAPI (Backend)
- Uvicorn (Server)
- Transformers (Hugging Face)
- PyTorch
- Sentence-Transformers
- Scikit-learn
- NumPy
- PyMuPDF

---

## 📂 Data Sources

- ESG Frameworks: BRSR, GRI, IFRS, CSRD
- Company Reports: Infosys, HDFC Bank, ICICI Bank
- Triplet Dataset: Anchor–Positive–Negative clause pairs for fine-tuning

---

## 🔄 Workflow

1. Collect ESG frameworks and company reports
2. Extract text from PDFs (PyMuPDF)
3. Segment text into clauses and disclosures
4. Generate embeddings using SAGE-BERT
5. Compute cosine similarity
6. Classify results:
   - Complete
   - Partial
   - Missing
7. Output structured compliance report

---

## 📊 Evaluation

- Similarity-based classification
- Coverage analysis across ESG pillars
- Validation using triplet accuracy
- Thresholds:
  - ≥ 0.75 → Strong Match
  - 0.5 – 0.75 → Partial Match
  - < 0.5 → Gap

---

## 📌 Key Contributions

- Fine-tuned ESG-specific similarity model (SAGE-BERT)
- Automated clause-level ESG compliance mapping
- Triplet-based training approach
- Scalable ESG analysis pipeline

---


## 🔮 Future Scope

- Multilingual ESG analysis
- Larger and diverse datasets
- Advanced transformer models (RoBERTa, DeBERTa)
- Interactive dashboard for visualization
- Integration with regulatory platforms

---

## 📁 Project Structure


<img width="1751" height="850" alt="SAGE (3)" src="https://github.com/user-attachments/assets/a2394564-9855-4f0f-a8db-a3ee8c787e7b" />
