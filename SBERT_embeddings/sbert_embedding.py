"""
Regulation Data Preprocessing Pipeline
Parses regulation PDFs and structures them into a standardized format
"""

import json
import re
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import PyPDF2
from pathlib import Path

class RegulationPreprocessor:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with embedding model"""
        self.embedding_model = SentenceTransformer(model_name)
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def parse_csrd_structure(self, text: str) -> Dict[str, Any]:
        """
        Parse CSRD regulation into structured format
        Customize this based on actual CSRD document structure
        """
        regulation = {
            "regulation_id": "CSRD_2023_v1",
            "name": "Corporate Sustainability Reporting Directive",
            "version": "2023-v1",
            "effective_date": "2024-01-01",
            "jurisdiction": "EU",
            "categories": []
        }
        
        # Pattern to identify disclosure requirements
        # This is a simplified example - adjust based on actual document structure
        disclosure_pattern = r"(?:ESRS\s+)?([A-Z]+)-(\d+)\s*[:\-]\s*(.+?)(?=(?:ESRS|$))"
        
        matches = re.finditer(disclosure_pattern, text, re.DOTALL | re.MULTILINE)
        
        category_map = {}
        
        for match in matches:
            category_code = match.group(1)
            disclosure_num = match.group(2)
            disclosure_text = match.group(3).strip()
            
            # Extract title (first line) and description
            lines = disclosure_text.split('\n')
            title = lines[0].strip()
            description = ' '.join(lines[1:]).strip()[:500]  # Limit description length
            
            disclosure_id = f"CSRD_{category_code}_{disclosure_num}"
            
            # Extract keywords using simple NLP
            keywords = self._extract_keywords(title + " " + description)
            
            disclosure = {
                "disclosure_id": disclosure_id,
                "title": title,
                "description": description,
                "mandatory": self._is_mandatory(disclosure_text),
                "keywords": keywords,
                "embedding": None  # Will be populated later
            }
            
            # Organize into categories
            if category_code not in category_map:
                category_map[category_code] = {
                    "category_id": f"{category_code}_01",
                    "name": self._get_category_name(category_code),
                    "subcategories": [
                        {
                            "subcategory_id": f"{category_code}_01_01",
                            "name": "General Disclosures",
                            "disclosures": []
                        }
                    ]
                }
            
            category_map[category_code]["subcategories"][0]["disclosures"].append(disclosure)
        
        regulation["categories"] = list(category_map.values())
        return regulation
    
    def parse_brsr_structure(self, text: str) -> Dict[str, Any]:
        """Parse BRSR regulation structure"""
        regulation = {
            "regulation_id": "BRSR_2023_v1",
            "name": "Business Responsibility and Sustainability Report",
            "version": "2023-v1",
            "effective_date": "2023-04-01",
            "jurisdiction": "India",
            "categories": []
        }
        
        # BRSR is organized into 9 principles (P1-P9)
        # Section A: General Disclosures
        # Section B: Management and Process Disclosures
        # Section C: Principle-wise Performance Disclosure
        
        principle_pattern = r"Principle\s+(\d+)[:\-]\s*(.+?)(?=Principle|\Z)"
        matches = re.finditer(principle_pattern, text, re.DOTALL | re.MULTILINE)
        
        for match in matches:
            principle_num = match.group(1)
            principle_text = match.group(2).strip()
            
            # Extract disclosure requirements from principle text
            disclosures = self._extract_brsr_disclosures(principle_text, principle_num)
            
            category = {
                "category_id": f"BRSR_P{principle_num}",
                "name": f"Principle {principle_num}",
                "subcategories": [
                    {
                        "subcategory_id": f"BRSR_P{principle_num}_01",
                        "name": "Essential Indicators",
                        "disclosures": disclosures["essential"]
                    },
                    {
                        "subcategory_id": f"BRSR_P{principle_num}_02",
                        "name": "Leadership Indicators",
                        "disclosures": disclosures["leadership"]
                    }
                ]
            }
            regulation["categories"].append(category)
        
        return regulation
    
    def _extract_brsr_disclosures(self, text: str, principle_num: str) -> Dict[str, List[Dict]]:
        """Extract essential and leadership indicators from BRSR principle"""
        disclosures = {"essential": [], "leadership": []}
        
        # Pattern to match indicator questions
        indicator_pattern = r"(\d+\.\s+.+?)(?=\d+\.|$)"
        matches = re.finditer(indicator_pattern, text, re.DOTALL)
        
        for i, match in enumerate(matches, 1):
            indicator_text = match.group(1).strip()
            keywords = self._extract_keywords(indicator_text)
            
            disclosure = {
                "disclosure_id": f"BRSR_P{principle_num}_{i:03d}",
                "title": indicator_text[:200],
                "description": indicator_text,
                "mandatory": True,
                "keywords": keywords,
                "embedding": None
            }
            
            # Classify as essential or leadership (simplified logic)
            if i <= 5:  # First 5 are typically essential
                disclosures["essential"].append(disclosure)
            else:
                disclosures["leadership"].append(disclosure)
        
        return disclosures
    
    def generate_embeddings(self, regulation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate embeddings for all disclosures"""
        for category in regulation["categories"]:
            for subcategory in category["subcategories"]:
                for disclosure in subcategory["disclosures"]:
                    # Combine title, description, and keywords for embedding
                    text_for_embedding = f"{disclosure['title']} {disclosure['description']} {' '.join(disclosure['keywords'])}"
                    embedding = self.embedding_model.encode(text_for_embedding)
                    disclosure["embedding"] = embedding.tolist()
        
        return regulation
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Remove common words and extract meaningful terms
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 
                    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                    'would', 'should', 'could', 'may', 'might', 'must', 'can'}
        
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        keywords = [w for w in words if w not in stopwords]
        
        # Get top 10 most frequent keywords
        from collections import Counter
        keyword_counts = Counter(keywords)
        return [kw for kw, _ in keyword_counts.most_common(10)]
    
    def _is_mandatory(self, text: str) -> bool:
        """Determine if disclosure is mandatory"""
        mandatory_keywords = ['must', 'shall', 'required', 'mandatory', 'obligatory']
        return any(keyword in text.lower() for keyword in mandatory_keywords)
    
    def _get_category_name(self, code: str) -> str:
        """Map category code to human-readable name"""
        category_names = {
            "ENV": "Environmental",
            "S": "Social",
            "G": "Governance",
            "E": "Environmental",
            "ESRS": "General"
        }
        return category_names.get(code, "Other")
    
    def save_regulation(self, regulation: Dict[str, Any], output_path: str):
        """Save structured regulation to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(regulation, f, indent=2, ensure_ascii=False)
        print(f"Regulation saved to {output_path}")


# Example usage
if __name__ == "__main__":
    preprocessor = RegulationPreprocessor()
    
    # Process CSRD
    csrd_text = preprocessor.extract_text_from_pdf("regulations/csrd.pdf")
    csrd_structured = preprocessor.parse_csrd_structure(csrd_text)
    csrd_with_embeddings = preprocessor.generate_embeddings(csrd_structured)
    preprocessor.save_regulation(csrd_with_embeddings, "output/csrd_structured.json")
    
    # Process BRSR
    # brsr_text = preprocessor.extract_text_from_pdf("regulations/BRSR_2023.pdf")
    # brsr_structured = preprocessor.parse_brsr_structure(brsr_text)
    # brsr_with_embeddings = preprocessor.generate_embeddings(brsr_structured)
    # preprocessor.save_regulation(brsr_with_embeddings, "output/brsr_structured.json")
    
    print("Regulation preprocessing complete!")