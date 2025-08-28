"""PDF reader module for extracting content from Jeep Patriot manual."""

import PyPDF2
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class PatriotManualReader:
    """Reads and processes the Jeep Patriot manual PDF."""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.manual_content = {}
        
    def extract_text(self) -> str:
        """Extract all text from the PDF."""
        try:
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                return text
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            return ""
    
    def extract_sections(self) -> Dict[str, str]:
        """Extract manual content organized by sections."""
        full_text = self.extract_text()
        
        # Common manual sections to look for
        sections = {
            "maintenance": "",
            "troubleshooting": "",
            "engine": "",
            "transmission": "",
            "electrical": "",
            "brakes": "",
            "suspension": "",
            "air_conditioning": "",
            "heating": ""
        }
        
        # Simple keyword-based section extraction
        lines = full_text.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower()
            
            # Identify section headers
            if any(keyword in line_lower for keyword in ["maintenance", "service"]):
                current_section = "maintenance"
            elif any(keyword in line_lower for keyword in ["troubleshoot", "problem", "diagnostic"]):
                current_section = "troubleshooting"
            elif "engine" in line_lower:
                current_section = "engine"
            elif "transmission" in line_lower:
                current_section = "transmission"
            elif "electrical" in line_lower:
                current_section = "electrical"
            elif "brake" in line_lower:
                current_section = "brakes"
            elif "suspension" in line_lower:
                current_section = "suspension"
            elif any(keyword in line_lower for keyword in ["air conditioning", "a/c", "hvac"]):
                current_section = "air_conditioning"
            elif "heating" in line_lower:
                current_section = "heating"
            
            # Add content to current section
            if current_section and line.strip():
                sections[current_section] += line + "\n"
        
        self.manual_content = sections
        return sections
    
    def search_manual(self, query: str) -> List[str]:
        """Search for specific content in the manual."""
        full_text = self.extract_text()
        query_lower = query.lower()
        
        matching_lines = []
        lines = full_text.split('\n')
        
        for i, line in enumerate(lines):
            if query_lower in line.lower():
                # Include context (previous and next lines)
                context_start = max(0, i - 2)
                context_end = min(len(lines), i + 3)
                context = '\n'.join(lines[context_start:context_end])
                matching_lines.append(context)
        
        return matching_lines[:10]  # Limit to top 10 matches
