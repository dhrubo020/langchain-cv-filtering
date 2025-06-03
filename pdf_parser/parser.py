import os
import fitz  # PyMuPDF
from typing import List, Dict, Optional
import re

# ----------- Extraction Logic -----------

import re

def clean_text(raw_text: str) -> str:
    # Normalize encoding issues
    text = raw_text.encode('utf-8', 'ignore').decode('utf-8', 'ignore') 

    # Replace bullet and dash-like characters with a newline or dash
    bullet_chars = ['·', '\u00b7', '\u2022']
    for ch in bullet_chars:
        text = text.replace(f'\n{ch}', '\n')
        text = text.replace(ch, '')  # standalone bullets

    dash_chars = ['–', '—', '−', '\u2013', '\u2014']
    for ch in dash_chars:
        text = text.replace(ch, '-')  # normalize to ASCII dash

    # Normalize dash formatting
    text = re.sub(r'-{2,}', '-', text)         # multiple dashes → single
    text = re.sub(r'\s*-\s*', ' - ', text)     # normalize spacing around dash

    # Normalize newlines
    text = re.sub(r'\r\n|\r|\n+', '\n', text)

    # Remove unwanted punctuation characters (but preserve dashes and slashes if needed)
    text = re.sub(r'[,_&]', '', text)

    # Remove common encoding artifacts
    text = re.sub(r'\s*c7\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*c2\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*b7\s*', '', text, flags=re.IGNORECASE)

    # Remove non-ASCII leftovers
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Remove lines starting with dashes or dots (replaced with newline if needed)
    text = re.sub(r'\n[\-\.\u00b7]+\s*', '\n', text)

    return text.strip()


def preprocess_text(raw_text: str) -> str:
    # Normalize text for consistent parsing
    return re.sub(r'\s+\n', '\n', raw_text).strip()

def extract_name(lines: List[str]) -> Optional[str]:
    # Rule 1: Look for explicit label like 'Name:'
    for line in lines:
        if re.match(r'(?i)^name\s*[:\-]', line):
            return re.sub(r'(?i)^name\s*[:\-]\s*', '', line).strip()

    # Rule 2: Check line 0 if it's likely a name (not email/phone/link)
    if lines:
        first_line = lines[0]
        if not re.search(r'@|http|www|\d', first_line):
            return first_line.strip()

    # Rule 3: Fallback – Find capitalized name-like patterns (Firstname Lastname)
    for line in lines[:5]:  # Search only in top few lines
        match = re.match(r'^([A-Z][a-z]+\s+[A-Z][a-z]+)$', line.strip())
        if match:
            return match.group(1)
    return None

def extract_email(text: str) -> Optional[str]:
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group(0) if match else None


def extract_phone(text: str) -> Optional[str]:
    match = re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    return match.group(0) if match else None


def extract_location(lines: List[str]) -> Optional[str]:
    for line in lines:
        if re.search(r'\b[A-Z][a-z]+,\s+[A-Z]{2}\b', line):
            return line.strip()
    return None


def extract_section(text: str, start: str, stop_keywords: List[str]) -> Optional[str]:
    pattern = rf'{start}([\s\S]+?)\b(?:{"|".join(stop_keywords)}|$)'
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def extract_skills(text: str) -> str:
    skills_block = extract_section(text, 'SKILLS', ['WORK EXPERIENCE', 'EDUCATION', 'CERTIFICATES'])
    if not skills_block:
        return ""
    skills = re.split(r'[,\n;•]+', skills_block)
    return " | ".join(s.strip() for s in skills if s.strip())

def extract_section(text: str, start: str, ends: List[str]) -> Optional[str]:
    pattern = rf'{start}([\s\S]*?)(?:{"|".join(ends)}|$)'
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else None

def extract_education(text: str) -> Dict:
    edu_block = extract_section(text, 'EDUCATION', ['SKILLS', 'WORK EXPERIENCE', 'CERTIFICATES'])
    if not edu_block:
        return {}
    clean_lines = [l.strip() for l in edu_block.split('\n') if l.strip()]
    return " | ".join(clean_lines) if clean_lines else None


def extract_work_experience(text: str, fallback_location: Optional[str] = None) -> List[Dict]:
    work_block = extract_section(text, 'WORK EXPERIENCE', ['EDUCATION', 'SKILLS', 'CERTIFICATES'])
    if not work_block:
        return []

    experiences = []
    blocks = re.split(r'\n(?=\w.*\s-\s)', work_block)

    for block in blocks:
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if not lines:
            continue

        title_line = lines[0]
        meta_line = lines[1] if len(lines) > 1 else ''
        description_lines = lines[2:] if len(lines) > 2 else []

        # Try to get duration
        date_match = re.search(r'([A-Za-z]+\s+\d{4})\s*-\s*(current|[A-Za-z]+\s+\d{4})', meta_line, re.IGNORECASE)
        duration = f"{date_match.group(1)} - {date_match.group(2)}" if date_match else ""

        # Try to get location
        location_match = re.search(r'\b([A-Z][a-z]+,\s+[A-Z]{2})\b', meta_line)
        location = location_match.group(0) if location_match else fallback_location or ""

        parts = [title_line]
        if duration or location:
            parts.append(f"{duration} | {location}".strip(" |"))
        if description_lines:
            parts.append(" ".join(description_lines))

        full_text = " | ".join([p for p in parts if p])

        experiences.append({"text": full_text})

    return experiences


def extract_structured_fields(text: str) -> Dict:
    text = preprocess_text(text)
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    return {
        "name": extract_name(lines),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "location": extract_location(lines),
        "skills": extract_skills(text),
        "education": extract_education(text),
        "work_experience": extract_work_experience(text, fallback_location=extract_location(lines)),
    }

# ----------- Main PDF Processing -----------

def extract_info_from_pdf(file_path: str) -> Dict:
    doc = fitz.open(file_path)
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    parsed = extract_structured_fields(full_text)
    return parsed

def extract_from_folder(pdf_folder: str) -> List[Dict]:
    results = []
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            info = extract_info_from_pdf(pdf_path)
            results.append(info)
    return results

# ----------- Example Usage -----------

if __name__ == "__main__":
    folder_path = "./pdf"  # Replace with your folder path
    all_data = extract_from_folder(folder_path)
    
    import json
    print(json.dumps(all_data, indent=2))
