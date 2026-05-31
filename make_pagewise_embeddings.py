### CORRECTED VERSION ###

import os
import torch
import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer

# ----------------------------
# SETTINGS & RELATIVE PATHS (FIX FOR CORRECTION 4)
# ----------------------------

# Dynamically gets the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Replaced absolute "D:\Dissertation_Project\..." paths with flexible relative paths
pdf_folder = os.path.join(BASE_DIR, "Chatbot", "Data_pdfs")
output_dir = os.path.join(BASE_DIR, "Chatbot", "Outputs")

chunk_size = 500
overlap = 150

# ----------------------------
# OUTPUT PATH
# ----------------------------

os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(
    output_dir,
    "all_pdfs_pagewise_embeddings.pt"
)

# ----------------------------
# LOAD EMBEDDING MODEL
# ----------------------------

embed_model = SentenceTransformer(
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
)

# ----------------------------
# FUNCTIONS
# ----------------------------

def load_pdf_pages(pdf_path):
    """
    Safely extracts pages from a PDF. 
    Prevents whole script failure if a PDF file is corrupted.
    """
    pages = []

    # --- FIX START: Handle PDF errors smoothly (CORRECTION 5) ---
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)

            for page in reader.pages:
                try:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
                    else:
                        pages.append("")
                except Exception as page_err:
                    print(f"   Warning: Could not read a specific page inside {os.path.basename(pdf_path)}. Skipping page. Error: {page_err}")
                    pages.append("")
                    
    except Exception as file_err:
        print(f"Error: Skipping corrupted, encrypted, or missing file: {pdf_path}. Details: {file_err}")
        return None # Return None so the main loop knows to skip this file safely
    # --- FIX END ---

    return pages


def split_text_into_chunks(text, chunk_size=500, overlap=150):
    """
    Splits input text into overlapping chunks safely.
    Prevents infinite loops by validating chunk_size and overlap parameters.
    """
    # --- FIX START: Prevent infinite loops (CORRECTION 1) ---
    chunk_size = int(chunk_size)
    overlap = int(overlap)

    if overlap >= chunk_size:
        print(f"Warning: Overlap ({overlap}) cannot be >= chunk_size ({chunk_size}). Reverting overlap to {chunk_size // 4}.")
        overlap = chunk_size // 4

    if chunk_size <= 0:
        print("Warning: chunk_size must be greater than 0. Falling back to default size of 500.")
        chunk_size = 500
        overlap = 100
    # --- FIX END ---

    step = chunk_size - overlap
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step
        
    return chunks


# ----------------------------
# MAIN
# ----------------------------

all_embeddings = []

# Ensure the PDF folder actually exists to avoid sudden runtime crashes
if not os.path.exists(pdf_folder):
    print(f"Error: The directory '{pdf_folder}' does not exist. Please place your source PDFs inside it.")
    pdf_files = []
else:
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

print("PDFs found:", pdf_files)

for pdf_name in pdf_files:

    pdf_path = os.path.join(pdf_folder, pdf_name)

    print("\nProcessing:", pdf_name)

    pages = load_pdf_pages(pdf_path)
    
    # --- FIX START: Skip over unreadable files gracefully (CORRECTION 5) ---
    if pages is None:
        continue
    # --- FIX END ---

    for page_number, page_text in enumerate(pages, start=1):

        chunks = split_text_into_chunks(
            page_text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk in chunks:

            if len(chunk.strip()) == 0:
                continue

            # --- FIX START: Normalize embedding at save time (CORRECTION 2) ---
            emb = embed_model.encode(chunk, normalize_embeddings=True)
            # --- FIX END ---

            all_embeddings.append({
                "pdf": pdf_name,
                "page": page_number,
                "chunk": chunk,
                "embedding": emb
            })

print("\nTotal chunks processed:", len(all_embeddings))

# Guard clause to ensure we don't save an empty file if data is missing
if all_embeddings:
    torch.save(all_embeddings, output_file)
    print("\nSuccessfully saved to:", output_file)
else:
    print("\nProcess finished with zero chunks. No data saved.")
