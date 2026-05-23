import os
import torch
import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer

# ----------------------------
# SETTINGS
# ----------------------------

pdf_folder = r"D:\Dissertation_Project\Chatbot\Data_pdfs"

chunk_size = 500
overlap = 150

# ----------------------------
# OUTPUT PATH
# ----------------------------

output_dir = r"D:\Dissertation_Project\Chatbot\Outputs"

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

    pages = []

    with open(pdf_path, "rb") as f:

        reader = PyPDF2.PdfReader(f)

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages.append(text)
            else:
                pages.append("")

    return pages


def split_text_into_chunks(text, chunk_size=500, overlap=150):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ----------------------------
# MAIN
# ----------------------------

all_embeddings = []

pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

print("PDFs found:", pdf_files)

for pdf_name in pdf_files:

    pdf_path = os.path.join(pdf_folder, pdf_name)

    print("\nProcessing:", pdf_name)

    pages = load_pdf_pages(pdf_path)

    for page_number, page_text in enumerate(pages, start=1):

        chunks = split_text_into_chunks(
            page_text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk in chunks:

            if len(chunk.strip()) == 0:
                continue

            emb = embed_model.encode(chunk)

            all_embeddings.append({
                "pdf": pdf_name,
                "page": page_number,
                "chunk": chunk,
                "embedding": emb
            })

print("\nTotal chunks:", len(all_embeddings))

torch.save(all_embeddings, output_file)

print("\nSaved to:", output_file)