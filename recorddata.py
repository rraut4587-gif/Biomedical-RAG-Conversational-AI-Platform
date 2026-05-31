import os
import torch
import PyPDF2
import numpy as np
from collections import defaultdict, Counter

# =========================================================
# LOAD DATA
# =========================================================

folder = r"D:\Dissertation_Project\Chatbot\Data_pdfs"

data = torch.load(
    r"D:\Dissertation_Project\Chatbot\Outputs\all_pdfs_pagewise_embeddings.pt",
    weights_only=False
)

# =========================================================
# INSPECT DATA STRUCTURE
# =========================================================

print("\nAVAILABLE KEYS INSIDE DATA:\n")
print(data[0].keys())

print("\nFIRST RECORD SAMPLE:\n")
print(data[0])

# =========================================================
# IMPORTANT:
# CHANGE THIS KEY IF NEEDED
# =========================================================

# Possible options:
# "chunk"
# "content"
# "text"
# "page_content"
# "sentence"

TEXT_KEY = "chunk"

# =========================================================
# STORAGE
# =========================================================

chunk_counts = Counter()

pages_in_embeddings = defaultdict(set)

embedding_shapes = defaultdict(list)

chunk_sizes = defaultdict(list)

# =========================================================
# PROCESS DATA
# =========================================================

for d in data:

    pdf_name = d["pdf"]

    chunk_counts[pdf_name] += 1

    pages_in_embeddings[pdf_name].add(d["page"])

    # embedding shape
    emb_shape = tuple(d["embedding"].shape)
    embedding_shapes[pdf_name].append(emb_shape)

    # =====================================================
    # GET CHUNK TEXT
    # =====================================================

    text = str(d.get(TEXT_KEY, ""))

    # character count
    char_len = len(text)

    # word count
    word_len = len(text.split())

    # save stats
    chunk_sizes[pdf_name].append({
        "chars": char_len,
        "words": word_len
    })

# =========================================================
# PRINT SUMMARY
# =========================================================

print("\nPDF SUMMARY\n")

print(
    f"{'S.No':5} | "
    f"{'PDF':40} | "
    f"{'Real':5} | "
    f"{'Embedded':8} | "
    f"{'Chunks':6} | "
    f"{'AvgWords':10} | "
    f"{'MinWords':10} | "
    f"{'MaxWords':10} | "
    f"{'AvgChars':10} | "
    f"{'Emb Shape':15} | "
    f"Missing"
)

print("-" * 220)

serial = 1

for f in os.listdir(folder):

    if not f.endswith(".pdf"):
        continue

    path = os.path.join(folder, f)

    # =====================================================
    # REAL PAGES
    # =====================================================

    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        real_pages = len(reader.pages)

    # =====================================================
    # EMBEDDED PAGES
    # =====================================================

    embedded_pages = sorted(pages_in_embeddings[f])

    embedded_count = len(embedded_pages)

    chunks = chunk_counts[f]

    # =====================================================
    # MISSING PAGES
    # =====================================================

    missing = [
        p for p in range(1, real_pages + 1)
        if p not in embedded_pages
    ]

    # =====================================================
    # EMBEDDING SHAPES
    # =====================================================

    unique_shapes = list(set(embedding_shapes[f]))

    # =====================================================
    # CHUNK SIZE STATS
    # =====================================================

    words_list = [x["words"] for x in chunk_sizes[f]]

    chars_list = [x["chars"] for x in chunk_sizes[f]]

    if len(words_list) > 0:

        avg_words = round(np.mean(words_list), 1)
        min_words = min(words_list)
        max_words = max(words_list)

        avg_chars = round(np.mean(chars_list), 1)

    else:

        avg_words = 0
        min_words = 0
        max_words = 0
        avg_chars = 0

    # =====================================================
    # PRINT
    # =====================================================

    print(
        f"{serial:5} | "
        f"{f[:40]:40} | "
        f"{real_pages:5} | "
        f"{embedded_count:8} | "
        f"{chunks:6} | "
        f"{avg_words:10} | "
        f"{min_words:10} | "
        f"{max_words:10} | "
        f"{avg_chars:10} | "
        f"{str(unique_shapes):15} | "
        f"{missing}"
    )

    serial += 1







# import os
# import torch
# import PyPDF2
# from collections import defaultdict, Counter

# folder = r"C:\Users\rraut\OneDrive\Desktop\Dissertation\Work_For_10-13_March\Data_PDFs"

# data = torch.load(
#     "all_pdfs_pagewise_embeddings.pt",
#     weights_only=False
# )

# chunk_counts = Counter([d["pdf"] for d in data])

# pages_in_embeddings = defaultdict(set)

# for d in data:
#     pages_in_embeddings[d["pdf"]].add(d["page"])


# print("\nPDF SUMMARY\n")

# print(
#     f"{'PDF':40} | {'Real':5} | {'Embedded':8} | {'Chunks':6} | Missing"
# )

# print("-" * 90)


# for f in os.listdir(folder):

#     if not f.endswith(".pdf"):
#         continue

#     path = os.path.join(folder, f)

#     # real pages
#     with open(path, "rb") as file:
#         reader = PyPDF2.PdfReader(file)
#         real_pages = len(reader.pages)

#     # embedded pages
#     embedded_pages = sorted(pages_in_embeddings[f])

#     embedded_count = len(embedded_pages)

#     chunks = chunk_counts[f]

#     # missing pages
#     missing = [
#         p for p in range(1, real_pages + 1)
#         if p not in embedded_pages
#     ]

#     print(
#         f"{f[:40]:40} | "
#         f"{real_pages:5} | "
#         f"{embedded_count:8} | "
#         f"{chunks:6} | "
#         f"{missing}"
#     )





# import os
# import torch
# import PyPDF2
# from collections import defaultdict, Counter

# folder = r"C:\Users\rraut\OneDrive\Desktop\Dissertation\Work_For_10-13_March\Data_PDFs"

# data = torch.load(
#     "all_pdfs_pagewise_embeddings.pt",
#     weights_only=False
# )

# chunk_counts = Counter([d["pdf"] for d in data])

# pages_in_embeddings = defaultdict(set)

# # NEW: store embedding shapes per PDF
# embedding_shapes = defaultdict(list)

# for d in data:
#     pages_in_embeddings[d["pdf"]].add(d["page"])
    
#     # get shape of embedding
#     emb_shape = tuple(d["embedding"].shape)
#     embedding_shapes[d["pdf"]].append(emb_shape)


# print("\nPDF SUMMARY\n")

# print(
#     f"{'S.No':5} | {'PDF':40} | {'Real':5} | {'Embedded':8} | {'Chunks':6} | {'Emb Shape':15} | Missing"
# )

# print("-" * 130)


# for i, f in enumerate(os.listdir(folder), start=1):

#     if not f.endswith(".pdf"):
#         continue

#     path = os.path.join(folder, f)

#     # real pages
#     with open(path, "rb") as file:
#         reader = PyPDF2.PdfReader(file)
#         real_pages = len(reader.pages)

#     # embedded pages
#     embedded_pages = sorted(pages_in_embeddings[f])
#     embedded_count = len(embedded_pages)

#     chunks = chunk_counts[f]

#     # missing pages
#     missing = [
#         p for p in range(1, real_pages + 1)
#         if p not in embedded_pages
#     ]

#     # embedding shapes
#     unique_shapes = list(set(embedding_shapes[f]))

#     print(
#         f"{i:5} | "
#         f"{f[:40]:40} | "
#         f"{real_pages:5} | "
#         f"{embedded_count:8} | "
#         f"{chunks:6} | "
#         f"{str(unique_shapes):15} | "
#         f"{missing}"
#     )



