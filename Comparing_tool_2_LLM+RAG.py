import streamlit as st
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama

# --- PAGE CONFIG ---
st.set_page_config(page_title="AMR Biomedical RAG Chatbot", layout="wide")
st.title("🧬 Infectious Diseases & AMR Research Chatbot (RAG System)")

# --- MODE SELECTION ---
mode = st.sidebar.radio(
    "Select Mode",
    ["Tool 1: Only LLM", "Tool 2: LLM + RAG"]
)

@st.cache_resource
def load_data():
    data = torch.load(
        r"D:\Dissertation_Project\Chatbot\Outputs\all_pdfs_pagewise_embeddings.pt",
        weights_only=False
    )

    embeddings = np.array([item["embedding"] for item in data])
    chunks = [item["chunk"] for item in data]
    pdfs = [item["pdf"] for item in data]
    pages = [item["page"] for item in data]

    embed_model = SentenceTransformer(
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    )

    # normalize embeddings for better cosine stability
    embeddings = embeddings / np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    return embeddings, chunks, pdfs, pages, embed_model

embeddings, chunks, pdfs, pages, embed_model = load_data()

# --- SEARCH FUNCTION ---
def search(query, top_k=10):
    q_emb = embed_model.encode(query, normalize_embeddings=True)

    scores = cosine_similarity([q_emb], embeddings)[0]
    top_idx = scores.argsort()[::-1][:top_k]

    results = []
    for i in top_idx:
        if scores[i] > 0.30:
            results.append({
                "text": chunks[i],
                "pdf": pdfs[i],
                "page": pages[i],
                "score": float(scores[i])
            })

    return results

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are a senior Biomedical AI Research Assistant specializing in:

- Infectious Diseases (bacterial, viral, fungal, parasitic)
- Antimicrobial Resistance (AMR)
- Pharmacology of antibiotics and antivirals
- Molecular mechanisms of drug resistance
- Clinical microbiology and pathogenesis

STRICT INSTRUCTIONS:
1. Answer ONLY in 6–8 medically precise sentences.
2. Use standard biomedical terminology.
3. Prioritize mechanisms, pathways, and molecular details.
4. If RAG context is provided, base your answer ONLY on it.
5. If context is insufficient, explicitly state:
   "Insufficient contextual evidence in provided literature."
6. Do NOT hallucinate drug names, resistance genes, or mechanisms.
7. Avoid general explanations.
8. Maintain a professional scientific tone suitable for a postgraduate researcher.
"""

# --- LLM FUNCTION ---
def ask_llm_stream(question, results, use_rag=True):

    history_lines = []

    if "messages" in st.session_state:
        for m in st.session_state.messages[-5:]:
            history_lines.append(
                f"{m['role'].upper()}: {m['content']}"
            )

    history_str = "\n".join(history_lines)

    if use_rag:
        context_text = "\n".join(
            [r['text'] for r in results]
        ) if results else "No relevant literature retrieved."

        instruction = (
            "Answer strictly using the provided infectious disease / "
            "AMR literature context."
        )

    else:
        context_text = "NO KNOWLEDGEBASE ACCESS."

        instruction = (
            "Answer using only general biomedical knowledge. "
            "Do NOT assume paper-specific evidence."
        )

    user_message = f"""
INSTRUCTION:
{instruction}

CHAT HISTORY:
{history_str}

RESEARCH CONTEXT:
{context_text}

QUESTION:
{question}
"""

    return ollama.chat(
        model="qwen2.5:1.5b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        stream=True,
        options={"temperature": 0.1}
    )

# --- SESSION MEMORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- CHAT DISPLAY ---
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.chat_input(
    "Ask about AMR mechanisms, pathogens, or drug resistance..."
)

if question:

    st.chat_message("user").write(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    results = search(question)

    # --- UI TRACE ---
    if mode == "Tool 1: Only LLM":

        use_rag = False

    else:
        use_rag = True

        with st.expander(
            "🔬 RAG Retrieval Results",
            expanded=True
        ):

            if results:
                for r in results:
                    st.info(
                        f"{r['pdf']} | "
                        f"Page {r['page']} | "
                        f"Score: {r['score']:.3f}"
                    )

            else:
                st.error(
                    "No relevant biomedical context retrieved."
                )

    # --- STREAM RESPONSE ---
    with st.chat_message("assistant"):

        placeholder = st.empty()
        full_response = ""

        for chunk in ask_llm_stream(
            question,
            results,
            use_rag=use_rag
        ):

            content = chunk["message"]["content"]

            full_response += content

            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })