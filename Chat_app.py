## FINAL WORKING SCRIPT ###

import streamlit as st
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama

st.set_page_config(page_title="Infection and AMR AI Chat Assistant", layout="centered")
st.title("🧬 Infection and AMR AI Chat Assistant")

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
    embed_model = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    return embeddings, chunks, pdfs, pages, embed_model

embeddings, chunks, pdfs, pages, embed_model = load_data()

# ============================================
# SEARCH 
# ============================================

def search(query, top_k=4):
    q_emb = embed_model.encode(query, normalize_embeddings=True, show_progress_bar=False)
    scores = cosine_similarity([q_emb], embeddings)[0]
    top_idx = scores.argsort()[::-1][:top_k]
    
    results = []
    for i in top_idx:
        # Lowered threshold slightly to ensure we get context if it exists
        if scores[i] > 0.30: 
            results.append({"text": chunks[i], "pdf": pdfs[i], "page": pages[i]})
    return results

# ============================================
# LLM LOGIC (STREAMING & LENGTH CONTROL)
# ============================================

def ask_llm_stream(question, results):
    # 1. Pull the last 5 messages from Streamlit's session state
    # This is the only way the model will 'remember' the influenza question.
    history_lines = []
    if "messages" in st.session_state:
        for m in st.session_state.messages[-5:]:
            history_lines.append(f"{m['role'].upper()}: {m['content']}")
    history_str = "\n".join(history_lines)

    # 2. Re-engineered System Prompt
    system_prompt = (
        "You are a Biomedical Research Assistant. You have access to two things:\n"
        "1. CHAT HISTORY: Use this to answer questions like 'What was my last question?'\n"
        "2. RESEARCH CONTEXT: Use this only for technical biomedical questions.\n"
        "STRICT RULES:\n"
        "- If the user asks about the conversation history, IGNORE the Research Context.\n"
        "- Your response MUST be a single paragraph of 7-8 lines.\n"
        "- Never use bullet points. Finish your final sentence completely."
    )

    # 3. Format the data for the model
    context_text = "\n".join([r['text'] for r in results]) if results else "No context found."
    
    user_message = (
        f"--- START OF CHAT HISTORY ---\n{history_str}\n--- END OF CHAT HISTORY ---\n\n"
        f"--- START OF RESEARCH CONTEXT ---\n{context_text}\n--- END OF RESEARCH CONTEXT ---\n\n"
        f"USER'S CURRENT QUESTION: {question}"
    )

    return ollama.chat(
        model="qwen2.5:1.5b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        stream=True,
        options={
            "temperature": 0.1, # Keep it very focused on the provided text
            "num_predict": 500, # Large enough to finish 10 lines
            "num_thread": 8
        }
    )
# ============================================
# UI & FLOW
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.chat_input("Ask about drug mechanisms, research, or my capabilities...")

if question:
    st.chat_message("user").write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Always run search; if no high-quality matches are found, 
        # results will be empty and the LLM will fall back to its internal identity.
        results = search(question)
        
        for chunk in ask_llm_stream(question, results):
            content = chunk["message"]["content"]
            full_response += content
            # Add a blinking cursor effect while typing
            response_placeholder.markdown(full_response + "▌")
        
        # Final clean render
        response_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})














# import streamlit as st
# import torch
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import ollama

# st.set_page_config(page_title="Infection and AMR AI Chat Assistant", layout="centered")
# st.title("🧬 Infection and AMR AI Chat Assistant")

# # ============================================
# # CACHED DATA & MODEL LOADING
# # ============================================
# @st.cache_resource
# def load_data():
#     data = torch.load("all_pdfs_pagewise_embeddings.pt", weights_only=False)
#     embeddings = np.array([item["embedding"] for item in data])
#     chunks = [item["chunk"] for item in data]
#     pdfs = [item["pdf"] for item in data]
#     pages = [item["page"] for item in data]
#     # Keep model light and cached
#     embed_model = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
#     return embeddings, chunks, pdfs, pages, embed_model

# embeddings, chunks, pdfs, pages, embed_model = load_data()

# # Initialize session states to prevent rendering loop traps
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "user_role" not in st.session_state:
#     st.session_state.user_role = "General Patient"

# # ============================================
# # ISOLATED SEARCH FUNCTION
# # ============================================
# def search(query, top_k=5):
#     q_emb = embed_model.encode(query, normalize_embeddings=True, show_progress_bar=False)
#     scores = cosine_similarity([q_emb], embeddings)[0] # Fixed indexing structure
#     top_idx = scores.argsort()[::-1][:top_k]
    
#     results = []
#     for i in top_idx:
#         if scores[i] > 0.30: 
#             results.append({"text": chunks[i], "pdf": pdfs[i], "page": pages[i]})
#     return results

# # ============================================
# # STREAM GENERATOR ENGINE
# # ============================================
# def ask_llm_stream(question, results_context):
#     history_lines = []
#     # Read fewer history lines to minimize prompt size processing time
#     for m in st.session_state.messages[-4:]: 
#         history_lines.append(f"{m['role'].upper()}: {m['content']}")
#     history_str = "\n".join(history_lines)

#     system_prompt = (
#         "You are an expert Infectious Disease and Antimicrobial Resistance (AMR) Assistant.\n"
#         "You have access to CHAT HISTORY and RESEARCH CONTEXT data items.\n\n"
#         "CRITICAL ASSIGNMENT - STEP 1 (CLASSIFICATION):\n"
#         "On the absolute first line of your output, classify the incoming query into one of three roles. "
#         "Type exactly 'ROLE: Medical Doctor', 'ROLE: Lab Researcher', or 'ROLE: General Patient'. "
#         "Do not put anything else on this line. Follow it immediately with a single line break.\n\n"
#         "CRITICAL ASSIGNMENT - STEP 2 (TONE ADAPTATION):\n"
#         "- If Medical Doctor: Provide clinically accurate, professional guidance regarding stewardship or adjustments.\n"
#         "- If Lab Researcher: Provide highly technical insights focusing on genetic mutations, pathways, and strains.\n"
#         "- If General Patient: Avoid all medical jargon, use reassuring basic terms, and warn them to consult their doctor.\n\n"
#         "CRITICAL FORMAT RULES:\n"
#         "- After the classification line, your answer MUST be exactly a single paragraph of 8-10 lines.\n"
#         "- Never use bullet points, numbered lists, markdown headers, or bold tags inside the paragraph.\n"
#         "- Conclude your final sentence completely."
#     )

#     user_message = (
#         f"--- START OF CHAT HISTORY ---\n{history_str}\n--- END OF CHAT HISTORY ---\n\n"
#         f"--- START OF RESEARCH CONTEXT ---\n{results_context}\n--- END OF RESEARCH CONTEXT ---\n\n"
#         f"USER'S CURRENT QUESTION: {question}"
#     )

#     return ollama.chat(
#         model="qwen2.5:1.5b",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_message}
#         ],
#         stream=True,
#         options={
#             "temperature": 0.1, 
#             "num_predict": 400, # Lowered from 1000 to keep it tightly bound to your line rule
#             "num_thread": 4     # Balanced multi-threading allocation profile
#         }
#     )

# # ============================================
# # UI RENDER CONTROLS
# # ============================================
# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

# if st.session_state.user_role == "Medical Doctor":
#     input_placeholder = "Dr., paste case details, MIC values, or renal dosing queries here..."
# elif st.session_state.user_role == "Lab Researcher":
#     input_placeholder = "Enter resistance genes (e.g., NDM-1), mutations, strains, or pathway queries..."
# else:
#     input_placeholder = "Ask about infection symptoms, antibiotic safety, or side effects..."

# question = st.chat_input(input_placeholder)

# if question:
#     st.chat_message("user").write(question)
#     st.session_state.messages.append({"role": "user", "content": question})

#     # 1. RUN SEARCH ONCE OUTSIDE OF THE GENERATION STREAM
#     search_matches = search(question)
#     context_text = "\n".join([r['text'] for r in search_matches]) if search_matches else "No context found."

#     with st.chat_message("assistant"):
#         response_placeholder = st.empty()
#         full_text = ""
#         role_parsed = False
        
#         # 2. Trigger the stream using the pre-computed text block
#         for chunk in ask_llm_stream(question, context_text):
#             full_text += chunk["message"]["content"]
            
#             # Fast parsing configuration setup
#             if not role_parsed and "ROLE:" in full_text and "\n" in full_text:
#                 lines = full_text.split("\n")
#                 first_line = lines[0]
                
#                 if "Medical Doctor" in first_line:
#                     st.session_state.user_role = "Medical Doctor"
#                 elif "Lab Researcher" in first_line:
#                     st.session_state.user_role = "Lab Researcher"
#                 else:
#                     st.session_state.user_role = "General Patient"
                
#                 role_parsed = True
            
#             if role_parsed:
#                 # Render content excluding the first raw classification row string
#                 display_text = "\n".join(full_text.split("\n")[1:])
#                 response_placeholder.markdown(display_text + "▌")
#             else:
#                 response_placeholder.markdown("Analyzing query profile... ▌")
        
#         # Final static cleanup rendering block
#         final_clean_output = "\n".join(full_text.split("\n")[1:]).strip() if role_parsed else full_text.strip()
#         response_placeholder.markdown(final_clean_output)
    
#     st.session_state.messages.append({"role": "assistant", "content": final_clean_output})
#     st.rerun()

















# import streamlit as st
# import torch
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import ollama

# # --- PAGE CONFIG ---
# st.set_page_config(page_title="Biomedical RAG Chatbot", layout="wide")
# st.title("🧬 Biomedical Research Chatbot (RAG)")

# @st.cache_resource
# def load_data():
#     # Loading your original .pt file
#     data = torch.load("all_pdfs_pagewise_embeddings.pt", weights_only=False)
#     embeddings = np.array([item["embedding"] for item in data])
#     chunks = [item["chunk"] for item in data]
#     pdfs = [item["pdf"] for item in data]
#     pages = [item["page"] for item in data]
#     # Domain-specific SapBERT as per your methodology
#     embed_model = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
#     return embeddings, chunks, pdfs, pages, embed_model

# embeddings, chunks, pdfs, pages, embed_model = load_data()

# # --- SEARCH LOGIC (Using NumPy Matrix) ---
# def search(query, top_k=10):
#     q_emb = embed_model.encode(query, normalize_embeddings=True, show_progress_bar=False)
#     # This is the "Cosine Similarity Matrix" calculation your mentor wanted to see
#     scores = cosine_similarity([q_emb], embeddings)[0]
#     top_idx = scores.argsort()[::-1][:top_k]
    
#     results = []
#     for i in top_idx:
#         # Using a 0.35 threshold to ensure high-quality matches for the report
#         if scores[i] > 0.35: 
#             results.append({
#                 "text": chunks[i], 
#                 "pdf": pdfs[i], 
#                 "page": pages[i], 
#                 "score": scores[i]
#             })
#     return results

# # --- LLM LOGIC (Streaming) ---
# def ask_llm_stream(question, results):
#     history_lines = []
#     if "messages" in st.session_state:
#         for m in st.session_state.messages[-5:]:
#             history_lines.append(f"{m['role'].upper()}: {m['content']}")
#     history_str = "\n".join(history_lines)

#     system_prompt = (
#         "You are a Biomedical Research Assistant. Use the provided Research Context to answer.\n"
#         "STRICT RULES:\n"
#         "- Answer in a single paragraph of 8-10 lines.\n"
#         "- Use professional medical terminology.\n"
#         "- If context is missing, rely on chat history or state you don't know."
#     )

#     context_text = "\n".join([r['text'] for r in results]) if results else "No context found."
    
#     user_message = (
#         f"--- CHAT HISTORY ---\n{history_str}\n"
#         f"--- RESEARCH CONTEXT ---\n{context_text}\n"
#         f"USER QUESTION: {question}"
#     )

#     return ollama.chat(
#         model="qwen2.5:1.5b",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_message}
#         ],
#         stream=True,
#         options={"temperature": 0.1, "num_thread": 8}
#     )

# # --- UI & INTERACTION ---
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat history
# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

# question = st.chat_input("Ask about drug mechanisms or research papers...")

# if question:
#     # 1. Display User Question
#     st.chat_message("user").write(question)
#     st.session_state.messages.append({"role": "user", "content": question})

#     # 2. RUN SEARCH & DISPLAY EVALUATION TRACE (SCREENSHOT THIS PART!)
#     results = search(question)
    
#     with st.expander("🔍 System Trace: Retrieval & Similarity Logic", expanded=True):
#         st.write("### Vector Search Results (NumPy Matrix)")
#         if results:
#             for i, res in enumerate(results):
#                 # This UI block proves traceability to your mentor
#                 st.info(f"**Result {i+1}** | **Source:** {res['pdf']} | **Page:** {res['page']} | **Similarity:** {res['score']:.4f}")
#                 st.caption(f"Content Snippet: {res['text'][:300]}...")
#         else:
#             st.warning("No context found above the 0.35 similarity threshold.")

#     # 3. GENERATE RESPONSE
#     with st.chat_message("assistant"):
#         response_placeholder = st.empty()
#         full_response = ""
        
#         for chunk in ask_llm_stream(question, results):
#             content = chunk["message"]["content"]
#             full_response += content
#             response_placeholder.markdown(full_response + "▌")
        
#         response_placeholder.markdown(full_response)
    
#     st.session_state.messages.append({"role": "assistant", "content": full_response})

   
   
   
   
   
   
 
