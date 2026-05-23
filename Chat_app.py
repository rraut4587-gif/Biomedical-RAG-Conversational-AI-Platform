# import streamlit as st
# import torch
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import ollama
# import time

# st.set_page_config(page_title="Biomedical RAG Chatbot", layout="wide")
# st.title("🧬 Biomedical Research Chatbot (RAG)")

# # ============================================
# # LOAD DATA
# # ============================================

# @st.cache_resource
# def load_data():
#     start_time = time.time()

#     data = torch.load("all_pdfs_pagewise_embeddings.pt", weights_only=False)
#     embeddings, chunks, pdfs, pages = [], [], [], []

#     for item in data:
#         embeddings.append(item["embedding"])
#         chunks.append(item["chunk"])
#         pdfs.append(item["pdf"])
#         pages.append(item["page"])

#     embeddings = np.array(embeddings)
#     embed_model = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

#     # warmup LLM
#     ollama.chat(
#         model="qwen2.5:1.5b",
#         messages=[{"role": "user", "content": "hi"}],
#         options={"num_predict": 1}
#     )

#     print(f"[Timing] Data load completed in {time.time() - start_time:.2f} sec")
#     return embeddings, chunks, pdfs, pages, embed_model

# embeddings, chunks, pdfs, pages, embed_model = load_data()

# # ============================================
# # SEARCH TOP-K
# # ============================================

# def search(query, top_k=5):
#     start_time = time.time()
#     q_emb = embed_model.encode(query, normalize_embeddings=True, show_progress_bar=False)
#     scores = cosine_similarity([q_emb], embeddings)[0]
#     top_idx = scores.argsort()[::-1][:top_k]

#     results = []
#     for i in top_idx:
#         results.append({
#             "text": chunks[i][:1000],  # reduced chunk size for speed
#             "pdf": pdfs[i],
#             "page": pages[i],
#             "score": scores[i]
#         })

#     print(f"[Timing] Search time: {time.time() - start_time:.2f} sec")
#     return results

# # ============================================
# # RERANK (better context quality)
# # ============================================

# def rerank(query, results):
#     start_time = time.time()
#     texts = [r["text"] for r in results]
#     q_emb = embed_model.encode(query, normalize_embeddings=True)
#     t_emb = embed_model.encode(texts, normalize_embeddings=True)
#     scores = cosine_similarity([q_emb], t_emb)[0]
#     order = scores.argsort()[::-1]
#     new_results = [results[i] for i in order]

#     print(f"[Timing] Rerank time: {time.time() - start_time:.2f} sec")
#     return new_results[:2]  # top-2 chunks for speed

# # ============================================
# # LLM QUERY
# # ============================================

# def ask_llm(question, results):
#     start_time = time.time()

#     # Join top chunks for context
#     context = "\n\n".join([r['text'] for r in results])
#     system_prompt = "You are a biomedical research assistant. Answer ONLY using the provided context. If answer not found, respond: 'Answer not found in context'."

#     # PRINT SYSTEM PROMPT AND CONTEXT IN TERMINAL
#     print("\n======= SYSTEM PROMPT =======")
#     print(system_prompt)
#     print("\n======= CONTEXT PASSED =======")
#     print(context)
#     print("==============================\n")

#     prompt = f"""
# You are a biomedical research assistant.

# STRICT RULES:
# - Use ONLY the provided context
# - Do NOT use outside knowledge
# - Do NOT guess
# - If answer not in context → say "Answer not found in context"
# - Write 8-10 lines
# - Explain clearly
# - Use scientific language but simple

# Context:
# {context}

# Question:
# {question}

# Answer:
# """

#     response = ollama.chat(
#         model="qwen2.5:1.5b",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": prompt}
#         ],
#         options={
#             "temperature": 0,
#             "num_predict": 600,       # detailed answer
#             "num_ctx": 4096,
#             "num_thread": 8,
#             "top_k": 20,
#             "top_p": 0.9,
#             "repeat_penalty": 1.1,
#         }
#     )

#     print(f"[Timing] LLM response time: {time.time() - start_time:.2f} sec")
#     return response["message"]["content"]

# # ============================================
# # CHAT MEMORY
# # ============================================

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

# # ============================================
# # INPUT
# # ============================================

# question = st.chat_input("Ask biomedical research question...")

# if question:
#     st.session_state.messages.append({"role": "user", "content": question})
#     st.chat_message("user").write(question)

#     with st.spinner("Searching papers ⚡"):
#         results = search(question)
#         results = rerank(question, results)
#         answer = ask_llm(question, results)

#     st.session_state.messages.append({"role": "assistant", "content": answer})
#     st.chat_message("assistant").write(answer)












# import streamlit as st
# import torch
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import ollama
# import time

# st.set_page_config(page_title="Biomedical RAG Chatbot", layout="wide")
# st.title("🧬 Biomedical Research Chatbot (RAG)")

# # ============================================
# # LOAD DATA
# # ============================================

# @st.cache_resource
# def load_data():
#     start_time = time.time()

#     data = torch.load("all_pdfs_pagewise_embeddings.pt", weights_only=False)
#     embeddings, chunks, pdfs, pages = [], [], [], []

#     for item in data:
#         embeddings.append(item["embedding"])
#         chunks.append(item["chunk"])
#         pdfs.append(item["pdf"])
#         pages.append(item["page"])

#     embeddings = np.array(embeddings)
#     embed_model = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

#     # warmup LLM
#     ollama.chat(
#         model="qwen2.5:1.5b",
#         messages=[{"role": "user", "content": "hi"}],
#         options={"num_predict": 1}
#     )

#     print(f"[Timing] Data load completed in {time.time() - start_time:.2f} sec")
#     return embeddings, chunks, pdfs, pages, embed_model

# embeddings, chunks, pdfs, pages, embed_model = load_data()

# # ============================================
# # SEARCH TOP-K
# # ============================================

# def search(query, top_k=5):
#     start_time = time.time()
#     q_emb = embed_model.encode(query, normalize_embeddings=True, show_progress_bar=False)
#     scores = cosine_similarity([q_emb], embeddings)[0]
#     top_idx = scores.argsort()[::-1][:top_k]

#     results = []
#     for i in top_idx:
#         results.append({
#             "text": chunks[i][:1000],  # reduce chunk size for speed
#             "pdf": pdfs[i],
#             "page": pages[i],
#             "score": scores[i]
#         })

#     print(f"[Timing] Search time: {time.time() - start_time:.2f} sec")
#     return results

# # ============================================
# # RERANK (better context quality)
# # ============================================

# def rerank(query, results):
#     start_time = time.time()
#     texts = [r["text"] for r in results]
#     q_emb = embed_model.encode(query, normalize_embeddings=True)
#     t_emb = embed_model.encode(texts, normalize_embeddings=True)
#     scores = cosine_similarity([q_emb], t_emb)[0]
#     order = scores.argsort()[::-1]
#     new_results = [results[i] for i in order]

#     print(f"[Timing] Rerank time: {time.time() - start_time:.2f} sec")
#     return new_results[:2]  # top-2 chunks for speed

# # ============================================
# # LLM QUERY (strict context-only)
# # ============================================

# def ask_llm(question, results):
#     start_time = time.time()

#     # Join top chunks for context
#     context = "\n\n".join([f"[PDF: {r['pdf']}, Page: {r['page']}]\n{r['text']}" for r in results])
#     system_prompt = (
#         "You are a biomedical research assistant. "
#         "STRICT RULE: Answer ONLY using the context provided from PDF chunks. "
#         "Do NOT guess, do NOT use outside knowledge. "
#         "If the answer is not in the context, respond exactly: 'Answer not found in context'."
#     )

#     prompt = f"""
# You are a biomedical research assistant.

# STRICT RULES:
# - Use ONLY the provided context from PDF chunks
# - Do NOT guess or use outside knowledge
# - If answer not in context → say "Answer not found in context"
# - Write 5-8 lines
# - Explain clearly
# - Use scientific language but simple

# Context:
# {context}

# Question:
# {question}

# Answer:
# """

#     response = ollama.chat(
#         model="qwen2.5:1.5b",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": prompt}
#         ],
#         options={
#             "temperature": 0,
#             "num_predict": 600,
#             "num_ctx": 4096,
#             "num_thread": 8,
#             "top_k": 20,
#             "top_p": 0.9,
#             "repeat_penalty": 1.1,
#         }
#     )

#     print(f"[Timing] LLM response time: {time.time() - start_time:.2f} sec")
#     return response["message"]["content"]

# # ============================================
# # CHAT MEMORY
# # ============================================

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

# # ============================================
# # INPUT
# # ============================================

# question = st.chat_input("Ask biomedical research question...")

# if question:
#     st.session_state.messages.append({"role": "user", "content": question})
#     st.chat_message("user").write(question)

#     with st.spinner("Searching papers ⚡"):
#         results = search(question)
#         results = rerank(question, results)

#         # Strict context check before LLM
#         if len(results) == 0 or max([r["score"] for r in results]) < 0.1:
#             answer = "Answer not found in context."
#         else:
#             answer = ask_llm(question, results)

#     st.session_state.messages.append({"role": "assistant", "content": answer})
#     st.chat_message("assistant").write(answer)













# import streamlit as st
# import torch
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import ollama
# import time

# st.set_page_config(page_title="Biomedical RAG Chatbot", layout="wide")
# st.title("🧬 Biomedical Research Chatbot (RAG)")

# # ============================================
# # LOAD DATA
# # ============================================

# @st.cache_resource
# def load_data():
#     start_time = time.time()

#     data = torch.load("all_pdfs_pagewise_embeddings.pt", weights_only=False)
#     embeddings, chunks, pdfs, pages = [], [], [], []

#     for item in data:
#         embeddings.append(item["embedding"])
#         chunks.append(item["chunk"])
#         pdfs.append(item["pdf"])
#         pages.append(item["page"])

#     embeddings = np.array(embeddings)
#     embed_model = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

#     # Warm up LLM
#     ollama.chat(
#         model="qwen2.5:1.5b",
#         messages=[{"role": "user", "content": "hi"}],
#         options={"num_predict": 1}
#     )

#     print(f"[Timing] Data load completed in {time.time() - start_time:.2f} sec")
#     return embeddings, chunks, pdfs, pages, embed_model

# embeddings, chunks, pdfs, pages, embed_model = load_data()

# # ============================================
# # SEARCH TOP-K
# # ============================================

# def search(query, top_k=5):
#     start_time = time.time()
#     q_emb = embed_model.encode(query, normalize_embeddings=True, show_progress_bar=False)
#     scores = cosine_similarity([q_emb], embeddings)[0]
#     top_idx = scores.argsort()[::-1][:top_k]

#     results = []
#     for i in top_idx:
#         results.append({
#             "text": chunks[i][:1000],
#             "pdf": pdfs[i],
#             "page": pages[i],
#             "score": scores[i]
#         })

#     print(f"[Timing] Search time: {time.time() - start_time:.2f} sec")
#     return results

# # ============================================
# # RERANK
# # ============================================

# def rerank(query, results):
#     start_time = time.time()
#     texts = [r["text"] for r in results]
#     q_emb = embed_model.encode(query, normalize_embeddings=True)
#     t_emb = embed_model.encode(texts, normalize_embeddings=True)
#     scores = cosine_similarity([q_emb], t_emb)[0]
#     order = scores.argsort()[::-1]
#     new_results = [results[i] for i in order]

#     print(f"[Timing] Rerank time: {time.time() - start_time:.2f} sec")
#     return new_results[:2]

# # ============================================
# # LLM QUERY (strict PDF-only)
# # ============================================

# def ask_llm(question, results):
#     start_time = time.time()

#     # Only create system_prompt and context here
#     context = "\n\n".join([f"[PDF: {r['pdf']}, Page: {r['page']}]\n{r['text']}" for r in results])

#     system_prompt = (
#         "You are a biomedical research assistant. "
#         "STRICT RULE: Answer ONLY using the context provided from PDF chunks. "
#         "Do NOT guess, do NOT use outside knowledge. "
#         "If the answer is not in the context, respond exactly: 'Answer not found in context'."
#     )

#     prompt = f"""
# You are a biomedical research assistant.
# Greet for only if user says Hi or Hello.
# STRICT RULES:
# - Use ONLY the provided context from PDF chunks
# - Do NOT guess or use outside knowledge
# - If answer not in context Use your knowledge base to answer the question,
#   but clearly indicate that the answer is based on your knowledge and not the provided context. 
#   For example, you can say "Based on my knowledge, the answer is....
# - Write 5-8 lines
# - Explain clearly
# - Use scientific language but simple

# Context:
# {context}

# Question:
# {question}

# Answer:
# """

#     # PRINT SYSTEM PROMPT AND CONTEXT
#     print("\n======= SYSTEM PROMPT =======")
#     print(system_prompt)
#     print("\n======= CONTEXT PASSED TO LLM =======")
#     print(context)
#     print("======================================\n")

#     response = ollama.chat(
#         model="qwen2.5:1.5b",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": prompt}
#         ],
#         options={
#             "temperature": 0,
#             "num_predict": 600,
#             "num_ctx": 4096,
#             "num_thread": 8,
#             "top_k": 20,
#             "top_p": 0.9,
#             "repeat_penalty": 1.1,
#         }
#     )

#     print(f"[Timing] LLM response time: {time.time() - start_time:.2f} sec")
#     return response["message"]["content"]

# # ============================================
# # CHAT MEMORY
# # ============================================

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

# # ============================================
# # INPUT
# # ============================================

# question = st.chat_input("Ask biomedical research question...")

# if question:
#     st.session_state.messages.append({"role": "user", "content": question})
#     st.chat_message("user").write(question)

#     with st.spinner("Searching papers ⚡"):
#         results = search(question)
#         results = rerank(question, results)

#         # Strict PDF context check
#         if len(results) == 0 or max([r["score"] for r in results]) < 0.1:
#             answer = "Answer not found in context."
#         else:
#             answer = ask_llm(question, results)

#     st.session_state.messages.append({"role": "assistant", "content": answer})
#     st.chat_message("assistant").write(answer)








### WORKING VERSION ######

# import streamlit as st
# import torch
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import ollama
# import time

# st.set_page_config(page_title="Biomedical RAG Chatbot", layout="wide")
# st.title("🧬 Biomedical Research Chatbot (RAG)")

# # ============================================
# # LOAD DATA
# # ============================================

# @st.cache_resource
# def load_data():
#     start_time = time.time()

#     data = torch.load("all_pdfs_pagewise_embeddings.pt", weights_only=False)
#     embeddings, chunks, pdfs, pages = [], [], [], []

#     for item in data:
#         embeddings.append(item["embedding"])
#         chunks.append(item["chunk"])
#         pdfs.append(item["pdf"])
#         pages.append(item["page"])

#     embeddings = np.array(embeddings)
#     embed_model = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

#     # Warm up LLM
#     ollama.chat(
#         model="qwen2.5:1.5b",
#         messages=[{"role": "user", "content": "hi"}],
#         options={"num_predict": 1}
#     )

#     print(f"[Timing] Data load completed in {time.time() - start_time:.2f} sec")
#     return embeddings, chunks, pdfs, pages, embed_model

# embeddings, chunks, pdfs, pages, embed_model = load_data()

# # ============================================
# # SEARCH TOP-K
# # ============================================

# def search(query, top_k=5):
#     start_time = time.time()

#     q_emb = embed_model.encode(query, normalize_embeddings=True, show_progress_bar=False)
#     scores = cosine_similarity([q_emb], embeddings)[0]
#     top_idx = scores.argsort()[::-1][:top_k]

#     results = []
#     for i in top_idx:
#         results.append({
#             "text": chunks[i][:1000],
#             "pdf": pdfs[i],
#             "page": pages[i],
#             "score": scores[i]
#         })

#     print(f"[Timing] Search time: {time.time() - start_time:.2f} sec")
#     return results

# # ============================================
# # RERANK
# # ============================================

# def rerank(query, results):
#     start_time = time.time()

#     texts = [r["text"] for r in results]
#     q_emb = embed_model.encode(query, normalize_embeddings=True)
#     t_emb = embed_model.encode(texts, normalize_embeddings=True)

#     scores = cosine_similarity([q_emb], t_emb)[0]
#     order = scores.argsort()[::-1]

#     new_results = [results[i] for i in order]

#     print(f"[Timing] Rerank time: {time.time() - start_time:.2f} sec")
#     return new_results[:3]

# # ============================================
# # LLM QUERY (Improved)
# # ============================================

# def ask_llm(question, results):
#     start_time = time.time()

#     context = "\n\n".join([
#         f"[PDF: {r['pdf']}, Page: {r['page']}]\n{r['text']}"
#         for r in results
#     ]) if results else "No relevant context retrieved."

#     system_prompt = (
#         "You are an expert biomedical research assistant.\n\n"
#         "RULES:\n"
#         "1. Give direct, clear, and scientific answers.\n"
#         "2. Do NOT mention missing context.\n"
#         "3. Do NOT say 'based on context' or 'it appears'.\n"
#         "4. Use provided context if relevant.\n"
#         "5. If context is insufficient, use your biomedical knowledge.\n"
#         "6. Always sound confident and informative.\n"
#         "7. Keep answers 5–8 lines.\n"
#         "8. Use simple but scientific language.\n"
#     )

#     prompt = f"""
# Answer the biomedical question clearly and directly.

# - Do NOT mention missing context
# - Avoid meta explanations
# - Be informative and scientific

# Context:
# {context}

# Question:
# {question}

# Answer:
# """

#     response = ollama.chat(
#         model="qwen2.5:1.5b",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": prompt}
#         ],
#         options={
#             "temperature": 0.2,
#             "num_predict": 500,
#             "num_ctx": 4096,
#             "num_thread": 8,
#             "top_k": 40,
#             "top_p": 0.9,
#             "repeat_penalty": 1.1,
#         }
#     )

#     print(f"[Timing] LLM response time: {time.time() - start_time:.2f} sec")
#     return response["message"]["content"]

# # ============================================
# # CHAT MEMORY
# # ============================================

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

# # ============================================
# # INPUT
# # ============================================

# question = st.chat_input("Ask biomedical research question...")

# if question:
#     st.session_state.messages.append({"role": "user", "content": question})
#     st.chat_message("user").write(question)

#     # ============================================
#     # HANDLE GENERIC QUESTIONS
#     # ============================================

#     if question.lower() in ["what can i ask you", "what can you do", "help"]:
#         answer = """You can ask me biomedical and research-related questions such as:

# • Disease mechanisms (cancer, infectious diseases)
# • Drug–target interactions
# • Gene mutations and molecular pathways
# • Clinical treatments and therapies
# • Research paper summaries
# • Immunology and microbiology concepts

# I provide clear, scientific, and easy-to-understand explanations."""
#     else:
#         with st.spinner("Searching papers ⚡"):
#             results = search(question)
#             results = rerank(question, results)

#             answer = ask_llm(question, results)

#     # ============================================
#     # DISPLAY ANSWER
#     # ============================================

#     st.session_state.messages.append({"role": "assistant", "content": answer})
#     st.chat_message("assistant").write(answer)

#     # ============================================
#     # SHOW SOURCES (VERY IMPORTANT FOR THESIS)
#     # ============================================

#     if 'results' in locals() and len(results) > 0:
#         st.markdown("### 📄 Sources")
#         for r in results:
#             st.caption(f"{r['pdf']} — Page {r['page']} (Score: {r['score']:.3f})")











# import streamlit as st
# import torch
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import ollama
# import time

# st.set_page_config(page_title="Biomedical RAG Chatbot", layout="wide")
# st.title("🧬 Biomedical Research Chatbot (RAG)")

# # ============================================
# # LOAD DATA
# # ============================================

# @st.cache_resource
# def load_data():
#     start_time = time.time()

#     data = torch.load("all_pdfs_pagewise_embeddings.pt", weights_only=False)
#     embeddings, chunks, pdfs, pages = [], [], [], []

#     for item in data:
#         embeddings.append(item["embedding"])
#         chunks.append(item["chunk"])
#         pdfs.append(item["pdf"])
#         pages.append(item["page"])

#     embeddings = np.array(embeddings)
#     embed_model = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

#     # Warmup
#     ollama.chat(
#         model="qwen2.5:1.5b",
#         messages=[{"role": "user", "content": "hi"}],
#         options={"num_predict": 1}
#     )

#     print(f"[Timing] Load: {time.time() - start_time:.2f}s")
#     return embeddings, chunks, pdfs, pages, embed_model

# embeddings, chunks, pdfs, pages, embed_model = load_data()

# # ============================================
# # SEARCH
# # ============================================

# def search(query, top_k=3):
#     q_emb = embed_model.encode(query, normalize_embeddings=True, show_progress_bar=False)
#     scores = cosine_similarity([q_emb], embeddings)[0]
#     top_idx = scores.argsort()[::-1][:top_k]

#     results = []
#     for i in top_idx:
#         results.append({
#             "text": chunks[i][:900],  # slightly increased
#             "pdf": pdfs[i],
#             "page": pages[i],
#             "score": scores[i]
#         })

#     return results

# # ============================================
# # RERANK (LIGHT)
# # ============================================

# def rerank(query, results):
#     texts = [r["text"] for r in results]

#     q_emb = embed_model.encode(query, normalize_embeddings=True)
#     t_emb = embed_model.encode(texts, normalize_embeddings=True)

#     scores = cosine_similarity([q_emb], t_emb)[0]
#     order = scores.argsort()[::-1]

#     return [results[i] for i in order][:2]

# # ============================================
# # LLM
# # ============================================

# def ask_llm(question, results):
#     start_time = time.time()

#     context = "\n\n".join([
#         f"[PDF: {r['pdf']} | Page: {r['page']}]\n{r['text']}"
#         for r in results
#     ]) if results else ""

#     # 🔥 PRINT CONTEXT TO TERMINAL
#     print("\n========== CONTEXT SENT TO LLM ==========")
#     print(context)
#     print("=========================================\n")

#     system_prompt = (
#         "You are an expert biomedical research assistant.\n"
#         "Provide detailed, clear, and scientific answers.\n"
#         "Do NOT mention context or missing information.\n"
#         "Do NOT give meta explanations.\n"
#         "Answer in 6-10 lines with proper explanation.\n"
#     )

#     prompt = f"""
# Answer the biomedical question clearly and in detail.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """

#     response = ollama.chat(
#         model="qwen2.5:1.5b",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": prompt}
#         ],
#         options={
#             "temperature": 0.2,
#             "num_predict": 260,   # increased (longer answer)
#             "num_ctx": 2048,      # keep low for speed
#             "top_k": 30,
#             "top_p": 0.9,
#             "repeat_penalty": 1.1,
#         }
#     )

#     print(f"[Timing] LLM: {time.time() - start_time:.2f}s")
#     return response["message"]["content"]

# # ============================================
# # MEMORY
# # ============================================

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

# # ============================================
# # INPUT
# # ============================================

# question = st.chat_input("Ask biomedical research question...")

# if question:
#     st.session_state.messages.append({"role": "user", "content": question})
#     st.chat_message("user").write(question)

#     q = question.lower().strip()

#     # ============================================
#     # INSTANT RESPONSES
#     # ============================================

#     if q in ["hi", "hello", "hey"]:
#         answer = "Hello! 👋 How can I help you with biomedical research today?"

#     elif q in ["what can i ask you", "what can you do", "help"]:
#         answer = """You can ask me:

# • Disease mechanisms (cancer, infectious diseases)
# • Drug–target interactions
# • Gene mutations and pathways
# • Clinical treatments and therapies
# • Research paper explanations

# I provide detailed and scientific answers."""

#     else:
#         start = time.time()

#         results = search(question)
#         results = rerank(question, results)

#         answer = ask_llm(question, results)

#         print(f"⚡ Total time: {time.time() - start:.2f}s")

#     # ============================================
#     # DISPLAY
#     # ============================================

#     st.session_state.messages.append({"role": "assistant", "content": answer})
#     st.chat_message("assistant").write(answer)








#### FINAL VERSION (SPEED OPTIMIZED) ####

# import streamlit as st
# import torch
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import ollama
# import time

# st.set_page_config(page_title="Biomedical RAG Chatbot", layout="wide")
# st.title("🧬 Biomedical Research Chatbot (RAG)")

# # ============================================
# # LOAD DATA
# # ============================================

# @st.cache_resource
# def load_data():
#     start_time = time.time()

#     data = torch.load("all_pdfs_pagewise_embeddings.pt", weights_only=False)
#     embeddings, chunks, pdfs, pages = [], [], [], []

#     for item in data:
#         embeddings.append(item["embedding"])
#         chunks.append(item["chunk"])
#         pdfs.append(item["pdf"])
#         pages.append(item["page"])

#     embeddings = np.array(embeddings)

#     embed_model = SentenceTransformer(
#         "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
#     )

#     # Warmup LLM
#     ollama.chat(
#         model="qwen2.5:1.5b",
#         messages=[{"role": "user", "content": "hi"}],
#         options={"num_predict": 1}
#     )

#     print(f"[Timing] Load: {time.time() - start_time:.2f}s")
#     return embeddings, chunks, pdfs, pages, embed_model

# embeddings, chunks, pdfs, pages, embed_model = load_data()

# # ============================================
# # SEARCH (OPTIMIZED)
# # ============================================

# def search(query, top_k=4):  
#     start_time = time.time()

#     q_emb = embed_model.encode(
#         query,
#         normalize_embeddings=True,
#         show_progress_bar=False
#     )

#     scores = cosine_similarity([q_emb], embeddings)[0]
#     top_idx = scores.argsort()[::-1][:top_k]

#     results = []
#     for i in top_idx:
#         results.append({
#             "text": chunks[i][:500],  # reduced chunk size
#             "pdf": pdfs[i],
#             "page": pages[i],
#             "score": scores[i]
#         })

#     print(f"[Timing] Search: {time.time() - start_time:.2f}s")
#     return results

# # ============================================
# # LLM (FAST + LONG ANSWER BALANCE)
# # ============================================

# def ask_llm(question, results):
#     start_time = time.time()

#     context = "\n\n".join([
#         f"[PDF: {r['pdf']} | Page: {r['page']}]\n{r['text']}"
#         for r in results
#     ]) if results else ""

#     # PRINT CONTEXT
#     print("\n========== CONTEXT SENT TO LLM ==========")
#     print(context)
#     print("=========================================\n")

#     system_prompt = (
#         "You are a biomedical assistant. "
#         "Give clear, direct, scientific answers in 7-10lines. "
#         "Do not give meta explanations."
#         "Do not answer the Non biomedical, Life science, Research related questions. If the question is not related to these topics, respond with 'I can only answer biomedical research questions."
#     )

#     prompt = f"""
# Question: {question}

# Context:
# {context}

# Answer:
# """

#     response = ollama.chat(
#         model="qwen2.5:1.5b",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": prompt}
#         ],
#         options={
#             "temperature": 0.2,
#             "num_predict": 500,   
#             "num_ctx": 1024,      
#             "num_thread": 8,
#             "top_k": 30,
#             "top_p": 0.9,
#             "repeat_penalty": 1.1,
#         }
#     )

#     print(f"[Timing] LLM: {time.time() - start_time:.2f}s")
#     return response["message"]["content"]

# # ============================================
# # MEMORY
# # ============================================

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

# # ============================================
# # INPUT
# # ============================================

# question = st.chat_input("Ask biomedical research question...")

# if question:
#     st.session_state.messages.append({"role": "user", "content": question})
#     st.chat_message("user").write(question)

#     q = question.lower().strip()

#     # ============================================
#     # INSTANT RESPONSES (0 sec)
#     # ============================================

#     if q in ["hi", "hello", "hey"]:
#         answer = "Hello! 👋 How can I help you with biomedical research today?"

#     elif q in ["what can i ask you", "what can you do", "help"]:
#         answer = """You can ask me:

# • Disease mechanisms (cancer, infectious diseases)
# • Drug–target interactions
# • Gene mutations and pathways
# • Clinical treatments and therapies
# • Research paper explanations

# I provide clear and scientific answers quickly."""

#     else:
#         start_time = time.time()

#         # SEARCH ONLY (rerank removed for speed)
#         results = search(question)

#         # LLM
#         answer = ask_llm(question, results)

#         print(f"⚡ Total time: {time.time() - start_time:.2f}s")

#     # ============================================
#     # DISPLAY
#     # ============================================

#     st.session_state.messages.append({"role": "assistant", "content": answer})
#     st.chat_message("assistant").write(answer)










# import streamlit as st
# import torch
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import ollama
# import time

# st.set_page_config(page_title="Biomedical RAG Chatbot", layout="wide")
# st.title("🧬 Biomedical Research Chatbot (RAG)")

# # ============================================
# # LOAD DATA
# # ============================================

# @st.cache_resource
# def load_data():
#     start_time = time.time()

#     data = torch.load("all_pdfs_pagewise_embeddings.pt", weights_only=False)
#     embeddings, chunks, pdfs, pages = [], [], [], []

#     for item in data:
#         embeddings.append(item["embedding"])
#         chunks.append(item["chunk"])
#         pdfs.append(item["pdf"])
#         pages.append(item["page"])

#     embeddings = np.array(embeddings)

#     embed_model = SentenceTransformer(
#         "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
#     )

#     # Warmup LLM
#     ollama.chat(
#         model="qwen2.5:1.5b",
#         messages=[{"role": "user", "content": "hi"}],
#         options={"num_predict": 1}
#     )

#     print(f"[Timing] Load: {time.time() - start_time:.2f}s")
#     return embeddings, chunks, pdfs, pages, embed_model

# embeddings, chunks, pdfs, pages, embed_model = load_data()

# # ============================================
# # SEARCH (WITH SCORE LOGGING)
# # ============================================

# def search(query, top_k=4):  
#     start_time = time.time()

#     q_emb = embed_model.encode(
#         query,
#         normalize_embeddings=True,
#         show_progress_bar=False
#     )

#     scores = cosine_similarity([q_emb], embeddings)[0]
#     top_idx = scores.argsort()[::-1][:top_k]


#     results = []
#     for rank, i in enumerate(top_idx, start=1):
#         result = {
#             "text": chunks[i][:500],
#             "pdf": pdfs[i],
#             "page": pages[i],
#             "score": scores[i]
#         }

#         results.append(result)

#         # 🔥 Terminal logging
#         print(f"\nRank {rank}")
#         print(f"PDF   : {pdfs[i]}")
#         print(f"Page  : {pages[i]}")
#         print(f"Score : {scores[i]:.4f}")
#         print(f"Text  : {chunks[i][:200]}...")

#     print("============================================\n")
#     print(f"[Timing] Search: {time.time() - start_time:.2f}s")

#     return results

# # ============================================
# # LLM (WITH CONTEXT + SCORE + FINAL OUTPUT)
# # ============================================

# def ask_llm(question, results):
#     start_time = time.time()

#     context = "\n\n".join([
#         f"[PDF: {r['pdf']} | Page: {r['page']} | Score: {r['score']:.4f}]\n{r['text']}"
#         for r in results
#     ]) if results else ""

#     # 🔥 PRINT CONTEXT
#     print("\n========== CONTEXT SENT TO LLM ==========")
#     print(context)
#     print("=========================================\n")

#     system_prompt = (
#         "You are a biomedical assistant. "
#         "Give clear, direct, scientific answers in 7-10 lines. "
#         "Do not give meta explanations. "
#         "Do not answer non-biomedical questions. "
#         "If unrelated, say: 'I can only answer biomedical research questions.'"
#     )

#     prompt = f"""
# Question: {question}

# Context:
# {context}

# Answer:
# """

#     response = ollama.chat(
#         model="qwen2.5:1.5b",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": prompt}
#         ],
#         options={
#             "temperature": 0.2,
#             "num_predict": 500,
#             "num_ctx": 1024,
#             "num_thread": 8,
#             "top_k": 30,
#             "top_p": 0.9,
#             "repeat_penalty": 1.1,
#         }
#     )

#     answer = response["message"]["content"]

#     # 🔥 PRINT FINAL ANSWER
#     print("\n========== FINAL LLM ANSWER ==========")
#     print(answer)
#     print("======================================\n")

#     print(f"[Timing] LLM: {time.time() - start_time:.2f}s")

#     return answer

# # ============================================
# # MEMORY
# # ============================================

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

# # ============================================
# # INPUT
# # ============================================

# question = st.chat_input("Ask biomedical research question...")

# if question:
#     st.session_state.messages.append({"role": "user", "content": question})
#     st.chat_message("user").write(question)

#     q = question.lower().strip()

#     # ============================================
#     # INSTANT RESPONSES
#     # ============================================

#     if q in ["hi", "hello", "hey"]:
#         answer = "Hello! 👋 How can I help you with biomedical research today?"

#     elif q in ["what can i ask you", "what can you do", "help"]:
#         answer = """You can ask me:

# • Disease mechanisms (cancer, infectious diseases)
# • Drug–target interactions
# • Gene mutations and pathways
# • Clinical treatments and therapies
# • Research paper explanations

# I provide clear and scientific answers quickly."""

#     else:
#         start_time = time.time()

#         # 🔍 SEARCH
#         results = search(question)

#         # 🤖 LLM
#         answer = ask_llm(question, results)

#         print(f"⚡ Total time: {time.time() - start_time:.2f}s")

#     # ============================================
#     # DISPLAY
#     # ============================================

#     st.session_state.messages.append({"role": "assistant", "content": answer})
#     st.chat_message("assistant").write(answer)











# import streamlit as st
# import torch
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import ollama
# import time

# st.set_page_config(page_title="Biomedical RAG Chatbot", layout="wide")
# st.title("🧬 Biomedical Research Chatbot (RAG)")

# # ============================================
# # LOAD DATA
# # ============================================

# @st.cache_resource
# def load_data():
#     start_time = time.time()

#     data = torch.load("all_pdfs_pagewise_embeddings.pt", weights_only=False)

#     embeddings, chunks, pdfs, pages = [], [], [], []

#     for item in data:
#         embeddings.append(item["embedding"])
#         chunks.append(item["chunk"])
#         pdfs.append(item["pdf"])
#         pages.append(item["page"])

#     embeddings = np.array(embeddings)

#     embed_model = SentenceTransformer(
#         "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
#     )

#     # Warmup LLM
#     ollama.chat(
#         model="qwen2.5:1.5b",
#         messages=[{"role": "user", "content": "hi"}],
#         options={"num_predict": 1}
#     )

#     print(f"[Timing] Load: {time.time() - start_time:.2f}s")
#     return embeddings, chunks, pdfs, pages, embed_model


# embeddings, chunks, pdfs, pages, embed_model = load_data()

# # ============================================
# # QUERY TYPE DETECTION
# # ============================================

# def detect_query_type(question: str):
#     q = question.lower()

#     if any(x in q for x in ["what is", "define", "meaning of", "explain what"]):
#         return "definition"

#     elif any(x in q for x in ["how", "mechanism", "pathway", "why does"]):
#         return "mechanism"

#     elif any(x in q for x in ["analyze", "interpret", "pg/ml", "fold change", "data", "result"]):
#         return "data_analysis"

#     elif any(x in q for x in ["difference", "compare", "vs", "distinguish"]):
#         return "comparison"

#     elif any(x in q for x in ["role of", "function of", "importance of"]):
#         return "functional"

#     else:
#         return "general"

# # ============================================
# # DYNAMIC SYSTEM PROMPT
# # ============================================

# def build_system_prompt(query_type: str):

#     base_prompt = """
# You are a biomedical research assistant with expertise in molecular biology, immunology, and disease mechanisms.

# Use the provided context as the primary source of information.
# If the context is insufficient, say:
# "I do not have enough information in the provided data to answer this question."

# Do not make unsupported claims.
# Do not mention context or sources explicitly.
# Do not provide diagnosis or treatment advice.

# Respond naturally and clearly.

# - If the user greets (e.g., "hi", "hello"), respond politely and offer help in biomedical research.
# - If the user asks what you can do, briefly describe your capabilities in biomedical research.

# Prioritize more relevant or detailed information when multiple data points are present.
# """

#     if query_type == "definition":
#         return base_prompt + """
# Provide a clear and concise explanation of the concept.
# """

#     elif query_type == "mechanism":
#         return base_prompt + """
# Explain the biological mechanism step-by-step, including pathways and key molecules if available.
# """

#     elif query_type == "data_analysis":
#         return base_prompt + """
# Interpret the data carefully. Explain values, units (e.g., pg/mL, fold change), and biological meaning.
# """

#     elif query_type == "comparison":
#         return base_prompt + """
# Compare the entities clearly, highlighting key similarities and differences.
# """

#     elif query_type == "functional":
#         return base_prompt + """
# Explain the biological role and significance.
# """

#     else:
#         return base_prompt

# # ============================================
# # SEARCH
# # ============================================

# def search(query, top_k=4):
#     start_time = time.time()

#     q_emb = embed_model.encode(
#         query,
#         normalize_embeddings=True,
#         show_progress_bar=False
#     )

#     scores = cosine_similarity([q_emb], embeddings)[0]
#     top_idx = scores.argsort()[::-1][:top_k]

#     results = []

#     for rank, i in enumerate(top_idx, start=1):
#         result = {
#             "text": chunks[i][:700],
#             "pdf": pdfs[i],
#             "page": pages[i],
#             "score": scores[i]
#         }

#         results.append(result)

#         print(f"\nRank {rank}")
#         print(f"PDF   : {pdfs[i]}")
#         print(f"Page  : {pages[i]}")
#         print(f"Score : {scores[i]:.4f}")
#         print(f"Text  : {chunks[i][:200]}...")

#     print("============================================")
#     print(f"[Timing] Search: {time.time() - start_time:.2f}s\n")

#     return results

# # ============================================
# # LLM
# # ============================================

# def ask_llm(question, results):
#     start_time = time.time()

#     context = "\n\n".join([
#         f"[Source: {r['pdf']} | Page: {r['page']} | Score: {r['score']:.4f}]\n{r['text']}"
#         for r in results
#     ]) if results else ""

#     print("\n========== CONTEXT SENT TO LLM ==========")
#     print(context)
#     print("=========================================\n")

#     query_type = detect_query_type(question)
#     print(f"[Query Type]: {query_type}")

#     system_prompt = build_system_prompt(query_type)

#     prompt = f"""
# Question: {question}

# Context:
# {context}

# Answer:
# """

#     response = ollama.chat(
#         model="qwen2.5:1.5b",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": prompt}
#         ],
#         options={
#             "temperature": 0.2,
#             "num_predict": 500,
#             "num_ctx": 2048,
#             "num_thread": 8,
#             "top_k": 30,
#             "top_p": 0.9,
#             "repeat_penalty": 1.1,
#         }
#     )

#     answer = response["message"]["content"]

#     print("\n========== FINAL ANSWER ==========")
#     print(answer)
#     print("=================================\n")
#     print(f"[Timing] LLM: {time.time() - start_time:.2f}s")

#     return answer

# # ============================================
# # MEMORY
# # ============================================

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

# # ============================================
# # INPUT (NO HARDCODING)
# # ============================================

# question = st.chat_input("Ask biomedical research question...")

# if question:
#     st.session_state.messages.append({"role": "user", "content": question})
#     st.chat_message("user").write(question)

#     start_time = time.time()

#     # 🔍 SEARCH
#     results = search(question)

#     # 🤖 LLM handles EVERYTHING
#     answer = ask_llm(question, results)

#     print(f"⚡ Total time: {time.time() - start_time:.2f}s")

#     # DISPLAY
#     st.session_state.messages.append({"role": "assistant", "content": answer})
#     st.chat_message("assistant").write(answer)









#####BEST WORKING VERSION WITH QUERY TYPE DETECTION + DYNAMIC SYSTEM PROMPT + SCORE LOGGING + FINAL ANSWER LOGGING#####

# import streamlit as st
# import torch
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import ollama
# import time

# st.set_page_config(page_title="Biomedical RAG Chatbot", layout="wide")
# st.title("🧬 Biomedical Research Chatbot (RAG)")

# # ============================================
# # LOAD DATA
# # ============================================

# @st.cache_resource
# def load_data():
#     start_time = time.time()

#     data = torch.load("all_pdfs_pagewise_embeddings.pt", weights_only=False)

#     embeddings, chunks, pdfs, pages = [], [], [], []

#     for item in data:
#         embeddings.append(item["embedding"])
#         chunks.append(item["chunk"])
#         pdfs.append(item["pdf"])
#         pages.append(item["page"])

#     embeddings = np.array(embeddings)

#     embed_model = SentenceTransformer(
#         "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
#     )

#     # Warmup LLM
#     ollama.chat(
#         model="qwen2.5:1.5b",
#         messages=[{"role": "user", "content": "hi"}],
#         options={"num_predict": 1}
#     )

#     print(f"[Timing] Load: {time.time() - start_time:.2f}s")
#     return embeddings, chunks, pdfs, pages, embed_model


# embeddings, chunks, pdfs, pages, embed_model = load_data()

# # ============================================
# # QUERY TYPE DETECTION
# # ============================================

# def detect_query_type(question: str):
#     q = question.lower()

#     if any(x in q for x in ["what is", "define", "meaning of", "explain what"]):
#         return "definition"

#     elif any(x in q for x in ["how", "mechanism", "pathway", "why does"]):
#         return "mechanism"

#     elif any(x in q for x in ["analyze", "interpret", "pg/ml", "fold change", "data", "result"]):
#         return "data_analysis"

#     elif any(x in q for x in ["difference", "compare", "vs", "distinguish"]):
#         return "comparison"

#     elif any(x in q for x in ["role of", "function of", "importance of"]):
#         return "functional"

#     else:
#         return "general"

# # ============================================
# # DYNAMIC SYSTEM PROMPT
# # ============================================

# def build_system_prompt(query_type: str):

#     base_prompt = """
# You are a biomedical research assistant with expertise in molecular biology, immunology, and disease mechanisms.

# Use the provided context as the primary source of information.
# If the context is insufficient, say:
# "I do not have enough information in the provided data to answer this question."

# Do not make unsupported claims.
# Do not mention context or sources explicitly.
# Do not provide diagnosis or treatment advice.

# # - If the user greets (e.g., "hi", "hello"), respond politely and offer help in biomedical research.
# # - If the user asks what you can do, briefly describe your capabilities in biomedical research.

# Respond naturally and clearly.

# Use previous conversation (if provided) to understand follow-up questions.
# """

#     if query_type == "definition":
#         return base_prompt + "Provide a clear and concise explanation."

#     elif query_type == "mechanism":
#         return base_prompt + "Explain the biological mechanism step-by-step."

#     elif query_type == "data_analysis":
#         return base_prompt + "Interpret the data and explain biological meaning."

#     elif query_type == "comparison":
#         return base_prompt + "Compare clearly with key similarities and differences."

#     elif query_type == "functional":
#         return base_prompt + "Explain biological role and significance."

#     else:
#         return base_prompt

# # ============================================
# # SEARCH
# # ============================================

# def search(query, top_k=4):
#     q_emb = embed_model.encode(
#         query,
#         normalize_embeddings=True,
#         show_progress_bar=False
#     )

#     scores = cosine_similarity([q_emb], embeddings)[0]
#     top_idx = scores.argsort()[::-1][:top_k]

#     results = []

#     for i in top_idx:
#         results.append({
#             "text": chunks[i][:700],
#             "pdf": pdfs[i],
#             "page": pages[i],
#             "score": scores[i]
#         })

#     return results

# # ============================================
# # MEMORY HANDLING (LAST 4)
# # ============================================

# def get_memory_context():
#     """Get last 4 Q&A pairs"""
#     memory = st.session_state.get("chat_memory", [])

#     memory_text = ""
#     for m in memory[-4:]:
#         memory_text += f"User: {m['question']}\nAssistant: {m['answer']}\n\n"

#     return memory_text.strip()

# def update_memory(question, answer):
#     if "chat_memory" not in st.session_state:
#         st.session_state.chat_memory = []

#     st.session_state.chat_memory.append({
#         "question": question,
#         "answer": answer
#     })

#     # Keep only last 4
#     st.session_state.chat_memory = st.session_state.chat_memory[-4:]

# # ============================================
# # LLM
# # ============================================

# def ask_llm(question, results):

#     context = "\n\n".join([
#         r["text"] for r in results
#     ]) if results else ""

#     memory_context = get_memory_context()

#     query_type = detect_query_type(question)
#     system_prompt = build_system_prompt(query_type)

#     prompt = f"""
# Previous Conversation:
# {memory_context}

# Question:
# {question}

# Context:
# {context}

# Answer:
# """

#     response = ollama.chat(
#         model="qwen2.5:1.5b",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": prompt}
#         ],
#         options={
#             "temperature": 0.2,
#             "num_predict": 500,
#             "num_ctx": 2048,
#             "num_thread": 8,
#         }
#     )

#     return response["message"]["content"]

# # ============================================
# # UI MEMORY
# # ============================================

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

# # ============================================
# # INPUT
# # ============================================

# question = st.chat_input("Ask biomedical research question...")

# if question:
#     st.session_state.messages.append({"role": "user", "content": question})
#     st.chat_message("user").write(question)

#     # SEARCH
#     results = search(question)

#     # LLM
#     answer = ask_llm(question, results)

#     # STORE MEMORY
#     update_memory(question, answer)

#     # DISPLAY
#     st.session_state.messages.append({"role": "assistant", "content": answer})
#     st.chat_message("assistant").write(answer)









##### CODE WITH MEMORY + QUERY TYPE DETECTION ########

# import streamlit as st
# import torch
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import ollama
# import time

# st.set_page_config(page_title="Biomedical RAG Chatbot", layout="wide")
# st.title("🧬 Biomedical Research Chatbot (RAG)")

# # ============================================
# # LOAD DATA
# # ============================================

# @st.cache_resource
# def load_data():
#     data = torch.load("all_pdfs_pagewise_embeddings.pt", weights_only=False)
#     embeddings = np.array([item["embedding"] for item in data])
#     chunks = [item["chunk"] for item in data]
#     pdfs = [item["pdf"] for item in data]
#     pages = [item["page"] for item in data]

#     embed_model = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
#     ollama.chat(model="qwen2.5:1.5b", messages=[{"role": "user", "content": "hi"}], options={"num_predict": 1})
#     return embeddings, chunks, pdfs, pages, embed_model

# embeddings, chunks, pdfs, pages, embed_model = load_data()

# # ============================================
# # LLM LOGIC (FIXED IDENTITY & LENGTH)
# # ============================================

# def ask_llm(question, results):
#     # 1. Define Identity based on your specific dataset
#     system_prompt = (
#         "You are an expert Biomedical Research Assistant. "
#         "Your knowledge base is a specialized collection of PDF documents focusing on: "
#         "Lung Cancer mechanisms, Drug-Target interactions (pharmacokinetics/pharmacodynamics), "
#         "clinical trial data, and molecular biology research. "
#         "When asked 'What can I ask you?' or 'What is your knowledge?', describe these specific topics. "
#         "Always provide complete, professional sentences. Never stop mid-sentence."
#     )

#     messages = [{"role": "system", "content": system_prompt}]
    
#     # 2. History for follow-ups
#     if "chat_history" in st.session_state:
#         messages.extend(st.session_state.chat_history[-8:])
    
#     # 3. Context Construction
#     if results:
#         context_str = "\n\n".join([f"[Source: {r['pdf']}, Page: {r['page']}] {r['text']}" for r in results])
#         user_content = f"Using the following research context, answer the question.\n\nContext:\n{context_str}\n\nQuestion: {question}"
#     else:
#         # For greetings or meta-questions
#         user_content = question

#     messages.append({"role": "user", "content": user_content})

#     # 4. Optimized Generation Settings
#     response = ollama.chat(
#         model="qwen2.5:1.5b",
#         messages=messages,
#         options={
#             "temperature": 0.3,
#             "num_ctx": 4096,     # Large window to prevent logic overflow
#             "num_predict": 600,  # Increased to prevent truncation
#             "top_p": 0.9,
#             "num_thread": 8
#         }
#     )
#     return response["message"]["content"]

# # ============================================
# # SEARCH
# # ============================================

# def search(query, top_k=4):
#     q_emb = embed_model.encode(query, normalize_embeddings=True, show_progress_bar=False)
#     scores = cosine_similarity([q_emb], embeddings)[0]
#     top_idx = scores.argsort()[::-1][:top_k]
    
#     results = []
#     for i in top_idx:
#         if scores[i] > 0.32: # Filter irrelevant noise
#             results.append({"text": chunks[i], "pdf": pdfs[i], "page": pages[i], "score": scores[i]})
#     return results

# # ============================================
# # UI & FLOW
# # ============================================

# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

# question = st.chat_input("Ask about Lung Cancer research, drug mechanisms, etc.")

# if question:
#     st.session_state.messages.append({"role": "user", "content": question})
#     st.chat_message("user").write(question)

#     with st.spinner("Processing..."):
#         # Check if question is a meta-question
#         meta_triggers = ["who are you", "what can i ask", "knowledge", "help", "capabilities"]
#         is_meta = any(word in question.lower() for word in meta_triggers)
        
#         if is_meta:
#             results = [] # Let the LLM use its System Prompt identity
#         else:
#             results = search(question)
        
#         answer = ask_llm(question, results)

#     # Final Display and State Update
#     st.session_state.chat_history.append({"role": "user", "content": question})
#     st.session_state.chat_history.append({"role": "assistant", "content": answer})
#     st.session_state.messages.append({"role": "assistant", "content": answer})
#     st.chat_message("assistant").write(answer)









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

   
   
   
   
   
   
  ### PERFORMANCE EVALUATING GRAPHIC DASHBOARD (ADDITIONAL CODE) ### 
# import streamlit as st
# import torch
# import numpy as np
# import json
# import pandas as pd
# from sentence_transformers import SentenceTransformer, util
# from sklearn.metrics.pairwise import cosine_similarity
# import ollama

# # --- PAGE CONFIG ---
# st.set_page_config(page_title="Biomedical RAG Chatbot", layout="wide")
# st.title("🧬 Biomedical Research Chatbot (RAG + Evaluation Dashboard)")

# # --- LOAD DATA ---
# @st.cache_resource
# def load_data():
#     data = torch.load("all_pdfs_pagewise_embeddings.pt", weights_only=False)

#     embeddings = np.array([item["embedding"] for item in data])
#     chunks = [item["chunk"] for item in data]
#     pdfs = [item["pdf"] for item in data]
#     pages = [item["page"] for item in data]

#     embed_model = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

#     return embeddings, chunks, pdfs, pages, embed_model


# embeddings, chunks, pdfs, pages, embed_model = load_data()

# # --- SEARCH LOGIC ---
# def search(query, top_k=10):
#     q_emb = embed_model.encode(query, normalize_embeddings=True, show_progress_bar=False)

#     scores = cosine_similarity([q_emb], embeddings)[0]
#     top_idx = scores.argsort()[::-1][:top_k]

#     results = []
#     for i in top_idx:
#         if scores[i] > 0.35:
#             results.append({
#                 "text": chunks[i],
#                 "pdf": pdfs[i],
#                 "page": pages[i],
#                 "score": float(scores[i])
#             })

#     return results


# # --- LLM ANSWER GENERATION ---
# def ask_llm_stream(question, results):
#     history_lines = []
#     if "messages" in st.session_state:
#         for m in st.session_state.messages[-5:]:
#             history_lines.append(f"{m['role'].upper()}: {m['content']}")

#     history_str = "\n".join(history_lines)

#     context_text = "\n".join([r["text"] for r in results]) if results else "No context found."

#     system_prompt = (
#         "You are a Biomedical Research Assistant.\n"
#         "Answer strictly using provided context.\n"
#         "Write 8-10 lines in professional biomedical language.\n"
#         "If context is missing, say you don't know."
#     )

#     user_message = f"""
# --- CHAT HISTORY ---
# {history_str}

# --- RESEARCH CONTEXT ---
# {context_text}

# USER QUESTION:
# {question}
# """

#     return ollama.chat(
#         model="qwen2.5:1.5b",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_message}
#         ],
#         stream=True,
#         options={"temperature": 0.1, "num_thread": 8}
#     )


# # --- LLM SELF EVALUATION ---
# def evaluate_answer(question, answer, context):
#     eval_prompt = f"""
# You are an evaluation system for biomedical QA.

# QUESTION:
# {question}

# ANSWER:
# {answer}

# CONTEXT:
# {context}

# Evaluate and return ONLY valid JSON:
# {{
#  "relevance": (0-10),
#  "accuracy": (0-10),
#  "completeness": (0-10),
#  "overall": (0-10)
# }}
# """

#     response = ollama.chat(
#         model="qwen2.5:1.5b",
#         messages=[{"role": "user", "content": eval_prompt}],
#         stream=False
#     )

#     try:
#         return json.loads(response["message"]["content"])
#     except:
#         return {
#             "relevance": 0,
#             "accuracy": 0,
#             "completeness": 0,
#             "overall": 0
#         }


# # --- SESSION INIT ---
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "metrics" not in st.session_state:
#     st.session_state.metrics = []


# # --- DISPLAY CHAT HISTORY ---
# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])


# # --- INPUT ---
# question = st.chat_input("Ask about biomedical research...")

# if question:

#     # USER MESSAGE
#     st.chat_message("user").write(question)
#     st.session_state.messages.append({"role": "user", "content": question})

#     # RETRIEVAL
#     results = search(question)

#     retrieval_conf = np.mean([r["score"] for r in results]) if results else 0

#     # TRACE UI
#     with st.expander("🔍 Retrieval System Trace", expanded=True):
#         st.metric("Retrieval Confidence", f"{retrieval_conf:.4f}")

#         if results:
#             for i, res in enumerate(results):
#                 st.info(
#                     f"Result {i+1} | {res['pdf']} | Page {res['page']} | Score {res['score']:.4f}"
#                 )
#                 st.caption(res["text"][:300] + "...")
#         else:
#             st.warning("No relevant context found.")

#     # LLM RESPONSE
#     with st.chat_message("assistant"):
#         placeholder = st.empty()
#         full_response = ""

#         for chunk in ask_llm_stream(question, results):
#             content = chunk["message"]["content"]
#             full_response += content
#             placeholder.markdown(full_response + "▌")

#         placeholder.markdown(full_response)

#     st.session_state.messages.append({"role": "assistant", "content": full_response})

#     # EVALUATION
#     context_text = "\n".join([r["text"] for r in results])
#     eval_json = evaluate_answer(question, full_response, context_text)

#     # METRICS STORAGE
#     st.session_state.metrics.append({
#         "question": question,
#         "retrieval": retrieval_conf,
#         "relevance": eval_json["relevance"],
#         "accuracy": eval_json["accuracy"],
#         "completeness": eval_json["completeness"],
#         "overall": eval_json["overall"]
#     })

#     # DASHBOARD
#     st.subheader("📊 Answer Quality Metrics")

#     col1, col2, col3 = st.columns(3)

#     col1.metric("Relevance", eval_json["relevance"])
#     col2.metric("Accuracy", eval_json["accuracy"])
#     col3.metric("Completeness", eval_json["completeness"])

#     st.metric("Overall Score", eval_json["overall"])


# # --- OPTIONAL GLOBAL ANALYTICS DASHBOARD ---
# if st.session_state.metrics:
#     st.subheader("📈 System Performance Trend")

#     df = pd.DataFrame(st.session_state.metrics)

#     st.line_chart(df[["retrieval", "accuracy", "overall"]])

#     st.dataframe(df)