import streamlit as st
import ollama

st.set_page_config(page_title="Tool 1: Standalone LLM", layout="wide")
st.title("🤖 Tool 1: Standalone LLM (No RAG)")
st.warning("This tool uses ONLY internal pre-trained knowledge. No PDFs are loaded.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.chat_input("Ask a question to the standalone model...")

if question:
    st.chat_message("user").write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Pure LLM call with NO context or history logic
        response = ollama.chat(
            model="qwen2.5:1.5b",
            messages=[
                {"role": "system", "content": "You are a Biomedical Research Assistant. Answer based on your general knowledge. STRICT RULES: Answer in one paragraph of 8-10 lines using professional medical terminology."},
                {"role": "user", "content": question}
            ],
            stream=True,
            options={"temperature": 0.7} # Higher temp to show its own "creative" knowledge
        )
        
        for chunk in response:
            content = chunk["message"]["content"]
            full_response += content
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
