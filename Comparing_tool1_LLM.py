import streamlit as st
import ollama

st.set_page_config(page_title="Tool 1: Standalone LLM", layout="wide")
st.title("🤖 Tool 1: Standalone LLM (No RAG)")
st.warning("This tool uses ONLY internal pre-trained knowledge. No PDFs are loaded.")

# Initialize message history array in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history interface UI
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.chat_input("Ask a question to the standalone model...")

if question:
    # 1. Render the user question to the web screen
    st.chat_message("user").write(question)
    
    # 2. Append the current question to your running session data array
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # --- FIX START: Construct the full message thread with memory (CORRECTION 6) ---
        # Add the foundational system prompt context at the start of the payload array
        llm_payload = [
            {
                "role": "system", 
                "content": "You are a Biomedical Research Assistant. Answer based on your general knowledge. "
                           "STRICT RULES: Answer in one paragraph of 8-10 lines using professional medical terminology. "
                           "You have full access to the current conversation history to answer follow-up questions."
            }
        ]
        
        # Append all previous interactions (User questions + Assistant answers) into the active context thread
        for msg in st.session_state.messages:
            llm_payload.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        # --- FIX END ---
        
        # Call the local Ollama daemon using your running chat history sequence
        response = ollama.chat(
            model="qwen2.5:1.5b",
            messages=llm_payload, # Now tracks past context turns dynamically
            stream=True,
            options={"temperature": 0.7} # Higher temp to show its own "creative" knowledge
        )
        
        # Stream the tokens iteratively to the app view
        for chunk in response:
            content = chunk["message"]["content"]
            full_response += content
            response_placeholder.markdown(full_response + "▌")
            
        response_placeholder.markdown(full_response)
    
    # Append the finalized string context block into memory for subsequent turns
    st.session_state.messages.append({"role": "assistant", "content": full_response})
