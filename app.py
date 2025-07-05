import os
import streamlit as st
from dotenv import load_dotenv
import requests
from utils import load_documents, chunk_text, embed_chunks, create_faiss_index, search_index, export_chat_to_pdf

# Load environment
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "mistralai/mixtral-8x7b-instruct"

# Load and process documents
@st.cache_resource
def prepare_data():
    docs = load_documents("documents")
    all_chunks = []
    for fname, text in docs:
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
    embeddings = embed_chunks(all_chunks)
    index = create_faiss_index(embeddings)
    return index, all_chunks

index, chunks = prepare_data()

# Streamlit UI
st.title("📚 AI Document Q&A Bot using Mistral")
st.markdown("Ask questions about your documents (PDF/TXT).")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("🧠 Your question:")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    matched_chunks = search_index(index, user_input, chunks, k=4)
    context = "\n\n".join(matched_chunks)

    prompt = f"""You are an AI assistant. Use the context below to answer the user's question.
Context:
{context}

Question: {user_input}
Answer:"""

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        res = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
        res.raise_for_status()
        response = res.json()
        answer = response["choices"][0]["message"]["content"]
    except Exception as e:
        answer = f"⚠️ API Error: {str(e)}"

    st.session_state.chat_history.append(("bot", answer))

# Display chat history
for sender, msg in st.session_state.chat_history:
    if sender == "user":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")

# Export chat history to PDF
if st.session_state.chat_history:
    pdf_buffer = export_chat_to_pdf(st.session_state.chat_history)
    st.download_button(
        label="📄 Export Chat as PDF",
        data=pdf_buffer,
        file_name="chat_history.pdf",
        mime="application/pdf"
    )
