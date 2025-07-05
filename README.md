# 📚 **AI Document Q&A Bot using Mistral**

An AI-powered chatbot that reads documents (PDF/TXT), retrieves relevant content using semantic search, and answers your questions using the powerful Mistral Mixtral-8x7B-Instruct model via OpenRouter.

---

## 🔍 **Features**

- ✅ Accepts `.pdf` and `.txt` files
- ✅ Extracts and chunks document text
- ✅ Generates embeddings using Sentence Transformers
- ✅ Retrieves relevant info using FAISS vector search
- ✅ Answers questions using Mistral (OpenRouter API)
- ✅ Interactive Streamlit chat interface
- ✅ Export chat history as a **structured PDF**

---

## 🧱 **Tech Stack**

| Component      | Tool / Library                     |
|----------------|------------------------------------|
| Interface      | Streamlit                          |
| Backend Logic  | Python, Requests                   |
| Embedding      | `sentence-transformers`            |
| Vector Search  | `faiss-cpu`                        |
| LLM Inference  | `mistralai/mixtral-8x7b-instruct` via OpenRouter |
| File Parsing   | PyPDF2                             |
| Export Format  | ReportLab for PDF generation       |

---

