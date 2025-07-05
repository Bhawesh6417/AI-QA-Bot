import os
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import textwrap

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_documents(doc_folder="documents"):
    docs = []
    for fname in os.listdir(doc_folder):
        path = os.path.join(doc_folder, fname)
        if fname.endswith(".pdf"):
            text = extract_text_from_pdf(path)
        elif fname.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            continue
        docs.append((fname, text))
    return docs

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def chunk_text(text, max_tokens=150):
    words = text.split()
    return [" ".join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]

def embed_chunks(chunks):
    return model.encode(chunks)

def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_index(index, query, chunks, k=4):
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec), k)
    return [chunks[i] for i in indices[0]]

def export_chat_to_pdf(chat_history):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 0.75 * inch
    max_width = width - 2 * margin
    y = height - margin
    line_height = 14
    wrap_chars = 100  # approx number of chars that fit in line

    c.setFont("Helvetica", 12)

    for role, message in chat_history:
        prefix = "You: " if role == "user" else "Bot: "
        wrapped_lines = textwrap.wrap(prefix + message, width=wrap_chars)

        for line in wrapped_lines:
            if y <= margin:
                c.showPage()
                c.setFont("Helvetica", 12)
                y = height - margin
            c.drawString(margin, y, line)
            y -= line_height

        y -= line_height  # extra space between messages

    c.save()
    buffer.seek(0)
    return buffer

