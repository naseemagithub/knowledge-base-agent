import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# --------------------
# Embedding model (local, free)
# --------------------
@st.cache_resource
def get_embedder():
    # Small, fast sentence-transformers model
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)

def read_pdfs(uploaded_files):
    text = ""
    for f in uploaded_files:
        reader = PdfReader(f)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
    return text

def chunk_text(text, chunk_size=800, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def get_embeddings(texts):
    model = get_embedder()
    # normalize embeddings so cosine similarity = dot product
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return np.array(embeddings, dtype="float32")

def cosine_similarity(query_vec, doc_matrix):
    # if normalized, cosine = dot product
    return doc_matrix @ query_vec

def build_prompt(question, chunks):
    context = "\n\n----\n\n".join(chunks)
    return f"""
You are a helpful assistant. Answer the question ONLY using the context.

CONTEXT:
{context}

QUESTION:
{question}

If the answer is not in the context, say clearly:
"I don't know based on the provided documents."
"""

def chat_completion(client, prompt):
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# --------------------
# Streamlit UI
# --------------------

st.set_page_config(
    page_title="Knowledge Base AI Agent (Groq + Local Embeddings)",
    page_icon="🤖"
)

st.title("🤖 Knowledge Base AI Agent")
st.write(
    "Upload PDFs → They are converted to local embeddings (free) → "
    "Groq Llama 3.1 answers using only your documents."
)

# Sidebar
st.sidebar.header("Settings")

api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    help="Create one at https://console.groq.com (starts with gsk_...)"
)

chunk_size = st.sidebar.slider("Chunk size", 400, 1200, 800)
overlap = st.sidebar.slider("Chunk overlap", 100, 400, 200)
top_k = st.sidebar.slider("Top K retrieved chunks", 2, 8, 4)

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("📚 Index Documents"):
    if not api_key:
        st.error("Please enter your Groq API Key.")
    elif not uploaded_files:
        st.error("Please upload at least one PDF.")
    else:
        with st.spinner("Reading and indexing... (first time may take ~1–2 minutes to download the embedding model)"):
            # just to validate key if needed
            client = get_client(api_key)

            text = read_pdfs(uploaded_files)
            if not text.strip():
                st.error("Could not extract any text from the PDFs.")
            else:
                chunks = chunk_text(text, chunk_size, overlap)
                embeddings = get_embeddings(chunks)

                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings

                st.success(f"Indexed {len(chunks)} chunks.")

st.markdown("---")

st.subheader("💬 Ask a Question")
question = st.text_input("Type your question:")

if st.button("🔍 Get Answer"):
    if not api_key:
        st.error("Please enter your Groq API Key.")
    elif not st.session_state.chunks or st.session_state.embeddings is None:
        st.error("Please index documents first.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            client = get_client(api_key)

            q_embed = get_embeddings([question])[0]
            sims = cosine_similarity(q_embed, st.session_state.embeddings)

            top_idx = np.argsort(sims)[::-1][:top_k]
            top_chunks = [st.session_state.chunks[i] for i in top_idx]

            prompt = build_prompt(question, top_chunks)
            answer = chat_completion(client, prompt)

        st.markdown("### ✅ Answer")
        st.write(answer)

        with st.expander("🔎 Retrieved Context"):
            for i, idx in enumerate(top_idx, 1):
                st.markdown(f"**Chunk {i} (Score: {sims[idx]:.3f})**")
                st.write(st.session_state.chunks[idx])
                st.markdown("---")
