import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

st.set_page_config(page_title="Local AI Chatbot", page_icon="🤖", layout="wide")

st.markdown("""
    <h1 style='text-align:center;'>🤖 Local AI Chatbot</h1>
    <p style='text-align:center;'>Ask questions based on your local text files!</p>
""", unsafe_allow_html=True)

# ------------------ Load Embedding Model ------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embedding_model()

# ------------------ Load & Chunk Text Data ------------------
texts = []
data_folder = 'data'
for file in os.listdir(data_folder):
    if file.endswith('.txt'):
        with open(os.path.join(data_folder, file), 'r', encoding='utf-8') as f:
            content = f.read()
            # Split into sentences for better FAISS retrieval
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            texts.extend(sentences)

# ------------------ Build FAISS Index ------------------
embeddings = embed_model.encode(texts)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype=np.float32))

# ------------------ Session State ------------------
if 'history' not in st.session_state:
    st.session_state.history = []

# ------------------ Factual Answer Extraction ------------------
def extract_answer(query, context):
    query_lower = query.lower()
    context_lower = context.lower()
    if "capital" in query_lower and "capital" in context_lower:
        for sentence in context.split('.'):
            if "capital" in sentence.lower():
                return sentence.strip()
    elif "states" in query_lower and "states" in context_lower:
        for sentence in context.split('.'):
            if "states" in sentence.lower():
                return sentence.strip()
    elif any(keyword in query_lower for keyword in ["languages", "official language"]):
        for sentence in context.split('.'):
            if "language" in sentence.lower():
                return sentence.strip()
    elif any(keyword in query_lower for keyword in ["festival", "celebration"]):
        for sentence in context.split('.'):
            if "festival" in sentence.lower() or "celebration" in sentence.lower():
                return sentence.strip()
    else:
        return context.strip()
    return "Sorry, I could not find the answer in my data."

# ------------------ User Input ------------------
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your question here:", "")
    submit_button = st.form_submit_button("Ask")

if submit_button and user_input:
    with st.spinner("Processing your question..."):
        # Embed user query
        query_emb = embed_model.encode([user_input])
        D, I = index.search(np.array(query_emb, dtype=np.float32), k=3)  # top-3 relevant sentences
        context = " ".join([texts[i] for i in I[0]])

        # Extract accurate answer
        answer = extract_answer(user_input, context)

        # Save conversation in session
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", answer))

# ------------------ Display Chat ------------------
chat_container = st.container()
with chat_container:
    for speaker, text in st.session_state.history:
        if speaker == "You":
            st.markdown(f"""
                <div style="text-align: right; margin:10px;">
                    <span style="background-color:#DCF8C6; padding:10px; border-radius:10px; display:inline-block;">**You:** {text}</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="text-align: left; margin:10px;">
                    <span style="background-color:#F1F0F0; padding:10px; border-radius:10px; display:inline-block;">**Bot:** {text}</span>
                </div>
            """, unsafe_allow_html=True)
            
