import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

st.set_page_config(page_title="Local AI Chatbot", page_icon="🤖")

st.title("Local AI Chatbot 🌟")
st.write("Ask questions based on your local text files!")

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embedding_model()

# Load local data files and split into sentences
texts = []
data_folder = 'data'
for file in os.listdir(data_folder):
    if file.endswith('.txt'):
        with open(os.path.join(data_folder, file), 'r', encoding='utf-8') as f:
            content = f.read()
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            texts.extend(sentences)

# Generate embeddings and build FAISS index
embeddings = embed_model.encode(texts)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype=np.float32))

# Initialize session state for conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

# User input
query = st.text_input("Your question:", key="query")

def extract_answer(query, context):
    """Simple rule-based extraction for factual questions"""
    query_lower = query.lower()
    context_lower = context.lower()

    # Check if query keywords are in context
    if "capital" in query_lower and "capital" in context_lower:
        for sentence in context.split('.'):
            if "capital" in sentence.lower():
                return sentence.strip()
    elif "states" in query_lower and "states" in context_lower:
        for sentence in context.split('.'):
            if "states" in sentence.lower():
                return sentence.strip()
    else:
        # Default: return full context
        return context.strip()
    return "Sorry, I could not find the answer in my data."

if st.button("Ask") and query:
    # Embed query and search FAISS index
    query_emb = embed_model.encode([query])
    D, I = index.search(np.array(query_emb, dtype=np.float32), k=1)
    context = texts[I[0][0]]

    # Extract precise answer
    answer = extract_answer(query, context)

    # Save conversation in session state
    st.session_state.history.append(("You", query))
    st.session_state.history.append(("Bot", answer))

# Display conversation
for speaker, text in st.session_state.history:
    if speaker == "You":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Bot:** {text}")
        
