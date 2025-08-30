import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from pathlib import Path
import openai  # For generative AI (RAG)

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Next-Level AI Chatbot", page_icon="🤖", layout="wide")

st.markdown("""
<h1 style='text-align:center;'>🤖 Next-Level Local AI Chatbot</h1>
<p style='text-align:center;'>Ask questions based on your local files and get intelligent answers!</p>
""", unsafe_allow_html=True)

# Set your OpenAI API key (for RAG generation)
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

# ------------------ Load Embedding Model ------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-mpnet-base-v2')

embed_model = load_embedding_model()

# ------------------ Load & Chunk Text Data ------------------
texts = []
data_folder = Path('data')
for file_path in data_folder.glob("*.txt"):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        sentences = [s.strip() for s in content.replace('\n', ' ').split('.') if s.strip()]
        texts.extend(sentences)

# ------------------ Build FAISS Index ------------------
embeddings = embed_model.encode(texts, convert_to_numpy=True)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype=np.float32))

# ------------------ Session State ------------------
if 'history' not in st.session_state:
    st.session_state.history = []

# ------------------ RAG-based Answer ------------------
def generate_answer_with_context(query, context):
    """
    If simple keyword extraction fails, use OpenAI GPT to generate answer
    based on the context from local files.
    """
    if not openai.api_key:
        return context.strip() if context else "No OpenAI API key found. Install key to get generative answers."
    
    prompt = f"""
    You are an AI assistant. Answer the question based on the following context:
    
    CONTEXT:
    {context}
    
    QUESTION: {query}
    
    Provide a concise and clear answer. If answer is not in the context, say 'Sorry, I could not find the answer in my data.'
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            max_tokens=200,
            temperature=0.2
        )
        answer = response['choices'][0]['message']['content']
        return answer
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# ------------------ Extract Answer ------------------
def extract_answer(query, context):
    query_lower = query.lower()
    # Keyword-based extraction
    if "capital" in query_lower:
        for s in context.split('.'):
            if "capital" in s.lower():
                return s.strip()
    elif "states" in query_lower:
        for s in context.split('.'):
            if "states" in s.lower():
                return s.strip()
    elif any(k in query_lower for k in ["language", "languages", "official language"]):
        for s in context.split('.'):
            if "language" in s.lower():
                return s.strip()
    elif any(k in query_lower for k in ["festival", "celebration"]):
        for s in context.split('.'):
            if "festival" in s.lower() or "celebration" in s.lower():
                return s.strip()
    # Fallback: use generative answer
    return generate_answer_with_context(query, context)

# ------------------ User Input ------------------
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your question here:", "")
    submit_button = st.form_submit_button("Ask")

if submit_button and user_input:
    with st.spinner("Processing your question..."):
        # Embed query
        query_emb = embed_model.encode([user_input], convert_to_numpy=True)
        D, I = index.search(np.array(query_emb, dtype=np.float32), k=5)
        context = " ".join([texts[i] for i in I[0]])

        answer = extract_answer(user_input, context)

        # Store conversation
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
            
