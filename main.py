import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

st.set_page_config(page_title="Local AI Chatbot", page_icon="🤖")

st.title("Local AI Chatbot 🌟")
st.write("Ask questions based on your local text files!")

# Load models (cache to avoid reloading on each interaction)
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
    return embed_model, tokenizer, gpt_model

embed_model, tokenizer, gpt_model = load_models()

# Load local data files
texts = []
data_folder = 'data'
for file in os.listdir(data_folder):
    if file.endswith('.txt'):
        with open(os.path.join(data_folder, file), 'r', encoding='utf-8') as f:
            texts.append(f.read())

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

if st.button("Ask") and query:
    # Embed query
    query_emb = embed_model.encode([query])
    D, I = index.search(np.array(query_emb, dtype=np.float32), k=1)
    context = texts[I[0][0]]

    # Generate response using GPT-2
    input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = gpt_model.generate(inputs, max_length=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Save conversation in session state
    st.session_state.history.append(("You", query))
    st.session_state.history.append(("Bot", response))

# Display conversation
for speaker, text in st.session_state.history:
    if speaker == "You":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Bot:** {text}")
