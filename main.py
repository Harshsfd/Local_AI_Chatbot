import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load models
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
    return embed_model, tokenizer, gpt_model

embed_model, tokenizer, gpt_model = load_models()

# Load data
texts = []
data_folder = 'data'
for file in os.listdir(data_folder):
    if file.endswith('.txt'):
        with open(os.path.join(data_folder, file), 'r', encoding='utf-8') as f:
            texts.append(f.read())

# Load FAISS index
index = faiss.read_index("embeddings/index.faiss")

# Streamlit UI
st.title("Local AI Chatbot 🌟")
st.write("Ask questions based on your local data files!")

query = st.text_input("Your question:")

if st.button("Ask") and query:
    # Embed query
    query_emb = embed_model.encode([query])
    D, I = index.search(np.array(query_emb, dtype=np.float32), k=1)
    context = texts[I[0][0]]
    
    # Generate GPT-2 response
    input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = gpt_model.generate(inputs, max_length=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    st.write("**Bot:**", response)
