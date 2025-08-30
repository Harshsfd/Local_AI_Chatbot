# main.py
from sentence_transformers import SentenceTransformer
import faiss
import os
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load data
texts = []
data_folder = 'data'
for file in os.listdir(data_folder):
    if file.endswith('.txt'):
        with open(os.path.join(data_folder, file), 'r', encoding='utf-8') as f:
            texts.append(f.read())

# Create embeddings
embeddings = embed_model.encode(texts)
dim = embeddings.shape[1]

# Build FAISS index
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype=np.float32))

# Chat loop
while True:
    query = input("You: ")
    if query.lower() in ['exit', 'quit']:
        break
    
    query_emb = embed_model.encode([query])
    D, I = index.search(np.array(query_emb, dtype=np.float32), k=1)
    context = texts[I[0][0]]
    
    # Generate response using GPT-2
    input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = gpt_model.generate(inputs, max_length=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("Bot:", response)
  
