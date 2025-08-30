from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
texts = []
data_folder = 'data'
for file in os.listdir(data_folder):
    if file.endswith('.txt'):
        with open(os.path.join(data_folder, file), 'r', encoding='utf-8') as f:
            texts.append(f.read())

# Create embeddings
embeddings = embed_model.encode(texts)

# Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype=np.float32))

# Save FAISS index
if not os.path.exists('embeddings'):
    os.makedirs('embeddings')
faiss.write_index(index, "embeddings/index.faiss")

print("FAISS index created and saved as embeddings/index.faiss")
