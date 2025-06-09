from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-base-en-v1.5')
emb = model.encode("Test sentence")
print(emb.shape)  # Should output (384,)