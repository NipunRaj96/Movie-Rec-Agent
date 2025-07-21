from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingHandler:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str):
        return self.model.encode([text])[0]

    def embed_texts(self, texts: List[str]):
        return self.model.encode(texts) 