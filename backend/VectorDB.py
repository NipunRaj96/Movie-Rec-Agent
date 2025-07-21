import faiss
import numpy as np
from typing import List

class VectorDBHandler:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.movie_ids = []

    def add_embeddings(self, embeddings: np.ndarray, ids: List[int]):
        self.index.add(embeddings)
        self.movie_ids.extend(ids)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        D, I = self.index.search(np.array([query_embedding]), top_k)
        return [self.movie_ids[i] for i in I[0]] 