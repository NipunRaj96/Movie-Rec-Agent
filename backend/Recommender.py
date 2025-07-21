import pandas as pd
from backend.Embedding import EmbeddingHandler
from backend.VectorDB import VectorDBHandler
from backend.LLM import LLMHandler

class Recommender:
    def __init__(self, data_path='data/movies.csv'):
        self.movies = pd.read_csv(data_path)
        self.embedder = EmbeddingHandler()
        self.embeddings = self.embedder.embed_texts(self.movies['summary'].tolist())
        self.vectordb = VectorDBHandler(dim=self.embeddings.shape[1])
        self.vectordb.add_embeddings(self.embeddings, self.movies.index.tolist())
        self.llm = LLMHandler()

    def recommend(self, user_query, top_k=5):
        query_emb = self.embedder.embed_text(user_query)
        top_ids = self.vectordb.search(query_emb, top_k)
        top_movies = self.movies.iloc[top_ids]
        movie_infos = [
            {
                "title": row["title"],
                "summary": row["summary"],
                "year": row["year"],
                "genre": row["genre"],
                "director": row["director"],
                "cast": row["cast"],
                "duration": row["duration"],
                "language": row["language"],
                "poster_url": row["poster_url"],
                "country": row["country"]
            }
            for _, row in top_movies.iterrows()
        ]
        recommendations = self.llm.get_recommendation(user_query, movie_infos)
        return recommendations 