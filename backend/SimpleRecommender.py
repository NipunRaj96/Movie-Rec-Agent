import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.Database import DatabaseManager

class SimpleRecommender:
    """Simple movie recommendation system without complex dependencies."""
    
    def __init__(self, database_path: str = "sqlite:///data/movieapp.db"):
        """Initialize the simple recommender system."""
        self.db = DatabaseManager(database_path)
        self.movies_df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Load data and initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize the recommender."""
        print("🚀 Initializing simple recommendation system...")
        self._load_movie_data()
        self._initialize_content_based()
        print("✅ Simple recommender system initialized successfully!")
    
    def _load_movie_data(self):
        """Load movie data from database safely."""
        # Get ALL movies without any limit
        movies = self.db.get_movies(limit=None)  # Ensure no limit is applied
        
        movie_data = []
        for movie in movies:
            movie_data.append({
                'id': movie.id,
                'tmdb_id': movie.tmdb_id,
                'title': movie.title,
                'summary': movie.summary or '',
                'year': movie.year,
                'genres': movie.primary_genre or '',
                'director': movie.director or '',
                'cast': movie.cast or '',
                'vote_average': movie.vote_average or 0.0,
                'vote_count': movie.vote_count or 0,
                'popularity': movie.popularity or 0.0,
                'runtime': movie.runtime or 0,
                'poster_url': movie.poster_url or '',
            })
        
        self.movies_df = pd.DataFrame(movie_data)
        print(f"📊 Loaded {len(self.movies_df)} movies for recommendation engine")
        
        # Debug: Show first few movies by genre
        if not self.movies_df.empty:
            genre_sample = self.movies_df.groupby('genres').head(2)
            print("🎬 Sample movies by genre:")
            for _, movie in genre_sample.iterrows():
                print(f"  {movie['genres']}: {movie['title']}")

    def _initialize_content_based(self):
        """Initialize content-based filtering."""
        if self.movies_df is None or self.movies_df.empty:
            print("⚠️ No movie data available for content-based filtering")
            return
        
        print(f"🔧 Initializing TF-IDF for {len(self.movies_df)} movies...")
        
        # Create content features
        content_features = []
        for _, movie in self.movies_df.iterrows():
            features = f"{movie['title']} {movie['summary']} {movie['genres']} {movie['director']} {movie['cast']}"
            content_features.append(features)
        
        # Create TF-IDF matrix
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(content_features)
        print(f"✅ Content-based filtering initialized with {self.tfidf_matrix.shape[0]} movies")
    
    def recommend(self, user_query: str, user_id: Optional[int] = None, top_k: int = 5) -> List[Dict]:
        """Get movie recommendations (API interface)."""
        return self.get_recommendations(user_query, top_k)

    def get_recommendations(self, query: str, top_k: int = 5) -> List[Dict]:
        """Get movie recommendations based on query."""
        try:
            if self.movies_df is None or self.movies_df.empty:
                return []
            
            # Transform query using TF-IDF
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate similarity with all movies
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top recommendations
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            recommendations = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Only include movies with some similarity
                    movie = self.movies_df.iloc[idx]
                    
                    # Simple explanation without LLM
                    explanation = self._generate_simple_explanation(query, movie.to_dict(), similarities[idx])
                    
                    recommendations.append({
                        'id': int(movie['id']),
                        'tmdb_id': int(movie['tmdb_id']),
                        'title': movie['title'],
                        'summary': movie['summary'],
                        'year': movie['year'],
                        'genres': [movie['genres']] if movie['genres'] else [],
                        'director': movie['director'],
                        'cast': movie['cast'],
                        'vote_average': float(movie['vote_average']),
                        'vote_count': int(movie['vote_count']),
                        'poster_url': movie['poster_url'],
                        'runtime': int(movie['runtime']),
                        'explanation': explanation,
                        'similarity_score': float(similarities[idx])
                    })
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
            return []
    
    def _generate_simple_explanation(self, query: str, movie: Dict, similarity: float) -> str:
        """Generate simple explanation for recommendation."""
        explanations = [
            f"This {movie['genres'].lower()} movie matches your interest in '{query}'.",
            f"Recommended because of similar themes to '{query}'.",
            f"Great match for your '{query}' preference with {movie['genres'].lower()} elements.",
            f"Selected based on content similarity to '{query}' (Director: {movie['director']})."
        ]
        
        # Pick explanation based on similarity score
        if similarity > 0.3:
            return explanations[0]
        elif similarity > 0.2:
            return explanations[1]
        elif similarity > 0.1:
            return explanations[2]
        else:
            return explanations[3] 