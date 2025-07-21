import pandas as pd
import numpy as np
import time
import json
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from backend.Embedding import EmbeddingHandler
from backend.VectorDB import VectorDBHandler
from backend.LLM import LLMHandler
from backend.Database import DatabaseManager, Movie, User, UserRating
from backend.Cache import embedding_cache, recommendation_cache, performance_monitor

class AdvancedRecommender:
    """Advanced movie recommendation system with multiple algorithms and personalization."""
    
    def __init__(self, database_path: str = "sqlite:///data/movieapp.db"):
        """Initialize the advanced recommender system."""
        self.db = DatabaseManager(database_path)
        self.embedder = EmbeddingHandler()
        self.llm = LLMHandler()
        
        # Initialize components
        self.vectordb = None
        self.movies_df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.svd_model = None
        self.movie_clusters = None
        self.content_similarity_matrix = None
        
        # Load data and initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all ML models and data structures."""
        print("🚀 Initializing advanced recommendation models...")
        
        # Load movie data
        self._load_movie_data()
        
        # Initialize different recommendation algorithms
        self._initialize_content_based()
        self._initialize_vector_search()
        self._initialize_collaborative_filtering()
        self._initialize_clustering()
        
        print("✅ Advanced recommender system initialized successfully!")
    
    def _load_movie_data(self):
        """Load movie data from database."""
        movies = self.db.get_movies()
        if not movies:
            # Load from CSV if database is empty
            self.db.load_movies_from_csv()
            movies = self.db.get_movies()
        
        # Convert to DataFrame for easier manipulation
        movie_data = []
        for movie in movies:
            # Safely handle genres to avoid session issues
            try:
                all_genres = ', '.join([g.name for g in movie.genres]) if hasattr(movie, 'genres') and movie.genres else ''
            except:
                all_genres = movie.primary_genre if movie.primary_genre else ''
            
            movie_data.append({
                'id': movie.id,
                'tmdb_id': movie.tmdb_id,
                'title': movie.title,
                'summary': movie.summary or '',
                'year': movie.year,
                'genres': movie.primary_genre or '',
                'all_genres': all_genres,
                'director': movie.director or '',
                'cast': movie.cast or '',
                'keywords': movie.keywords or '',
                'vote_average': movie.vote_average,
                'vote_count': movie.vote_count,
                'popularity': movie.popularity,
                'runtime': movie.runtime,
                'poster_url': movie.poster_url or '',
                'explanation': ''
            })
        
        self.movies_df = pd.DataFrame(movie_data)
        print(f"📊 Loaded {len(self.movies_df)} movies for recommendation engine")
    
    def _initialize_content_based(self):
        """Initialize content-based filtering using TF-IDF."""
        if self.movies_df is None or len(self.movies_df) == 0:
            return
        
        # Combine text features for content analysis
        self.movies_df['combined_features'] = (
            self.movies_df['summary'].fillna('') + ' ' +
            self.movies_df['all_genres'].fillna('') + ' ' +
            self.movies_df['director'].fillna('') + ' ' +
            self.movies_df['keywords'].fillna('') + ' ' +
            self.movies_df['cast'].fillna('')
        )
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Fit and transform the combined features
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies_df['combined_features'])
        
        # Compute content similarity matrix
        self.content_similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        print("✅ Content-based filtering initialized")
    
    def _initialize_vector_search(self):
        """Initialize semantic vector search using embeddings."""
        if self.movies_df is None or len(self.movies_df) == 0:
            return
        
        # Generate embeddings for movie summaries
        summaries = self.movies_df['summary'].fillna('').tolist()
        embeddings = []
        
        for i, summary in enumerate(summaries):
            if i % 10 == 0:
                print(f"Generating embeddings: {i+1}/{len(summaries)}")
            
            # Check cache first
            cached_embedding = embedding_cache.get_embedding(summary)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                embedding = self.embedder.embed_text(summary)
                embeddings.append(embedding)
                embedding_cache.set_embedding(summary, embedding)
        
        embeddings = np.array(embeddings)
        
        # Initialize vector database
        self.vectordb = VectorDBHandler(dim=embeddings.shape[1])
        self.vectordb.add_embeddings(embeddings, self.movies_df.index.tolist())
        
        print("✅ Vector search initialized")
    
    def _initialize_collaborative_filtering(self):
        """Initialize collaborative filtering using matrix factorization."""
        # Get user ratings from database
        session = self.db.get_session()
        try:
            ratings = session.query(UserRating).all()
            if len(ratings) < 10:  # Not enough ratings for collaborative filtering
                print("⚠️ Not enough user ratings for collaborative filtering")
                return
            
            # Create user-item matrix
            rating_data = []
            for rating in ratings:
                rating_data.append({
                    'user_id': rating.user_id,
                    'movie_id': rating.movie_id,
                    'rating': rating.rating
                })
            
            if not rating_data:
                return
            
            ratings_df = pd.DataFrame(rating_data)
            user_item_matrix = ratings_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
            
            # Apply SVD for dimensionality reduction
            self.svd_model = TruncatedSVD(n_components=min(50, min(user_item_matrix.shape) - 1))
            self.svd_model.fit(user_item_matrix)
            
            print("✅ Collaborative filtering initialized")
            
        finally:
            session.close()
    
    def _initialize_clustering(self):
        """Initialize movie clustering for diversity."""
        if self.movies_df is None or len(self.movies_df) == 0:
            return
        
        # Create feature matrix for clustering
        feature_columns = ['vote_average', 'vote_count', 'popularity', 'runtime']
        available_columns = [col for col in feature_columns if col in self.movies_df.columns]
        
        if not available_columns:
            return
        
        cluster_features = self.movies_df[available_columns].fillna(0)
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(cluster_features)
        
        # Apply K-means clustering
        n_clusters = min(10, len(self.movies_df) // 5)
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.movie_clusters = kmeans.fit_predict(normalized_features)
            self.movies_df['cluster'] = self.movie_clusters
            
            print(f"✅ Movie clustering initialized ({n_clusters} clusters)")
    
    def get_content_recommendations(self, movie_id: int, top_k: int = 10) -> List[Dict]:
        """Get recommendations based on content similarity."""
        if self.content_similarity_matrix is None:
            return []
        
        try:
            # Find movie index
            movie_idx = self.movies_df[self.movies_df['id'] == movie_id].index[0]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.content_similarity_matrix[movie_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top similar movies (excluding the movie itself)
            movie_indices = [i[0] for i in sim_scores[1:top_k+1]]
            
            recommendations = []
            for idx in movie_indices:
                movie_row = self.movies_df.iloc[idx]
                recommendations.append({
                    'id': movie_row['id'],
                    'title': movie_row['title'],
                    'similarity_score': sim_scores[idx][1],
                    'method': 'content_based'
                })
            
            return recommendations
            
        except (IndexError, KeyError):
            return []
    
    def get_semantic_recommendations(self, user_query: str, top_k: int = 10) -> List[Dict]:
        """Get recommendations using semantic search."""
        if self.vectordb is None:
            return []
        
        start_time = time.time()
        
        # Check cache first
        cached_results = recommendation_cache.get_recommendations(user_query)
        if cached_results:
            performance_monitor.record_query_time(user_query, time.time() - start_time)
            return cached_results
        
        try:
            # Get query embedding
            query_embedding = self.embedder.embed_text(user_query)
            
            # Search similar movies
            similar_indices = self.vectordb.search(query_embedding, top_k * 2)  # Get more for filtering
            
            recommendations = []
            for idx in similar_indices[:top_k]:
                if idx < len(self.movies_df):
                    movie_row = self.movies_df.iloc[idx]
                    recommendations.append({
                        'id': movie_row['id'],
                        'title': movie_row['title'],
                        'summary': movie_row['summary'],
                        'genres': movie_row['all_genres'],
                        'vote_average': movie_row['vote_average'],
                        'year': movie_row['year'],
                        'poster_url': movie_row['poster_url'],
                        'method': 'semantic_search'
                    })
            
            # Cache results
            recommendation_cache.set_recommendations(user_query, recommendations)
            
            execution_time = time.time() - start_time
            performance_monitor.record_query_time(user_query, execution_time)
            
            return recommendations
            
        except Exception as e:
            print(f"Error in semantic recommendations: {e}")
            return []
    
    def get_collaborative_recommendations(self, user_id: int, top_k: int = 10) -> List[Dict]:
        """Get recommendations using collaborative filtering."""
        if self.svd_model is None:
            return []
        
        # This is a simplified version - in practice, you'd use the trained model
        # to predict ratings for unwatched movies
        session = self.db.get_session()
        try:
            # Get user's ratings
            user_ratings = session.query(UserRating).filter(UserRating.user_id == user_id).all()
            
            if not user_ratings:
                return []
            
            # For now, return movies similar to highly rated ones
            high_rated_movies = [r.movie_id for r in user_ratings if r.rating >= 8.0]
            
            if not high_rated_movies:
                return []
            
            # Get content-based recommendations for highly rated movies
            all_recommendations = []
            for movie_id in high_rated_movies[:3]:  # Limit to top 3
                content_recs = self.get_content_recommendations(movie_id, top_k // 3)
                all_recommendations.extend(content_recs)
            
            # Remove duplicates and return top K
            seen_ids = set()
            unique_recommendations = []
            for rec in all_recommendations:
                if rec['id'] not in seen_ids:
                    seen_ids.add(rec['id'])
                    rec['method'] = 'collaborative'
                    unique_recommendations.append(rec)
            
            return unique_recommendations[:top_k]
            
        finally:
            session.close()
    
    def get_hybrid_recommendations(self, user_query: str, user_id: Optional[int] = None, top_k: int = 5) -> List[Dict]:
        """Get hybrid recommendations combining multiple methods."""
        start_time = time.time()
        
        # Get semantic recommendations
        semantic_recs = self.get_semantic_recommendations(user_query, top_k)
        
        # Get collaborative recommendations if user is provided
        collaborative_recs = []
        if user_id:
            collaborative_recs = self.get_collaborative_recommendations(user_id, top_k)
        
        # Combine and diversify recommendations
        all_recommendations = semantic_recs + collaborative_recs
        
        # Remove duplicates
        seen_ids = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec['id'] not in seen_ids:
                seen_ids.add(rec['id'])
                unique_recommendations.append(rec)
        
        # Add diversity using clustering if available
        if hasattr(self, 'movie_clusters') and self.movie_clusters is not None:
            unique_recommendations = self._diversify_recommendations(unique_recommendations)
        
        # Generate AI explanations
        final_recommendations = self._generate_explanations(user_query, unique_recommendations[:top_k])
        
        execution_time = time.time() - start_time
        performance_monitor.record_query_time(user_query, execution_time)
        
        return final_recommendations
    
    def _diversify_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Diversify recommendations using clustering."""
        if not recommendations:
            return recommendations
        
        # Group recommendations by cluster
        cluster_groups = {}
        for rec in recommendations:
            try:
                movie_row = self.movies_df[self.movies_df['id'] == rec['id']].iloc[0]
                cluster = movie_row.get('cluster', 0)
                if cluster not in cluster_groups:
                    cluster_groups[cluster] = []
                cluster_groups[cluster].append(rec)
            except (IndexError, KeyError):
                continue
        
        # Select recommendations from different clusters
        diversified = []
        max_per_cluster = max(1, len(recommendations) // len(cluster_groups)) if cluster_groups else 1
        
        for cluster, cluster_recs in cluster_groups.items():
            diversified.extend(cluster_recs[:max_per_cluster])
        
        # Fill remaining slots if needed
        remaining_slots = len(recommendations) - len(diversified)
        if remaining_slots > 0:
            for rec in recommendations:
                if rec not in diversified and remaining_slots > 0:
                    diversified.append(rec)
                    remaining_slots -= 1
        
        return diversified
    
    def _generate_explanations(self, user_query: str, recommendations: List[Dict]) -> List[Dict]:
        """Generate AI explanations for recommendations."""
        if not recommendations:
            return recommendations
        
        try:
            # Prepare movie data for LLM
            movie_infos = []
            for rec in recommendations:
                movie_info = {
                    'title': rec.get('title', ''),
                    'summary': rec.get('summary', ''),
                    'genres': rec.get('genres', ''),
                    'year': rec.get('year', ''),
                    'vote_average': rec.get('vote_average', 0),
                    'method': rec.get('method', 'hybrid')
                }
                movie_infos.append(movie_info)
            
            # Get LLM explanations
            llm_recommendations = self.llm.get_recommendation(user_query, movie_infos)
            
            # Merge explanations with recommendations
            for i, rec in enumerate(recommendations):
                if i < len(llm_recommendations):
                    llm_rec = llm_recommendations[i]
                    rec['explanation'] = llm_rec.get('explanation', 'Great movie recommendation for you!')
                else:
                    rec['explanation'] = 'Recommended based on your preferences.'
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating explanations: {e}")
            # Return recommendations without explanations
            for rec in recommendations:
                rec['explanation'] = 'Recommended based on advanced algorithms.'
            return recommendations
    
    def recommend(self, user_query: str, user_id: Optional[int] = None, top_k: int = 5) -> List[Dict]:
        """Main recommendation method."""
        try:
            return self.get_hybrid_recommendations(user_query, user_id, top_k)
        except Exception as e:
            print(f"Error in recommendation: {e}")
            # Fallback to simple semantic search
            return self.get_semantic_recommendations(user_query, top_k)
    
    def get_stats(self) -> Dict:
        """Get system statistics."""
        stats = {
            'total_movies': len(self.movies_df) if self.movies_df is not None else 0,
            'algorithms_available': [],
            'cache_stats': embedding_cache.cache.get_stats(),
            'performance_stats': performance_monitor.get_performance_stats()
        }
        
        if self.content_similarity_matrix is not None:
            stats['algorithms_available'].append('content_based')
        if self.vectordb is not None:
            stats['algorithms_available'].append('semantic_search')
        if self.svd_model is not None:
            stats['algorithms_available'].append('collaborative_filtering')
        if hasattr(self, 'movie_clusters') and self.movie_clusters is not None:
            stats['algorithms_available'].append('clustering')
        
        return stats 