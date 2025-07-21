from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, joinedload
from sqlalchemy.sql import func
from datetime import datetime
import json
import os
import pandas as pd
from typing import List, Dict, Optional

Base = declarative_base()

# Association table for movie genres (many-to-many)
movie_genres = Table('movie_genres', Base.metadata,
    Column('movie_id', Integer, ForeignKey('movies.id')),
    Column('genre_id', Integer, ForeignKey('genres.id'))
)

# Association table for user favorite movies
user_favorites = Table('user_favorites', Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('movie_id', Integer, ForeignKey('movies.id')),
    Column('created_at', DateTime, default=datetime.utcnow)
)

# Association table for user watchlist
user_watchlist = Table('user_watchlist', Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('movie_id', Integer, ForeignKey('movies.id')),
    Column('created_at', DateTime, default=datetime.utcnow)
)

class Genre(Base):
    __tablename__ = 'genres'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    tmdb_id = Column(Integer, unique=True)
    
    # Relationships
    movies = relationship("Movie", secondary=movie_genres, back_populates="genres")

class Collection(Base):
    __tablename__ = 'collections'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    tmdb_id = Column(Integer, unique=True)
    
    # Relationships
    movies = relationship("Movie", back_populates="collection")

class Movie(Base):
    __tablename__ = 'movies'
    
    id = Column(Integer, primary_key=True)
    tmdb_id = Column(Integer, unique=True, nullable=False)
    title = Column(String(500), nullable=False)
    summary = Column(Text)
    release_date = Column(String(20))
    year = Column(String(4))
    primary_genre = Column(String(50))
    director = Column(String(200))
    writers = Column(Text)
    cast = Column(Text)
    runtime = Column(Integer)
    language = Column(String(10))
    spoken_languages = Column(Text)
    production_countries = Column(Text)
    production_companies = Column(Text)
    budget = Column(Integer, default=0)
    revenue = Column(Integer, default=0)
    vote_average = Column(Float, default=0.0)
    vote_count = Column(Integer, default=0)
    popularity = Column(Float, default=0.0)
    adult = Column(Boolean, default=False)
    poster_url = Column(Text)
    backdrop_url = Column(Text)
    imdb_id = Column(String(20))
    tagline = Column(Text)
    status = Column(String(20))
    keywords = Column(Text)
    reviews_sample = Column(Text)  # JSON string
    homepage = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign Keys
    collection_id = Column(Integer, ForeignKey('collections.id'))
    
    # Relationships
    genres = relationship("Genre", secondary=movie_genres, back_populates="movies")
    collection = relationship("Collection", back_populates="movies")
    ratings = relationship("UserRating", back_populates="movie")
    user_favorites = relationship("User", secondary=user_favorites, back_populates="favorite_movies")
    user_watchlist = relationship("User", secondary=user_watchlist, back_populates="watchlist_movies")

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(200), nullable=False)
    first_name = Column(String(50))
    last_name = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    is_active = Column(Boolean, default=True)
    preferences = Column(Text)  # JSON string for user preferences
    
    # Relationships
    ratings = relationship("UserRating", back_populates="user")
    recommendation_history = relationship("RecommendationHistory", back_populates="user")
    favorite_movies = relationship("Movie", secondary=user_favorites, back_populates="user_favorites")
    watchlist_movies = relationship("Movie", secondary=user_watchlist, back_populates="user_watchlist")

class UserRating(Base):
    __tablename__ = 'user_ratings'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    movie_id = Column(Integer, ForeignKey('movies.id'), nullable=False)
    rating = Column(Float, nullable=False)  # 1-10 scale
    review = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="ratings")
    movie = relationship("Movie", back_populates="ratings")

class RecommendationHistory(Base):
    __tablename__ = 'recommendation_history'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    query = Column(Text, nullable=False)
    recommended_movies = Column(Text, nullable=False)  # JSON string of movie IDs
    feedback = Column(String(20))  # 'positive', 'negative', 'neutral'
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="recommendation_history")

class DatabaseManager:
    def __init__(self, database_url: str = "sqlite:///data/movieapp.db"):
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        os.makedirs(os.path.dirname(database_url.replace("sqlite:///", "")), exist_ok=True)
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get a database session."""
        return self.SessionLocal()
    
    def load_movies_from_csv(self, csv_path: str = "data/movies.csv"):
        """Load movies from CSV into the database."""
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        session = self.get_session()
        
        try:
            # Clear existing data
            session.query(Movie).delete()
            session.query(Genre).delete()
            session.query(Collection).delete()
            session.commit()
            
            # Process genres
            all_genres = set()
            for _, row in df.iterrows():
                if pd.notna(row.get('genres')):
                    genres = [g.strip() for g in str(row['genres']).split(',')]
                    all_genres.update(genres)
            
            # Add genres to database
            genre_objects = {}
            for genre_name in all_genres:
                if genre_name:
                    genre = Genre(name=genre_name)
                    session.add(genre)
                    session.flush()
                    genre_objects[genre_name] = genre
            
            # Process collections
            collection_objects = {}
            for _, row in df.iterrows():
                collection_name = row.get('collection', '')
                if pd.notna(collection_name) and collection_name and collection_name not in collection_objects:
                    collection = Collection(name=collection_name)
                    session.add(collection)
                    session.flush()
                    collection_objects[collection_name] = collection
            
            # Add movies
            for _, row in df.iterrows():
                # Handle collection
                collection = None
                collection_name = row.get('collection', '')
                if pd.notna(collection_name) and collection_name:
                    collection = collection_objects.get(collection_name)
                
                movie = Movie(
                    tmdb_id=int(row.get('tmdb_id', 0)),
                    title=str(row.get('title', '')),
                    summary=str(row.get('summary', '')),
                    release_date=str(row.get('release_date', '')),
                    year=str(row.get('year', '')),
                    primary_genre=str(row.get('primary_genre', '')),
                    director=str(row.get('director', '')),
                    writers=str(row.get('writers', '')),
                    cast=str(row.get('cast', '')),
                    runtime=int(row.get('runtime', 0)) if pd.notna(row.get('runtime')) else 0,
                    language=str(row.get('language', '')),
                    spoken_languages=str(row.get('spoken_languages', '')),
                    production_countries=str(row.get('production_countries', '')),
                    production_companies=str(row.get('production_companies', '')),
                    budget=int(row.get('budget', 0)) if pd.notna(row.get('budget')) else 0,
                    revenue=int(row.get('revenue', 0)) if pd.notna(row.get('revenue')) else 0,
                    vote_average=float(row.get('vote_average', 0.0)) if pd.notna(row.get('vote_average')) else 0.0,
                    vote_count=int(row.get('vote_count', 0)) if pd.notna(row.get('vote_count')) else 0,
                    popularity=float(row.get('popularity', 0.0)) if pd.notna(row.get('popularity')) else 0.0,
                    adult=bool(row.get('adult', False)),
                    poster_url=str(row.get('poster_url', '')),
                    backdrop_url=str(row.get('backdrop_url', '')),
                    imdb_id=str(row.get('imdb_id', '')),
                    tagline=str(row.get('tagline', '')),
                    status=str(row.get('status', '')),
                    keywords=str(row.get('keywords', '')),
                    reviews_sample=str(row.get('reviews_sample', '')),
                    homepage=str(row.get('homepage', '')),
                    collection=collection
                )
                
                # Add genre relationships
                if pd.notna(row.get('genres')):
                    genres = [g.strip() for g in str(row['genres']).split(',')]
                    for genre_name in genres:
                        if genre_name and genre_name in genre_objects:
                            movie.genres.append(genre_objects[genre_name])
                
                session.add(movie)
            
            session.commit()
            movie_count = session.query(Movie).count()
            genre_count = session.query(Genre).count()
            print(f"✅ Successfully loaded {movie_count} movies and {genre_count} genres into database")
            
        except Exception as e:
            session.rollback()
            print(f"❌ Error loading movies: {e}")
            raise
        finally:
            session.close()
    
    def get_movies(self, limit: int = None, genre: str = None) -> List[Movie]:
        """Get movies with optional filtering."""
        session = self.get_session()
        try:
            # Eager load genres to avoid session issues
            query = session.query(Movie).options(joinedload(Movie.genres))
            
            if genre:
                query = query.join(Movie.genres).filter(Genre.name == genre)
            
            if limit:
                query = query.limit(limit)
            
            movies = query.all()
            
            # Force loading of all relationships while session is active
            for movie in movies:
                _ = list(movie.genres)  # Force load genres
            
            return movies
        finally:
            session.close()
    
    def get_movie_by_id(self, movie_id: int) -> Optional[Movie]:
        """Get a movie by ID."""
        session = self.get_session()
        try:
            return session.query(Movie).filter(Movie.id == movie_id).first()
        finally:
            session.close()
    
    def search_movies(self, query: str, limit: int = 20) -> List[Movie]:
        """Search movies by title, summary, or other fields."""
        session = self.get_session()
        try:
            return session.query(Movie).filter(
                Movie.title.contains(query) | 
                Movie.summary.contains(query) |
                Movie.keywords.contains(query) |
                Movie.cast.contains(query) |
                Movie.director.contains(query)
            ).limit(limit).all()
        finally:
            session.close()
    
    def get_genres(self) -> List[Genre]:
        """Get all genres."""
        session = self.get_session()
        try:
            return session.query(Genre).all()
        finally:
            session.close()
    
    def create_user(self, username: str, email: str, password_hash: str, **kwargs) -> User:
        """Create a new user."""
        session = self.get_session()
        try:
            user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                **kwargs
            )
            session.add(user)
            session.commit()
            return user
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        session = self.get_session()
        try:
            return session.query(User).filter(User.username == username).first()
        finally:
            session.close()
    
    def add_user_rating(self, user_id: int, movie_id: int, rating: float, review: str = None) -> UserRating:
        """Add or update user rating for a movie."""
        session = self.get_session()
        try:
            # Check if rating already exists
            existing_rating = session.query(UserRating).filter(
                UserRating.user_id == user_id,
                UserRating.movie_id == movie_id
            ).first()
            
            if existing_rating:
                existing_rating.rating = rating
                existing_rating.review = review
                existing_rating.updated_at = datetime.utcnow()
                user_rating = existing_rating
            else:
                user_rating = UserRating(
                    user_id=user_id,
                    movie_id=movie_id,
                    rating=rating,
                    review=review
                )
                session.add(user_rating)
            
            session.commit()
            return user_rating
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
    
    def add_recommendation_history(self, user_id: int, query: str, recommended_movies: List[int], feedback: str = None):
        """Add recommendation to history."""
        session = self.get_session()
        try:
            history = RecommendationHistory(
                user_id=user_id,
                query=query,
                recommended_movies=json.dumps(recommended_movies),
                feedback=feedback
            )
            session.add(history)
            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_user_stats(self, user_id: int) -> Dict:
        """Get user statistics."""
        session = self.get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                return {}
            
            stats = {
                'total_ratings': session.query(UserRating).filter(UserRating.user_id == user_id).count(),
                'average_rating': session.query(func.avg(UserRating.rating)).filter(UserRating.user_id == user_id).scalar() or 0,
                'favorite_movies_count': len(user.favorite_movies),
                'watchlist_count': len(user.watchlist_movies),
                'recommendations_count': session.query(RecommendationHistory).filter(RecommendationHistory.user_id == user_id).count()
            }
            return stats
        finally:
            session.close() 