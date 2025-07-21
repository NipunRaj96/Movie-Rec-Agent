from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
import hashlib
import jwt
import time
from datetime import datetime, timedelta
import logging
import os
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import our backend modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.SimpleRecommender import SimpleRecommender
from backend.Database import DatabaseManager, User, UserRating
from backend.Cache import cache_manager, performance_monitor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="🎭 CineSphere API",
    description="Professional Movie Discovery Platform with Advanced Recommendation Algorithms",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security
security = HTTPBearer(auto_error=False)
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"

# Initialize components
db = DatabaseManager()
# Initialize recommender system
recommender = None
try:
    recommender = SimpleRecommender("sqlite:///../data/movieapp.db")
    logger.info("✅ Simple recommender system initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize recommender: {e}")
    recommender = None

# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class MovieRating(BaseModel):
    movie_id: int
    rating: float
    review: Optional[str] = None

class RecommendationRequest(BaseModel):
    query: str
    user_id: Optional[int] = None
    top_k: Optional[int] = 5

class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    query: str
    execution_time: float
    method: str
    total_results: int

# Utility functions
def hash_password(password: str) -> str:
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash."""
    return hash_password(password) == hashed

def create_jwt_token(user_id: int, username: str) -> str:
    """Create JWT token for user."""
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str) -> Optional[Dict]:
    """Verify JWT token and return payload."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Dict]:
    """Get current user from JWT token."""
    if not credentials:
        return None
    
    payload = verify_jwt_token(credentials.credentials)
    if not payload:
        return None
    
    return payload

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "🎭 CineSphere Movie Discovery Platform",
        "version": "2.0.0",
        "features": [
            "Semantic search with AI embeddings",
            "Content-based filtering",
            "Collaborative filtering",
            "Hybrid recommendations",
            "User management and ratings",
            "Performance monitoring",
            "Intelligent caching"
        ],
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "recommendations": "/api/v1/recommend",
            "auth": "/api/v1/auth/"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "database": "healthy",
            "recommender": "healthy" if recommender else "unhealthy",
            "cache": "healthy"
        }
    }
    
    # Check database
    try:
        movies = db.get_movies(limit=1)
        health_status["components"]["database"] = "healthy"
    except Exception as e:
        health_status["components"]["database"] = "unhealthy"
        health_status["status"] = "degraded"
    
    # Check cache
    try:
        cache_stats = cache_manager.get_stats()
        health_status["components"]["cache"] = "healthy"
        health_status["cache_stats"] = cache_stats
    except Exception as e:
        health_status["components"]["cache"] = "unhealthy"
        health_status["status"] = "degraded"
    
    return health_status

# Authentication endpoints
@app.post("/api/v1/auth/register")
@limiter.limit("5/minute")
async def register(request: Request, user_data: UserCreate):
    """Register a new user."""
    try:
        # Check if user already exists
        existing_user = db.get_user_by_username(user_data.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        # Create new user
        password_hash = hash_password(user_data.password)
        user = db.create_user(
            username=user_data.username,
            email=user_data.email,
            password_hash=password_hash,
            first_name=user_data.first_name,
            last_name=user_data.last_name
        )
        
        # Create JWT token
        token = create_jwt_token(user.id, user.username)
        
        return {
            "message": "User registered successfully",
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email
            },
            "token": token
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@app.post("/api/v1/auth/login")
@limiter.limit("10/minute")
async def login(request: Request, login_data: UserLogin):
    """Login user."""
    try:
        user = db.get_user_by_username(login_data.username)
        if not user or not verify_password(login_data.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Update last login
        session = db.get_session()
        try:
            user.last_login = datetime.utcnow()
            session.commit()
        finally:
            session.close()
        
        # Create JWT token
        token = create_jwt_token(user.id, user.username)
        
        return {
            "message": "Login successful",
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email
            },
            "token": token
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

# Movie and recommendation endpoints
@app.get("/api/v1/movies")
async def get_movies(
    limit: Optional[int] = 20,
    genre: Optional[str] = None,
    search: Optional[str] = None
):
    """Get movies with optional filtering."""
    try:
        if search:
            movies = db.search_movies(search, limit)
        else:
            movies = db.get_movies(limit, genre)
        
        movie_list = []
        for movie in movies:
            movie_list.append({
                "id": movie.id,
                "tmdb_id": movie.tmdb_id,
                "title": movie.title,
                "summary": movie.summary,
                "year": movie.year,
                "genres": [g.name for g in movie.genres],
                "director": movie.director,
                "cast": movie.cast,
                "vote_average": movie.vote_average,
                "vote_count": movie.vote_count,
                "poster_url": movie.poster_url,
                "runtime": movie.runtime
            })
        
        return {
            "movies": movie_list,
            "total": len(movie_list),
            "filters": {
                "limit": limit,
                "genre": genre,
                "search": search
            }
        }
        
    except Exception as e:
        logger.error(f"Get movies error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch movies"
        )

@app.get("/api/v1/genres")
async def get_genres():
    """Get all available genres."""
    try:
        genres = db.get_genres()
        return {
            "genres": [{"id": g.id, "name": g.name} for g in genres],
            "total": len(genres)
        }
    except Exception as e:
        logger.error(f"Get genres error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch genres"
        )

@app.post("/api/v1/recommend")
@limiter.limit("30/minute")
async def get_recommendations(
    request: Request,
    req: RecommendationRequest,
    current_user: Optional[Dict] = Depends(get_current_user)
) -> RecommendationResponse:
    """Get movie recommendations."""
    if not recommender:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation service is not available"
        )
    
    try:
        start_time = time.time()
        
        # Get user ID from token if available
        user_id = current_user.get("user_id") if current_user else req.user_id
        
        # Get recommendations
        recommendations = recommender.recommend(
            user_query=req.query,
            user_id=user_id,
            top_k=req.top_k
        )
        
        execution_time = time.time() - start_time
        
        return RecommendationResponse(
            recommendations=recommendations,
            query=req.query,
            execution_time=round(execution_time, 3),
            method="hybrid_advanced",
            total_results=len(recommendations)
        )
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations"
        )

@app.post("/api/v1/ratings")
async def add_rating(
    rating_data: MovieRating,
    current_user: Dict = Depends(get_current_user)
):
    """Add or update a movie rating."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    try:
        user_rating = db.add_user_rating(
            user_id=current_user["user_id"],
            movie_id=rating_data.movie_id,
            rating=rating_data.rating,
            review=rating_data.review
        )
        
        return {
            "message": "Rating added successfully",
            "rating": {
                "id": user_rating.id,
                "movie_id": user_rating.movie_id,
                "rating": user_rating.rating,
                "review": user_rating.review,
                "created_at": user_rating.created_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Add rating error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add rating"
        )

@app.get("/api/v1/user/stats")
async def get_user_stats(current_user: Dict = Depends(get_current_user)):
    """Get user statistics."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    try:
        stats = db.get_user_stats(current_user["user_id"])
        return {
            "user_id": current_user["user_id"],
            "username": current_user["username"],
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"User stats error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch user statistics"
        )

@app.get("/api/v1/system/stats")
async def get_system_stats():
    """Get system performance statistics."""
    try:
        if recommender:
            recommender_stats = recommender.get_stats()
        else:
            recommender_stats = {"error": "Recommender not available"}
        
        return {
            "system": {
                "uptime": "N/A",
                "version": "2.0.0",
                "status": "healthy"
            },
            "recommender": recommender_stats,
            "cache": cache_manager.get_stats(),
            "performance": performance_monitor.get_performance_stats()
        }
        
    except Exception as e:
        logger.error(f"System stats error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch system statistics"
        )

@app.delete("/api/v1/cache/clear")
async def clear_cache(current_user: Dict = Depends(get_current_user)):
    """Clear system cache (admin only for now)."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    try:
        cache_manager.clear()
        return {"message": "Cache cleared successfully"}
        
    except Exception as e:
        logger.error(f"Clear cache error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache"
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 