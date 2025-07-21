import pickle
import hashlib
import time
import os
import json
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import threading

class CacheManager:
    """A simple but effective file-based caching system."""
    
    def __init__(self, cache_dir: str = "data/cache", ttl: int = 3600):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            ttl: Time to live in seconds (default 1 hour)
        """
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.memory_cache = {}
        self.memory_cache_timestamps = {}
        self.lock = threading.RLock()
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Clean up old cache files on startup
        self._cleanup_old_files()
    
    def _get_cache_key(self, key: str) -> str:
        """Generate a safe filename from cache key."""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get full path to cache file."""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _cleanup_old_files(self):
        """Remove expired cache files."""
        current_time = time.time()
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(self.cache_dir, filename)
                try:
                    file_time = os.path.getmtime(filepath)
                    if current_time - file_time > self.ttl:
                        os.remove(filepath)
                except (OSError, IOError):
                    continue
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Store a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional override)
        
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                cache_key = self._get_cache_key(key)
                cache_path = self._get_cache_path(cache_key)
                
                # Store in memory cache
                self.memory_cache[key] = value
                self.memory_cache_timestamps[key] = time.time()
                
                # Store in file cache
                cache_data = {
                    'value': value,
                    'timestamp': time.time(),
                    'ttl': ttl or self.ttl
                }
                
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
                
                return True
            except Exception as e:
                print(f"Cache set error for key {key}: {e}")
                return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value if found and not expired, None otherwise
        """
        with self.lock:
            # Try memory cache first
            if key in self.memory_cache:
                if time.time() - self.memory_cache_timestamps[key] < self.ttl:
                    return self.memory_cache[key]
                else:
                    # Expired, remove from memory cache
                    del self.memory_cache[key]
                    del self.memory_cache_timestamps[key]
            
            # Try file cache
            try:
                cache_key = self._get_cache_key(key)
                cache_path = self._get_cache_path(cache_key)
                
                if not os.path.exists(cache_path):
                    return None
                
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Check if expired
                age = time.time() - cache_data['timestamp']
                if age > cache_data['ttl']:
                    os.remove(cache_path)
                    return None
                
                # Add back to memory cache
                self.memory_cache[key] = cache_data['value']
                self.memory_cache_timestamps[key] = cache_data['timestamp']
                
                return cache_data['value']
                
            except Exception as e:
                print(f"Cache get error for key {key}: {e}")
                return None
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                # Remove from memory cache
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    del self.memory_cache_timestamps[key]
                
                # Remove from file cache
                cache_key = self._get_cache_key(key)
                cache_path = self._get_cache_path(cache_key)
                
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                
                return True
            except Exception as e:
                print(f"Cache delete error for key {key}: {e}")
                return False
    
    def clear(self) -> bool:
        """Clear all cache."""
        with self.lock:
            try:
                # Clear memory cache
                self.memory_cache.clear()
                self.memory_cache_timestamps.clear()
                
                # Clear file cache
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.pkl'):
                        os.remove(os.path.join(self.cache_dir, filename))
                
                return True
            except Exception as e:
                print(f"Cache clear error: {e}")
                return False
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self.lock:
            try:
                file_count = len([f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')])
                memory_count = len(self.memory_cache)
                
                cache_size = 0
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.pkl'):
                        cache_size += os.path.getsize(os.path.join(self.cache_dir, filename))
                
                return {
                    'memory_cache_count': memory_count,
                    'file_cache_count': file_count,
                    'total_cache_size_bytes': cache_size,
                    'cache_directory': self.cache_dir,
                    'default_ttl': self.ttl
                }
            except Exception as e:
                print(f"Cache stats error: {e}")
                return {}

class EmbeddingCache:
    """Specialized cache for movie embeddings."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.embedding_prefix = "embedding_"
        self.similarity_prefix = "similarity_"
    
    def get_embedding(self, text: str) -> Optional[Any]:
        """Get cached embedding for text."""
        key = f"{self.embedding_prefix}{hashlib.md5(text.encode()).hexdigest()}"
        return self.cache.get(key)
    
    def set_embedding(self, text: str, embedding: Any, ttl: int = 86400) -> bool:
        """Cache embedding for text (24 hour TTL by default)."""
        key = f"{self.embedding_prefix}{hashlib.md5(text.encode()).hexdigest()}"
        return self.cache.set(key, embedding, ttl)
    
    def get_similarity_results(self, query: str, top_k: int) -> Optional[List]:
        """Get cached similarity search results."""
        key = f"{self.similarity_prefix}{hashlib.md5(f'{query}_{top_k}'.encode()).hexdigest()}"
        return self.cache.get(key)
    
    def set_similarity_results(self, query: str, top_k: int, results: List, ttl: int = 3600) -> bool:
        """Cache similarity search results (1 hour TTL by default)."""
        key = f"{self.similarity_prefix}{hashlib.md5(f'{query}_{top_k}'.encode()).hexdigest()}"
        return self.cache.set(key, results, ttl)

class RecommendationCache:
    """Specialized cache for recommendation results."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.recommendation_prefix = "recommendation_"
    
    def get_recommendations(self, user_query: str, user_id: Optional[int] = None) -> Optional[List]:
        """Get cached recommendations."""
        cache_key = f"{user_query}_{user_id}" if user_id else user_query
        key = f"{self.recommendation_prefix}{hashlib.md5(cache_key.encode()).hexdigest()}"
        return self.cache.get(key)
    
    def set_recommendations(self, user_query: str, recommendations: List, user_id: Optional[int] = None, ttl: int = 1800) -> bool:
        """Cache recommendations (30 minutes TTL by default)."""
        cache_key = f"{user_query}_{user_id}" if user_id else user_query
        key = f"{self.recommendation_prefix}{hashlib.md5(cache_key.encode()).hexdigest()}"
        return self.cache.set(key, recommendations, ttl)

class PerformanceMonitor:
    """Monitor and cache performance metrics."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.metrics_prefix = "metrics_"
    
    def record_query_time(self, query: str, execution_time: float):
        """Record query execution time."""
        timestamp = datetime.now().isoformat()
        metrics_key = f"{self.metrics_prefix}query_times"
        
        # Get existing metrics
        existing_metrics = self.cache.get(metrics_key) or []
        
        # Add new metric
        new_metric = {
            'query': query,
            'execution_time': execution_time,
            'timestamp': timestamp
        }
        existing_metrics.append(new_metric)
        
        # Keep only last 1000 entries
        if len(existing_metrics) > 1000:
            existing_metrics = existing_metrics[-1000:]
        
        # Cache for 24 hours
        self.cache.set(metrics_key, existing_metrics, ttl=86400)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        metrics_key = f"{self.metrics_prefix}query_times"
        metrics = self.cache.get(metrics_key) or []
        
        if not metrics:
            return {'total_queries': 0, 'average_time': 0, 'last_24h_queries': 0}
        
        total_queries = len(metrics)
        average_time = sum(m['execution_time'] for m in metrics) / total_queries if metrics else 0
        
        # Count queries in last 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        last_24h_queries = sum(1 for m in metrics if datetime.fromisoformat(m['timestamp']) > cutoff_time)
        
        return {
            'total_queries': total_queries,
            'average_time': round(average_time, 3),
            'last_24h_queries': last_24h_queries,
            'slowest_query': max(metrics, key=lambda x: x['execution_time']) if metrics else None,
            'fastest_query': min(metrics, key=lambda x: x['execution_time']) if metrics else None
        }

# Global cache instances
cache_manager = CacheManager()
embedding_cache = EmbeddingCache(cache_manager)
recommendation_cache = RecommendationCache(cache_manager)
performance_monitor = PerformanceMonitor(cache_manager) 