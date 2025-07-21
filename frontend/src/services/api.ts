import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

class ApiService {
  private api: any;
  private token: string | null = null;

  constructor() {
    this.api = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add request interceptor to include auth token
    this.api.interceptors.request.use(
      (config: any) => {
        if (this.token) {
          config.headers.Authorization = `Bearer ${this.token}`;
        }
        return config;
      },
      (error: any) => Promise.reject(error)
    );

    // Add response interceptor for error handling
    this.api.interceptors.response.use(
      (response: any) => response,
      (error: any) => {
        if (error.response?.status === 401) {
          this.removeToken();
          window.location.href = '/auth';
        }
        return Promise.reject(error);
      }
    );

    // Load token from localStorage on initialization
    this.loadToken();
  }

  setToken(token: string): void {
    this.token = token;
    localStorage.setItem('auth_token', token);
  }

  removeToken(): void {
    this.token = null;
    localStorage.removeItem('auth_token');
  }

  private loadToken(): void {
    const storedToken = localStorage.getItem('auth_token');
    if (storedToken) {
      this.token = storedToken;
    }
  }

  isAuthenticated(): boolean {
    return !!this.token;
  }

  // Authentication endpoints
  async register(userData: RegisterData): Promise<AuthResponse> {
    const response = await this.api.post('/api/v1/auth/register', userData);
    if (response.data.token) {
      this.setToken(response.data.token);
    }
    return response.data;
  }

  async login(loginData: LoginData): Promise<AuthResponse> {
    const response = await this.api.post('/api/v1/auth/login', loginData);
    if (response.data.token) {
      this.setToken(response.data.token);
    }
    return response.data;
  }

  logout(): void {
    this.removeToken();
  }

  // Movie endpoints
  async getMovies(params?: GetMoviesParams): Promise<MoviesResponse> {
    const response = await this.api.get('/api/v1/movies', { params });
    return response.data;
  }

  async getGenres(): Promise<GenresResponse> {
    const response = await this.api.get('/api/v1/genres');
    return response.data;
  }

  // Recommendation endpoint
  async getRecommendations(recommendationData: RecommendationRequest): Promise<RecommendationResponse> {
    const response = await this.api.post('/api/v1/recommend', recommendationData);
    return response.data;
  }

  // Rating endpoints
  async addRating(ratingData: RatingData): Promise<RatingResponse> {
    const response = await this.api.post('/api/v1/ratings', ratingData);
    return response.data;
  }

  // User stats
  async getUserStats(): Promise<UserStatsResponse> {
    const response = await this.api.get('/api/v1/user/stats');
    return response.data;
  }

  // System stats
  async getSystemStats(): Promise<SystemStatsResponse> {
    const response = await this.api.get('/api/v1/system/stats');
    return response.data;
  }

  // Health check
  async healthCheck(): Promise<HealthResponse> {
    const response = await this.api.get('/health');
    return response.data;
  }

  // Clear cache
  async clearCache(): Promise<{ message: string }> {
    const response = await this.api.delete('/api/v1/cache/clear');
    return response.data;
  }
}

// Type definitions
export interface RegisterData {
  username: string;
  email: string;
  password: string;
  first_name?: string;
  last_name?: string;
}

export interface LoginData {
  username: string;
  password: string;
}

export interface AuthResponse {
  message: string;
  user: {
    id: number;
    username: string;
    email: string;
  };
  token: string;
}

export interface Movie {
  id: number;
  tmdb_id: number;
  title: string;
  summary: string;
  year: string;
  genres: string[];
  director: string;
  cast: string;
  vote_average: number;
  vote_count: number;
  poster_url: string;
  runtime: number;
  explanation?: string;
}

export interface GetMoviesParams {
  limit?: number;
  genre?: string;
  search?: string;
}

export interface MoviesResponse {
  movies: Movie[];
  total: number;
  filters: GetMoviesParams;
}

export interface Genre {
  id: number;
  name: string;
}

export interface GenresResponse {
  genres: Genre[];
  total: number;
}

export interface RecommendationRequest {
  query: string;
  user_id?: number;
  top_k?: number;
}

export interface RecommendationResponse {
  recommendations: Movie[];
  query: string;
  execution_time: number;
  method: string;
  total_results: number;
}

export interface RatingData {
  movie_id: number;
  rating: number;
  review?: string;
}

export interface RatingResponse {
  message: string;
  rating: {
    id: number;
    movie_id: number;
    rating: number;
    review?: string;
    created_at: string;
  };
}

export interface UserStatsResponse {
  user_id: number;
  username: string;
  stats: {
    total_ratings: number;
    average_rating: number;
    favorite_movies_count: number;
    watchlist_count: number;
    recommendations_count: number;
  };
}

export interface SystemStatsResponse {
  system: {
    uptime: string;
    version: string;
    status: string;
  };
  recommender: any;
  cache: any;
  performance: any;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  components: {
    database: string;
    recommender: string;
    cache: string;
  };
  cache_stats?: any;
}

// Create and export a singleton instance
const apiService = new ApiService();
export default apiService; 