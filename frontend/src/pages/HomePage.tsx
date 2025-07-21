import React, { useState } from 'react';
import {
  Container,
  Typography,
  Box,
  TextField,
  Button,
  Card,
  CardContent,
  CardMedia,
  Chip,
  Rating,
  CircularProgress,
  Alert,
  Slider,
} from '@mui/material';
import {
  Search as SearchIcon,
  Star as StarIcon,
  AccessTime as TimeIcon,
  Person as PersonIcon,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import apiService from '../services/api';
import type { Movie, RecommendationResponse } from '../services/api';

interface HomePageProps {
  isAuthenticated: boolean;
}

const HomePage: React.FC<HomePageProps> = ({ isAuthenticated }) => {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [recommendations, setRecommendations] = useState<Movie[]>([]);
  const [executionTime, setExecutionTime] = useState<number>(0);
  const [error, setError] = useState<string>('');

  const handleSearch = async () => {
    if (!query.trim()) {
      setError('Please enter a search query');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      const response: RecommendationResponse = await apiService.getRecommendations({
        query: query.trim(),
        top_k: topK,
      });
      
      setRecommendations(response.recommendations);
      setExecutionTime(response.execution_time);
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to get recommendations');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Hero Section */}
      <Box textAlign="center" mb={6}>
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <Typography
            variant="h1"
            component="h1"
            gutterBottom
            sx={{
              fontSize: { xs: '2.5rem', md: '4rem' },
              fontWeight: 700,
              background: 'linear-gradient(45deg, #4F8BF9 30%, #FF6B6B 90%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            🎭 CineSphere
          </Typography>
          <Typography
            variant="h5"
            color="text.secondary"
            sx={{ mb: 4, maxWidth: '800px', mx: 'auto' }}
          >
            Your personalized movie discovery platform powered by advanced recommendation algorithms
          </Typography>
        </motion.div>

        {/* Search Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <Card
            sx={{
              maxWidth: 800,
              mx: 'auto',
              p: 3,
              background: 'rgba(26, 31, 46, 0.8)',
              backdropFilter: 'blur(20px)',
            }}
          >
            <CardContent>
              <Typography variant="h6" gutterBottom>
                What kind of movie are you in the mood for?
              </Typography>
              
              <Box sx={{ mb: 3 }}>
                <TextField
                  fullWidth
                  variant="outlined"
                  placeholder="e.g., 'A mind-bending sci-fi thriller like Inception' or 'Romantic comedy for date night'"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyPress={handleKeyPress}
                  sx={{ mb: 2 }}
                />
                
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                  <Typography variant="body2" sx={{ minWidth: 'max-content' }}>
                    Number of recommendations:
                  </Typography>
                  <Slider
                    value={topK}
                    onChange={(_, newValue) => setTopK(newValue as number)}
                    min={1}
                    max={10}
                    marks
                    valueLabelDisplay="auto"
                    sx={{ flexGrow: 1 }}
                  />
                </Box>
              </Box>

              <Button
                variant="contained"
                size="large"
                startIcon={loading ? <CircularProgress size={20} /> : <SearchIcon />}
                onClick={handleSearch}
                disabled={loading}
                fullWidth
                sx={{ py: 1.5 }}
              >
                {loading ? 'Finding Movies...' : 'Get Recommendations'}
              </Button>

              {error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  {error}
                </Alert>
              )}

              {executionTime > 0 && (
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  Found {recommendations.length} recommendations in {executionTime.toFixed(2)} seconds
                </Typography>
              )}
            </CardContent>
          </Card>
        </motion.div>
      </Box>

      {/* Recommendations Grid */}
      {recommendations.length > 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6 }}
        >
          <Typography variant="h4" gutterBottom sx={{ mb: 4 }}>
            Recommended Movies
          </Typography>
          
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: {
                xs: '1fr',
                sm: 'repeat(2, 1fr)',
                md: 'repeat(3, 1fr)',
              },
              gap: 3,
            }}
          >
            {recommendations.map((movie, index) => (
              <motion.div
                key={movie.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                whileHover={{ scale: 1.03 }}
              >
                <Card
                  sx={{
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    position: 'relative',
                    overflow: 'hidden',
                  }}
                >
                    {movie.poster_url && (
                      <CardMedia
                        component="img"
                        height="300"
                        image={movie.poster_url}
                        alt={movie.title}
                        sx={{ objectFit: 'cover' }}
                      />
                    )}
                    
                    <CardContent sx={{ flexGrow: 1 }}>
                      <Typography variant="h6" gutterBottom>
                        {movie.title}
                      </Typography>
                      
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                        <Rating
                          value={movie.vote_average / 2}
                          readOnly
                          size="small"
                          icon={<StarIcon fontSize="inherit" />}
                        />
                        <Typography variant="body2" color="text.secondary">
                          {movie.vote_average?.toFixed(1)}/10
                        </Typography>
                      </Box>

                      <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
                        <Chip
                          label={movie.year}
                          size="small"
                          variant="outlined"
                        />
                        {movie.runtime > 0 && (
                          <Chip
                            icon={<TimeIcon />}
                            label={`${movie.runtime} min`}
                            size="small"
                            variant="outlined"
                          />
                        )}
                      </Box>

                      {movie.genres && movie.genres.length > 0 && (
                        <Box sx={{ mb: 2 }}>
                          {movie.genres.slice(0, 3).map((genre, idx) => (
                            <Chip
                              key={idx}
                              label={genre}
                              size="small"
                              sx={{ mr: 0.5, mb: 0.5 }}
                            />
                          ))}
                        </Box>
                      )}

                      {movie.director && (
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                          <PersonIcon fontSize="small" />
                          <Typography variant="body2" color="text.secondary">
                            {movie.director}
                          </Typography>
                        </Box>
                      )}

                      {movie.explanation && (
                        <Box
                          sx={{
                            mt: 2,
                            p: 2,
                            bgcolor: 'rgba(79, 139, 249, 0.1)',
                            borderRadius: 2,
                            borderLeft: '4px solid',
                            borderColor: 'primary.main',
                          }}
                        >
                          <Typography variant="body2" fontStyle="italic">
                            {movie.explanation}
                          </Typography>
                        </Box>
                      )}
                    </CardContent>
                  </Card>
                </motion.div>
            ))}
          </Box>
        </motion.div>
      )}

      {/* Welcome Message for Unauthenticated Users */}
      {!isAuthenticated && recommendations.length === 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <Card sx={{ maxWidth: 600, mx: 'auto', mt: 4 }}>
            <CardContent sx={{ textAlign: 'center', py: 4 }}>
              <Typography variant="h5" gutterBottom>
                Welcome to CineSphere! 🎭
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                Get started by describing what kind of movie you'd like to watch.
                Sign in to access personalized recommendations and save your favorite movies!
              </Typography>
              <Button
                variant="outlined"
                size="large"
                href="/auth"
              >
                Sign In for Better Recommendations
              </Button>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </Container>
  );
};

export default HomePage; 