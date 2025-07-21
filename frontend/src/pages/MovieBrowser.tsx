import React from 'react';
import { Container, Typography, Box } from '@mui/material';

interface MovieBrowserProps {
  isAuthenticated: boolean;
}

const MovieBrowser: React.FC<MovieBrowserProps> = ({ isAuthenticated }) => {
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box textAlign="center">
        <Typography variant="h4" gutterBottom>
          Movie Browser
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Movie browsing functionality will be implemented here.
        </Typography>
      </Box>
    </Container>
  );
};

export default MovieBrowser; 