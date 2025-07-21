import React from 'react';
import { Container, Typography, Box } from '@mui/material';

interface AnalyticsProps {
  isAuthenticated: boolean;
}

const Analytics: React.FC<AnalyticsProps> = ({ isAuthenticated }) => {
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box textAlign="center">
        <Typography variant="h4" gutterBottom>
          Analytics Dashboard
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Analytics functionality will be implemented here.
        </Typography>
      </Box>
    </Container>
  );
};

export default Analytics; 