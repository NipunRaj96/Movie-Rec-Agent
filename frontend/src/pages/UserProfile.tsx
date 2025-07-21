import React from 'react';
import { Container, Typography, Box } from '@mui/material';

const UserProfile: React.FC = () => {
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box textAlign="center">
        <Typography variant="h4" gutterBottom>
          User Profile
        </Typography>
        <Typography variant="body1" color="text.secondary">
          User profile functionality will be implemented here.
        </Typography>
      </Box>
    </Container>
  );
};

export default UserProfile; 