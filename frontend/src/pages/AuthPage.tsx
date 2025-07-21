import React from 'react';
import { Container, Typography, Box, Card, CardContent } from '@mui/material';
import { motion } from 'framer-motion';

interface AuthPageProps {
  onAuthChange: (authStatus: boolean) => void;
}

const AuthPage: React.FC<AuthPageProps> = ({ onAuthChange }) => {
  return (
    <Container maxWidth="sm" sx={{ py: 8 }}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <Card>
          <CardContent sx={{ textAlign: 'center', py: 6 }}>
            <Typography variant="h4" gutterBottom color="primary">
              🔐 Authentication
            </Typography>
            <Typography variant="h6" color="text.secondary" sx={{ mb: 3 }}>
              Coming Soon
            </Typography>
            <Typography variant="body1" color="text.secondary">
              User authentication and profile management features are currently under development. 
              For now, enjoy exploring movies with our AI-powered recommendation system!
            </Typography>
          </CardContent>
        </Card>
      </motion.div>
    </Container>
  );
};

export default AuthPage; 