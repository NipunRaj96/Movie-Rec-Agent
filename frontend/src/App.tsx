import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box } from '@mui/material';
import { motion } from 'framer-motion';

// Components
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import AuthPage from './pages/AuthPage';

// Services
import apiService from './services/api';

// Create dark theme
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#4F8BF9',
      light: '#7BA7FB',
      dark: '#2E5BF6',
    },
    secondary: {
      main: '#FF6B6B',
      light: '#FF8E8E',
      dark: '#FF4848',
    },
    background: {
      default: '#0A0E1A',
      paper: '#1A1F2E',
    },
    text: {
      primary: '#FFFFFF',
      secondary: '#B0BEC5',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 700,
      fontSize: '3rem',
      background: 'linear-gradient(45deg, #4F8BF9 30%, #FF6B6B 90%)',
      backgroundClip: 'text',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
    },
    h2: {
      fontWeight: 600,
      fontSize: '2.25rem',
    },
    h3: {
      fontWeight: 600,
      fontSize: '1.75rem',
    },
    h4: {
      fontWeight: 600,
      fontSize: '1.5rem',
    },
    h5: {
      fontWeight: 500,
      fontSize: '1.25rem',
    },
    h6: {
      fontWeight: 500,
      fontSize: '1rem',
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'linear-gradient(135deg, #1A1F2E 0%, #2A3441 100%)',
          border: '1px solid #4F8BF9',
          borderRadius: 16,
          boxShadow: '0 8px 32px rgba(79, 139, 249, 0.2)',
          transition: 'all 0.3s ease-in-out',
          '&:hover': {
            transform: 'translateY(-4px)',
            boxShadow: '0 12px 48px rgba(79, 139, 249, 0.3)',
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          textTransform: 'none',
          fontWeight: 600,
          padding: '12px 24px',
          transition: 'all 0.3s ease-in-out',
        },
        contained: {
          background: 'linear-gradient(45deg, #4F8BF9 30%, #2E5BF6 90%)',
          '&:hover': {
            background: 'linear-gradient(45deg, #2E5BF6 30%, #4F8BF9 90%)',
            transform: 'translateY(-2px)',
            boxShadow: '0 8px 24px rgba(79, 139, 249, 0.4)',
          },
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 12,
            '& fieldset': {
              borderColor: '#4F8BF9',
            },
            '&:hover fieldset': {
              borderColor: '#7BA7FB',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#4F8BF9',
            },
          },
        },
      },
    },
  },
});

// Animation variants
const pageVariants = {
  initial: {
    opacity: 0,
    y: 20,
  },
  in: {
    opacity: 1,
    y: 0,
  },
  out: {
    opacity: 0,
    y: -20,
  },
};

const pageTransition = {
  type: 'tween' as const,
  ease: 'anticipate' as const,
  duration: 0.5,
};

function App() {
  const [isAuthenticated, setIsAuthenticated] = React.useState(false);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    // Check authentication status on app load
    setIsAuthenticated(apiService.isAuthenticated());
    setLoading(false);
  }, []);

  const handleAuthChange = (authStatus: boolean) => {
    setIsAuthenticated(authStatus);
  };

  if (loading) {
    return (
      <ThemeProvider theme={darkTheme}>
        <CssBaseline />
        <Box
          display="flex"
          justifyContent="center"
          alignItems="center"
          minHeight="100vh"
          bgcolor="background.default"
        >
          <motion.div
            animate={{
              rotate: 360,
            }}
            transition={{
              duration: 1,
              repeat: Infinity,
              ease: 'linear',
            }}
          >
            <Box
              sx={{
                width: 60,
                height: 60,
                borderRadius: '50%',
                border: '4px solid',
                borderColor: 'primary.main',
                borderTopColor: 'transparent',
              }}
            />
          </motion.div>
        </Box>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Router>
        <Box
          sx={{
            minHeight: '100vh',
            background: 'linear-gradient(135deg, #0A0E1A 0%, #1A1F2E 50%, #2A3441 100%)',
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          {/* Background decoration */}
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: `
                radial-gradient(circle at 20% 80%, rgba(79, 139, 249, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255, 107, 107, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 40% 40%, rgba(79, 139, 249, 0.05) 0%, transparent 50%)
              `,
              zIndex: 0,
            }}
          />

          <Box sx={{ position: 'relative', zIndex: 1 }}>
            <Navbar
              isAuthenticated={isAuthenticated}
              onAuthChange={handleAuthChange}
            />

            <motion.div
              key={window.location.pathname}
              initial="initial"
              animate="in"
              exit="out"
              variants={pageVariants}
              transition={pageTransition}
            >
              <Routes>
                <Route
                  path="/"
                  element={<HomePage isAuthenticated={isAuthenticated} />}
                />
                <Route
                  path="/auth"
                  element={<AuthPage onAuthChange={handleAuthChange} />}
                />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </motion.div>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;
