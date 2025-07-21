import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  IconButton,
  Menu,
  MenuItem,
  Avatar,
} from '@mui/material';
import {
  Movie as MovieIcon,
  Analytics as AnalyticsIcon,
  Person as PersonIcon,
  Home as HomeIcon,
  Logout as LogoutIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import apiService from '../services/api';

interface NavbarProps {
  isAuthenticated: boolean;
  onAuthChange: (authStatus: boolean) => void;
}

const Navbar: React.FC<NavbarProps> = ({ isAuthenticated, onAuthChange }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);

  const handleProfileMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = () => {
    apiService.logout();
    onAuthChange(false);
    handleMenuClose();
    navigate('/');
  };

  const navigationItems = [
    { label: 'Home', path: '/', icon: <HomeIcon /> },
  ];

  return (
    <AppBar
      position="sticky"
      sx={{
        backgroundColor: 'rgba(26, 31, 46, 0.9)',
        backdropFilter: 'blur(20px)',
        borderBottom: '1px solid rgba(79, 139, 249, 0.2)',
      }}
    >
      <Toolbar sx={{ justifyContent: 'space-between' }}>
        {/* Logo */}
        <motion.div
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <Typography
            variant="h4"
            component="div"
            sx={{
              fontWeight: 700,
              background: 'linear-gradient(45deg, #4F8BF9 30%, #FF6B6B 90%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              cursor: 'pointer',
            }}
            onClick={() => navigate('/')}
          >
            🎭 CineSphere
          </Typography>
        </motion.div>

        {/* Navigation Items */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          {navigationItems.map((item) => (
            <motion.div
              key={item.path}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Button
                startIcon={item.icon}
                onClick={() => navigate(item.path)}
                sx={{
                  color: location.pathname === item.path ? 'primary.main' : 'text.primary',
                  fontWeight: location.pathname === item.path ? 600 : 400,
                  borderBottom: location.pathname === item.path ? '2px solid' : 'none',
                  borderColor: 'primary.main',
                  borderRadius: 0,
                  px: 2,
                }}
              >
                {item.label}
              </Button>
            </motion.div>
          ))}

          {/* Authentication Section */}
          {isAuthenticated ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <IconButton
                  onClick={handleProfileMenuOpen}
                  sx={{ p: 0 }}
                >
                  <Avatar sx={{ bgcolor: 'primary.main' }}>
                    <PersonIcon />
                  </Avatar>
                </IconButton>
              </motion.div>
              <Menu
                anchorEl={anchorEl}
                open={Boolean(anchorEl)}
                onClose={handleMenuClose}
                anchorOrigin={{
                  vertical: 'bottom',
                  horizontal: 'right',
                }}
                transformOrigin={{
                  vertical: 'top',
                  horizontal: 'right',
                }}
              >
                <MenuItem onClick={() => { navigate('/profile'); handleMenuClose(); }}>
                  <PersonIcon sx={{ mr: 1 }} />
                  Profile
                </MenuItem>
                <MenuItem onClick={handleLogout}>
                  <LogoutIcon sx={{ mr: 1 }} />
                  Logout
                </MenuItem>
              </Menu>
            </Box>
          ) : (
            <motion.div
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Button
                variant="contained"
                onClick={() => navigate('/auth')}
                sx={{ ml: 2 }}
              >
                Auth - Coming Soon
              </Button>
            </motion.div>
          )}
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar; 