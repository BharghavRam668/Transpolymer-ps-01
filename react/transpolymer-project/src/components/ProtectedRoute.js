// components/ProtectedRoute.js
import React from 'react';
import { Navigate } from 'react-router-dom';

const ProtectedRoute = ({ children, role }) => {
  const user = JSON.parse(localStorage.getItem('user'));
  const admin = JSON.parse(localStorage.getItem('admin'));

  if (role === 'admin') return admin ? children : <Navigate to="/login" />;
  return user ? children : <Navigate to="/login" />;
};

export default ProtectedRoute;
