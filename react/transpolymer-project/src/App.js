import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';

// Import components
import Login from './components/Login';
import SignUp from './components/SignUp';
import ForgotPassword from './components/ForgotPassword';
import AboutUs from './components/AboutUs';
import Dashboard from './components/Dashboard';
import AdminDashboard from './components/AdminDashboard';
import Navbar from './components/Navbar';
import LandingPage from './components/landingpage';
import UserProfile from './components/UserProfile';
import ResetRequestForm from './components/ResetRequestForm';
import ProtectedRoute from './components/ProtectedRoute';

function AppWrapper() {
  return (
    <Router>
      <App />
    </Router>
  );
}

function App() {
  const location = useLocation();

  // Hide navbar on both user and admin dashboards
  const hideNavbarOnRoutes = ['/dashboard', '/admin-dashboard'];
  const shouldHideNavbar = hideNavbarOnRoutes.includes(location.pathname);

  return (
    <div className="app-container">
      {!shouldHideNavbar && <Navbar />}

      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<SignUp />} />
        <Route path="/forgot-password" element={<ForgotPassword />} />
        <Route path="/about" element={<AboutUs />} />
        
        {/* ✅ Protected Routes */}
        <Route path="/dashboard" element={
          <ProtectedRoute role="user">
            <Dashboard />
          </ProtectedRoute>
        } />
        <Route path="/admin-dashboard" element={
          <ProtectedRoute role="admin">
            <AdminDashboard />
          </ProtectedRoute>
        } />

        <Route path="/profile" element={<UserProfile />} />
        <Route path="/" element={<LandingPage />} />
        <Route path="/reset" element={<ResetRequestForm />} />
      </Routes>
    </div>
  );
}

export default AppWrapper;
