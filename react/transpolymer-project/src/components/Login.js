import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import './login.css';
import { FaEye, FaEyeSlash } from 'react-icons/fa';

function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isAdmin, setIsAdmin] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const navigate = useNavigate();

  const handleToggle = (adminMode) => {
    setIsAdmin(adminMode);
    setEmail('');
    setPassword('');
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!email.trim() || !password.trim()) {
      setError('Please enter both email and password');
      return;
    }
    setError('');
    const endpoint = isAdmin
      ? 'http://localhost:3000/api/admin/login'
      : 'http://localhost:3000/api/users/login';

    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json();
      if (res.ok) {
        if (isAdmin) {
          localStorage.setItem('admin', JSON.stringify(data.admin));
          navigate('/admin-dashboard');
        } else {
          localStorage.setItem('user', JSON.stringify(data.user));
          navigate('/dashboard');
        }
      } else {
        setError(data.message || 'Invalid email or password');
      }
    } catch (err) {
      console.error(err);
      setError('Network error. Please ensure the server is running.');
    }
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <div className="login-brand"><h2>Transpolymer</h2></div>
        <div className="login-title"><h3>Welcome</h3></div>

        <div className="toggle-tabs">
          <button
            type="button"
            className={!isAdmin ? "toggle-tab active" : "toggle-tab"}
            onClick={() => handleToggle(false)}
          >
            User
          </button>
          <button
            type="button"
            className={isAdmin ? "toggle-tab active" : "toggle-tab"}
            onClick={() => handleToggle(true)}
          >
            Admin
          </button>
        </div>

        {error && <div className="error-message">{error}</div>}

        <form onSubmit={handleSubmit} className="login-form">
          <div className="input-group">
            <input
              type="email"
              className="form-input"
              value={email}
              onChange={e => setEmail(e.target.value)}
              placeholder="Email Address"
              required
            />
          </div>

          <div className="input-group" style={{ position: 'relative' }}>
            <input
              type={showPassword ? 'text' : 'password'}
              className="form-input"
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="Password"
              required
              style={{ paddingRight: 40 }}
            />
            <button
              type="button"
              onClick={() => setShowPassword(v => !v)}
              aria-label={showPassword ? 'Hide password' : 'Show password'}
              style={{
                position: 'absolute',
                right: 10,
                top: '50%',
                transform: 'translateY(-50%)',
                background: 'none',
                border: 'none',
                outline: 'none',
                cursor: 'pointer',
                fontSize: 20,
                lineHeight: 1,
              }}
            >
              {showPassword ? <FaEyeSlash /> : <FaEye />}
            </button>
          </div>

          <div className="forgot-password">
            <Link to="/reset">Forgot Password?</Link>
          </div>

          <button type="submit" className="login-button">Log In</button>
        </form>

        {!isAdmin && (
          <p className="signup-prompt">
            Don’t have an account? <Link to="/signup">Sign Up</Link>
          </p>
        )}
      </div>
    </div>
  );
}

export default Login;
