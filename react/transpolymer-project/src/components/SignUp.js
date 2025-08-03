// SignUp.js
import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import './signup.css';
import { FaEye, FaEyeSlash } from 'react-icons/fa';

function SignUp() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
  });
  const [error, setError] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);

  const handleChange = e => {
    const { name, value } = e.target;
    setFormData(f => ({ ...f, [name]: value }));
  };

  const handleSubmit = async e => {
    e.preventDefault();
    const { username, email, password, confirmPassword } = formData;
    if (!username || !email || !password || !confirmPassword) {
      setError('All fields are required');
      return;
    }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      setError('Please enter a valid email address');
      return;
    }
    if (password.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }
    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    setError('');
    try {
      const res = await fetch('http://localhost:3000/api/users/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: username, email, password }),
      });
      const data = await res.json();
      if (res.ok) navigate('/login');
      else setError(data.message || 'Registration failed');
    } catch {
      setError('Network error. Please try again.');
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-form">
        <div className="auth-brand"><h2>TRANSPOLYMER</h2></div>
        <div className="auth-title"><h3>Create Account</h3></div>

        {error && <div className="error-message">{error}</div>}

        <form onSubmit={handleSubmit} autoComplete="off">
          <div className="input-field-container">
            <label className="section-label">Username</label>
            <input
              type="text"
              name="username"
              className="text-input"
              placeholder="Choose a username"
              value={formData.username}
              onChange={handleChange}
              required
            />
          </div>

          <div className="input-field-container">
            <label className="section-label">Email</label>
            <input
              type="email"
              name="email"
              className="text-input"
              placeholder="Enter your email"
              value={formData.email}
              onChange={handleChange}
              required
            />
          </div>

          <div className="input-field-container">
            <label className="section-label">Password</label>
            <div style={{ display: 'flex', alignItems: 'center', border: '1px solid #ccc', borderRadius: 6, padding: '6px 10px' }}>
              <input
                type={showPassword ? 'text' : 'password'}
                name="password"
                className="text-input"
                placeholder="Create a password"
                value={formData.password}
                onChange={handleChange}
                style={{ flex: 1, border: 'none', outline: 'none' }}
                required
              />
              <button
                type="button"
                onClick={() => setShowPassword(v => !v)}
                aria-label={showPassword ? 'Hide password' : 'Show password'}
                style={{ background: 'none', border: 'none', outline: 'none', cursor: 'pointer', fontSize: 20 }}
              >
                {showPassword ? <FaEyeSlash /> : <FaEye />}
              </button>
            </div>
          </div>

          <div className="input-field-container">
            <label className="section-label">Confirm Password</label>
            <div style={{ display: 'flex', alignItems: 'center', border: '1px solid #ccc', borderRadius: 6, padding: '6px 10px' }}>
              <input
                type={showConfirm ? 'text' : 'password'}
                name="confirmPassword"
                className="text-input"
                placeholder="Confirm your password"
                value={formData.confirmPassword}
                onChange={handleChange}
                style={{ flex: 1, border: 'none', outline: 'none' }}
                required
              />
              <button
                type="button"
                onClick={() => setShowConfirm(v => !v)}
                aria-label={showConfirm ? 'Hide confirm password' : 'Show confirm password'}
                style={{ background: 'none', border: 'none', outline: 'none', cursor: 'pointer', fontSize: 20 }}
              >
                {showConfirm ? <FaEyeSlash /> : <FaEye />}
              </button>
            </div>
          </div>

          <button type="submit" className="submit-button">Create Account</button>
        </form>

        <div className="auth-links">
          Already have an account? <Link to="/login" className="login-link">Login</Link>
        </div>
      </div>
    </div>
  );
}

export default SignUp;
