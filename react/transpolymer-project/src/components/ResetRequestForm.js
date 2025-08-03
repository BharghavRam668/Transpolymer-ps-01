// ResetRequestForm.js
import React, { useState } from 'react';
import './forgotpassword.css';
import { FaEye, FaEyeSlash } from 'react-icons/fa';

function ResetRequestForm() {
  const [email, setEmail] = useState('');
  const [desiredPassword, setDesiredPassword] = useState('');
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showPassword, setShowPassword] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (desiredPassword.length < 8) {
      setError('Password must be at least 8 characters long.');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('http://localhost:3000/api/reset-request', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, desiredPassword }),
      });

      if (response.ok) {
        setSubmitted(true);
      } else {
        const errData = await response.json();
        setError(errData.message || 'Something went wrong.');
      }
    } catch {
      setError('Network error. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-form">
        <h2 className="auth-title">Request Password Reset</h2>

        {submitted ? (
          <div className="success-message">
            ✅ Request sent successfully! We'll contact you shortly.
          </div>
        ) : (
          <form onSubmit={handleSubmit}>
            <div className="input-field-container">
              <label className="section-label">Email</label>
              <input
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                required
                className="text-input"
                placeholder="Enter your email"
              />
            </div>

            <div className="input-field-container">
              <label className="section-label">Desired New Password</label>
              <div style={{ position: 'relative' }}>
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={desiredPassword}
                  onChange={e => setDesiredPassword(e.target.value)}
                  required
                  className="text-input"
                  placeholder="Create a new password"
                  minLength={8}
                  style={{ paddingRight: '40px' }}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(v => !v)}
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                  style={{
                    position: 'absolute',
                    right: '10px',
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
            </div>

            {error && <div className="error-message">❌ {error}</div>}

            <button type="submit" className="submit-button" disabled={loading}>
              {loading ? 'Sending...' : 'Send Request'}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}

export default ResetRequestForm;
