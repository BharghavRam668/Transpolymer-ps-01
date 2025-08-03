import React from 'react';
import { Link } from 'react-router-dom';
import './forgotpassword.css';

function ForgotPassword() {
  return (
    <div className="auth-container">
      <div className="auth-form">
        <div className="auth-brand">
          <h2>TRANSPOLYMER</h2>
        </div>

        <div className="auth-title">
          <h3>Reset Password</h3>
        </div>

        <div className="info-message">
          If you’ve forgotten your password, please contact our support team at:
          <br />
          <a href="mailto:info@transpolymer.com" className="email-link">info@transpolymer.com</a>
        </div>

        <div className="auth-links">
          <span>Remember your password? </span>
          <Link to="/login" className="login-link">Back to Login</Link>
        </div>
      </div>
    </div>
  );
}

export default ForgotPassword;
