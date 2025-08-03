import React, { useState, useEffect } from 'react';
import './RequestDemo.css';
import { Link } from 'react-router-dom';

const RequestDemo = () => {
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    document.querySelectorAll('.mobile-links a').forEach(link => {
      link.addEventListener('click', () => setMenuOpen(false));
    });
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    const nameInput = document.getElementById('name');
    const emailInput = document.getElementById('email');

    if (nameInput.checkValidity() && emailInput.checkValidity()) {
      alert('Your request submitted successfully');
      setTimeout(() => {
        window.location.href = 'hello.html';
      }, 1000);
    } else {
      alert('Please fill in all required fields.');
    }
  };

  return (
    <>
      <nav>
        <div className="nav-container">
          <a href="hello.html" className="logo">
            <span className="logo-icon"></span> TRANSPOLYMER
          </a>
          <ul className="nav-links">
            <li><a href="#features">Features</a></li>
            <li><a href="#how-it-works">How It Works</a></li>
            <li><a href="#use-cases">Use Cases</a></li>
            <li><a href="#testimonals">Testimonials</a></li>
            <li><Link to="/landingpage" className="nav-cta">Get Started</Link></li>
          </ul>
          <button className="mobile-menu-btn" onClick={() => setMenuOpen(true)}>
            ☰
          </button>
        </div>
      </nav>

      <div className={`mobile-menu ${menuOpen ? 'open' : ''}`}>
        <button className="mobile-menu-close" onClick={() => setMenuOpen(false)}>×</button>
        <ul className="mobile-links">
          <li><a href="#features">Features</a></li>
          <li><a href="#how-it-works">How It Works</a></li>
          <li><a href="#use-cases">Use Cases</a></li>
          <li><a href="#testimonals">Testimonials</a></li>
          <li><a href="login.html" className="nav-cta">Get Started</a></li>
        </ul>
      </div>

      <div className="demo-section">
        <div className="demo-container">
          <div className="demo-header">
            <h2 className="demo-title">Request a Demo</h2>
            <p className="demo-subtitle">
              Discover how Transploymer can transform your polymer engineering process.
            </p>
          </div>
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="name">Full Name</label>
              <input type="text" id="name" name="name" required />
            </div>
            <div className="form-group">
              <label htmlFor="email">Email Address</label>
              <input type="email" id="email" name="email" required />
            </div>
            <div className="form-group">
              <label htmlFor="company">Company Name</label>
              <input type="text" id="company" name="company" />
            </div>
            <div className="form-group">
              <label htmlFor="message">Message (Optional)</label>
              <textarea id="message" name="message"></textarea>
            </div>
            <button className="demo-btn" type="submit">Submit Request</button>
          </form>
        </div>
      </div>
    </>
  );
};

export default RequestDemo;
