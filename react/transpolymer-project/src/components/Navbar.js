import React, { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import './landingpage.css';
import '@fortawesome/fontawesome-free/css/all.min.css';

const Navbar = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [profileOpen, setProfileOpen]           = useState(false);
  const toggleMobileMenu = () => setIsMobileMenuOpen(o => !o);

  const navigate  = useNavigate();
  const { pathname } = useLocation();
  const isAdminDashboard = pathname === '/admin-dashboard';

  // read stored user/admin
  const admin = JSON.parse(localStorage.getItem('user')) || {};

  const handleLogout = () => {
    document.cookie = 'token=;expires=Thu,01 Jan 1970 00:00:00 GMT;path=/;';
    localStorage.removeItem('user');
    navigate('/login');
  };

  return (
    <>
      <nav className={`navbar${isAdminDashboard ? ' admin-nav' : ''}`}>
        <div className="nav-container">
          <Link to="/" className="logo">
            <span className="logo-icon"><i className="fas fa-atom"></i></span>
            TRANSPOLYMER
          </Link>

          <ul className="nav-links">
            <li><Link to="/#features">Features</Link></li>
            <li><Link to="/#how-it-works">How It Works</Link></li>
            <li><Link to="/#use-cases">Use Cases</Link></li>

            {isAdminDashboard ? (
              <li className="profile-wrapper">
                <button
                  className="nav-cta"
                  onClick={() => setProfileOpen(o => !o)}
                >
                  Admin Profile ▼
                </button>
                {profileOpen && (
                  <div className="profile-dropdown">
                    <p><strong>{admin.username}</strong></p>
                    <p>{admin.email}</p>
                    <hr />
                    <button className="logout-btn" onClick={handleLogout}>
                      Log Out
                    </button>
                  </div>
                )}
              </li>
            ) : (
              <li>
                <Link to="/login" className="nav-cta">Get Started</Link>
              </li>
            )}
          </ul>

          <button className="mobile-menu-btn" onClick={toggleMobileMenu}>
            <i className="fas fa-bars"></i>
          </button>
        </div>
      </nav>

      {/* Mobile menu overlay */}
      <div className={`mobile-menu${isMobileMenuOpen ? ' open' : ''}`}>
        <button className="mobile-menu-close" onClick={toggleMobileMenu}>
          <i className="fas fa-times"></i>
        </button>
        <ul className="mobile-links">
          <li><Link to="/#features" onClick={toggleMobileMenu}>Features</Link></li>
          <li><Link to="/#how-it-works" onClick={toggleMobileMenu}>How It Works</Link></li>
          <li><Link to="/#use-cases" onClick={toggleMobileMenu}>Use Cases</Link></li>

          {isAdminDashboard ? (
            <li className="profile-wrapper">
              <button
                className="nav-cta"
                onClick={() => {
                  setProfileOpen(o => !o);
                  setIsMobileMenuOpen(false);
                }}
              >
                Admin Profile ▼
              </button>
              {profileOpen && (
                <div className="profile-dropdown">
                  <p><strong>{admin.username}</strong></p>
                  <p>{admin.email}</p>
                  <hr />
                  <button className="logout-btn" onClick={handleLogout}>
                    Log Out
                  </button>
                </div>
              )}
            </li>
          ) : (
            <li>
              <Link to="/login" className="nav-cta" onClick={toggleMobileMenu}>
                Get Started
              </Link>
            </li>
          )}
        </ul>
      </div>

      <div
        className={`overlay${isMobileMenuOpen ? ' active' : ''}`}
        onClick={toggleMobileMenu}
      />
    </>
  );
};

export default Navbar;
