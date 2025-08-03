import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './UserProfile.css';
import axios from 'axios';

function UserProfile() {
  const navigate = useNavigate();
  const storedUser = JSON.parse(localStorage.getItem("user") || "null");
  const [user, setUser] = useState({
    username: storedUser?.username || "",
    email:    storedUser?.email    || "",
    lastLogin: null,
    previousLogins: [],
    role: "",
    profilePicture: null,
    twoFactorEnabled: false,
    loginDevices: [],
    createdAt: null,
  });

  const [newPassword, setNewPassword]           = useState("");
  const [confirmPassword, setConfirmPassword]   = useState("");
  const [message, setMessage]                   = useState("");
  const [messageType, setMessageType]           = useState("");
  const [loading, setLoading]                   = useState(true);
  const [activeTab, setActiveTab]               = useState("info");
  //const [showLoginHistory, setShowLoginHistory] = useState(false);

 useEffect(() => {
    setLoading(true);
    axios
      .get("/api/users/profile", { withCredentials: true })
      .then((res) => {
        setUser(res.data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Profile fetch error:", err);
        setLoading(false);
        // Kick back to login if cookie is missing / expired
        if (err.response?.status === 401) navigate("/login");
      });
  }, [navigate]);

  const updatePassword = () => {
    setMessage('');
    if (newPassword !== confirmPassword) {
      setMessage("Passwords do not match");
      setMessageType('error');
      return;
    }
    if (newPassword.length < 8) {
      setMessage("Password must be at least 8 characters long");
      setMessageType('error');
      return;
    }
    setLoading(true);
    axios.post('/api/users/update-password', { newPassword }, { withCredentials: true })
      .then(res => {
        setMessage(res.data.message || "Password updated successfully");
        setMessageType('success');
        setNewPassword('');
        setConfirmPassword('');
        setLoading(false);
      })
      .catch(err => {
        const errorMsg = err.response?.data?.message || "Error updating password";
        setMessage(errorMsg);
        setMessageType('error');
        setLoading(false);
      });
  };
  const handleBackToDashboard = () => {
    navigate('/dashboard');
  };

  // ... UI code remains unchanged

  return (
    <div className="user-profile-container">
    <div className="user-profile">
      <div className="profile-header">
        <h2><i className="fas fa-user"></i> Profile</h2>
        <button className="back-button" onClick={handleBackToDashboard}>
          <i className="fas fa-arrow-left"></i>
        </button>
      </div>

      {loading ? (
        <div className="loading-spinner">
          <i className="fas fa-spinner fa-spin"></i>
          Loading profile...
        </div>
      ) : (
        <>
          {message && (
            <div className={`message ${messageType}`}>
              <i className={`fas ${messageType === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'}`}></i>
              {message}
              <button className="close-message" onClick={() => setMessage('')}>&times;</button>
            </div>
          )}

          <div className="profile-navbar">
            <ul>
              <li
                className={activeTab === 'info' ? 'active' : ''}
                onClick={() => setActiveTab('info')}
              >
                <i className="fas fa-id-card"></i> Info
              </li>
            </ul>
          </div>

          {activeTab === 'info' && (
            <div className="profile-section">
              <h3>Personal Information</h3>
              <div className="profile-picture">
                {user.profilePicture ? (
                  <img src={user.profilePicture} alt="Profile" />
                ) : (
                  <div className="default-avatar">{user.username[0]}</div>
                )}
              </div>
              <div className="profile-info">
                <div className="info-item">
                  <div className="info-label"><i className="fas fa-user"></i> Username</div>
                  <div className="info-value">{user.username}</div>
                </div>
                <div className="info-item">
                  <div className="info-label"><i className="fas fa-envelope"></i> Email</div>
                  <div className="info-value">{user.email}</div>
                </div>
                <div className="info-item">
                  <div className="info-label"><i className="fas fa-calendar-alt"></i> Joined</div>
                  <div className="info-value">{user.createdAt ? new Date(user.createdAt).toLocaleDateString() : 'N/A'}</div>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  </div>
  );
}

export default UserProfile;