import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './Dashboard.css';

function Dashboard({ onLogout }) {
  const navigate = useNavigate();

  // ---------- STATE ----------
  const [smiles, setSmiles] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [molecularWeight, setMolecularWeight] = useState(null);
  const [structureImage, setStructureImage] = useState(null);
  const [searchHistory, setSearchHistory] = useState([]);
  const [historyVisible, setHistoryVisible] = useState(true);
  const [showChatbot, setShowChatbot] = useState(false);
  const [showPolymerChatbot, setShowPolymerChatbot] = useState(false);
  const [userMessage, setUserMessage] = useState('');
  const [chatMessages, setChatMessages] = useState([]);
  const [polymerName, setPolymerName] = useState('');
  const [polymerChatHistory, setPolymerChatHistory] = useState([]);
  const [processingMessage, setProcessingMessage] = useState(false);
  const [showProfileDropdown, setShowProfileDropdown] = useState(false);

  // ---------- AUTH ----------
  const user = JSON.parse(localStorage.getItem('user') || 'null');
  const userLoggedIn = !!user;

  // ---------- PROPERTY MAP ----------
  const propertyMap = {
    band_gap: { name: 'Band Gap', unit: 'eV' },
    dielectric_constant: { name: 'Dielectric Constant', unit: '' },
    refractive_index: { name: 'Refractive Index', unit: '' },
    electron_affinity: { name: 'Electron Affinity', unit: 'eV' },
    ionization_energy: { name: 'Ionization Energy', unit: 'eV' },
    crystallization_tendency: { name: 'Crystallization Tendency', unit: '' }
  };

  // ---------- HISTORY LOAD ----------
  useEffect(() => {
    if (userLoggedIn) {
      fetch('http://localhost:3000/api/search-history', { credentials: 'include' })
        .then(r => r.json())
        .then(data => setSearchHistory(data.map(d => d.smiles)))
        .catch(err => console.error('History fetch error:', err));
    } else {
      const saved = localStorage.getItem('smilesSearchHistory');
      if (saved) setSearchHistory(JSON.parse(saved));
    }

    const vis = localStorage.getItem('historyVisibility');
    if (vis !== null) setHistoryVisible(JSON.parse(vis));
  }, [userLoggedIn]);

  // ---------- PERSIST LOCAL GUEST HISTORY ----------
  useEffect(() => {
    if (!userLoggedIn) localStorage.setItem('smilesSearchHistory', JSON.stringify(searchHistory));
  }, [searchHistory]);

  // ---------- PERSIST SIDEBAR VISIBILITY ----------
  useEffect(() => {
    localStorage.setItem('historyVisibility', JSON.stringify(historyVisible));
  }, [historyVisible]);

  // ---------- CHATBASE WIDGET ----------
  useEffect(() => {
    const scriptWrapper = document.createElement('script');
    scriptWrapper.innerHTML = `(function(){if(!window.chatbase||window.chatbase("getState")!=="initialized"){window.chatbase=(...a)=>{if(!window.chatbase.q){window.chatbase.q=[]}window.chatbase.q.push(a)};window.chatbase=new Proxy(window.chatbase,{get(t,p){if(p==="q")return t.q;return(...a)=>t(p,...a)}})}const onLoad=()=>{const s=document.createElement("script");s.src="https://www.chatbase.co/embed.min.js";s.id="YW9okURRl8lKfP2CrmwWz";s.domain="www.chatbase.co";document.body.appendChild(s)};document.readyState==="complete"?onLoad():window.addEventListener("load",onLoad);})();`;
    document.body.appendChild(scriptWrapper);
  }, []);

  // ---------- PREDICT ----------
  const predictProperties = async (smilesString) => {
    setLoading(true);
    setError(null);
    setResults(null);
    setMolecularWeight(null);
    setStructureImage(null);

    // optimistic sidebar update
    if (!searchHistory.includes(smilesString)) {
      setSearchHistory(prev => [smilesString, ...prev].slice(0, 10));
    }

    try {
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ smiles: smilesString })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'API request failed');

      setResults(Object.entries(data.results).map(([k, v]) => ({
        id: k,
        name: propertyMap[k]?.name || k,
        value: v,
        unit: propertyMap[k]?.unit || ''
      })));
      setMolecularWeight(data.molecular_weight);
      setStructureImage(data.structure_image);

      if (userLoggedIn) {
        fetch('http://localhost:3000/api/search-history', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify({ smiles: smilesString })
        }).catch(err => console.error('Save history error:', err));
      }
    } catch (err) {
      setError(err.message || 'Failed to predict properties. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!smiles.trim()) return;
    predictProperties(smiles);
  };

  // ---------- CLEAR + SIDEBAR HANDLERS ----------
  const clearSearchHistory = () => {
    setSearchHistory([]);
    if (!userLoggedIn) localStorage.removeItem('smilesSearchHistory');
  };

  const toggleHistoryVisibility = () => setHistoryVisible(prev => !prev);

  const handleHistoryItemClick = (item) => {
    setSmiles(item);
    predictProperties(item);
  };

  // ---------- CHAT HANDLERS ----------
  const handleChatSubmit = (e) => {
    e.preventDefault();
    if (!userMessage.trim()) return;

    setChatMessages(prev => [...prev, { sender: 'user', text: userMessage }]);
    setProcessingMessage(true);
    setUserMessage('');

    setTimeout(() => {
      let response;
      if (userMessage.toLowerCase().includes('band gap')) {
        response = "Band gap is the energy difference between the top of the valence band and the bottom of the conduction band...";
      } else if (userMessage.toLowerCase().includes('dielectric')) {
        response = "The dielectric constant measures a material's ability to store electrical energy in an electric field...";
      } else {
        response = "I can help explain various molecular properties. Ask about band gap, dielectric constant, etc.";
      }

      setChatMessages(prev => [...prev, { sender: 'bot', text: response }]);
      setProcessingMessage(false);
    }, 1000);
  };

  const handlePolymerChatSubmit = async (e) => {
    e.preventDefault();
    if (!polymerName.trim()) return;

    const userMessage = polymerName;
    setPolymerChatHistory(prev => [...prev, { sender: 'user', text: userMessage }]);
    setProcessingMessage(true);
    setPolymerName('');

    try {
      const response = await fetch("http://localhost:8000/polymer-to-smiles", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ polymer_name: userMessage })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Conversion failed");
      }

      setPolymerChatHistory(prev => [
        ...prev,
        {
          sender: 'bot',
          text: `Polymer: ${userMessage}\nSMILES notation: ${data.smiles}\nMolecular Weight: ${data.molecular_weight.toFixed(2)} g/mol`,
          structureImage: data.structure_image
        }
      ]);
    } catch (err) {
      setPolymerChatHistory(prev => [
        ...prev,
        {
          sender: 'bot',
          text: `I couldn't find SMILES notation for "${userMessage}". Please try a different polymer name or common abbreviation.`
        }
      ]);
    } finally {
      setProcessingMessage(false);
    }
  };

  const clearChatHistory = () => {
    setChatMessages([]);
  };

  const clearPolymerChatHistory = () => {
    setPolymerChatHistory([]);
  };

  const toggleProfileDropdown = () => {
    setShowProfileDropdown(prev => !prev);
  };

  const navigateToProfile = () => {
    setShowProfileDropdown(false);
    navigate('/profile');
  };

  const handleLogout = () => {
    setShowProfileDropdown(false);
    if (onLogout) {
      onLogout();
    }
    navigate('/login');
  };

  const mainContentClass = historyVisible ? 'main-content with-sidebar' : 'main-content';

  // ---------- UI (unchanged from original) ----------
  return (
    <div className="dashboard-container">
      <div className="atom-animation">
        <div className="atom">
          <div className="nucleus"></div>
          <div className="orbit one"></div>
          <div className="orbit two"></div>
          <div className="orbit three"></div>
          <div className="electron"></div>
        </div>
      </div>

      <header className="dashboard-header">
        <div className="header-content">
          <div className="logo-container" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '28px', color: '#ff7a59' }}>
              <i className="fas fa-atom"></i>
            </span>
            <h1 style={{ margin: 0, color: '#ffffff', fontWeight: 'bold' }}>
              TRANSPOLYMER <span style={{ fontWeight: 'normal', fontSize: '20px' }}>| Properties Predictor</span>
            </h1>
          </div>
          
          <div className="header-actions">
            <button 
              className="nav-button home-button"
              onClick={() => navigate('/')}
            >
              <i className="fas fa-home"></i> Home
            </button>
            <button 
              className="nav-button polymer-button" 
              onClick={() => {
                setShowPolymerChatbot(!showPolymerChatbot);
                if (showChatbot) setShowChatbot(false);
              }}
            >
              <i className="fas fa-project-diagram"></i> {showPolymerChatbot ? 'Hide Polymer Tool' : 'Polymer Tool'}
            </button>
            <button 
              className="nav-button profile-button"
              onClick={toggleProfileDropdown}
            >
              <i className="fas fa-user-circle"></i> Profile
            </button>
          </div>
        </div>
      </header>

      {showProfileDropdown && (
        <div className="profile-dropdown-container">
          <div className="profile-dropdown">
            <div className="profile-dropdown-header" style={{ color: "#000" }}>
              <i className="fas fa-user-circle"></i>
              <span  style={{ color: "#fff" }}>User Profile</span>
            </div>
            <ul className="profile-menu">
              <li onClick={navigateToProfile} style={{ color: "#000" }}>
                <i className="fas fa-user"  style={{ marginRight: "8px", color: "#000" }}></i> View Profile
              </li>
              <li onClick={handleLogout} style={{ color: "#000" }}>
                <i className="fas fa-sign-out-alt"  style={{ marginRight: "8px", color: "#000" }} ></i> Logout
              </li>
            </ul>
          </div>
        </div>
      )}

      {showPolymerChatbot && (
        <div className="chatbot-panel polymer-chatbot slide-in">
          <div className="chatbot-header">
            <div className="chatbot-title">
              <i className="fas fa-project-diagram"></i>
              <h3>Polymer to SMILES Converter</h3>
            </div>
            <div className="chatbot-controls">
              <button onClick={clearPolymerChatHistory} className="clear-history" title="Clear conversation">
                <i className="fas fa-trash-alt"></i>
              </button>
              <button onClick={() => setShowPolymerChatbot(false)} className="close-chatbot" title="Close">
                <i className="fas fa-times"></i>
              </button>
            </div>
          </div>
          <div className="chat-messages">
            {polymerChatHistory.length === 0 ? (
              <div className="welcome-message">
                <div className="welcome-icon">
                  <i className="fas fa-atom"></i>
                </div>
                <p>Welcome to the Polymer Assistant! I can convert polymer names to SMILES notation.</p>
                <div className="examples">
                  <p>Try asking for:</p>
                  <ul>
                    <li>"polyethylene"</li>
                    <li>"convert PVC to SMILES"</li>
                    <li>"What polymers do you know?"</li>
                  </ul>
                </div>
              </div>
            ) : (
              polymerChatHistory.map((msg, index) => (
                <div key={index} className={`message ${msg.sender}`}>
                  {msg.sender === 'user' && (
                    <div className="message-sender">
                      <i className="fas fa-user"></i>
                    </div>
                  )}
                  {msg.sender === 'bot' && (
                    <div className="message-sender">
                      <i className="fas fa-robot"></i>
                    </div>
                  )}
                  <div className="message-content">
                    {msg.text.split('\n').map((line, i) => (
                      <p key={i}>
                        {line.includes('SMILES notation') && msg.sender === 'bot' && line.includes(':') ? (
                          <>
                            {line.split(':')[0]}:{' '}
                            <code 
                              className="clickable-smiles"
                              onClick={() => {
                                const smilesStr = line.split(':')[1].trim();
                                console.log('SMILES from polymer chatbot:', smilesStr);
                                setSmiles(smilesStr);
                                predictProperties(smilesStr);
                                setShowPolymerChatbot(false);
                              }}
                              title="Click to use this SMILES"
                            >
                              {line.split(':')[1].trim()}
                            </code>
                          </>
                        ) : (
                          line
                        )}
                      </p>
                    ))}
                    {msg.structureImage && (
                      <div className="structure-image-container">
                        <img src={msg.structureImage} alt="Polymer Structure" className="structure-image" />
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
            {processingMessage && (
              <div className="message bot processing">
                <div className="message-sender">
                  <i className="fas fa-robot"></i>
                </div>
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
          </div>
          <form onSubmit={handlePolymerChatSubmit} className="chat-input-form">
            <input
              type="text"
              value={polymerName}
              onChange={(e) => setPolymerName(e.target.value)}
              placeholder="Enter polymer name (e.g., polyethylene)..."
            />
            <button type="submit" className="send-button">
              <i className="fas fa-paper-plane"></i>
            </button>
          </form>
        </div>
      )}

      {showChatbot && (
        <div className="chatbot-panel slide-in">
          <div className="chatbot-header">
            <div className="chatbot-title">
              <i className="fas fa-robot"></i>
              <h3>Molecular Assistant</h3>
            </div>
            <div className="chatbot-controls">
              <button onClick={clearChatHistory} className="clear-history" title="Clear conversation">
                <i className="fas fa-trash-alt"></i>
              </button>
              <button onClick={() => setShowChatbot(false)} className="close-chatbot" title="Close">
                <i className="fas fa-times"></i>
              </button>
            </div>
          </div>
          <div className="chat-messages">
            {chatMessages.length === 0 ? (
              <div className="welcome-message">
                <div className="welcome-icon">
                  <i className="fas fa-microscope"></i>
                </div>
                <p>Hello! I'm your molecular assistant. I can help you understand molecular properties and predictions.</p>
                <div className="examples">
                  <p>Try asking about:</p>
                  <ul>
                    <li>"What is dielectric constant?"</li>
                    <li>"Explain band gap"</li>
                    <li>"How to interpret these results?"</li>
                  </ul>
                </div>
              </div>
            ) : (
              chatMessages.map((msg, index) => (
                <div key={index} className={`message ${msg.sender}`}>
                  {msg.sender === 'user' && (
                    <div className="message-sender">
                      <i className="fas fa-user"></i>
                    </div>
                  )}
                  {msg.sender === 'bot' && (
                    <div className="message-sender">
                      <i className="fas fa-robot"></i>
                    </div>
                  )}
                  <div className="message-content">
                    {msg.text}
                  </div>
                </div>
              ))
            )}
            {processingMessage && (
              <div className="message bot processing">
                <div className="message-sender">
                  <i className="fas fa-robot"></i>
                </div>
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
          </div>
          <form onSubmit={handleChatSubmit} className="chat-input-form">
            <input
              type="text"
              value={userMessage}
              onChange={(e) => setUserMessage(e.target.value)}
              placeholder="Ask about molecular properties..."
            />
            <button type="submit" className="send-button">
              <i className="fas fa-paper-plane"></i>
            </button>
          </form>
        </div>
      )}

      <div className="content-container">
        {historyVisible && (
          <div className="history-sidebar">
            <div className="history-header">
              <div className="history-title">
                <i className="fas fa-history"></i>
                <h3>Recent Searches</h3>
              </div>
              <div className="history-controls">
                {searchHistory.length > 0 && (
                  <button onClick={clearSearchHistory} className="clear-history-btn" title="Clear history">
                    <i className="fas fa-trash-alt"></i>
                  </button>
                )}
              </div>
            </div>
            {searchHistory.length > 0 ? (
              <ul className="history-list">
                {searchHistory.map((item, index) => (
                  <li 
                    key={index} 
                    onClick={() => handleHistoryItemClick(item)}
                    className={item === smiles ? 'active' : ''}
                  >
                    <span className="smiles-icon">
                      <i className="fas fa-atom"></i>
                    </span>
                    <span className="smiles-text">{item}</span>
                    {item === smiles && (
                      <span className="active-indicator">
                        <i className="fas fa-arrow-left"></i>
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            ) : (
              <div className="no-history">
                <i className="fas fa-search"></i>
                <p>No search history yet</p>
              </div>
            )}
            <div className="history-tip">
              <div className="tip-icon">
                <i className="fas fa-lightbulb"></i>
              </div>
              <p>Use the Polymer Tool for common polymers like polyethylene or PVC</p>
            </div>
          </div>
        )}
        
        <div className="history-toggle-container">
          <button 
            onClick={toggleHistoryVisibility} 
            className="history-toggle-btn" 
          >
            <span className="toggle-text">
              {historyVisible ? "" : ""}
            </span>
            <i className={`fas ${historyVisible ? 'fa-eye-slash' : 'fa-eye'}`}></i>
          </button>
        </div>

        <div className={mainContentClass}>
          <div className="input-card card-animation">
            <div className="card-header">
              <div className="card-icon">
                <i className="fas fa-flask"></i>
              </div>
              <h2>Properties Prediction</h2>
              <div className="card-description">
                Enter a Polymer SMILES string to predict its properties
              </div>
            </div>

            <form onSubmit={handleSubmit} className="input-form">
              <div className="input-field-container">
                <label className="section-label">
                  <i className="fas fa-code"></i> SMILES Representation
                </label>
                <div className="input-with-button">
                  <input
                    type="text"
                    value={smiles}
                    onChange={(e) => {
                      console.log('SMILES input changed:', e.target.value);
                      setSmiles(e.target.value);
                    }}
                    placeholder="E.g., C1=CC=CC=C1 (Benzene)"
                    className="text-input"
                  />
                  <button type="submit" disabled={loading} className="submit-button pulse-animation">
                    {loading ? (
                      <span className="loading-text">
                        <i className="fas fa-spinner fa-spin"></i> Processing
                      </span>
                    ) : (
                      <>
                        <i className="fas fa-chart-line"></i> Predict Properties
                      </>
                    )}
                  </button>
                </div>
                <div className="input-hint">
                  <i className="fas fa-info-circle"></i> Example SMILES: CCO (Ethanol), C#N (Hydrogen Cyanide)
                </div>
                <div className="polymer-converter-hint">
                  <button 
                    type="button" 
                    className="polymer-hint-button"
                    onClick={() => {
                      setShowPolymerChatbot(true);
                      setShowChatbot(false);
                    }}
                  >
                    <i className="fas fa-question-circle"></i> Don't know SMILES? Use Polymer Tool
                  </button>
                </div>
              </div>
            </form>
          </div>

          {error && (
            <div className="error-container slide-in">
              <div className="error-icon">
                <i className="fas fa-exclamation-triangle"></i>
              </div>
              <div className="error-message">{error}</div>
            </div>
          )}

          {results && (
            <div className="results-card card-animation">
              <div className="card-header">
                <div className="card-icon">
                  <i className="fas fa-chart-bar"></i>
                </div>
                <h2>Predicted Properties</h2>
                <div className="subheader">Analysis Results for <span className="smiles-result">{smiles}</span></div>
              </div>

              {structureImage && (
                <div className="structure-image-container">
                  <h3 className="structure-title">Polymer Structure</h3>
                  <img src={structureImage} alt="Polymer Structure" className="structure-image" />
                </div>
              )}

              {molecularWeight && (
                <div className="molecular-weight-container">
                  <h3 className="molecular-weight-title">Molecular Weight</h3>
                  <p className="molecular-weight-value">{molecularWeight.toFixed(2)} g/mol</p>
                </div>
              )}

              <div className="results-grid">
                {results.map((property) => (
                  <div key={property.id} className="property-card">
                    <div className="property-header">
                      <div className="property-name">{property.name}</div>
                    </div>
                    <div className="property-value-container">
                      <span className="property-value">{property.value}</span>
                      {property.unit && (
                        <span className="property-unit">{property.unit}</span>
                      )}
                    </div>
                    <div className="property-visualization">
                      <div 
                        className="property-bar" 
                        style={{ width: `${Math.min(100, Math.abs(property.value) * 10)}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="results-footer">
                <div className="results-timestamp">
                  <i className="fas fa-clock"></i> Generated at: {new Date().toLocaleString()}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Dashboard;