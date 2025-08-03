import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { Link } from 'react-router-dom';
import './landingpage.css';
import '@fortawesome/fontawesome-free/css/all.min.css';


const Loader = () => {
    const [isHidden, setIsHidden] = useState(false);

    useEffect(() => {
        const timer = setTimeout(() => {
            setIsHidden(true);
        }, 1000);
        return () => clearTimeout(timer);
    }, []);

    return (
        <div className={`loader ${isHidden ? 'hidden' : ''}`}>
            <div className="loader-content">
                <div className="loader-icon">
                    <div className="loader-circle"></div>
                </div>
                <div className="loader-text">TRANSPLOYMER</div>
            </div>
        </div>
    );
};

const CursorFollower = () => {
    const [position, setPosition] = useState({ x: 0, y: 0 });

    useEffect(() => {
        const handleMouseMove = (e) => {
            setPosition({ x: e.clientX - 20, y: e.clientY - 20 });
        };
        document.addEventListener('mousemove', handleMouseMove);
        return () => document.removeEventListener('mousemove', handleMouseMove);
    }, []);

    return (
        <div
            className="cursor-follower"
            style={{ left: `${position.x}px`, top: `${position.y}px`, opacity: 1 }}
        ></div>
    );
};

const Hero = () => {
    return (
        <section className="hero">
            <div className="hero-blob"></div>
            <div className="particles-container" id="particles-hero"></div>
            <div className="container">
                <div className="hero-content">
                    <div className="hero-text">
                        <h1>
                            Revolutionizing <span className="highlight">Polymer Engineering</span> with Transformer AI
                        </h1>
                        <p className="subtitle">
                            Discover how our advanced transformer neural networks are redefining polymer design, simulation, and optimization for <span className="gradient-text">next-generation materials</span>.
                        </p>
                        <div className="cta-buttons">
                            <Link to="/about" className="btn btn-secondary">About Us</Link>
                        </div>
                        <div className="trusted-by">
                            <p className="trusted-text">Trusted by industry leaders</p>
                            <div className="trusted-logos">
                                <svg width="100" height="30" viewBox="0 0 100 30" className="trusted-logo">
                                    <rect x="5" y="10" width="90" height="10" rx="2" fill="#718096" />
                                </svg>
                                <svg width="100" height="30" viewBox="0 0 100 30" className="trusted-logo">
                                    <circle cx="15" cy="15" r="10" fill="#718096" />
                                    <rect x="30" y="10" width="60" height="10" rx="2" fill="#718096" />
                                </svg>
                                <svg width="100" height="30" viewBox="0 0 100 30" className="trusted-logo">
                                    <polygon points="15,5 25,25 5,25" fill="#718096" />
                                    <rect x="30" y="10" width="60" height="10" rx="2" fill="#718096" />
                                </svg>
                                <svg width="100" height="30" viewBox="0 0 100 30" className="trusted-logo">
                                    <rect x="5" y="5" width="20" height="20" fill="#718096" />
                                    <rect x="30" y="10" width="60" height="10" rx="2" fill="#718096" />
                                </svg>
                            </div>
                        </div>
                    </div>
                    <div className="hero-image">
                        <div className="molecule-3d">
                            <svg width="500" height="400" viewBox="0 0 500 400" xmlns="http://www.w3.org/2000/svg">
                                <defs>
                                    <linearGradient id="gradientBg" x1="0%" y1="0%" x2="100%" y2="100%">
                                        <stop offset="0%" stopColor="#1a5f7a" stopOpacity="0.1" />
                                        <stop offset="100%" stopColor="#57c4e5" stopOpacity="0.2" />
                                    </linearGradient>
                                    <linearGradient id="gradientAccent" x1="0%" y1="0%" x2="100%" y2="100%">
                                        <stop offset="0%" stopColor="#ff7a59" />
                                        <stop offset="100%" stopColor="#ff957f" />
                                    </linearGradient>
                                    <linearGradient id="gradientPrimary" x1="0%" y1="0%" x2="100%" y2="100%">
                                        <stop offset="0%" stopColor="#1a5f7a" />
                                        <stop offset="100%" stopColor="#2e7d9b" />
                                    </linearGradient>
                                </defs>
                                <circle cx="400" cy="100" r="80" fill="url(#gradientBg)" />
                                <circle cx="100" cy="350" r="60" fill="url(#gradientBg)" />
                                <g transform="translate(100, 50)">
                                    <g transform="translate(0, 0)" className="molecule-group">
                                        <circle cx="150" cy="70" r="8" fill="url(#gradientPrimary)" className="atom" />
                                        <circle cx="190" cy="100" r="8" fill="url(#gradientPrimary)" className="atom" />
                                        <circle cx="190" cy="140" r="8" fill="url(#gradientPrimary)" className="atom" />
                                        <circle cx="150" cy="170" r="8" fill="url(#gradientPrimary)" className="atom" />
                                        <circle cx="110" cy="140" r="8" fill="url(#gradientPrimary)" className="atom" />
                                        <circle cx="110" cy="100" r="8" fill="url(#gradientPrimary)" className="atom" />
                                        <line x1="150" y1="70" x2="190" y2="100" stroke="#1a5f7a" strokeWidth="3" className="bond" />
                                        <line x1="190" y1="100" x2="190" y2="140" stroke="#1a5f7a" strokeWidth="3" className="bond" />
                                        <line x1="190" y1="140" x2="150" y2="170" stroke="#1a5f7a" strokeWidth="3" className="bond" />
                                        <line x1="150" y1="170" x2="110" y2="140" stroke="#1a5f7a" strokeWidth="3" className="bond" />
                                        <line x1="110" y1="140" x2="110" y2="100" stroke="#1a5f7a" strokeWidth="3" className="bond" />
                                        <line x1="110" y1="100" x2="150" y2="70" stroke="#1a5f7a" strokeWidth="3" className="bond" />
                                        <circle cx="230" cy="120" r="8" fill="url(#gradientPrimary)" className="atom" />
                                        <line x1="190" y1="120" x2="230" y2="120" stroke="#1a5f7a" strokeWidth="3" className="bond" />
                                        <circle cx="70" cy="120" r="8" fill="url(#gradientPrimary)" className="atom" />
                                        <line x1="110" y1="120" x2="70" y2="120" stroke="#1a5f7a" strokeWidth="3" className="bond" />
                                        <circle cx="250" cy="160" r="8" fill="url(#gradientPrimary)" className="atom" />
                                        <line x1="230" y1="120" x2="250" y2="160" stroke="#1a5f7a" strokeWidth="3" className="bond" />
                                        <circle cx="50" cy="160" r="8" fill="url(#gradientPrimary)" className="atom" />
                                        <line x1="70" y1="120" x2="50" y2="160" stroke="#1a5f7a" strokeWidth="3" className="bond" />
                                        <circle cx="270" cy="90" r="8" fill="url(#gradientAccent)" className="atom special" />
                                        <line x1="230" y1="120" x2="270" y2="90" stroke="#1a5f7a" strokeWidth="3" className="bond" />
                                    </g>
                                </g>
                            </svg>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
};

const Features = () => {
    return (
        <section className="features" id="features">
            <div className="falling-elements"></div>
            <div className="container">
                <div className="section-header">
                    <h2 className="section-title">
                        Transformative <span className="gradient-text">Features</span>
                    </h2>
                    <p className="section-subtitle">
                        Our platform combines cutting-edge AI with polymer science to deliver unprecedented capabilities in material engineering.
                    </p>
                </div>
                <div className="feature-grid">
                    <div className="feature-card">
                        <div className="feature-icon"><i className="fas fa-brain"></i></div>
                        <h3 className="feature-title">Advanced AI Modeling</h3>
                        <p>Deploy transformer neural networks specifically trained on polymer structures to predict properties with accuracy exceeding traditional methods.</p>
                    </div>
                    <div className="feature-card">
                        <div className="feature-icon"><i className="fas fa-flask"></i></div>
                        <h3 className="feature-title">Rapid Prototyping</h3>
                        <p>Reduce development cycles by 70% with our AI-powered simulation engine that accurately predicts polymer behavior before physical synthesis.</p>
                    </div>
                    <div className="feature-card">
                        <div className="feature-icon"><i className="fas fa-chart-line"></i></div>
                        <h3 className="feature-title">Property Optimization</h3>
                        <p>Automatically discover optimal polymer configurations for specific applications with our multi-objective optimization algorithms.</p>
                    </div>
                    <div className="feature-card">
                        <div className="feature-icon"><i className="fas fa-cloud"></i></div>
                        <h3 className="feature-title">Cloud Integration</h3>
                        <p>Access our platform anywhere with seamless cloud integration, enabling collaborative polymer development across global teams.</p>
                    </div>
                    <div className="feature-card">
                        <div className="feature-icon"><i className="fas fa-shield-alt"></i></div>
                        <h3 className="feature-title">Data Security</h3>
                        <p>Rest easy knowing your proprietary polymer research is protected with enterprise-grade encryption and security protocols.</p>
                    </div>
                    <div className="feature-card">
                        <div className="feature-icon"><i className="fas fa-sync-alt"></i></div>
                        <h3 className="feature-title">Continuous Learning</h3>
                        <p>Our AI models continually improve with each polymer simulation, creating an ever-expanding knowledge base for better predictions.</p>
                    </div>
                </div>
            </div>
            <svg className="molecule molecule1" width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                <circle cx="50" cy="50" r="10" fill="#1a5f7a" opacity="0.3" />
                <circle cx="75" cy="50" r="10" fill="#1a5f7a" opacity="0.3" />
                <circle cx="25" cy="50" r="10" fill="#1a5f7a" opacity="0.3" />
                <line x1="50" y1="50" x2="75" y2="50" stroke="#1a5f7a" strokeWidth="3" opacity="0.3" />
                <line x1="50" y1="50" x2="25" y2="50" stroke="#1a5f7a" strokeWidth="3" opacity="0.3" />
            </svg>
            <svg className="molecule molecule2" width="80" height="80" viewBox="0 0 80 80" xmlns="http://www.w3.org/2000/svg">
                <polygon points="40,10 65,30 55,60 25,60 15,30" stroke="#1a5f7a" fill="none" strokeWidth="3" opacity="0.3" />
                <circle cx="40" cy="10" r="5" fill="#ff7a59" opacity="0.3" />
                <circle cx="65" cy="30" r="5" fill="#1a5f7a" opacity="0.3" />
                <circle cx="55" cy="60" r="5" fill="#1a5f7a" opacity="0.3" />
                <circle cx="25" cy="60" r="5" fill="#1a5f7a" opacity="0.3" />
                <circle cx="15" cy="30" r="5" fill="#1a5f7a" opacity="0.3" />
            </svg>
            <svg className="molecule molecule3" width="120" height="60" viewBox="0 0 120 60" xmlns="http://www.w3.org/2000/svg">
                <circle cx="20" cy="30" r="8" fill="#1a5f7a" opacity="0.3" />
                <circle cx="50" cy="30" r="8" fill="#1a5f7a" opacity="0.3" />
                <circle cx="80" cy="30" r="8" fill="#1a5f7a" opacity="0.3" />
                <circle cx="110" cy="30" r="8" fill="#ff7a59" opacity="0.3" />
                <line x1="20" y1="30" x2="50" y2="30" stroke="#1a5f7a" strokeWidth="3" opacity="0.3" />
                <line x1="50" y1="30" x2="80" y2="30" stroke="#1a5f7a" strokeWidth="3" opacity="0.3" />
                <line x1="80" y1="30" x2="110" y2="30" stroke="#1a5f7a" strokeWidth="3" opacity="0.3" />
            </svg>
        </section>
    );
};

const HowItWorks = () => {
    return (
        <section className="how-it-works" id="how-it-works">
            <div className="particles-container" id="particles-hiw"></div>
            <div className="container">
                <div className="section-header">
                    <h2 className="section-title">
                        How <span className="gradient-text">Transploymer</span> Works
                    </h2>
                    <p className="section-subtitle">
                        Our streamlined process makes advanced polymer engineering accessible and efficient for R&D teams of all sizes.
                    </p>
                </div>
                <div className="process-flow">
                    <div className="process-line"></div>
                    <div className="process-step">
                        <div className="step-number">1</div>
                        <h3 className="step-title">Molecular Input</h3>
                        <p className="step-description">Users provide SMILES strings or polymer names via a clean and intuitive interface. These inputs represent the structural formulae of the polymers, which serve as the starting point for analysis.</p>
                    </div>
                    <div className="process-step">
                        <div className="step-number">2</div>
                        <h3 className="step-title">Custom Tokenization</h3>
                        <p className="step-description">The input is processed through a domain-specific Byte-Pair Encoding (BPE) tokenizer, specially trained for polymer data. This ensures that the chemical structure is segmented meaningfully for Transformer-based analysis.</p>
                    </div>
                    <div className="process-step">
                        <div className="step-number">3</div>
                        <h3 className="step-title">Transformer-Based Feature Extraction</h3>
                        <p className="step-description">A custom-built Transformer model, trained from scratch, learns molecular patterns and encodes the input into rich, high-dimensional feature representations. This is the system’s core engine for understanding polymer behavior.</p>
                    </div>
                    <div className="process-step">
                        <div className="step-number">4</div>
                        <h3 className="step-title">Multi-Target Property Prediction</h3>
                        <p className="step-description">These features are passed to six parallel regression heads, each designed to predict a specific property: Eea (Electron affinity)   Egb (Energy gap)   Ei (Ionization energy)   EPS (Permittivity)   Nc (Number 0 + 0; Nc (Number of carbon atoms)   Xc (Crystallinity index)</p>
                    </div>
                    <div className="process-step">
                        <div className="step-number">5</div>
                        <h3 className="step-title">Application-Ready Output</h3>
                        <p className="step-description">Predicted properties are displayed through the MERN-stack frontend, helping researchers select polymers best suited for specific applications—whether it's high-strength materials, biocompatible surfaces, or conductive films.</p>
                    </div>
                </div>
            </div>
        </section>
    );
};

const UseCases = () => {
    return (
        <section className="use-cases" id="use-cases">
            <div className="falling-elements"></div>
            <div className="container">
                <div className="section-header">
                    <h2 className="section-title">
                        Transforming <span className="gradient-text">Industries</span>
                    </h2>
                    <p className="section-subtitle">
                        See how Transploymer is revolutionizing polymer applications across diverse sectors.
                    </p>
                </div>
                <div className="case-container from-left">
                    <div className="case-image">
                        <svg width="400" height="300" viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg">
                            <defs>
                                <linearGradient id="caseGradient1" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" stopColor="#1a5f7a" stopOpacity="0.1" />
                                    <stop offset="100%" stopColor="#57c4e5" stopOpacity="0.3" />
                                </linearGradient>
                            </defs>
                            <rect width="400" height="300" fill="url(#caseGradient1)" rx="10" />
                            <circle cx="200" cy="150" r="100" fill="white" stroke="#1a5f7a" strokeWidth="2" />
                            <circle cx="200" cy="150" r="80" fill="white" stroke="#1a5f7a" strokeWidth="2" strokeDasharray="5,5" />
                            <circle cx="200" cy="150" r="60" fill="#f8f9fa" stroke="#1a5f7a" strokeWidth="2" />
                            <path d="M170,150 Q200,120 230,150 T290,150" stroke="#ff7a59" strokeWidth="3" fill="none" />
                            <path d="M110,150 Q140,180 170,150" stroke="#ff7a59" strokeWidth="3" fill="none" />
                            <circle cx="170" cy="150" r="5" fill="#1a5f7a" />
                            <circle cx="230" cy="150" r="5" fill="#1a5f7a" />
                            <circle cx="290" cy="150" r="5" fill="#1a5f7a" />
                            <circle cx="110" cy="150" r="5" fill="#1a5f7a" />
                        </svg>
                    </div>
                    <div className="case-content">
                        <h3 className="case-title">Medical Devices</h3>
                        <ul className="case-list">
                            <li>
                                <span className="case-list-icon"><i className="fas fa-check-circle"></i></span>
                                Predict biocompatible polymers with tunable degradation kinetics for implantable systems
                            </li>
                            <li>
                                <span className="case-list-icon"><i className="fas fa-check-circle"></i></span>
                                Identify polymer candidates with surface chemistries favorable for antimicrobial activity
                            </li>
                            <li>
                                <span className="case-list-icon"><i className="fas fa-check-circle"></i></span>
                                Select high-strength, elastomeric polymers suitable for next-generation prosthetics and biomedical wearables
                            </li>
                        </ul>
                        <a href="#" className="btn btn-primary">Explore Medical Applications</a>
                    </div>
                </div>
                <div className="case-container from-right">
                    <div className="case-content">
                        <h3 className="case-title">Material Science</h3>
                        <ul className="case-list">
                            <li>
                                <span className="case-list-icon"><i className="fas fa-check-circle"></i></span>
                                Discover and engineer materials with tailored properties for specific applications across industries
                            </li>
                            <li>
                                <span className="case-list-icon"><i className="fas fa-check-circle"></i></span>
                                Optimize material design by analyzing structure–property relationships from SMILES representations
                            </li>
                            <li>
                                <span className="case-list-icon"><i className="fas fa-check-circle"></i></span>
                                Reduce trial-and-error R&D using data-driven models trained on large-scale polymer datasets
                            </li>
                        </ul>
                        <a href="#" className="btn btn-primary">Explore Material Applications</a>
                    </div>
                    <div className="case-image">
                        <svg width="400" height="300" viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg">
                            <defs>
                                <linearGradient id="caseGradient2" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" stopColor="#57c4e5" stopOpacity="0.2" />
                                    <stop offset="100%" stopColor="#1a5f7a" stopOpacity="0.1" />
                                </linearGradient>
                            </defs>
                            <rect width="400" height="300" fill="url(#caseGradient2)" rx="10" />
                            <rect x="100" y="100" width="200" height="150" rx="5" fill="white" stroke="#1a5f7a" strokeWidth="2" />
                            <rect x="120" y="70" width="160" height="30" rx="5" fill="white" stroke="#1a5f7a" strokeWidth="2" />
                            <path d="M120,70 L100,100" stroke="#1a5f7a" strokeWidth="2" fill="none" />
                            <path d="M280,70 L300,100" stroke="#1a5f7a" strokeWidth="2" fill="none" />
                            <circle cx="150" cy="150" r="30" fill="#f8f9fa" stroke="#1a5f7a" strokeWidth="2" />
                            <path d="M140,150 L160,150 M150,140 L150,160" stroke="#1a5f7a" strokeWidth="2" />
                            <circle cx="250" cy="150" r="30" fill="#f8f9fa" stroke="#1a5f7a" strokeWidth="2" />
                            <path d="M250,150 Q260,140 270,150 T290,150" stroke="#ff7a59" strokeWidth="2" fill="none" transform="scale(0.6) translate(235, 125)" />
                        </svg>
                    </div>
                </div>
            </div>
        </section>
    );
};



const CtaSection = () => {
    return (
        <section className="cta-section">
            <div className="cta-floating-circle cta-circle-1"></div>
            <div className="cta-floating-circle cta-circle-2"></div>
            <div className="container">
                <h2 className="cta-title">Ready to Transform Your Polymer Engineering?</h2>
                <p className="cta-subtitle">
                    Join leading innovators who are using Transploymer to create the materials of tomorrow, today.
                </p>
                <Link to="/login" className="cta-btn">Get Started Now</Link>
            </div>
        </section>
    );
};

const Footer = () => {
    return (
        <footer>
            <div className="container">
                <div className="footer-content">
                    <div>
                        <a href="#" className="footer-logo">
                            <span className="footer-logo-icon"><i className="fas fa-atom"></i></span>
                            TRANSPOLYMER
                        </a>
                        <p className="footer-description">
                            Transploymer is revolutionizing polymer engineering with transformer AI, enabling faster, smarter, and more sustainable material innovation.
                        </p>
                        <div className="footer-social">
                            <a href="#" className="social-icon"><i className="fab fa-twitter"></i></a>
                            <a href="#" className="social-icon"><i className="fab fa-linkedin-in"></i></a>
                            <a href="#" className="social-icon"><i className="fab fa-github"></i></a>
                        </div>
                    </div>
                    <div className="footer-links">
    
                    </div>
                </div>
               
            </div>
        </footer>
    );
};

const LandingPage = () => {


    const location = useLocation();

    useEffect(() => {
      if (location.hash) {
        const id = location.hash.replace('#', '');
        const el = document.getElementById(id);
        if (el) el.scrollIntoView({ behavior: 'smooth' });
     }
}, [location]);


    useEffect(() => {
        const animateOnScroll = () => {
            const elements = document.querySelectorAll('.feature-card, .process-step, .case-container, .cta-title, .cta-subtitle, .cta-btn');
            elements.forEach((el) => {
                const rect = el.getBoundingClientRect();
                if (rect.top <= window.innerHeight * 0.8) {
                    el.classList.add('animated');
                }
            });

            const processLine = document.querySelector('.process-line');
            if (processLine) {
                const rect = processLine.getBoundingClientRect();
                if (rect.top <= window.innerHeight * 0.8) {
                    processLine.classList.add('animated');
                }
            }
        };

        window.addEventListener('scroll', animateOnScroll);
        animateOnScroll();

        const fallingContainers = document.querySelectorAll('.falling-elements');
        fallingContainers.forEach((container) => {
            for (let i = 0; i < 10; i++) {
                const element = document.createElement('div');
                const type = ['hexagon', 'circle', 'square'][Math.floor(Math.random() * 3)];
                element.classList.add('falling-element', type);
                if (type === 'hexagon') {
                    element.innerHTML = `<svg viewBox="0 0 40 35"><polygon points="20,5 35,15 35,30 20,35 5,30 5,15" stroke="#1a5f7a" fill="none" stroke-width="2" /></svg>`;
                }
                element.style.left = `${Math.random() * 100}%`;
                element.style.animationDelay = `${Math.random() * 10}s`;
                container.appendChild(element);
            }
        });

        const particleContainers = document.querySelectorAll('.particles-container');
        particleContainers.forEach((container) => {
            for (let i = 0; i < 20; i++) {
                const particle = document.createElement('div');
                particle.classList.add('particle');
                particle.style.width = `${Math.random() * 5 + 5}px`;
                particle.style.height = particle.style.width;
                particle.style.backgroundColor = `rgba(87, 196, 229, ${Math.random() * 0.3 + 0.1})`;
                particle.style.left = `${Math.random() * 100}%`;
                particle.style.top = `${Math.random() * 100}%`;
                particle.style.animation = `float ${Math.random() * 10 + 5}s ease-in-out infinite alternate`;
                container.appendChild(particle);
            }
        });

        return () => {
            window.removeEventListener('scroll', animateOnScroll);
        };
    }, []);

    return (
        <>
            <Loader />
            <CursorFollower />
            <Hero />
            <Features />
            <HowItWorks />
            <UseCases />
            <CtaSection />
            <Footer />
        </>
    );
};

export default LandingPage;