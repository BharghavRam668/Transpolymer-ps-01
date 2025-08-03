import React from 'react';
import './App.css'
function AboutUs() {
  return (
    <div className="about-container">
      <section className="about-section">
        <h3>Our Mission</h3>
        <p>
        The TransPolymer project aims to revolutionize polymer design by leveraging a Transformer-based language model to predict critical polymer properties, such as electrolyte conductivity, band gap, and dielectric constant, with high accuracy and efficiency. By tokenizing polymer sequences using chemically aware SMILES notation and pretraining on a vast dataset of 5 million sequences, followed by finetuning on specific benchmarks, TransPolymer enables rapid, data-driven material development, reducing the reliance on costly and time-intensive experimental methods. The project outperforms traditional models like Random Forests and Graph Neural Networks, achieving state-of-the-art results (e.g., an R² of 0.69 for polymer electrolyte conductivity) across ten benchmarks, demonstrating strong generalization and less overfitting. Ultimately, TransPolymer seeks to advance polymer informatics, guiding rational design for applications like energy storage and sustainable materials, though its real-world impact hinges on experimental validation of its computational predictions.
        </p>
      </section>

      <section className="about-section">
        <h3>The Technology</h3>
        <div className="tech-cards">
          <div className="tech-card">
            <h4>Machine Learning</h4>
            <p>
              Our platform utilizes state-of-the-art machine learning algorithms trained on vast databases of polymer
              structures and properties. These models can predict various properties with high accuracy, reducing the
              need for extensive laboratory testing.
            </p>
          </div>
          <div className="tech-card">
            <h4>Molecular Dynamics</h4>
            <p>
              We integrate molecular dynamics simulations to provide atomic-level insights into polymer behavior,
              allowing for detailed understanding of structure-property relationships.
            </p>
          </div>
          <div className="tech-card">
            <h4>QSPR Models</h4>
            <p>
              Quantitative Structure-Property Relationship models establish connections between polymer chemical
              structures and their macroscopic properties, enabling precise prediction of thermal, mechanical, and
              electrical characteristics.
            </p>
          </div>
        </div>
      </section>

      <section className="about-section">
        <h3>Our Team</h3>
        <div className="team-members">
          <div className="team-member">
            <div className="member-photo placeholder"></div>
            <h4>Katkuri Pardiv Reddy</h4>
            <p className="member-title"></p>
            <p className="member-bio">
              Computer Science and Engineering
            </p>
          </div>
          <div className="team-member">
            <div className="member-photo placeholder"></div>
            <h4>Prathik Reddy Mettu</h4>
            <p className="member-title"></p>
            <p className="member-bio">
            Computer Science and Engineering with Internet of Things 
            </p>
          </div>
          <div className="team-member">
            <div className="member-photo placeholder"></div>
            <h4>A.Bharghav Ram </h4>
            <p className="member-title"></p>
            <p className="member-bio">
              Computer Science and Engineering with Data Science
            </p>
          </div>
          <div className="team-member">
            <div className="member-photo placeholder"></div>
            <h4>Terala Samiksha Gupta</h4>
            <p className="member-title"></p>
            <p className="member-bio">
            Computer Science and Engineering with Data Science
            </p>
          </div>
          <div className="team-member">
            <div className="member-photo placeholder"></div>
            <h4>D.L Sravya</h4>
            <p className="member-title"></p>
            <p className="member-bio">
            Computer Science and Engineering with Data Science
            </p>
          </div>
        </div>
      </section>
      <section className="about-section contact-section">
        <h3>Contact Us</h3>
        <p>
          Interested in learning more about TransPolymer or exploring potential collaborations? Reach out to our team at{' '}
          <a href="mailto:info@transpolymer.com">info@transpolymer.com</a>
        </p>
        <div className="office-locations">
          <div className="office">
            <h4>Keshav Memorial college of Engineering</h4>
            <address>
              Ibrahimpatnam<br />
              Hyderabad<br />
            </address>
          </div>
        </div>
      </section>
    </div>
  );
}

export default AboutUs;