import React from 'react';
import Layout from '@theme/Layout';
import styles from './about.module.css';

const timeline = [
  {
    period: 'Jun 2024 - Present',
    company: 'Freelancer',
    role: 'Algo Trading Developer',
    tasks: [
      'Design and deploy cryptocurrency trading bot systems',
      'Operate and optimize cryptocurrency trading strategies',
      'Develop indicators/strategies for trading bots',
    ],
  },
  {
    period: 'Oct 2022 - Jun 2024',
    company: 'NTQ Solution JSC',
    role: 'Team Lead',
    tasks: [
      'Develop backend for social network platform and voice call system',
      'Design and implement API and WebSocket for real-time communication',
      'Build socket system for fast and stable data transmission',
      'Optimize system performance, reduce latency',
      'Lead development team, code review, testing',
    ],
  },
  {
    period: 'Oct 2021 - Oct 2022',
    company: 'CMC Global',
    role: 'Tester, BA, Pre-Sale',
    tasks: [
      'Evaluate functional/non-functional requirements from customers',
      'Analyze, design, and implement business requirements',
      'Testing, code review, document review',
    ],
  },
  {
    period: 'Oct 2019 - Oct 2021',
    company: 'System-Gear Vietnam',
    role: 'Developer, Tester, BA, Pre-Sale',
    tasks: [
      'Consult and implement ERP-Odoo for small businesses',
      'Integrate Odoo with third-party systems',
      'Develop work management mobile application',
    ],
  },
  {
    period: 'Oct 2016 - Oct 2019',
    company: 'Self-employed',
    role: 'Business Owner',
    tasks: [
      'Build and operate villa & spiritual works design/construction business',
      'Manage business, finance, marketing, sales, operations',
      'Directly consult, design, supervise construction',
    ],
  },
  {
    period: 'Sep 2014 - Sep 2016',
    company: 'Hoang Son',
    role: 'IT Manager',
    tasks: [
      'Manage IT systems for furniture & appliance supermarket chain',
      'Build and deploy sales management software',
      'Technical support and employee training',
    ],
  },
  {
    period: 'Sep 2009 - Sep 2014',
    company: 'FPT Software',
    role: 'Developer, Tester, Designer, Team Leader',
    tasks: [
      'Analyze customer requirements, evaluate feasibility',
      'Design system architecture, backend programming, API development',
      'Unit testing, integration testing, code review',
      'Write technical documentation and user manuals',
    ],
  },
];

const desiredRoles = [
  { title: 'Senior Backend Engineer', sub: 'Python/Django', desc: 'Build high-performance backend, microservices architecture, integrate AI in algo trading.' },
  { title: 'Algo Trading Developer', sub: 'Forex, Crypto, Options', desc: 'Design automated trading bots, strategy backtesting, risk management systems.' },
  { title: 'System Architect', sub: 'Trading & eCommerce', desc: 'Architect scalable financial platforms, CI/CD optimization, tech stack leadership.' },
  { title: 'Software Engineer', sub: 'Odoo & ERP', desc: 'Customize ERP-Odoo, workflow automation, deploy ERP for SMBs.' },
];

const skills = ['Python', 'C#.NET', 'PHP', 'Django', 'RESTful API', 'WebSocket', 'PostgreSQL', 'Redis', 'Docker', 'CI/CD', 'Git', 'BDD', 'TDD', 'ERP-Odoo', 'Algo Trading'];

const projects = [
  'Cryptocurrency and options trading bot',
  'Forex trading bot (FTMO, The5ers)',
  'Analysis, reporting & risk management tools',
  'Python packages for testing',
  'Python packages for trading strategy backtesting',
  'Rule engines for trading strategy optimization',
];

export default function About() {
  return (
    <Layout title="About" description="Nguyen Doan Cuong - Senior Backend & Algo Trading Developer">
      <main className={styles.page}>
        {/* Hero */}
        <section className={styles.hero}>
          <div className="container">
            <h1 className={styles.name}>Nguyen Doan Cuong</h1>
            <p className={styles.tagline}>10+ years in Software Development — Algo Trading · Backend · System Architecture</p>
            <div className={styles.contact}>
              <a href="mailto:cuongnd86@gmail.com" className={styles.contactLink}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 7-10 7L2 7"/></svg>
                cuongnd86@gmail.com
              </a>
              <span className={styles.contactLink}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/></svg>
                090 696 1286
              </span>
              <a href="https://www.linkedin.com/in/cuong-nd/" target="_blank" rel="noopener" className={styles.contactLink}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
                LinkedIn
              </a>
            </div>
          </div>
        </section>

        {/* Intro */}
        <section className={styles.section}>
          <div className="container">
            <h2 className={styles.sectionTitle}>Introduction</h2>
            <p className={styles.introText}>
              With over 10 years in software development, I focus on building advanced technology solutions 
              in <strong>Algorithmic Trading</strong>, <strong>eCommerce</strong>, <strong>ERP</strong>, and <strong>Blockchain</strong>. 
              I bring hands-on experience from FPT Software to leading teams at NTQ Solution, 
              combined with deep domain knowledge in financial markets and automated trading systems.
            </p>
          </div>
        </section>

        {/* Desired Roles */}
        <section className={styles.sectionAlt}>
          <div className="container">
            <h2 className={styles.sectionTitle}>Looking For</h2>
            <div className={styles.roleGrid}>
              {desiredRoles.map((role) => (
                <div key={role.title} className={styles.roleCard}>
                  <h3>{role.title}</h3>
                  <span className={styles.roleSub}>{role.sub}</span>
                  <p>{role.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Skills */}
        <section className={styles.section}>
          <div className="container">
            <h2 className={styles.sectionTitle}>Technical Skills</h2>
            <div className={styles.skillTags}>
              {skills.map((s) => (
                <span key={s} className={styles.skillTag}>{s}</span>
              ))}
            </div>
          </div>
        </section>

        {/* Timeline */}
        <section className={styles.sectionAlt}>
          <div className="container">
            <h2 className={styles.sectionTitle}>Work Experience</h2>
            <div className={styles.timeline}>
              {timeline.map((item, i) => (
                <div key={i} className={styles.timelineItem}>
                  <div className={styles.timelineDot} />
                  <div className={styles.timelineContent}>
                    <span className={styles.timelinePeriod}>{item.period}</span>
                    <h3 className={styles.timelineRole}>{item.role} <span>at {item.company}</span></h3>
                    <ul className={styles.timelineTasks}>
                      {item.tasks.map((t, j) => <li key={j}>{t}</li>)}
                    </ul>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Education */}
        <section className={styles.section}>
          <div className="container">
            <h2 className={styles.sectionTitle}>Education</h2>
            <div className={styles.eduGrid}>
              <div className={styles.eduCard}>
                <span className={styles.eduYear}>2012 – 2014</span>
                <h3>Master of Business Administration</h3>
                <p>TMU (Thuong Mai University)</p>
              </div>
              <div className={styles.eduCard}>
                <span className={styles.eduYear}>2004 – 2008</span>
                <h3>Software Engineer</h3>
                <p>Industrial University</p>
              </div>
            </div>
          </div>
        </section>

        {/* Projects */}
        <section className={styles.sectionAlt}>
          <div className="container">
            <h2 className={styles.sectionTitle}>Personal Projects</h2>
            <div className={styles.projectGrid}>
              {projects.map((p, i) => (
                <div key={i} className={styles.projectCard}>
                  <span className={styles.projectNum}>{String(i + 1).padStart(2, '0')}</span>
                  <p>{p}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Interests */}
        <section className={styles.section}>
          <div className="container">
            <h2 className={styles.sectionTitle}>Interests</h2>
            <div className={styles.interestRow}>
              <span className={styles.interestTag}>Buddhism</span>
              <span className={styles.interestTag}>Trading</span>
              <span className={styles.interestTag}>Programming</span>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
