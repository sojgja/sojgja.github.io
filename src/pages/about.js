import React from 'react';
import Layout from '@theme/Layout';
import styles from './about.module.css';

const experience = [
  {
    period: '2024 – Now',
    role: 'Algo Trading Developer',
    company: 'Freelancer',
    desc: 'Design and deploy high-frequency cryptocurrency trading bots on Binance, Bybit, OKX. Build strategy backtesting engines with pandas/numpy, real-time market data pipelines via WebSocket, and risk management systems with dynamic position sizing and stop-loss automation. Achieve consistent returns through systematic quantitative strategies.',
  },
  {
    period: '2022 – 2024',
    role: 'Team Lead — Backend',
    company: 'NTQ Solution JSC',
    desc: 'Led a team of 5 engineers building a social network platform with real-time voice calling. Architected WebSocket infrastructure handling 10K+ concurrent connections with sub-100ms latency. Designed RESTful APIs, optimized PostgreSQL queries reducing response times by 60%, and implemented CI/CD pipelines cutting deployment time from hours to minutes.',
  },
  {
    period: '2021 – 2022',
    role: 'Tester, BA, Pre-Sale',
    company: 'CMC Global',
    desc: 'Bridged the gap between clients and engineering teams. Analyzed functional/non-functional requirements for enterprise projects, designed test strategies, and led code reviews ensuring 95%+ coverage. Contributed to pre-sale proposals that won 3 major contracts.',
  },
  {
    period: '2019 – 2021',
    role: 'Full-stack Developer, BA',
    company: 'System-Gear Vietnam',
    desc: 'Delivered end-to-end ERP-Odoo solutions for small and medium businesses — from requirement gathering to deployment. Customized accounting, inventory, and HR modules. Built a cross-platform mobile app for workforce management used by 500+ daily active users.',
  },
  {
    period: '2016 – 2019',
    role: 'Founder & Business Owner',
    company: 'Self-employed',
    desc: 'Founded and operated a villa design & construction company. Managed 20+ employees across sales, design, and construction. Grew revenue 3x in two years through systematic operations and customer-centric approach. This entrepreneurial journey sharpened my business acumen and leadership.',
  },
  {
    period: '2014 – 2016',
    role: 'IT Manager',
    company: 'Hoang Son',
    desc: 'Oversaw IT infrastructure for a furniture & electronics retail chain with 5 branches. Architected and deployed a centralized sales management system integrating POS, inventory, and accounting — reducing manual work by 70%. Built reporting dashboards for executive decision-making.',
  },
  {
    period: '2009 – 2014',
    role: 'Developer, Team Leader',
    company: 'FPT Software',
    desc: 'Started as a developer and grew to lead teams of up to 8 engineers. Designed system architecture for enterprise applications serving Japanese clients. Specialized in backend development, API design, and database optimization. Mentored 12+ junior engineers, many of whom became senior developers.',
  },
];

const skills = [
  { cat: 'Architecture', items: ['System Design', 'Software Architecture', 'Solution Architecture', 'Microservices'] },
  { cat: 'Cloud & Infra', items: ['AWS', 'Docker', 'CI/CD', 'Linux'] },
  { cat: 'Backend', items: ['RESTful API', 'WebSocket', 'High Concurrency', 'Database Design'] },
  { cat: 'Leadership', items: ['Team Leadership', 'Code Review', 'Mentoring', 'Agile/Scrum'] },
  { cat: 'Domain', items: ['Algo Trading', 'System Design', 'ERP-Odoo', 'Quantitative Finance'] },
];

export default function About() {
  return (
    <Layout title="About" description="Nguyen Doan Cuong - Senior Backend & Algo Trading Developer">
      <main className={styles.page}>
        <div className={styles.container}>
          {/* Header */}
          <header className={styles.header}>
            <div className={styles.avatar}>
              <span>NC</span>
            </div>
            <h1 className={styles.name}>Nguyen Doan Cuong</h1>
            <p className={styles.headline}>Senior Backend Engineer & Algo Trading Developer</p>
            <p className={styles.subhead}>16 years of building — from enterprise systems at FPT Software to algorithmic trading bots and leading engineering teams</p>
            <div className={styles.contact}>
              <a href="tel:0906961286" className={styles.link}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{verticalAlign:'middle',marginRight:6}}><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/></svg>
                090 696 1286
              </a>
              <a href="https://www.linkedin.com/in/cuong-nd/" target="_blank" rel="noopener" className={styles.link}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style={{verticalAlign:'middle',marginRight:6}}><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
                LinkedIn
              </a>
            </div>
          </header>

          {/* Intro */}
          <section className={styles.section}>
            <p className={styles.intro}>
              With over 15 years in software engineering, I've progressed from writing production code at <strong>FPT Software</strong> to architecting enterprise systems at <strong>NTQ Solution</strong>, and now designing cutting-edge <strong>algorithmic trading platforms</strong>. My focus has shifted from coding to <strong>system architecture, solution design, and technical leadership</strong>.
            </p>
            <p className={styles.intro}>
              I specialize in <strong>software architecture</strong> — designing scalable, resilient systems that handle millions of requests. I define technical roadmaps, evaluate trade-offs between monolith vs microservices, choose the right infrastructure (AWS, Docker, CI/CD), and ensure systems are built for performance, reliability, and maintainability.
            </p>
            <p className={styles.intro}>
              Beyond architecture, I bring deep domain expertise in <strong>algorithmic trading</strong> and <strong>quantitative finance</strong>. Combined with entrepreneurial experience from founding my own company, I solve problems at the intersection of business, technology, and strategy.
            </p>
          </section>

          {/* Skills */}
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Skills</h2>
            <div className={styles.skillsCol}>
              {skills.map((g) => (
                <div key={g.cat} className={styles.skillRow}>
                  <span className={styles.skillLabel}>{g.cat}</span>
                  <div className={styles.skillTags}>
                    {g.items.map((s) => <span key={s} className={styles.skillTag}>{s}</span>)}
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Experience */}
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Experience</h2>
            <div className={styles.timeline}>
              {experience.map((item, i) => (
                <div key={i} className={styles.timelineItem}>
                  <div className={styles.timelineDot} />
                  <div className={styles.timelineContent}>
                    <span className={styles.timelinePeriod}>{item.period}</span>
                    <h3 className={styles.timelineRole}>{item.role}</h3>
                    <p className={styles.timelineCompany}>{item.company}</p>
                    <p className={styles.timelineDesc}>{item.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Education */}
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Education</h2>
            <div className={styles.eduGrid}>
              <div className={styles.eduCard}>
                <span className={styles.eduYear}>2012 – 2014</span>
                <h3>Master of Business Administration</h3>
                <p>Thuong Mai University</p>
              </div>
              <div className={styles.eduCard}>
                <span className={styles.eduYear}>2004 – 2008</span>
                <h3>Bachelor of Software Engineering</h3>
                <p>Industrial University of Ho Chi Minh City</p>
              </div>
            </div>
          </section>

          {/* Interests */}
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Interests</h2>
            <div className={styles.interests}>
              <span className={styles.interestTag}>Buddhism</span>
              <span className={styles.interestTag}>Algorithmic Trading</span>
              <span className={styles.interestTag}>Software Architecture</span>
              <span className={styles.interestTag}>Quantitative Finance</span>
              <span className={styles.interestTag}>Open Source</span>
            </div>
          </section>

          {/* Footer */}
          <footer className={styles.footer}>
            <p>Always building. Always learning. Always shipping.</p>
          </footer>
        </div>
      </main>
    </Layout>
  );
}