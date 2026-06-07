/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Trading System',
      items: ['trading-system'],
    },
    {
      type: 'category',
      label: 'Backend',
      items: ['django-backend'],
    },
    {
      type: 'category',
      label: 'SDK & Patterns',
      items: ['sdk-design-pattern'],
    },
  ],
};

module.exports = sidebars;
