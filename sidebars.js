/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  docsSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Backend',
      collapsible: true,
      collapsed: false,
      items: ['backend/django', 'backend/api-design'],
    },
    {
      type: 'category',
      label: 'SQL',
      collapsible: true,
      collapsed: false,
      items: ['sql/oracle-minus', 'sql/join-patterns'],
    },
    {
      type: 'category',
      label: 'Trading',
      collapsible: true,
      collapsed: false,
      items: ['trading/orderbook', 'trading/strategy'],
    },
    {
      type: 'category',
      label: 'System Design',
      collapsible: true,
      collapsed: false,
      items: ['system-design/architecture'],
    },
    {
      type: 'category',
      label: 'SDK & Patterns',
      collapsible: true,
      collapsed: false,
      items: ['sdk/design-patterns'],
    },
  ],
};

module.exports = sidebars;
