// @ts-check
const { themes } = require('prism-react-renderer');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'soi gia',
  tagline: '',
  favicon: 'img/favicon.svg',

  url: process.env.SITE_URL || 'https://sojgja.github.io',
  baseUrl: '/',
  organizationName: 'sojgja',
  projectName: 'sojgja.github.io',

  onBrokenLinks: 'ignore',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          editUrl: 'https://github.com/sojgja/sojgja.github.io/tree/master/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  plugins: [
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      /** @type {import('@easyops-cn/docusaurus-search-local').PluginOptions} */
      ({
        hashed: true,
        language: ['en'],
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
        indexDocs: true,
        indexBlog: false,
        indexPages: false,
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      docs: {
        sidebar: {
          hideable: true,
        },
      },
      image: 'img/social-card.svg',
      navbar: {
        title: 'sojgja',
        logo: {
          alt: 'sojgja logo',
          src: 'img/logo.svg',
        },
        items: [
          { to: '/docs/2026/hello-2026', label: 'Help', position: 'left' },
          { to: '/docs/book/book-intro', label: 'Book', position: 'left' },
          { to: '/docs/series/series-intro', label: 'Series', position: 'left' },
          { to: '/about', label: 'About', position: 'left' },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: '📅 2026',
            items: [
              { label: '👋 Hello 2026', to: '/docs/2026/hello-2026' },
              { label: '🎯 Humble Object', to: '/docs/2026/humble-object' },
            ],
          },
          {
            title: '📚 Book',
            items: [
              { label: '❤️ Lời nói đầu', to: '/docs/book/book-intro' },
              { label: '🧹 Clean Code', to: '/docs/book/clean-code' },
              { label: '🧠 Pragmatic Programmer', to: '/docs/book/pragmatic-programmer' },
            ],
          },
          {
            title: '🧭 Navigate',
            items: [
              { label: '🏠 Home', to: '/' },
              { label: '👤 About', to: '/about' },
              { label: '⚡ Quick Refs', to: '/docs/category/-quick-refs' },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} sojgja. Built with Docusaurus.`,
      },
      prism: {
        theme: themes.github,
        darkTheme: themes.dracula,
      },
    }),
};

module.exports = config;
