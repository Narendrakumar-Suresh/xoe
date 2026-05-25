import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'xoe',
  description: 'Write PyTorch. Run JAX.',

  themeConfig: {
    logo: '/logo.png',
    siteTitle: false,
    nav: [
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'API', link: '/api/tensor' },
      { text: 'GitHub', link: 'https://github.com/Narendrakumar-Suresh/xoe' },
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'Guide',
          items: [
            { text: 'Getting Started', link: '/guide/getting-started' },
            { text: 'Tensors', link: '/guide/tensors' },
            { text: 'Autograd', link: '/guide/autograd' },
            { text: 'Training Loop', link: '/guide/training-loop' },
          ],
        },
      ],
      '/api/': [
        {
          text: 'API Reference',
          items: [
            { text: 'Tensor', link: '/api/tensor' },
            { text: 'nn', link: '/api/nn' },
            { text: 'Optim', link: '/api/optim' },
            { text: 'Random', link: '/api/random' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/Narendrakumar-Suresh/xoe' },
    ],
  },

  appearance: 'dark',
})
