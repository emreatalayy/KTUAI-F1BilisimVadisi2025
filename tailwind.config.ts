import type { Config } from 'tailwindcss'

export default {
  darkMode: 'class',
  content: [
    './index.html',
    './src/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        arka: '#F7F7F8',
        kart: '#FFFFFF',
        metin: '#111111',
        metinIkincil: '#6B7280',
        vurgu: '#111111',
        cizgi: '#E5E7EB',
        basari: '#16A34A',
        uyarı: '#F59E0B',
        hata: '#DC2626',
      },
      boxShadow: {
        hafif: '0 4px 20px rgba(17,17,17,0.08)',
      },
      borderRadius: {
        kart: '24px',
        input: '20px',
        buton: '32px',
        toast: '16px',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', 'Apple Color Emoji', 'Segoe UI Emoji'],
      },
      spacing: {
        13: '52px',
      },
      fontSize: {
        body: ['16px', '24px'],
        heading: ['24px', '32px'],
        label: ['13px', '18px'],
      },
      screens: {
        'lg2': '1280px',
      },
      maxWidth: {
        1200: '1200px',
      }
    },
  },
  plugins: [],
} satisfies Config


