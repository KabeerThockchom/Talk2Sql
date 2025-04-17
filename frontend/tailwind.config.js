/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: {
          light: '#FAFAF7',
          dark: '#191919'
        },
        primary: {
          DEFAULT: '#CC785C',
          hover: '#B86A50'
        },
        secondary: {
          DEFAULT: '#CC785C',
          hover: '#B86A50'
        },
        neutral: {
          100: '#FAFAF7',
          200: '#F0F0EB',
          300: '#E5E4DF',
          400: '#BFBFBA',
          500: '#91918D',
          600: '#666663',
          700: '#40403E',
          800: '#262625',
          900: '#191919'
        },
        accent: {
          kraft: '#D4A27F',
          manila: '#EBDBBC'
        }
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        serif: ['Tiempos Text', 'serif']
      },
      boxShadow: {
        'input': '0 1px 2px rgba(0, 0, 0, 0.05)',
        'card': '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
      }
    },
  },
  plugins: [],
};