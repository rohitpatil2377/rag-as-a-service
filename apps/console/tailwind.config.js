// RAG as a Service console — extends the shared Acme brand preset (design/tokens), so the
// console inherits the exact palette / Sora+Inter / spacing / radii from the single
// source of truth. See design/tokens/README.md.
import brandPreset from '../../design/tokens/tailwind.preset.js'

/** @type {import('tailwindcss').Config} */
export default {
  presets: [brandPreset],
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      boxShadow: {
        card: '0 4px 6px -1px rgba(13,27,58,.04)',
        cardHover: '0 10px 25px -5px rgba(13,27,58,.08)',
      },
    },
  },
  plugins: [],
}
