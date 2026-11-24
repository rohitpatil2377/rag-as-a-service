// Acme Tailwind preset — consumed by apps/console (Phase-1) so the UI inherits the
// brand system from day one. Authority: Acme Brand & Identity Guidelines Book v2.
// Generated from tokens.json — do not hand-edit.
module.exports = {
  theme: {
    extend: {
      colors: {
        acme: {
          white: '#FFFFFF', ivory: '#FAF7F2',
          saffron: '#E87722', 'saffron-light': '#FFA940', 'saffron-deep': '#C95F0E',
          emerald: '#0F7B3E', 'emerald-light': '#2E9D5F', 'emerald-deep': '#0A5C2D',
          navy: '#1A2B4A', ink: '#0D1B3A',
          gold: '#C9A961', 'gold-light': '#E8D094', teal: '#0EA5E9',
        },
      },
      fontFamily: {
        display: ['Sora', 'sans-serif'],   // H1–H6 (locked)
        body: ['Inter', 'sans-serif'],      // body/UI (locked)
        mono: ['JetBrains Mono', 'monospace'],
      },
      borderRadius: { sm: '6px', md: '8px', lg: '12px' }, // max 12px
      spacing: { 1: '4px', 2: '8px', 3: '12px', 4: '16px', 5: '24px', 6: '32px', 7: '48px', 8: '64px', 9: '96px' },
    },
  },
};
