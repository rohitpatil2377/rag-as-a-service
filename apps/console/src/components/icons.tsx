// Inline SVG icon set (sovereign — no icon font / no external requests).
// Brand §13: 24px grid, 1.75 stroke, rounded caps/joins, currentColor.
import type { SVGProps } from 'react'

type P = SVGProps<SVGSVGElement> & { size?: number }

function Svg({ size = 20, children, ...p }: P & { children: React.ReactNode }) {
  return (
    <svg
      width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round" aria-hidden {...p}
    >
      {children}
    </svg>
  )
}

export const Icon = {
  dashboard: (p: P) => <Svg {...p}><rect x="3" y="3" width="7" height="9" rx="1" /><rect x="14" y="3" width="7" height="5" rx="1" /><rect x="14" y="12" width="7" height="9" rx="1" /><rect x="3" y="16" width="7" height="5" rx="1" /></Svg>,
  database: (p: P) => <Svg {...p}><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5" /><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" /></Svg>,
  upload: (p: P) => <Svg {...p}><path d="M12 16V4M7 9l5-5 5 5" /><path d="M4 20h16" /></Svg>,
  ask: (p: P) => <Svg {...p}><circle cx="11" cy="11" r="7" /><path d="M21 21l-4-4" /></Svg>,
  review: (p: P) => <Svg {...p}><path d="M9 11l3 3 8-8" /><path d="M20 12v6a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h9" /></Svg>,
  audit: (p: P) => <Svg {...p}><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" /><path d="M14 3v6h6M8 13h8M8 17h5" /></Svg>,
  shield: (p: P) => <Svg {...p}><path d="M12 2l8 4v6c0 5-3.5 8-8 10-4.5-2-8-5-8-10V6z" /><path d="M9 12l2 2 4-4" /></Svg>,
  check: (p: P) => <Svg {...p}><path d="M20 6L9 17l-5-5" /></Svg>,
  flag: (p: P) => <Svg {...p}><path d="M4 21V4M4 4h13l-2 4 2 4H4" /></Svg>,
  minus: (p: P) => <Svg {...p}><circle cx="12" cy="12" r="9" /><path d="M8 12h8" /></Svg>,
  alert: (p: P) => <Svg {...p}><path d="M12 9v4M12 17h.01" /><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" /></Svg>,
  x: (p: P) => <Svg {...p}><path d="M18 6L6 18M6 6l12 12" /></Svg>,
  sun: (p: P) => <Svg {...p}><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5 19 19M19 5l-1.5 1.5M6.5 17.5 5 19" /></Svg>,
  moon: (p: P) => <Svg {...p}><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" /></Svg>,
  logout: (p: P) => <Svg {...p}><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" /><path d="M16 17l5-5-5-5M21 12H9" /></Svg>,
  chevron: (p: P) => <Svg {...p}><path d="M9 18l6-6-6-6" /></Svg>,
  sidebar: (p: P) => <Svg {...p}><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M9 4v16" /></Svg>,
  eye: (p: P) => <Svg {...p}><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" /><circle cx="12" cy="12" r="3" /></Svg>,
  eyeOff: (p: P) => <Svg {...p}><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.4 18.4 0 0 1 5.06-5.94M9.9 4.24A9.1 9.1 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" /><path d="M1 1l22 22" /></Svg>,
  doc: (p: P) => <Svg {...p}><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" /><path d="M14 3v6h6" /></Svg>,
  quote: (p: P) => <Svg {...p}><path d="M7 7h4v6H7z M13 7h4v6h-4z" /></Svg>,
  connectors: (p: P) => <Svg {...p}><path d="M9 2v6M15 2v6" /><rect x="7" y="8" width="10" height="6" rx="2" /><path d="M12 14v4a3 3 0 0 0 3 3h2" /></Svg>,
  settings: (p: P) => <Svg {...p}><line x1="4" y1="7" x2="20" y2="7" /><circle cx="9" cy="7" r="2.4" /><line x1="4" y1="17" x2="20" y2="17" /><circle cx="15" cy="17" r="2.4" /></Svg>,
  activity: (p: P) => <Svg {...p}><path d="M3 12h4l3 8 4-16 3 8h4" /></Svg>,
  cloud: (p: P) => <Svg {...p}><path d="M7 18a4 4 0 0 1 0-8 5 5 0 0 1 9.6-1.5A3.5 3.5 0 0 1 18 18z" /></Svg>,
  key: (p: P) => <Svg {...p}><circle cx="7.5" cy="15.5" r="4.5" /><path d="M10.7 12.3 21 2M17 6l2 2M14 9l2 2" /></Svg>,
  layers: (p: P) => <Svg {...p}><path d="M12 3 3 8l9 5 9-5-9-5z" /><path d="M3 13l9 5 9-5M3 16.5l9 5 9-5" /></Svg>,
  plus: (p: P) => <Svg {...p}><path d="M12 5v14M5 12h14" /></Svg>,
  info: (p: P) => <Svg {...p}><circle cx="12" cy="12" r="9" /><path d="M12 11v5M12 8h.01" /></Svg>,
  play: (p: P) => <Svg {...p}><path d="M8 5v14l11-7z" /></Svg>,
  refresh: (p: P) => <Svg {...p}><path d="M20 11a8 8 0 1 0-2.3 5.6" /><path d="M20 4v6h-6" /></Svg>,
  edit: (p: P) => <Svg {...p}><path d="M12 20h9" /><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z" /></Svg>,
  stop: (p: P) => <Svg {...p}><rect x="6" y="6" width="12" height="12" rx="1.5" /></Svg>,
  copy: (p: P) => <Svg {...p}><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15V5a2 2 0 0 1 2-2h8" /></Svg>,
  thumbUp: (p: P) => <Svg {...p}><path d="M7 11v9H4a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1z" /><path d="M7 11l4-8a2 2 0 0 1 2 1.5V9h5a2 2 0 0 1 2 2.3l-1.2 6A2 2 0 0 1 16.8 19H7" /></Svg>,
  thumbDown: (p: P) => <Svg {...p}><path d="M17 13V4h3a1 1 0 0 1 1 1v7a1 1 0 0 1-1 1z" /><path d="M17 13l-4 8a2 2 0 0 1-2-1.5V15H6a2 2 0 0 1-2-2.3l1.2-6A2 2 0 0 1 7.2 5H17" /></Svg>,
  book: (p: P) => <Svg {...p}><path d="M4 4v15a1 1 0 0 0 1 1h13" /><path d="M6 4h11a1 1 0 0 1 1 1v13H7a1 1 0 0 1-1-1z" /></Svg>,
  scale: (p: P) => <Svg {...p}><path d="M12 3v18M7 21h10" /><path d="M12 6l-7 2 3 5a3 3 0 0 1-6 0l3-5M12 6l7 2-3 5a3 3 0 0 0 6 0l-3-5" /></Svg>,
  // --- Navigation-distinct glyphs -----------------------------------------
  // Every sidebar module owns ONE glyph (a repeated icon makes the rail unreadable),
  // so these exist purely to break former collisions: Knowledge Graph / Data & SQL /
  // Workflows / Quality / Federation / Agents / Team / Ask each got their own.
  graph: (p: P) => <Svg {...p}><circle cx="12" cy="5" r="2.4" /><circle cx="5.5" cy="18" r="2.4" /><circle cx="18.5" cy="18" r="2.4" /><path d="M10.6 7 7 15.7M13.4 7 17 15.7M8 18h8" /></Svg>,
  chat: (p: P) => <Svg {...p}><path d="M20 15a3 3 0 0 1-3 3H9l-5 3.5V6a3 3 0 0 1 3-3h10a3 3 0 0 1 3 3z" /><path d="M8.5 10.5h.01M12 10.5h.01M15.5 10.5h.01" /></Svg>,
  table: (p: P) => <Svg {...p}><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M3 9.5h18M3 15h18M9.5 9.5V20" /></Svg>,
  workflow: (p: P) => <Svg {...p}><rect x="3" y="3" width="7" height="5" rx="1.5" /><rect x="3" y="16" width="7" height="5" rx="1.5" /><rect x="14" y="16" width="7" height="5" rx="1.5" /><path d="M6.5 8v8" /><path d="M6.5 12h9a2 2 0 0 1 2 2v2" /></Svg>,
  gauge: (p: P) => <Svg {...p}><path d="M3.5 18a8.5 8.5 0 1 1 17 0" /><path d="M12 18l4.6-5.2" /><circle cx="12" cy="18" r="1.1" /></Svg>,
  globe: (p: P) => <Svg {...p}><circle cx="12" cy="12" r="9" /><path d="M3 12h18" /><path d="M12 3c3 2.7 3 15.3 0 18M12 3c-3 2.7-3 15.3 0 18" /></Svg>,
  agent: (p: P) => <Svg {...p}><rect x="4" y="8" width="16" height="11" rx="3" /><path d="M12 4.6V8" /><circle cx="12" cy="3.3" r="1.2" /><path d="M9 13h.01M15 13h.01M9.75 16.25h4.5" /></Svg>,
  users: (p: P) => <Svg {...p}><circle cx="9" cy="8" r="3.3" /><path d="M3 20.5a6 6 0 0 1 12 0" /><path d="M16.2 5.2a3.3 3.3 0 0 1 0 6.1" /><path d="M21 20.5a6 6 0 0 0-3.4-4.9" /></Svg>,
  trash: (p: P) => <Svg {...p}><path d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2M6 7l1 13a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1l1-13" /></Svg>,
}
