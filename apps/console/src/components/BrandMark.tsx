// Product logo mark — a document with a folded corner and three citation bars, which is the
// one promise the platform makes: an answer always points back at a source. Static,
// dependency-free, theme-agnostic (the white bars sit on the amber body, so it reads on both
// an ivory plate and the dark sidebar rail). Square, so the `.logo-plate` wrapper stays
// proportional at every call size.
export function BrandMark({ size = 32, title = 'RAG as a Service' }: { size?: number; title?: string }) {
  return (
    <svg viewBox="0 0 64 64" width={size} height={size} role="img" aria-label={title} style={{ display: 'block' }}>
      {/* Page body. The top-right corner is cut rather than rounded — that cut is what the
          fold facet below sits in, and together they read as a sheet of paper. */}
      <path fill="#FDBC2F" d="M14 2 H36 L52 18 V58 Q52 62 48 62 H14 Q10 62 10 58 V6 Q10 2 14 2 Z" />
      {/* Folded corner, one tone lighter so the fold catches the light. */}
      <path fill="#FFDC82" d="M36 2 L52 18 H36 Z" />
      {/* Spine shadow — keeps the mark from flattening out at 30px on the dark rail. */}
      <path fill="#E08A32" d="M14 2 H18 V62 H14 Q10 62 10 58 V6 Q10 2 14 2 Z" />
      {/* Citation bars. The last is short: a citation, not a full line of prose. */}
      <rect x="23" y="28" width="22" height="5" rx="2.5" fill="#FEFEFE" />
      <rect x="23" y="39" width="22" height="5" rx="2.5" fill="#FEFEFE" />
      <rect x="23" y="50" width="13" height="5" rx="2.5" fill="#FEFEFE" />
    </svg>
  )
}
