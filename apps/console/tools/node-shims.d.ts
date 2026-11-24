// Minimal ambient declarations for the two Node APIs the check scripts use.
//
// The console is a browser app and deliberately does not depend on @types/node — pulling it in
// just to type-check a standalone script would change package.json and the lockfile, and so the
// `npm ci` that runs inside the image build. These few lines keep `tools/` inside the type-check
// project (which is the point: it is how a drift between a check and the page it guards gets
// caught at `tsc` time) without that cost.
declare module 'node:fs' {
  export function readFileSync(path: string, encoding: 'utf8'): string
}
declare module 'node:path' {
  export function resolve(...segments: string[]): string
}
declare const process: {
  argv: string[]
  // `env`/`cwd` back the repo-relative enum lookup in audit-logic.check.ts, so the script
  // works in any clone instead of a hard-coded absolute path.
  env: Record<string, string | undefined>
  cwd(): string
  exit(code?: number): never
}
