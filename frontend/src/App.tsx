import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import './App.css'

/* ───────────────────────── Types (mirror src/placement/api.py) ───────────────────────── */
type Status = 'idle' | 'submitting' | 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
interface Placed { macro_id: number; x: number; y: number }
interface Netlist {
  version: string
  grid_w: number
  grid_h: number
  macros: { id: number; w: number; h: number }[]
  nets: { id: number; macro_ids: number[]; weight: number }[]
}
interface JobStatus {
  run_id: string; status: string; progress: number; step: number; total_steps: number
  current_hpwl: number | null; placement: Placed[]; error: string | null
  current_congestion?: number | null; current_objective?: number | null
  congestion?: number | null; objective?: number | null
}
interface TracePoint {
  step: number
  hpwl: number
  congestion?: number | null
  objective?: number | null
}
interface RunRec {
  alg: string; status: Status; runId: string | null; seed: number; netlist: Netlist
  progress: number; step: number; total: number; hpwl: number | null
  congestion: number | null; objective: number | null
  placement: Placed[]; error: string | null; elapsedMs: number | null
  trace: TracePoint[]
}
interface MacroRow { id: string; w: string; h: string }
interface NetRow { id: string; ids: string; weight: string }
interface HistoryRow {
  run_id: string; status: string; progress: number
  current_hpwl: number | null; current_congestion?: number | null; current_objective?: number | null
  placement: Placed[]; error: string | null
}
interface RudyData {
  run_id?: string
  grid_w: number
  grid_h: number
  demand: number[][]
  congestion_score: number
  max_demand: number
}
interface BenchmarkSummary {
  name: string
  grid_w: number
  grid_h: number
  macro_count: number
  net_count: number
  total_macro_area: number
  grid_area: number
  area_utilization: number
}

/* ───────────────────────── Constants & defaults ───────────────────────── */
const DEFAULT_API = 'http://127.0.0.1:8000'
const POLL_MS = 200
const RUN_TIMEOUT_MS = 180_000
const RL_RE = /^(rl|ppo|maskable_ppo|rl_ppo|reinforcement\w*)$/i
const ALGS = [
  { key: 'random', label: 'Random' },
  { key: 'greedy', label: 'Greedy' },
  { key: 'sa', label: 'Simulated Annealing' },
]
const DEFAULT_MACROS: MacroRow[] = [
  { id: '0', w: '6', h: '4' }, { id: '1', w: '4', h: '4' }, { id: '2', w: '5', h: '3' },
  { id: '3', w: '3', h: '5' }, { id: '4', w: '4', h: '3' }, { id: '5', w: '3', h: '3' },
]
const DEFAULT_NETS: NetRow[] = [
  { id: '0', ids: '0, 1', weight: '2' }, { id: '1', ids: '1, 2, 3', weight: '1' },
  { id: '2', ids: '0, 4', weight: '1' }, { id: '3', ids: '2, 5', weight: '1.5' },
  { id: '4', ids: '3, 4, 5', weight: '1' }, { id: '5', ids: '0, 5', weight: '1' },
]

/* ───────────────────────── Helpers ───────────────────────── */
const toInt = (s: string) => (s.trim() !== '' && Number.isInteger(Number(s)) ? Number(s) : NaN)
const toNum = (s: string) => (s.trim() !== '' ? Number(s) : NaN)
const fmt = (n: number | null | undefined, d = 3) =>
  n == null || Number.isNaN(n) ? '—' : n.toLocaleString(undefined, { maximumFractionDigits: d })
const fmtMs = (ms: number | null) => (ms == null ? '—' : ms < 1000 ? `${ms.toFixed(0)} ms` : `${(ms / 1000).toFixed(2)} s`)
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms))
const isAbort = (e: unknown) => e instanceof DOMException && e.name === 'AbortError'
const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e))
const hue = (id: number) => (id * 137.508 + 200) % 360
const algLabel = (k: string) => ALGS.find((a) => a.key === k)?.label ?? (RL_RE.test(k) ? 'Reinforcement Learning' : k)

function detailText(d: unknown): string {
  if (d == null) return ''
  if (typeof d === 'string') return d
  if (Array.isArray(d))
    return d
      .map((x) => {
        if (x && typeof x === 'object') {
          const o = x as Record<string, unknown>
          const loc = Array.isArray(o.loc) ? o.loc.join('.') : ''
          return `${loc ? loc + ': ' : ''}${String(o.msg ?? JSON.stringify(o))}`
        }
        return String(x)
      })
      .join('; ')
  if (typeof d === 'object') {
    const o = d as Record<string, unknown>
    const errs = Array.isArray(o.errors) ? o.errors.map(String) : []
    return [String(o.message ?? ''), errs.join(' ')].filter(Boolean).join(' ') || JSON.stringify(d)
  }
  return String(d)
}

async function api<T>(base: string, path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(base + path, {
      ...init,
      headers: init?.body ? { 'Content-Type': 'application/json' } : undefined,
    })
  } catch (e) {
    if (isAbort(e)) throw e
    throw new Error(`Cannot reach the API at ${base}. Is the backend running (uvicorn) and is the URL correct?`)
  }
  const text = await res.text()
  let body: unknown = text
  try { body = text ? JSON.parse(text) : null } catch { /* keep raw text */ }
  if (!res.ok) {
    const d = body && typeof body === 'object' ? (body as Record<string, unknown>).detail : body
    throw new Error(`HTTP ${res.status}: ${detailText(d) || res.statusText}`)
  }
  return body as T
}

function buildNetlist(gw: string, gh: string, macros: MacroRow[], nets: NetRow[]): { netlist: Netlist | null; errors: string[] } {
  const errors: string[] = []
  const W = toInt(gw), H = toInt(gh)
  if (!(W > 0)) errors.push('Grid width must be a positive integer.')
  if (!(H > 0)) errors.push('Grid height must be a positive integer.')
  if (macros.length === 0) errors.push('Add at least one macro.')
  const ms: Netlist['macros'] = []
  const seen = new Set<number>()
  let area = 0
  macros.forEach((m, i) => {
    const id = toInt(m.id), w = toInt(m.w), h = toInt(m.h)
    if (!(id >= 0)) return void errors.push(`Macro row ${i + 1}: ID must be an integer ≥ 0.`)
    if (seen.has(id)) errors.push(`Macro ID ${id} is duplicated.`)
    seen.add(id)
    if (!(w > 0) || !(h > 0)) return void errors.push(`Macro ${id}: width and height must be positive integers.`)
    if (W > 0 && H > 0 && (w > W || h > H)) errors.push(`Macro ${id} (${w}×${h}) exceeds the grid (${W}×${H}).`)
    area += w * h
    ms.push({ id, w, h })
  })
  if (W > 0 && H > 0 && area > W * H) errors.push(`Total macro area (${area}) exceeds grid area (${W * H}); no legal placement exists.`)
  const ns: Netlist['nets'] = []
  const netSeen = new Set<number>()
  nets.forEach((n, i) => {
    const id = toInt(n.id), weight = toNum(n.weight)
    const ids = n.ids.split(/[\s,;]+/).filter(Boolean).map(Number)
    if (!(id >= 0)) return void errors.push(`Net row ${i + 1}: ID must be an integer ≥ 0.`)
    if (netSeen.has(id)) errors.push(`Net ID ${id} is duplicated.`)
    netSeen.add(id)
    if (ids.length < 2 || ids.some((x) => !Number.isInteger(x))) return void errors.push(`Net ${id}: list at least 2 integer macro IDs, e.g. "0, 1".`)
    const missing = ids.filter((x) => !seen.has(x))
    if (missing.length) errors.push(`Net ${id} references undefined macro ID(s): ${[...new Set(missing)].join(', ')}.`)
    if (!(weight > 0) || !Number.isFinite(weight)) return void errors.push(`Net ${id}: weight must be a number > 0.`)
    ns.push({ id, macro_ids: ids, weight })
  })
  if (errors.length) return { netlist: null, errors }
  return { netlist: { version: '1.0', grid_w: W, grid_h: H, macros: ms, nets: ns }, errors }
}

/** Independent client-side sanity check of a returned placement (bounds, overlap, completeness). */
function checkLegal(nl: Netlist, pl: Placed[]): string[] {
  const issues: string[] = []
  const dims = new Map(nl.macros.map((m) => [m.id, m]))
  const rects: { id: number; x: number; y: number; w: number; h: number }[] = []
  for (const p of pl) {
    const m = dims.get(p.macro_id)
    if (!m) { issues.push(`Unknown macro ${p.macro_id}`); continue }
    if (p.x < 0 || p.y < 0 || p.x + m.w > nl.grid_w || p.y + m.h > nl.grid_h) issues.push(`Macro ${m.id} out of bounds`)
    rects.push({ id: m.id, x: p.x, y: p.y, w: m.w, h: m.h })
  }
  for (let i = 0; i < rects.length; i++)
    for (let j = i + 1; j < rects.length; j++) {
      const a = rects[i], b = rects[j]
      if (a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h) issues.push(`Macros ${a.id} and ${b.id} overlap`)
    }
  if (rects.length !== nl.macros.length) issues.push(`Placed ${rects.length}/${nl.macros.length} macros`)
  return issues
}

const csv = (v: unknown) => {
  const s = String(v ?? '')
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}
function download(name: string, mime: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: mime }))
  const a = document.createElement('a')
  a.href = url
  a.download = name
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

/** Submit one run and poll until it finishes. Reports every real server snapshot through patch(). */
async function runOne(
  base: string, alg: string, nl: Netlist, seed: number, params: Record<string, unknown>,
  signal: AbortSignal, patch: (p: Partial<RunRec>) => void,
) {
  const t0 = performance.now()
  const trace: TracePoint[] = []
  try {
    patch({ status: 'submitting', trace: [] })
    const start = await api<{ run_id: string }>(base, '/api/placements/run', {
      method: 'POST', body: JSON.stringify({ netlist: nl, algorithm: alg, seed, params }), signal,
    })
    patch({ runId: start.run_id, status: 'pending' })
    let fails = 0
    for (; ;) {
      await sleep(POLL_MS)
      if (signal.aborted) throw new DOMException('aborted', 'AbortError')
      if (performance.now() - t0 > RUN_TIMEOUT_MS) throw new Error('Timed out waiting for the run to finish.')
      let js: JobStatus
      try {
        js = await api<JobStatus>(base, `/api/placements/${start.run_id}`, { signal })
        fails = 0
      } catch (e) {
        if (isAbort(e) || ++fails >= 3) throw e
        continue
      }
      const done = js.status === 'completed' || js.status === 'failed'
      const hpwl = js.current_hpwl ?? null
      const congestion = js.current_congestion ?? js.congestion ?? null
      const objective = js.current_objective ?? js.objective ?? null

      if (hpwl != null && (trace.length === 0 || trace[trace.length - 1].step !== js.step || trace[trace.length - 1].hpwl !== hpwl)) {
        trace.push({ step: js.step, hpwl, congestion, objective })
      }

      patch({
        status: (['pending', 'running', 'completed', 'failed'].includes(js.status) ? js.status : 'running') as Status,
        progress: js.progress, step: js.step, total: js.total_steps,
        hpwl, congestion, objective,
        placement: js.placement ?? [], error: js.error,
        trace: [...trace],
        ...(done ? { elapsedMs: performance.now() - t0 } : {}),
      })
      if (done) return
    }
  } catch (e) {
    patch(isAbort(e)
      ? { status: 'cancelled', error: 'Cancelled in the browser (the backend job may still finish).', elapsedMs: performance.now() - t0 }
      : { status: 'failed', error: errMsg(e), elapsedMs: performance.now() - t0 })
  }
}

/* ───────────────────────── SVG Floorplan & Heatmap ───────────────────────── */
function Floorplan({
  nl, placement, showGrid, showNets, showLabels, showHeatmap, rudy, compact,
}: {
  nl: Netlist; placement: Placed[]; showGrid: boolean; showNets: boolean; showLabels: boolean
  showHeatmap?: boolean; rudy?: RudyData | null; compact?: boolean
}) {
  const { grid_w: W, grid_h: H } = nl
  const p = Math.max(W, H) * 0.075
  const fs = p * 0.42
  const step = [1, 2, 5, 10, 20, 50, 100, 200, 500].find((s) => W / s <= 12 && H / s <= 12) ?? 1000
  const pos = new Map(placement.map((q) => [q.macro_id, q]))
  const dims = new Map(nl.macros.map((m) => [m.id, m]))
  const center = (id: number) => {
    const m = dims.get(id), q = pos.get(id)
    return m && q ? { x: q.x + m.w / 2, y: q.y + m.h / 2 } : null
  }
  const ticks = (n: number) => Array.from({ length: Math.floor(n / step) + 1 }, (_, i) => i * step)
  const minor = W * H <= 2500

  return (
    <svg className="fp-svg" viewBox={`${-p} ${-p} ${W + p * 1.5} ${H + p * 1.5}`} role="img"
      aria-label={`Floorplan ${W} by ${H} grid with ${placement.length} placed macros`}>
      <rect x={0} y={0} width={W} height={H} className="fp-chip" />

      {/* RUDY Wire Density Heatmap Cells */}
      {showHeatmap && rudy && rudy.demand && rudy.demand.map((row, y) =>
        row.map((val, x) => {
          if (val <= 0.001) return null
          const maxD = rudy.max_demand > 0 ? rudy.max_demand : 1.0
          const ratio = Math.min(1.0, val / maxD)
          // Gradient from transparent blue (220) to orange-red (0)
          const hueVal = Math.max(0, 220 - ratio * 220)
          const alpha = 0.2 + ratio * 0.65
          return (
            <rect
              key={`rudy-${x}-${y}`}
              x={x}
              y={y}
              width={1}
              height={1}
              fill={`hsla(${hueVal}, 95%, 48%, ${alpha})`}
              vectorEffect="non-scaling-stroke"
            >
              <title>{`Cell (${x}, ${y}): RUDY wire demand = ${val.toFixed(2)}`}</title>
            </rect>
          )
        })
      )}

      {showGrid && (
        <g>
          {minor && Array.from({ length: W - 1 }, (_, i) => <line key={`vx${i}`} x1={i + 1} y1={0} x2={i + 1} y2={H} className="fp-minor" vectorEffect="non-scaling-stroke" />)}
          {minor && Array.from({ length: H - 1 }, (_, i) => <line key={`hy${i}`} x1={0} y1={i + 1} x2={W} y2={i + 1} className="fp-minor" vectorEffect="non-scaling-stroke" />)}
          {ticks(W).map((t) => <line key={`mx${t}`} x1={t} y1={0} x2={t} y2={H} className="fp-major" vectorEffect="non-scaling-stroke" />)}
          {ticks(H).map((t) => <line key={`my${t}`} x1={0} y1={t} x2={t} y2={H} className="fp-major" vectorEffect="non-scaling-stroke" />)}
        </g>
      )}

      {!compact && ticks(W).map((t) => <text key={`tx${t}`} x={t} y={-fs * 0.5} fontSize={fs} textAnchor="middle" className="fp-tick">{t}</text>)}
      {!compact && ticks(H).map((t) => <text key={`ty${t}`} x={-fs * 0.5} y={t + fs * 0.35} fontSize={fs} textAnchor="end" className="fp-tick">{t}</text>)}

      {/* Macros */}
      {placement.map((q) => {
        const m = dims.get(q.macro_id)
        if (!m) return null
        const label = `M${m.id}`
        const f = Math.min(m.h * 0.34, m.w / (label.length * 0.62))
        const two = m.h >= 3 && m.w >= 3.5 && !compact
        const macroOpacity = showHeatmap ? 0.88 : 0.94
        return (
          <g key={q.macro_id}>
            <rect x={q.x} y={q.y} width={m.w} height={m.h} rx={Math.min(m.w, m.h) * 0.06}
              fill={`hsl(${hue(m.id)} 62% 62% / ${macroOpacity})`}
              stroke={showHeatmap ? '#ffffff' : 'currentColor'}
              strokeWidth={showHeatmap ? Math.max(0.2, Math.min(W, H) * 0.006) : undefined}
              className="fp-macro" vectorEffect="non-scaling-stroke">
              <title>{`Macro ${m.id}: ${m.w}×${m.h} at (${q.x}, ${q.y})`}</title>
            </rect>
            {showLabels && f > 0.35 && (
              <text x={q.x + m.w / 2} y={q.y + m.h / 2 + (two ? -f * 0.1 : f * 0.35)} fontSize={f} textAnchor="middle" className="fp-label">
                {label}
                {two && <tspan x={q.x + m.w / 2} dy={f * 1.05} fontSize={f * 0.62} fontWeight={400}>{m.w}×{m.h}</tspan>}
              </text>
            )}
          </g>
        )
      })}

      {/* Nets */}
      {showNets && nl.nets.map((n) => {
        const cs = n.macro_ids.map(center).filter((c): c is { x: number; y: number } => c !== null)
        if (cs.length < 2) return null
        const cx = cs.reduce((s, c) => s + c.x, 0) / cs.length
        const cy = cs.reduce((s, c) => s + c.y, 0) / cs.length
        return (
          <g key={`n${n.id}`} className="fp-net" strokeWidth={Math.min(3, 0.8 + n.weight * 0.4)}>
            <title>{`Net ${n.id} (weight ${n.weight}): macros ${n.macro_ids.join(', ')}`}</title>
            {cs.map((c, i) => <line key={i} x1={cx} y1={cy} x2={c.x} y2={c.y} vectorEffect="non-scaling-stroke" />)}
            <circle cx={cx} cy={cy} r={Math.max(W, H) * 0.008} className="fp-netdot" />
          </g>
        )
      })}
      <rect x={0} y={0} width={W} height={H} className="fp-border" vectorEffect="non-scaling-stroke" />
    </svg>
  )
}

/* ───────────────────────── Optimization Trace / Convergence Chart ───────────────────────── */
function ConvergenceChart({ trace, totalSteps, compact }: { trace: TracePoint[]; totalSteps: number; compact?: boolean }) {
  if (trace.length === 0) return null

  const W = compact ? 260 : 540
  const H = compact ? 110 : 160
  const padL = compact ? 36 : 46
  const padR = 14
  const padT = 16
  const padB = compact ? 22 : 28

  const plotW = W - padL - padR
  const plotH = H - padT - padB

  const maxStep = Math.max(totalSteps, ...trace.map((p) => p.step), 1)
  const hpwls = trace.map((p) => p.hpwl)
  const minH = Math.min(...hpwls)
  const maxH = Math.max(...hpwls)
  const hRange = maxH - minH > 0.001 ? maxH - minH : 1.0

  const scaleX = (s: number) => padL + (s / maxStep) * plotW
  const scaleY = (v: number) => padT + plotH - ((v - minH) / hRange) * plotH

  const pts = trace.map((p) => `${scaleX(p.step).toFixed(1)},${scaleY(p.hpwl).toFixed(1)}`).join(' ')

  return (
    <div className="trace-chart">
      <div className="trace-head">
        <strong>Optimization Convergence Curve</strong>
        <span className="hint">HPWL: initial {fmt(trace[0]?.hpwl, 1)} → best {fmt(minH, 1)}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" role="img" aria-label="Convergence chart">
        {/* Background gridlines */}
        <line x1={padL} y1={padT} x2={padL + plotW} y2={padT} stroke="var(--border)" strokeDasharray="3 3" />
        <line x1={padL} y1={padT + plotH / 2} x2={padL + plotW} y2={padT + plotH / 2} stroke="var(--border)" strokeDasharray="3 3" />
        <line x1={padL} y1={padT + plotH} x2={padL + plotW} y2={padT + plotH} stroke="var(--border)" />
        <line x1={padL} y1={padT} x2={padL} y2={padT + plotH} stroke="var(--border)" />

        {/* Y Axis ticks */}
        <text x={padL - 6} y={padT + 4} textAnchor="end" className="chart-tick">{fmt(maxH, 0)}</text>
        <text x={padL - 6} y={padT + plotH} textAnchor="end" className="chart-tick">{fmt(minH, 0)}</text>

        {/* X Axis ticks */}
        <text x={padL} y={H - 5} textAnchor="start" className="chart-tick">Step 0</text>
        <text x={padL + plotW} y={H - 5} textAnchor="end" className="chart-tick">Step {maxStep}</text>

        {/* Polyline */}
        <polyline fill="none" stroke="var(--accent)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" points={pts} />

        {/* Trace dots */}
        {trace.map((p, i) => (
          <circle key={i} cx={scaleX(p.step)} cy={scaleY(p.hpwl)} r={compact ? 2.5 : 3.5} fill="var(--accent)" stroke="#ffffff" strokeWidth="1">
            <title>{`Step ${p.step}: HPWL ${p.hpwl.toFixed(1)}${p.congestion != null ? `, Congestion ${p.congestion.toFixed(2)}` : ''}`}</title>
          </circle>
        ))}
      </svg>
    </div>
  )
}

/* ───────────────────────── Main App ───────────────────────── */
export default function App() {
  const [apiUrl, setApiUrl] = useState(DEFAULT_API)
  const [conn, setConn] = useState<{ state: 'checking' | 'ok' | 'down'; msg: string }>({ state: 'checking', msg: 'Checking…' })
  const [registry, setRegistry] = useState<string[] | null>(null)
  const [benchmarks, setBenchmarks] = useState<BenchmarkSummary[]>([])

  const [gw, setGw] = useState('20')
  const [gh, setGh] = useState('16')
  const [macros, setMacros] = useState<MacroRow[]>(DEFAULT_MACROS)
  const [nets, setNets] = useState<NetRow[]>(DEFAULT_NETS)
  const [alg, setAlg] = useState('greedy')
  const [seed, setSeed] = useState('42')
  const [saIter, setSaIter] = useState('1000')
  const [saTemp, setSaTemp] = useState('100')
  const [saCool, setSaCool] = useState('0.95')

  const [runs, setRuns] = useState<Record<string, RunRec>>({})
  const [view, setView] = useState('greedy')
  const [busy, setBusy] = useState<null | 'single' | 'compare'>(null)
  const [notice, setNotice] = useState<{ kind: 'error' | 'ok'; lines: string[] } | null>(null)
  const [showGrid, setShowGrid] = useState(true)
  const [showNets, setShowNets] = useState(true)
  const [showLabels, setShowLabels] = useState(true)
  const [showHeatmap, setShowHeatmap] = useState(false)
  const [rudyData, setRudyData] = useState<Record<string, RudyData>>({})
  const [history, setHistory] = useState<HistoryRow[] | null>(null)
  const [histErr, setHistErr] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const base = apiUrl.trim().replace(/\/+$/, '')
  const built = useMemo(() => buildNetlist(gw, gh, macros, nets), [gw, gh, macros, nets])
  const curKey = built.netlist ? JSON.stringify(built.netlist) : ''
  const rlKey = registry?.find((n) => RL_RE.test(n)) ?? null
  const isAvail = (k: string) => (registry ? registry.includes(k) : true)

  /* connection, algorithm registry, and benchmark suites discovery */
  const checkApi = useCallback(async () => {
    setConn({ state: 'checking', msg: 'Checking…' })
    try {
      const h = await api<{ status: string; version: string }>(base, '/health')
      setConn({ state: 'ok', msg: `Connected · API v${h.version}` })
    } catch (e) {
      setConn({ state: 'down', msg: errMsg(e) })
      setRegistry(null)
      return
    }

    // Try direct algorithms endpoint first
    try {
      const algRes = await api<{ algorithms: string[] }>(base, '/api/algorithms')
      setRegistry(algRes.algorithms)
    } catch {
      // Fallback probe
      try {
        await api(base, '/api/placements/run', {
          method: 'POST',
          body: JSON.stringify({
            algorithm: '__registry_probe__', seed: 0, params: {},
            netlist: { version: '1.0', grid_w: 2, grid_h: 1, macros: [{ id: 0, w: 1, h: 1 }, { id: 1, w: 1, h: 1 }], nets: [{ id: 0, macro_ids: [0, 1], weight: 1 }] },
          }),
        })
        setRegistry(null)
      } catch (e) {
        const m = errMsg(e).match(/Available:\s*\[([^\]]*)\]/)
        setRegistry(m ? [...m[1].matchAll(/'([^']+)'/g)].map((x) => x[1]) : null)
      }
    }

    // Load available standard benchmarks
    try {
      const bm = await api<{ benchmarks: BenchmarkSummary[] }>(base, '/api/benchmarks')
      setBenchmarks(bm.benchmarks ?? [])
    } catch {
      setBenchmarks([])
    }
  }, [base])

  useEffect(() => {
    const t = setTimeout(() => void checkApi(), 400)
    return () => clearTimeout(t)
  }, [checkApi])

  /* run orchestration */
  const params = (a: string): Record<string, unknown> =>
    a === 'sa' || a === 'simulated_annealing'
      ? { max_iterations: toInt(saIter), initial_temperature: toNum(saTemp), cooling_rate: toNum(saCool), alpha: 1.0, beta: 1.0 }
      : { alpha: 1.0, beta: 1.0 }

  const runBatch = async (algs: string[]) => {
    if (busy) return
    const nl = built.netlist
    if (!nl) return setNotice({ kind: 'error', lines: built.errors })
    const sd = toInt(seed)
    if (Number.isNaN(sd)) return setNotice({ kind: 'error', lines: ['Seed must be an integer.'] })
    if (algs.includes('sa') || algs.includes('simulated_annealing')) {
      const it = toInt(saIter), t = toNum(saTemp), c = toNum(saCool)
      const bad = [!(it > 0) && 'SA max iterations must be a positive integer.', !(t > 0) && 'SA initial temperature must be > 0.', !(c > 0 && c < 1) && 'SA cooling rate must be between 0 and 1 (exclusive).'].filter(Boolean) as string[]
      if (bad.length) return setNotice({ kind: 'error', lines: bad })
    }
    setNotice(null)
    const ac = new AbortController()
    abortRef.current = ac
    setBusy(algs.length > 1 ? 'compare' : 'single')
    const blank = (a: string): RunRec => ({
      alg: a, status: 'idle', runId: null, seed: sd, netlist: nl,
      progress: 0, step: 0, total: nl.macros.length, hpwl: null,
      congestion: null, objective: null, placement: [], error: null, elapsedMs: null, trace: [],
    })
    setRuns((r) => { const n = { ...r }; algs.forEach((a) => (n[a] = blank(a))); return n })
    for (const a of algs) {
      setView(a)
      const patch = (p: Partial<RunRec>) => setRuns((r) => ({ ...r, [a]: { ...r[a], ...p } }))
      if (ac.signal.aborted) { patch({ status: 'cancelled', error: 'Cancelled before start.' }); continue }
      await runOne(base, a, nl, sd, params(a), ac.signal, patch)
    }
    setBusy(null)
  }

  const compareAlgs = [...ALGS.map((a) => a.key).filter(isAvail), ...(rlKey ? [rlKey] : [])]
  const allKeys = [...ALGS.map((a) => a.key), rlKey ?? 'rl']

  const validate = async () => {
    if (!built.netlist) return setNotice({ kind: 'error', lines: built.errors })
    try {
      const r = await api<{ is_valid: boolean; macro_count: number; net_count: number; grid_w: number; grid_h: number; errors: string[] }>(
        base, '/api/netlist/validate', { method: 'POST', body: JSON.stringify(built.netlist) })
      setNotice(r.is_valid
        ? { kind: 'ok', lines: [`Backend validation passed: ${r.macro_count} macros, ${r.net_count} nets on a ${r.grid_w}×${r.grid_h} grid.`] }
        : { kind: 'error', lines: r.errors })
    } catch (e) { setNotice({ kind: 'error', lines: [errMsg(e)] }) }
  }

  const loadHistory = async () => {
    try { setHistory(await api<HistoryRow[]>(base, '/api/placements')); setHistErr(null) }
    catch (e) { setHistErr(errMsg(e)) }
  }

  /* config editing */
  const setM = (i: number, k: keyof MacroRow, v: string) => setMacros((m) => m.map((r, j) => (j === i ? { ...r, [k]: v } : r)))
  const setN = (i: number, k: keyof NetRow, v: string) => setNets((m) => m.map((r, j) => (j === i ? { ...r, [k]: v } : r)))
  const nextId = (rows: { id: string }[]) => String(Math.max(-1, ...rows.map((r) => toInt(r.id)).filter((x) => x >= 0)) + 1)
  const addMacro = () => setMacros((m) => [...m, { id: nextId(m), w: '3', h: '3' }])
  const addNet = () => setNets((n) => [...n, { id: nextId(n), ids: '', weight: '1' }])
  const resetExample = () => { setGw('20'); setGh('16'); setMacros(DEFAULT_MACROS); setNets(DEFAULT_NETS); setSeed('42'); setNotice(null) }

  const loadBenchmarkPreset = async (name: string) => {
    try {
      const nl = await api<Netlist>(base, `/api/benchmarks/${name}`)
      setGw(String(nl.grid_w))
      setGh(String(nl.grid_h))
      setMacros(nl.macros.map((m) => ({ id: String(m.id), w: String(m.w), h: String(m.h) })))
      setNets(nl.nets.map((x) => ({ id: String(x.id), ids: x.macro_ids.join(', '), weight: String(x.weight ?? 1) })))
      setNotice({ kind: 'ok', lines: [`Loaded benchmark preset '${name}': ${nl.macros.length} macros, ${nl.nets.length} nets, ${nl.grid_w}×${nl.grid_h} grid.`] })
    } catch (e) {
      setNotice({ kind: 'error', lines: [`Failed to load benchmark '${name}': ${errMsg(e)}`] })
    }
  }

  const importNetlist = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    e.target.value = ''
    if (!f) return
    f.text().then((t) => {
      const j = JSON.parse(t) as { netlist?: Netlist } & Partial<Netlist>
      const n = j.netlist ?? j
      if (!n || !Array.isArray(n.macros) || !Array.isArray(n.nets) || !n.grid_w || !n.grid_h) throw new Error('Missing grid_w, grid_h, macros or nets.')
      setGw(String(n.grid_w)); setGh(String(n.grid_h))
      setMacros(n.macros.map((m) => ({ id: String(m.id), w: String(m.w), h: String(m.h) })))
      setNets(n.nets.map((x) => ({ id: String(x.id), ids: x.macro_ids.join(', '), weight: String(x.weight ?? 1) })))
      setNotice({ kind: 'ok', lines: [`Imported ${f.name}.`] })
    }).catch((err) => setNotice({ kind: 'error', lines: [`Import failed: ${errMsg(err)}`] }))
  }

  /* derived view data */
  const rec = runs[view]
  const previewNl: Netlist = { version: '1.0', grid_w: toInt(gw) > 0 ? toInt(gw) : 20, grid_h: toInt(gh) > 0 ? toInt(gh) : 16, macros: [], nets: [] }
  const shownNl = rec?.netlist ?? previewNl
  const stale = (r: RunRec) => curKey !== JSON.stringify(r.netlist)
  const finished = Object.values(runs).filter((r) => r.status === 'completed' && r.hpwl != null)
  const best = finished.length ? Math.min(...finished.map((r) => r.hpwl as number)) : null
  const issues = rec?.status === 'completed' ? checkLegal(rec.netlist, rec.placement) : []
  const running = !!rec && ['submitting', 'pending', 'running'].includes(rec.status)

  /* Fetch RUDY data when a placement is completed */
  const activeRunId = rec?.runId
  useEffect(() => {
    if (rec?.status === 'completed' && activeRunId && !rudyData[activeRunId]) {
      api<RudyData>(base, `/api/placements/${activeRunId}/rudy`)
        .then((data) => setRudyData((prev) => ({ ...prev, [activeRunId]: data })))
        .catch(() => { /* silent fallback */ })
    }
  }, [rec?.status, activeRunId, base, rudyData])

  const curRudy = activeRunId ? rudyData[activeRunId] : null

  /* exports */
  const exportable = Object.values(runs).filter((r) => r.status === 'completed')
  const exportJson = () => download('macro-placement-results.json', 'application/json', JSON.stringify({
    exported_at: new Date().toISOString(), api_url: base,
    results: Object.values(runs).filter((r) => r.status === 'completed' || r.status === 'failed').map((r) => ({
      algorithm: r.alg, run_id: r.runId, status: r.status, seed: r.seed,
      hpwl: r.status === 'completed' ? r.hpwl : null,
      congestion: r.status === 'completed' ? r.congestion : null,
      objective: r.status === 'completed' ? r.objective : null,
      steps: r.step, total_steps: r.total, client_wall_time_ms: r.elapsedMs, error: r.error,
      netlist: r.netlist, placement: r.placement,
    })),
  }, null, 2))

  const exportPlacementsCsv = () => {
    const rows = ['algorithm,run_id,macro_id,x,y,w,h']
    exportable.forEach((r) => r.placement.forEach((q) => {
      const m = r.netlist.macros.find((x) => x.id === q.macro_id)
      rows.push([r.alg, r.runId, q.macro_id, q.x, q.y, m?.w, m?.h].map(csv).join(','))
    }))
    download('macro-placements.csv', 'text/csv', rows.join('\n') + '\n')
  }

  const exportMetricsCsv = () => {
    const rows = ['algorithm,status,run_id,seed,hpwl,congestion,objective,steps,total_steps,client_wall_time_ms,error']
    Object.values(runs).forEach((r) => rows.push([
      r.alg, r.status, r.runId, r.seed,
      r.status === 'completed' ? r.hpwl : '',
      r.status === 'completed' ? r.congestion : '',
      r.status === 'completed' ? r.objective : '',
      r.step, r.total, r.elapsedMs?.toFixed(1), r.error,
    ].map(csv).join(',')))
    download('macro-metrics.csv', 'text/csv', rows.join('\n') + '\n')
  }

  const exportNetlist = () => built.netlist
    ? download('netlist.json', 'application/json', JSON.stringify(built.netlist, null, 2))
    : setNotice({ kind: 'error', lines: built.errors })

  const hasRuns = Object.keys(runs).length > 0
  const stCls = (s: string) => `st st-${s}`

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden>▦</span>
          <div><h1>Macro Placement Studio</h1><p>Graph-theoretic floorplanning · Random · Greedy · SA · RL</p></div>
        </div>
        <div className={`conn conn-${conn.state}`} title={conn.msg}>
          <span className="dot" /> <span className="conn-msg">{conn.msg}</span>
          <button className="btn sm" onClick={() => void checkApi()}>Recheck</button>
        </div>
      </header>

      <div className="layout">
        {/* ───────── Sidebar ───────── */}
        <aside className="sidebar">
          <section className="panel">
            <h2>Connection</h2>
            <label className="field"><span>API base URL</span>
              <input value={apiUrl} onChange={(e) => setApiUrl(e.target.value)} spellCheck={false} />
            </label>
            <p className="hint">{registry ? <>Backend algorithms: <code>{registry.join(', ')}</code></> : 'Backend algorithm list unavailable.'}</p>
          </section>

          {/* Standard Benchmark Presets */}
          <section className="panel">
            <h2>Standard Benchmarks</h2>
            <p className="hint">Select a synthetic benchmark to auto-fill the grid and circuit:</p>
            <div className="btn-row wrap">
              {benchmarks.length > 0 ? (
                benchmarks.map((b) => (
                  <button key={b.name} className="btn sm" onClick={() => void loadBenchmarkPreset(b.name)}>
                    {b.name} ({b.macro_count}m)
                  </button>
                ))
              ) : (
                <>
                  <button className="btn sm" onClick={() => void loadBenchmarkPreset('toy_4')}>toy_4 (4m)</button>
                  <button className="btn sm" onClick={() => void loadBenchmarkPreset('medium_8')}>medium_8 (8m)</button>
                  <button className="btn sm" onClick={() => void loadBenchmarkPreset('dense_10')}>dense_10 (10m)</button>
                </>
              )}
            </div>
          </section>

          <section className="panel">
            <h2>Chip grid</h2>
            <div className="row2">
              <label className="field"><span>Width</span><input inputMode="numeric" value={gw} onChange={(e) => setGw(e.target.value)} /></label>
              <label className="field"><span>Height</span><input inputMode="numeric" value={gh} onChange={(e) => setGh(e.target.value)} /></label>
            </div>
          </section>

          <section className="panel">
            <div className="panel-head"><h2>Macros <small>{macros.length}</small></h2><button className="btn sm" onClick={addMacro}>+ Add</button></div>
            <div className="table-edit">
              <div className="th"><span>ID</span><span>W</span><span>H</span><span /></div>
              {macros.map((m, i) => (
                <div className="tr" key={i}>
                  <input aria-label="Macro ID" value={m.id} onChange={(e) => setM(i, 'id', e.target.value)} />
                  <input aria-label="Macro width" value={m.w} onChange={(e) => setM(i, 'w', e.target.value)} />
                  <input aria-label="Macro height" value={m.h} onChange={(e) => setM(i, 'h', e.target.value)} />
                  <button className="btn icon" aria-label="Remove macro" onClick={() => setMacros((x) => x.filter((_, j) => j !== i))}>✕</button>
                </div>
              ))}
              {macros.length === 0 && <p className="hint">No macros yet.</p>}
            </div>
          </section>

          <section className="panel">
            <div className="panel-head"><h2>Nets <small>{nets.length}</small></h2><button className="btn sm" onClick={addNet}>+ Add</button></div>
            <div className="table-edit nets">
              <div className="th"><span>ID</span><span>Macro IDs</span><span>Wt</span><span /></div>
              {nets.map((n, i) => (
                <div className="tr" key={i}>
                  <input aria-label="Net ID" value={n.id} onChange={(e) => setN(i, 'id', e.target.value)} />
                  <input aria-label="Connected macro IDs" placeholder="0, 1, 2" value={n.ids} onChange={(e) => setN(i, 'ids', e.target.value)} />
                  <input aria-label="Net weight" value={n.weight} onChange={(e) => setN(i, 'weight', e.target.value)} />
                  <button className="btn icon" aria-label="Remove net" onClick={() => setNets((x) => x.filter((_, j) => j !== i))}>✕</button>
                </div>
              ))}
              {nets.length === 0 && <p className="hint">No nets yet.</p>}
            </div>
          </section>

          <section className="panel">
            <h2>Algorithm</h2>
            <label className="field"><span>Method</span>
              <select value={alg} onChange={(e) => { setAlg(e.target.value); if (runs[e.target.value]) setView(e.target.value) }}>
                {ALGS.map((a) => <option key={a.key} value={a.key} disabled={!isAvail(a.key)}>{a.label}{isAvail(a.key) ? '' : ' (not registered)'}</option>)}
                <option value={rlKey ?? 'rl'} disabled={!rlKey}>Reinforcement Learning{rlKey ? '' : ' — unavailable'}</option>
              </select>
            </label>
            {!rlKey && <p className="hint">RL is not registered in the backend’s algorithm registry, so it cannot be run.</p>}
            <label className="field"><span>Random seed</span><input inputMode="numeric" value={seed} onChange={(e) => setSeed(e.target.value)} /></label>
            {alg === 'sa' || alg === 'simulated_annealing'
              ? (
                <div className="row3">
                  <label className="field"><span>Max iters</span><input value={saIter} onChange={(e) => setSaIter(e.target.value)} /></label>
                  <label className="field"><span>Init temp</span><input value={saTemp} onChange={(e) => setSaTemp(e.target.value)} /></label>
                  <label className="field"><span>Cooling</span><input value={saCool} onChange={(e) => setSaCool(e.target.value)} /></label>
                </div>
              )
              : <p className="hint">Select Simulated Annealing to edit its parameters (also used by Compare).</p>}
          </section>

          <section className="panel">
            <h2>Netlist</h2>
            {built.errors.length > 0
              ? <ul className="issues">{built.errors.map((e, i) => <li key={i}>{e}</li>)}</ul>
              : <p className="ok-text">✓ Input valid · {built.netlist?.macros.length} macros, {built.netlist?.nets.length} nets</p>}
            <div className="btn-row">
              <button className="btn" onClick={() => void validate()}>Validate with API</button>
              <button className="btn" onClick={resetExample}>Reset example</button>
              <button className="btn" onClick={exportNetlist}>Save netlist</button>
              <button className="btn" onClick={() => fileRef.current?.click()}>Load netlist</button>
              <input ref={fileRef} type="file" accept="application/json,.json" hidden onChange={importNetlist} />
            </div>
          </section>
        </aside>

        {/* ───────── Main Content ───────── */}
        <main className="main">
          {notice && (
            <div className={`banner banner-${notice.kind}`} role="alert">
              <div>{notice.lines.map((l, i) => <div key={i}>{l}</div>)}</div>
              <button className="btn icon" aria-label="Dismiss" onClick={() => setNotice(null)}>✕</button>
            </div>
          )}

          <section className="panel runbar">
            <div className="btn-row">
              <button className="btn primary" disabled={!!busy || !isAvail(alg)} onClick={() => void runBatch([alg])}>
                {busy === 'single' ? 'Running…' : `▶ Run ${algLabel(alg)}`}
              </button>
              <button className="btn accent" disabled={!!busy} onClick={() => void runBatch(compareAlgs)}>
                {busy === 'compare' ? 'Comparing…' : `⇄ Compare ${compareAlgs.length} algorithms`}
              </button>
              <button className="btn" disabled={!busy} onClick={() => abortRef.current?.abort()}>■ Cancel</button>
              <button className="btn" disabled={!!busy || !hasRuns} onClick={() => { setRuns({}); setRudyData({}); setNotice(null) }}>Clear results</button>
            </div>
            <p className="hint">Compare runs {compareAlgs.map(algLabel).join(', ')} sequentially on the same netlist, seed and parameters. Elapsed time is measured in your browser (submit → finished).</p>
          </section>

          <section className="panel">
            <div className="panel-head wrap">
              <h2>Floorplan <small>{shownNl.grid_w}×{shownNl.grid_h}</small></h2>
              <div className="tabs" role="tablist">
                {allKeys.map((k) => {
                  const r = runs[k]
                  return (
                    <button key={k} role="tab" aria-selected={view === k} disabled={!r} className={`tab ${view === k ? 'active' : ''}`} onClick={() => setView(k)}
                      title={r ? r.status : RL_RE.test(k) && !rlKey ? 'RL unavailable in backend' : 'Not run yet'}>
                      {r && <span className={`pip pip-${r.status}`} />}{RL_RE.test(k) ? 'RL' : k === 'sa' ? 'SA' : algLabel(k)}
                    </button>
                  )
                })}
              </div>
              <div className="toggles">
                <label><input type="checkbox" checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} /> Grid</label>
                <label><input type="checkbox" checked={showNets} onChange={(e) => setShowNets(e.target.checked)} /> Nets</label>
                <label><input type="checkbox" checked={showLabels} onChange={(e) => setShowLabels(e.target.checked)} /> Labels</label>
                <label><input type="checkbox" checked={showHeatmap} onChange={(e) => setShowHeatmap(e.target.checked)} /> RUDY Heatmap</label>
              </div>
            </div>

            <div className="stage">
              <Floorplan
                nl={shownNl}
                placement={rec?.placement ?? []}
                showGrid={showGrid}
                showNets={showNets}
                showLabels={showLabels}
                showHeatmap={showHeatmap}
                rudy={curRudy}
              />
              {!rec && <div className="overlay"><strong>No placement yet</strong><span>Configure a netlist and press Run. Placements and RUDY density maps are rendered from API responses.</span></div>}
              {rec && running && <div className="overlay light"><div className="spinner" /><span>{rec.status === 'submitting' ? 'Submitting…' : `${algLabel(rec.alg)} · ${Math.round(rec.progress * 100)}% (step ${rec.step}/${rec.total})`}</span></div>}
              {rec?.status === 'failed' && <div className="overlay err"><strong>Run failed</strong><span>{rec.error}</span></div>}
              {rec?.status === 'cancelled' && <div className="overlay"><strong>Cancelled</strong><span>{rec.error}</span></div>}
              {rec?.status === 'completed' && rec.placement.length === 0 && <div className="overlay"><strong>Empty result</strong><span>The API returned no placed macros.</span></div>}
            </div>

            {showHeatmap && curRudy && (
              <div className="heatmap-legend">
                <span>RUDY Wire Demand:</span>
                <span className="heat-box low">0.0 (low)</span>
                <div className="heatmap-bar" />
                <span className="heat-box high">{curRudy.max_demand.toFixed(1)} (hotspot)</span>
                <span className="hint">· Congestion score C_score: <strong>{curRudy.congestion_score.toFixed(3)}</strong></span>
              </div>
            )}

            {rec && stale(rec) && <p className="warn-text">⚠ The configuration changed since this result was produced; the layout shows the original netlist.</p>}
            <p className="hint">Origin (0, 0) is the top-left corner; x grows right, y grows down. Net lines run from each net’s centroid to its macro centers.</p>

            {/* Optimization Convergence Curve */}
            {rec?.trace && rec.trace.length > 1 && (
              <ConvergenceChart trace={rec.trace} totalSteps={rec.total} />
            )}
          </section>

          {/* Metric Cards (Expanded for HPWL, Congestion, and Combined Objective) */}
          <section className="cards">
            <div className="card"><span>HPWL{rec && rec.status !== 'completed' ? ' (partial)' : ''}</span><strong>{fmt(rec?.hpwl)}</strong></div>
            <div className="card"><span>Congestion (C_score)</span><strong>{fmt(rec?.congestion, 3)}</strong></div>
            <div className="card"><span>Objective J(P)</span><strong>{fmt(rec?.objective, 3)}</strong></div>
            <div className="card"><span>Status</span><strong>{rec ? <span className={stCls(rec.status)}>{rec.status}</span> : '—'}</strong></div>
            <div className="card"><span>Progress</span><strong>{rec ? `${rec.step}/${rec.total}` : '—'}</strong>{rec && <div className="bar"><i style={{ width: `${rec.progress * 100}%` }} /></div>}</div>
            <div className="card"><span>Elapsed (client)</span><strong>{fmtMs(rec?.elapsedMs ?? null)}</strong></div>
            <div className="card"><span>Legality check</span><strong>{rec?.status === 'completed' ? (issues.length ? <span className="st st-failed">issues</span> : <span className="st st-completed">legal</span>) : '—'}</strong>{issues.length > 0 && <small>{issues.slice(0, 2).join('; ')}</small>}</div>
            <div className="card"><span>Run ID</span><strong className="mono">{rec?.runId ? rec.runId.slice(0, 8) : '—'}</strong>{rec && <small>seed {rec.seed}</small>}</div>
          </section>

          {/* ───────── Compare ───────── */}
          <section className="panel">
            <div className="panel-head"><h2>Comparison</h2></div>
            <div className="compare-grid">
              {allKeys.map((k) => {
                const r = runs[k]
                const na = (RL_RE.test(k) && !rlKey) || !isAvail(k)
                const rRudy = r?.runId ? rudyData[r.runId] : null
                return (
                  <article key={k} className={`cmp ${r?.status === 'completed' && r.hpwl === best && finished.length > 1 ? 'best' : ''} ${na ? 'na' : ''}`}>
                    <header><strong>{algLabel(k)}</strong>{na ? <span className="st st-na">unavailable</span> : <span className={stCls(r?.status ?? 'idle')}>{r?.status ?? 'not run'}</span>}</header>
                    {na ? (
                      <p className="hint">{RL_RE.test(k) ? 'Not registered in the backend registry (src/placement/jobs.py), so no RL result is shown.' : 'Not registered in the backend.'}</p>
                    ) : r ? (
                      <>
                        <Floorplan
                          nl={r.netlist}
                          placement={r.placement}
                          showGrid={showGrid}
                          showNets={false}
                          showLabels={showLabels}
                          showHeatmap={showHeatmap}
                          rudy={rRudy}
                          compact
                        />
                        <div className="cmp-meta">
                          <span>HPWL</span><strong>{r.status === 'completed' ? fmt(r.hpwl) : '—'}</strong>
                        </div>
                        <div className="cmp-meta">
                          <span>Congestion</span><strong>{r.status === 'completed' ? fmt(r.congestion, 2) : '—'}</strong>
                        </div>
                        <div className="cmp-meta">
                          <span>Objective</span><strong>{r.status === 'completed' ? fmt(r.objective, 2) : '—'}</strong>
                        </div>
                        {r.trace.length > 1 && (
                          <ConvergenceChart trace={r.trace} totalSteps={r.total} compact />
                        )}
                        {r.status === 'failed' && <p className="err-text">{r.error}</p>}
                        <button className="btn sm" onClick={() => setView(k)}>View large</button>
                      </>
                    ) : <p className="hint">Not run yet — press Compare.</p>}
                  </article>
                )
              })}
            </div>

            <div className="table-wrap">
              <table className="metrics">
                <thead>
                  <tr>
                    <th>Algorithm</th>
                    <th>Status</th>
                    <th className="num">HPWL</th>
                    <th className="num">Congestion</th>
                    <th className="num">Objective J(P)</th>
                    <th className="num">vs best</th>
                    <th className="num">Steps</th>
                    <th className="num">Elapsed</th>
                    <th>Run ID</th>
                  </tr>
                </thead>
                <tbody>
                  {allKeys.map((k) => {
                    const r = runs[k]
                    const na = (RL_RE.test(k) && !rlKey) || !isAvail(k)
                    const ok = r?.status === 'completed' && r.hpwl != null
                    return (
                      <tr key={k} className={ok && r.hpwl === best && finished.length > 1 ? 'best' : ''}>
                        <td>{algLabel(k)}</td>
                        <td>{na ? <span className="st st-na">unavailable</span> : <span className={stCls(r?.status ?? 'idle')}>{r?.status ?? 'not run'}</span>}</td>
                        <td className="num">{ok ? fmt(r.hpwl) : '—'}</td>
                        <td className="num">{ok ? fmt(r.congestion, 3) : '—'}</td>
                        <td className="num">{ok ? fmt(r.objective, 3) : '—'}</td>
                        <td className="num">{ok && best != null && best > 0 ? (r.hpwl === best ? 'best' : `+${(((r.hpwl as number) / best - 1) * 100).toFixed(1)}%`) : '—'}</td>
                        <td className="num">{r ? `${r.step}/${r.total}` : '—'}</td>
                        <td className="num">{fmtMs(r?.elapsedMs ?? null)}</td>
                        <td className="mono">{r?.runId?.slice(0, 8) ?? '—'}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <p className="hint">All metrics (HPWL, RUDY Congestion score, and combined objective J(P) = α·HPWL + β·C_score) are computed via canonical PlacementEngine objectives.</p>
          </section>

          {/* ───────── Export ───────── */}
          <section className="panel">
            <div className="panel-head"><h2>Export</h2></div>
            <div className="btn-row">
              <button className="btn" disabled={!hasRuns} onClick={exportJson}>Results · JSON</button>
              <button className="btn" disabled={!exportable.length} onClick={exportPlacementsCsv}>Placements · CSV</button>
              <button className="btn" disabled={!hasRuns} onClick={exportMetricsCsv}>Metrics · CSV</button>
            </div>
            <p className="hint">{hasRuns ? `${exportable.length} completed run(s) available.` : 'Run an algorithm to enable exports.'}</p>
          </section>

          {/* ───────── Backend history ───────── */}
          <section className="panel">
            <div className="panel-head"><h2>Backend run history</h2><button className="btn sm" onClick={() => void loadHistory()}>{history ? 'Refresh' : 'Load'}</button></div>
            {histErr && <p className="err-text">{histErr}</p>}
            {history && (history.length === 0 ? <p className="hint">The backend has no runs yet.</p> : (
              <div className="table-wrap"><table className="metrics">
                <thead><tr><th>Run ID</th><th>Status</th><th className="num">Progress</th><th className="num">HPWL</th><th className="num">Placed</th></tr></thead>
                <tbody>{history.map((h) => (
                  <tr key={h.run_id}><td className="mono">{h.run_id.slice(0, 8)}</td><td><span className={stCls(h.status)}>{h.status}</span></td>
                    <td className="num">{Math.round(h.progress * 100)}%</td><td className="num">{fmt(h.current_hpwl)}</td><td className="num">{h.placement.length}</td></tr>
                ))}</tbody>
              </table></div>
            ))}
          </section>
        </main>
      </div>
    </div>
  )
}
