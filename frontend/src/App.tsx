import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import './App.css'
import { ExperimentsView } from './ExperimentsView'

/* ───────────────────────── Types (mirror src/placement/api.py) ───────────────────────── */
type Status = 'idle' | 'submitting' | 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
type ViewMode = 'placement' | 'connectivity' | 'congestion'
type BottomTab = 'convergence' | 'comparison' | 'history' | 'export'
type ConfigSubTab = 'design' | 'macros' | 'nets'

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
  { key: 'greedy', label: 'Greedy' },
  { key: 'sa', label: 'Simulated Annealing' },
  { key: 'random', label: 'Random' },
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
const fmt = (n: number | null | undefined, d = 2) =>
  n == null || Number.isNaN(n) ? '—' : n.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d })
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
    throw new Error(`Cannot reach API at ${base}. Ensure the backend (uvicorn) is running.`)
  }
  const text = await res.text()
  let body: unknown = text
  try { body = text ? JSON.parse(text) : null } catch { /* keep raw */ }
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
    if (W > 0 && H > 0 && (w > W || h > H)) errors.push(`Macro ${id} (${w}×${h}) exceeds grid boundary (${W}×${H}).`)
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

function checkLegal(nl: Netlist, pl: Placed[]): string[] {
  const issues: string[] = []
  const dims = new Map(nl.macros.map((m) => [m.id, m]))
  const rects: { id: number; x: number; y: number; w: number; h: number }[] = []
  for (const p of pl) {
    const m = dims.get(p.macro_id)
    if (!m) { issues.push(`Unknown macro ${p.macro_id}`); continue }
    if (p.x < 0 || p.y < 0 || p.x + m.w > nl.grid_w || p.y + m.h > nl.grid_h) issues.push(`Macro M${m.id} out of bounds`)
    rects.push({ id: m.id, x: p.x, y: p.y, w: m.w, h: m.h })
  }
  for (let i = 0; i < rects.length; i++)
    for (let j = i + 1; j < rects.length; j++) {
      const a = rects[i], b = rects[j]
      if (a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h) issues.push(`Macros M${a.id} and M${b.id} overlap`)
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
      if (performance.now() - t0 > RUN_TIMEOUT_MS) throw new Error('Timed out waiting for placement completion.')
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
      ? { status: 'cancelled', error: 'Placement cancelled.', elapsedMs: performance.now() - t0 }
      : { status: 'failed', error: errMsg(e), elapsedMs: performance.now() - t0 })
  }
}

/* ───────────────────────── SVG Floorplan & Heatmap ───────────────────────── */
function Floorplan({
  nl, placement, showGrid, showNets, showLabels, showHeatmap, rudy, compact, selectedMacroId, onSelectMacro,
}: {
  nl: Netlist; placement: Placed[]; showGrid: boolean; showNets: boolean; showLabels: boolean
  showHeatmap?: boolean; rudy?: RudyData | null; compact?: boolean
  selectedMacroId?: number | null; onSelectMacro?: (id: number | null) => void
}) {
  const { grid_w: W, grid_h: H } = nl
  const p = Math.max(W, H) * 0.08
  const fs = p * 0.42
  const step = [1, 2, 5, 10, 20, 50, 100, 200, 500].find((s) => W / s <= 14 && H / s <= 14) ?? 1000
  const pos = new Map(placement.map((q) => [q.macro_id, q]))
  const dims = new Map(nl.macros.map((m) => [m.id, m]))
  const center = (id: number) => {
    const m = dims.get(id), q = pos.get(id)
    return m && q ? { x: q.x + m.w / 2, y: q.y + m.h / 2 } : null
  }
  const ticks = (n: number) => Array.from({ length: Math.floor(n / step) + 1 }, (_, i) => i * step)
  const minor = W * H <= 2500

  // Filter connected nets when a macro is selected
  const connectedNets = useMemo(() => {
    if (selectedMacroId == null) return new Set<number>()
    return new Set(nl.nets.filter((n) => n.macro_ids.includes(selectedMacroId)).map((n) => n.id))
  }, [nl.nets, selectedMacroId])

  return (
    <svg
      className="fp-svg"
      viewBox={`${-p} ${-p} ${W + p * 1.5} ${H + p * 1.5}`}
      role="img"
      aria-label={`Floorplan ${W} by ${H} grid with ${placement.length} placed macros`}
      onClick={(e) => {
        if (e.target === e.currentTarget || (e.target as SVGElement).classList.contains('fp-chip')) {
          onSelectMacro?.(null)
        }
      }}
    >
      <defs>
        <pattern id="eda-grid-minor" width="1" height="1" patternUnits="userSpaceOnUse">
          <path d="M 1 0 L 0 0 0 1" fill="none" stroke="var(--grid-minor)" strokeWidth="0.04" />
        </pattern>
      </defs>

      {/* Chip Die Substrate */}
      <rect x={0} y={0} width={W} height={H} className="fp-chip" />

      {/* RUDY Wire Demand Heatmap */}
      {showHeatmap && rudy && rudy.demand && rudy.demand.map((row, y) =>
        row.map((val, x) => {
          if (val <= 0.001) return null
          const maxD = rudy.max_demand > 0 ? rudy.max_demand : 1.0
          const ratio = Math.min(1.0, val / maxD)
          // Cold blue-cyan (210) to warning amber (45) to hot red (0)
          const hueVal = Math.max(0, 210 - ratio * 210)
          const alpha = 0.22 + ratio * 0.65
          return (
            <rect
              key={`rudy-${x}-${y}`}
              x={x}
              y={y}
              width={1}
              height={1}
              fill={`hsla(${hueVal}, 96%, 50%, ${alpha})`}
              vectorEffect="non-scaling-stroke"
            >
              <title>{`Cell (${x}, ${y}): Wire Demand = ${val.toFixed(2)}`}</title>
            </rect>
          )
        })
      )}

      {/* Grid Lines */}
      {showGrid && (
        <g className="fp-grid-group">
          {minor && <rect x={0} y={0} width={W} height={H} fill="url(#eda-grid-minor)" />}
          {ticks(W).map((t) => (
            <line key={`mx${t}`} x1={t} y1={0} x2={t} y2={H} className="fp-major" vectorEffect="non-scaling-stroke" />
          ))}
          {ticks(H).map((t) => (
            <line key={`my${t}`} x1={0} y1={t} x2={W} y2={t} className="fp-major" vectorEffect="non-scaling-stroke" />
          ))}
        </g>
      )}

      {/* Rulers / Coordinate Labels */}
      {!compact && (
        <g className="fp-rulers" aria-hidden="true">
          {ticks(W).map((t) => (
            <text key={`tx${t}`} x={t} y={-fs * 0.45} fontSize={fs * 0.9} textAnchor="middle" className="fp-tick">
              {t}
            </text>
          ))}
          {ticks(H).map((t) => (
            <text key={`ty${t}`} x={-fs * 0.4} y={t + fs * 0.35} fontSize={fs * 0.9} textAnchor="end" className="fp-tick">
              {t}
            </text>
          ))}
          <text x={-fs * 0.4} y={-fs * 0.45} fontSize={fs * 0.8} textAnchor="end" className="fp-tick-origin">
            (0,0)
          </text>
        </g>
      )}

      {/* Macros */}
      {placement.map((q) => {
        const m = dims.get(q.macro_id)
        if (!m) return null
        const label = `M${m.id}`
        const isSelected = selectedMacroId === m.id
        const f = Math.min(m.h * 0.28, m.w / (label.length * 0.72), 1.6)
        const two = m.h >= 2.8 && m.w >= 3.2 && !compact
        const macroOpacity = showHeatmap ? 0.88 : 0.94

        return (
          <g
            key={q.macro_id}
            className={`fp-macro-group ${isSelected ? 'selected' : ''}`}
            onClick={(e) => {
              e.stopPropagation()
              onSelectMacro?.(isSelected ? null : m.id)
            }}
            style={{ cursor: 'pointer' }}
          >
            <rect
              x={q.x}
              y={q.y}
              width={m.w}
              height={m.h}
              fill={`hsl(${hue(m.id)} 62% 62% / ${macroOpacity})`}
              stroke={isSelected ? '#E5E7EB' : 'rgba(10, 16, 36, 0.45)'}
              strokeWidth={isSelected ? 0.35 : 0.12}
              className="fp-macro"
              vectorEffect="non-scaling-stroke"
            >
              <title>{`Macro M${m.id} (${m.w}×${m.h}) placed at (${q.x}, ${q.y})`}</title>
            </rect>
            {showLabels && f > 0.32 && (
              <text
                x={q.x + m.w / 2}
                y={q.y + m.h / 2 + (two ? -f * 0.12 : f * 0.35)}
                fontSize={f}
                textAnchor="middle"
                className="fp-label"
              >
                {label}
                {two && (
                  <tspan x={q.x + m.w / 2} dy={f * 1.05} fontSize={f * 0.65} fontWeight={400}>
                    {m.w}×{m.h}
                  </tspan>
                )}
              </text>
            )}
          </g>
        )
      })}

      {/* Net Flylines */}
      {showNets && nl.nets.map((n) => {
        const cs = n.macro_ids.map(center).filter((c): c is { x: number; y: number } => c !== null)
        if (cs.length < 2) return null
        const cx = cs.reduce((s, c) => s + c.x, 0) / cs.length
        const cy = cs.reduce((s, c) => s + c.y, 0) / cs.length

        const isHighlighted = selectedMacroId != null && connectedNets.has(n.id)
        const isDimmed = selectedMacroId != null && !connectedNets.has(n.id)

        const strokeCol = isHighlighted ? 'var(--net-highlight)' : 'var(--net-color)'
        const strokeW = isHighlighted
          ? Math.min(1.8, 0.8 + n.weight * 0.25)
          : Math.min(1.2, 0.5 + n.weight * 0.15)
        const netOpacity = isDimmed ? 0.1 : isHighlighted ? 0.95 : 0.55

        if (cs.length === 2) {
          return (
            <g
              key={`n${n.id}`}
              className={`fp-net ${isHighlighted ? 'highlighted' : ''}`}
              stroke={strokeCol}
              strokeWidth={strokeW}
              opacity={netOpacity}
            >
              <title>{`Net ${n.id} (weight ${n.weight}): macros ${n.macro_ids.join(', ')}`}</title>
              <line x1={cs[0].x} y1={cs[0].y} x2={cs[1].x} y2={cs[1].y} vectorEffect="non-scaling-stroke" />
            </g>
          )
        }

        return (
          <g
            key={`n${n.id}`}
            className={`fp-net ${isHighlighted ? 'highlighted' : ''}`}
            stroke={strokeCol}
            strokeWidth={strokeW}
            opacity={netOpacity}
          >
            <title>{`Net ${n.id} (weight ${n.weight}): macros ${n.macro_ids.join(', ')}`}</title>
            {cs.map((c, i) => (
              <line key={i} x1={cx} y1={cy} x2={c.x} y2={c.y} vectorEffect="non-scaling-stroke" />
            ))}
          </g>
        )
      })}

      {/* Chip Die Outer Border */}
      <rect x={0} y={0} width={W} height={H} className="fp-border" vectorEffect="non-scaling-stroke" />
    </svg>
  )
}

/* ───────────────────────── Optimization Trace / Convergence Chart ───────────────────────── */
function ConvergenceChart({ trace, totalSteps, compact }: { trace: TracePoint[]; totalSteps: number; compact?: boolean }) {
  if (trace.length === 0) return null

  const W = compact ? 280 : 640
  const H = compact ? 120 : 180
  const padL = compact ? 38 : 50
  const padR = 16
  const padT = 18
  const padB = compact ? 24 : 32

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
  const initialHpwl = trace[0]?.hpwl ?? maxH
  const reductionPct = initialHpwl > 0 ? (((initialHpwl - minH) / initialHpwl) * 100).toFixed(1) : '0'

  return (
    <div className="trace-chart">
      <div className="trace-head">
        <div className="trace-title">
          <span>OPTIMIZATION CONVERGENCE</span>
          <span className="mono text-accent">HPWL Trace</span>
        </div>
        <div className="trace-stats mono text-xs">
          <span>Init: <strong>{fmt(initialHpwl, 1)}</strong></span>
          <span>Best: <strong className="text-ok">{fmt(minH, 1)}</strong></span>
          <span className="badge badge-ok">-{reductionPct}%</span>
        </div>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" role="img" aria-label="Convergence chart">
        {/* Background Grid */}
        <line x1={padL} y1={padT} x2={padL + plotW} y2={padT} stroke="var(--border)" strokeDasharray="3 3" />
        <line x1={padL} y1={padT + plotH / 2} x2={padL + plotW} y2={padT + plotH / 2} stroke="var(--border)" strokeDasharray="3 3" />
        <line x1={padL} y1={padT + plotH} x2={padL + plotW} y2={padT + plotH} stroke="var(--border)" />
        <line x1={padL} y1={padT} x2={padL} y2={padT + plotH} stroke="var(--border)" />

        {/* Ticks */}
        <text x={padL - 8} y={padT + 4} textAnchor="end" className="chart-tick">{fmt(maxH, 0)}</text>
        <text x={padL - 8} y={padT + plotH} textAnchor="end" className="chart-tick">{fmt(minH, 0)}</text>
        <text x={padL} y={H - 8} textAnchor="start" className="chart-tick">Step 0</text>
        <text x={padL + plotW} y={H - 8} textAnchor="end" className="chart-tick">Step {maxStep}</text>

        {/* Trace Curve */}
        <polyline fill="none" stroke="var(--accent)" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" points={pts} />

        {/* Trace Points */}
        {trace.map((p, i) => (
          <circle key={i} cx={scaleX(p.step)} cy={scaleY(p.hpwl)} r={compact ? 2 : 3} fill="var(--accent)" stroke="#090d16" strokeWidth="1">
            <title>{`Step ${p.step}: HPWL ${p.hpwl.toFixed(1)}${p.congestion != null ? `, Congestion ${p.congestion.toFixed(2)}` : ''}`}</title>
          </circle>
        ))}
      </svg>
    </div>
  )
}

/* ───────────────────────── Main App ───────────────────────── */
export default function App() {
  const [activeTab, setActiveTab] = useState<'placement' | 'experiments'>('placement')
  const [apiUrl, setApiUrl] = useState(DEFAULT_API)
  const [conn, setConn] = useState<{ state: 'checking' | 'ok' | 'down'; msg: string }>({ state: 'checking', msg: 'Connecting…' })
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

  // View Modes & Display Toggles
  const [viewMode, setViewMode] = useState<ViewMode>('placement')
  const [showGrid, setShowGrid] = useState(true)
  const [showNets, setShowNets] = useState(true)
  const [showLabels, setShowLabels] = useState(true)
  const [showHeatmap, setShowHeatmap] = useState(false)

  // Interactive Selection
  const [selectedMacroId, setSelectedMacroId] = useState<number | null>(null)

  // Bottom Analysis Drawer Tab
  const [bottomTab, setBottomTab] = useState<BottomTab>('convergence')
  const [configTab, setConfigTab] = useState<ConfigSubTab>('design')

  const [rudyData, setRudyData] = useState<Record<string, RudyData>>({})
  const [history, setHistory] = useState<HistoryRow[] | null>(null)
  const [histErr, setHistErr] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const base = apiUrl.trim().replace(/\/+$/, '')
  const built = useMemo(() => buildNetlist(gw, gh, macros, nets), [gw, gh, macros, nets])
  const rlKey = registry?.find((n) => RL_RE.test(n)) ?? null
  const isAvail = (k: string) => (registry ? registry.includes(k) : true)

  /* Apply preset view mode changes */
  const handleViewModeChange = (mode: ViewMode) => {
    setViewMode(mode)
    if (mode === 'placement') {
      setShowGrid(true)
      setShowNets(false)
      setShowLabels(true)
      setShowHeatmap(false)
    } else if (mode === 'connectivity') {
      setShowGrid(true)
      setShowNets(true)
      setShowLabels(true)
      setShowHeatmap(false)
    } else if (mode === 'congestion') {
      setShowGrid(true)
      setShowNets(false)
      setShowLabels(true)
      setShowHeatmap(true)
    }
  }

  /* Connection check */
  const checkApi = useCallback(async () => {
    setConn({ state: 'checking', msg: 'Checking API…' })
    try {
      const h = await api<{ status: string; version: string }>(base, '/health')
      setConn({ state: 'ok', msg: `Connected · v${h.version}` })
    } catch (e) {
      setConn({ state: 'down', msg: 'API Offline' })
      setRegistry(null)
      return
    }

    try {
      const algRes = await api<{ algorithms: string[] }>(base, '/api/algorithms')
      setRegistry(algRes.algorithms)
    } catch {
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

    try {
      const bm = await api<{ benchmarks: BenchmarkSummary[] }>(base, '/api/benchmarks')
      setBenchmarks(bm.benchmarks ?? [])
    } catch {
      setBenchmarks([])
    }
  }, [base])

  useEffect(() => {
    const t = setTimeout(() => void checkApi(), 300)
    return () => clearTimeout(t)
  }, [checkApi])

  /* Run orchestration */
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

  /* Config handlers */
  const setM = (i: number, k: keyof MacroRow, v: string) => setMacros((m) => m.map((r, j) => (j === i ? { ...r, [k]: v } : r)))
  const setN = (i: number, k: keyof NetRow, v: string) => setNets((m) => m.map((r, j) => (j === i ? { ...r, [k]: v } : r)))
  const nextId = (rows: { id: string }[]) => String(Math.max(-1, ...rows.map((r) => toInt(r.id)).filter((x) => x >= 0)) + 1)
  const addMacro = () => setMacros((m) => [...m, { id: nextId(m), w: '4', h: '3' }])
  const addNet = () => setNets((n) => [...n, { id: nextId(n), ids: '', weight: '1' }])
  const resetExample = () => { setGw('20'); setGh('16'); setMacros(DEFAULT_MACROS); setNets(DEFAULT_NETS); setSeed('42'); setNotice(null) }

  const loadBenchmarkPreset = async (name: string) => {
    try {
      const nl = await api<Netlist>(base, `/api/benchmarks/${name}`)
      setGw(String(nl.grid_w))
      setGh(String(nl.grid_h))
      setMacros(nl.macros.map((m) => ({ id: String(m.id), w: String(m.w), h: String(m.h) })))
      setNets(nl.nets.map((x) => ({ id: String(x.id), ids: x.macro_ids.join(', '), weight: String(x.weight ?? 1) })))
      setSelectedMacroId(null)
      setNotice({ kind: 'ok', lines: [`Loaded benchmark '${name}': ${nl.macros.length} macros, ${nl.nets.length} nets (${nl.grid_w}×${nl.grid_h}).`] })
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
      setSelectedMacroId(null)
      setNotice({ kind: 'ok', lines: [`Imported netlist from ${f.name}.`] })
    }).catch((err) => setNotice({ kind: 'error', lines: [`Import failed: ${errMsg(err)}`] }))
  }

  /* Derived view data */
  const rec = runs[view]
  const previewNl: Netlist = { version: '1.0', grid_w: toInt(gw) > 0 ? toInt(gw) : 20, grid_h: toInt(gh) > 0 ? toInt(gh) : 16, macros: [], nets: [] }
  const currentConfigSignature = built.netlist ? JSON.stringify(built.netlist) : ''
  const isStale = rec != null && (
    built.errors.length > 0 ||
    currentConfigSignature === '' ||
    JSON.stringify(rec.netlist) !== currentConfigSignature
  )
  const shownNl = built.netlist ?? previewNl
  const isRunCurrent = (r: RunRec | undefined) =>
    r != null && currentConfigSignature !== '' && JSON.stringify(r.netlist) === currentConfigSignature
  const currentFinished = Object.values(runs).filter(
    (r) => isRunCurrent(r) && r.status === 'completed' && r.hpwl != null
  )
  const bestObj = currentFinished.length ? Math.min(...currentFinished.map((r) => (r.objective ?? r.hpwl) as number)) : null
  const issues = rec?.status === 'completed' && !isStale ? checkLegal(rec.netlist, rec.placement) : []
  const running = !!rec && ['submitting', 'pending', 'running'].includes(rec.status)

  /* Automatically deselect macro if it no longer exists in current configuration */
  useEffect(() => {
    if (selectedMacroId != null && built.netlist) {
      if (!built.netlist.macros.some((m) => m.id === selectedMacroId)) {
        setSelectedMacroId(null)
      }
    }
  }, [selectedMacroId, built.netlist])

  /* RUDY Fetch */
  const activeRunId = rec?.runId
  useEffect(() => {
    if (rec?.status === 'completed' && activeRunId && !rudyData[activeRunId]) {
      api<RudyData>(base, `/api/placements/${activeRunId}/rudy`)
        .then((data) => setRudyData((prev) => ({ ...prev, [activeRunId]: data })))
        .catch(() => { /* silent fallback */ })
    }
  }, [rec?.status, activeRunId, base, rudyData])

  const curRudy = !isStale && activeRunId ? rudyData[activeRunId] : null

  /* Selected Macro Details */
  const selectedMacro = useMemo(() => {
    if (selectedMacroId == null) return null
    const m = shownNl.macros.find((x) => x.id === selectedMacroId)
    const p = !isStale ? rec?.placement.find((x) => x.macro_id === selectedMacroId) : undefined
    const connected = shownNl.nets.filter((n) => n.macro_ids.includes(selectedMacroId))
    const isOutOfBounds = p ? p.x < 0 || p.y < 0 || p.x + (m?.w ?? 0) > shownNl.grid_w || p.y + (m?.h ?? 0) > shownNl.grid_h : false
    return { macro: m, placed: p, connectedNets: connected, isOutOfBounds }
  }, [selectedMacroId, shownNl, isStale, rec?.placement])

  /* Exports */
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
    <div className="eda-app">
      {/* ───────── Top Navigation Bar ───────── */}
      <header className="eda-topbar">
        <div className="eda-brand">
          <div className="eda-logo-icon">◈</div>
          <div className="eda-brand-text">
            <h1>MACRO PLACEMENT LAB</h1>
            <p>Graph-Theoretic Floorplanning Research Platform</p>
          </div>
        </div>

        <nav className="eda-nav-tabs" role="tablist" aria-label="Main Navigation">
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'placement'}
            className={`eda-nav-tab ${activeTab === 'placement' ? 'active' : ''}`}
            onClick={() => setActiveTab('placement')}
          >
            <span className="tab-glyph">▦</span> Placement Studio
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'experiments'}
            className={`eda-nav-tab ${activeTab === 'experiments' ? 'active' : ''}`}
            onClick={() => setActiveTab('experiments')}
          >
            <span className="tab-glyph">📊</span> Research Experiments
          </button>
        </nav>

        <div className="eda-top-right">
          <div className={`eda-conn eda-conn-${conn.state}`} title={conn.msg}>
            <span className="eda-conn-dot" />
            <span className="eda-conn-label">{conn.state === 'ok' ? conn.msg : conn.state === 'checking' ? 'Checking…' : 'API Offline'}</span>
            <button className="btn btn-xs" onClick={() => void checkApi()} title="Re-check API connection">
              Recheck
            </button>
          </div>
        </div>
      </header>

      {/* ───────── Main Content Area ───────── */}
      {activeTab === 'placement' ? (
        <div className="eda-workspace-grid">
          {/* ────── Column 1: Configuration Panel ────── */}
          <aside className="eda-sidebar">
            {/* Design Presets Bar */}
            <section className="eda-panel">
              <div className="panel-title">
                <span>BENCHMARK PRESETS</span>
              </div>
              <div className="preset-pill-row">
                {benchmarks.length > 0 ? (
                  benchmarks.map((b) => (
                    <button
                      key={b.name}
                      className="btn-preset"
                      onClick={() => void loadBenchmarkPreset(b.name)}
                      title={`${b.macro_count} macros, ${b.grid_w}×${b.grid_h} grid, ${b.net_count} nets`}
                    >
                      {b.name} <span className="preset-count">{b.macro_count}m</span>
                    </button>
                  ))
                ) : (
                  <>
                    <button className="btn-preset" onClick={() => void loadBenchmarkPreset('toy_4')}>toy_4 4m</button>
                    <button className="btn-preset" onClick={() => void loadBenchmarkPreset('medium_8')}>medium_8 8m</button>
                    <button className="btn-preset" onClick={() => void loadBenchmarkPreset('dense_10')}>dense_10 10m</button>
                  </>
                )}
              </div>
            </section>

            {/* Config Sub-Tabs */}
            <section className="eda-panel flex-1">
              <div className="config-subtabs">
                <button
                  className={`cfg-tab ${configTab === 'design' ? 'active' : ''}`}
                  onClick={() => setConfigTab('design')}
                >
                  Canvas & Rules
                </button>
                <button
                  className={`cfg-tab ${configTab === 'macros' ? 'active' : ''}`}
                  onClick={() => setConfigTab('macros')}
                >
                  Macros ({macros.length})
                </button>
                <button
                  className={`cfg-tab ${configTab === 'nets' ? 'active' : ''}`}
                  onClick={() => setConfigTab('nets')}
                >
                  Nets ({nets.length})
                </button>
              </div>

              {configTab === 'design' && (
                <div className="cfg-tab-pane">
                  <div className="field-grid-2">
                    <label className="eda-field">
                      <span>Grid Width (X)</span>
                      <input inputMode="numeric" value={gw} onChange={(e) => setGw(e.target.value)} />
                    </label>
                    <label className="eda-field">
                      <span>Grid Height (Y)</span>
                      <input inputMode="numeric" value={gh} onChange={(e) => setGh(e.target.value)} />
                    </label>
                  </div>

                  <div className="eda-field" style={{ marginTop: '10px' }}>
                    <span>Random Seed</span>
                    <input inputMode="numeric" value={seed} onChange={(e) => setSeed(e.target.value)} />
                  </div>

                  <div className="eda-field" style={{ marginTop: '10px' }}>
                    <span>API Base URL</span>
                    <input value={apiUrl} onChange={(e) => setApiUrl(e.target.value)} spellCheck={false} />
                  </div>

                  {built.errors.length > 0 ? (
                    <div className="notice-error-box">
                      {built.errors.map((err, idx) => (
                        <div key={idx}>⚠ {err}</div>
                      ))}
                    </div>
                  ) : (
                    <div className="notice-ok-box">
                      ✓ Valid: {built.netlist?.macros.length} macros, {built.netlist?.nets.length} nets ({gw}×{gh} grid).
                    </div>
                  )}

                  <div className="btn-action-row" style={{ marginTop: '12px' }}>
                    <button className="btn btn-sm" onClick={() => void validate()}>Validate API</button>
                    <button className="btn btn-sm" onClick={resetExample}>Reset Example</button>
                    <button className="btn btn-sm" onClick={exportNetlist}>Save JSON</button>
                    <button className="btn btn-sm" onClick={() => fileRef.current?.click()}>Load JSON</button>
                    <input ref={fileRef} type="file" accept="application/json,.json" hidden onChange={importNetlist} />
                  </div>
                </div>
              )}

              {configTab === 'macros' && (
                <div className="cfg-tab-pane">
                  <div className="table-header-row">
                    <span>ID</span>
                    <span>Width</span>
                    <span>Height</span>
                    <button className="btn-xs-add" onClick={addMacro}>+ Add</button>
                  </div>
                  <div className="table-rows-scroll">
                    {macros.map((m, i) => (
                      <div className="table-input-row" key={i}>
                        <input aria-label="Macro ID" value={m.id} onChange={(e) => setM(i, 'id', e.target.value)} />
                        <input aria-label="Macro width" value={m.w} onChange={(e) => setM(i, 'w', e.target.value)} />
                        <input aria-label="Macro height" value={m.h} onChange={(e) => setM(i, 'h', e.target.value)} />
                        <button className="btn-row-del" onClick={() => setMacros((x) => x.filter((_, j) => j !== i))}>✕</button>
                      </div>
                    ))}
                    {macros.length === 0 && <p className="hint">No macros added yet.</p>}
                  </div>
                </div>
              )}

              {configTab === 'nets' && (
                <div className="cfg-tab-pane">
                  <div className="table-header-row nets">
                    <span>Net ID</span>
                    <span>Macro IDs</span>
                    <span>Weight</span>
                    <button className="btn-xs-add" onClick={addNet}>+ Add</button>
                  </div>
                  <div className="table-rows-scroll">
                    {nets.map((n, i) => (
                      <div className="table-input-row nets" key={i}>
                        <input aria-label="Net ID" value={n.id} onChange={(e) => setN(i, 'id', e.target.value)} />
                        <input aria-label="Macro IDs" placeholder="0, 1" value={n.ids} onChange={(e) => setN(i, 'ids', e.target.value)} />
                        <input aria-label="Weight" value={n.weight} onChange={(e) => setN(i, 'weight', e.target.value)} />
                        <button className="btn-row-del" onClick={() => setNets((x) => x.filter((_, j) => j !== i))}>✕</button>
                      </div>
                    ))}
                    {nets.length === 0 && <p className="hint">No nets added yet.</p>}
                  </div>
                </div>
              )}
            </section>

            {/* Algorithm & Run Controls */}
            <section className="eda-panel">
              <div className="panel-title">
                <span>SOLVER METHOD</span>
              </div>
              <div className="eda-field">
                <select value={alg} onChange={(e) => { setAlg(e.target.value); if (runs[e.target.value]) setView(e.target.value) }}>
                  {ALGS.map((a) => (
                    <option key={a.key} value={a.key} disabled={!isAvail(a.key)}>
                      {a.label} {!isAvail(a.key) ? '(not registered)' : ''}
                    </option>
                  ))}
                  <option value={rlKey ?? 'rl'} disabled={!rlKey}>
                    Reinforcement Learning (Maskable PPO) {!rlKey ? '— unavailable' : ''}
                  </option>
                </select>
              </div>

              {alg === 'sa' && (
                <div className="field-grid-3" style={{ marginTop: '8px' }}>
                  <label className="eda-field">
                    <span>Max Iters</span>
                    <input value={saIter} onChange={(e) => setSaIter(e.target.value)} />
                  </label>
                  <label className="eda-field">
                    <span>Init Temp</span>
                    <input value={saTemp} onChange={(e) => setSaTemp(e.target.value)} />
                  </label>
                  <label className="eda-field">
                    <span>Cooling</span>
                    <input value={saCool} onChange={(e) => setSaCool(e.target.value)} />
                  </label>
                </div>
              )}

              {/* Primary Action Buttons */}
              <div className="run-controls-stack" style={{ marginTop: '12px' }}>
                <button
                  className="btn btn-primary btn-run"
                  disabled={!!busy || !isAvail(alg)}
                  onClick={() => void runBatch([alg])}
                >
                  {busy === 'single' ? 'Optimizing…' : `▶ Run ${algLabel(alg)}`}
                </button>

                <button
                  className="btn btn-accent btn-compare"
                  disabled={!!busy}
                  onClick={() => void runBatch(compareAlgs)}
                >
                  {busy === 'compare' ? 'Evaluating All…' : `⇄ Compare ${compareAlgs.length} Algorithms`}
                </button>

                <div className="btn-pair">
                  <button className="btn btn-danger" disabled={!busy} onClick={() => abortRef.current?.abort()}>
                    ■ Cancel
                  </button>
                  <button className="btn" disabled={!!busy || !hasRuns} onClick={() => { setRuns({}); setRudyData({}); setNotice(null) }}>
                    Clear
                  </button>
                </div>
              </div>
            </section>
          </aside>

          {/* ────── Column 2: Floorplan Hero Canvas ────── */}
          <main className="eda-floorplan-hero">
            {notice && (
              <div className={`eda-alert eda-alert-${notice.kind}`}>
                <div>{notice.lines.map((l, idx) => <div key={idx}>{l}</div>)}</div>
                <button className="btn-close" onClick={() => setNotice(null)}>✕</button>
              </div>
            )}

            {/* Floorplan Workspace Toolbar */}
            <div className="floorplan-toolbar">
              <div className="toolbar-left">
                <span className="die-badge mono">
                  CANVAS: {shownNl.grid_w}×{shownNl.grid_h}
                </span>

                {/* Algorithm Switcher Tabs */}
                <div className="alg-tabs">
                  {allKeys.map((k) => {
                    const r = runs[k]
                    return (
                      <button
                        key={k}
                        className={`alg-tab-btn ${view === k ? 'active' : ''}`}
                        disabled={!r}
                        onClick={() => setView(k)}
                        title={r ? `${algLabel(k)} (${r.status})` : 'Not run yet'}
                      >
                        {r && <span className={`tab-pip pip-${r.status}`} />}
                        {RL_RE.test(k) ? 'RL' : k === 'sa' ? 'SA' : algLabel(k)}
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* View Modes & Toggles */}
              <div className="toolbar-right">
                <div className="view-mode-group">
                  <button
                    className={`btn-mode ${viewMode === 'placement' ? 'active' : ''}`}
                    onClick={() => handleViewModeChange('placement')}
                    title="Placement geometry and macro IDs"
                  >
                    Placement
                  </button>
                  <button
                    className={`btn-mode ${viewMode === 'connectivity' ? 'active' : ''}`}
                    onClick={() => handleViewModeChange('connectivity')}
                    title="Flylines from net centroids to macros"
                  >
                    Connectivity
                  </button>
                  <button
                    className={`btn-mode ${viewMode === 'congestion' ? 'active' : ''}`}
                    onClick={() => handleViewModeChange('congestion')}
                    title="RUDY wire routing demand heatmap overlay"
                  >
                    Congestion
                  </button>
                </div>

                <div className="layer-toggles">
                  <label><input type="checkbox" checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} /> Grid</label>
                  <label><input type="checkbox" checked={showNets} onChange={(e) => setShowNets(e.target.checked)} /> Nets</label>
                  <label><input type="checkbox" checked={showLabels} onChange={(e) => setShowLabels(e.target.checked)} /> Labels</label>
                  <label><input type="checkbox" checked={showHeatmap} onChange={(e) => setShowHeatmap(e.target.checked)} /> RUDY</label>
                </div>
              </div>
            </div>

            {/* Central SVG Die Stage */}
            <div className="die-stage">
              <Floorplan
                nl={shownNl}
                placement={isStale ? [] : (rec?.placement ?? [])}
                showGrid={showGrid}
                showNets={showNets}
                showLabels={showLabels}
                showHeatmap={showHeatmap}
                rudy={curRudy}
                selectedMacroId={selectedMacroId}
                onSelectMacro={setSelectedMacroId}
              />

              {/* Stale Configuration Empty State */}
              {isStale && (
                <div className="die-overlay stale-config-state">
                  <div className="empty-glyph">⟲</div>
                  <h3>CONFIGURATION CHANGED</h3>
                  <p>The displayed placement belongs to a previous configuration.</p>
                  <div className="stale-config-summary mono">
                    Current configuration: <strong>{built.netlist ? built.netlist.macros.length : macros.length} macros</strong> ({built.netlist?.grid_w ?? gw}×{built.netlist?.grid_h ?? gh})
                  </div>
                  <p className="stale-config-hint">Generate a new placement to continue analysis.</p>
                  <button
                    className="btn btn-primary btn-sm"
                    disabled={!!busy || built.errors.length > 0 || !isAvail(alg)}
                    onClick={() => void runBatch([alg])}
                  >
                    ▶ Run {algLabel(alg)} for Current Netlist
                  </button>
                </div>
              )}

              {/* Empty State */}
              {!rec && !isStale && (
                <div className="die-overlay empty-state">
                  <div className="empty-glyph">◇</div>
                  <h3>NO PLACEMENT GENERATED</h3>
                  <p>Select a benchmark preset or edit custom netlist dimensions, then run an optimization algorithm.</p>
                  <button className="btn btn-primary btn-sm" onClick={() => void runBatch(['greedy'])}>
                    ▶ Run Greedy Benchmark
                  </button>
                </div>
              )}

              {/* Optimizing State Overlay */}
              {rec && !isStale && running && (
                <div className="die-overlay running-state">
                  <div className="spinner-eda" />
                  <div className="optimizing-text">
                    <h4>OPTIMIZING PLACEMENT</h4>
                    <p className="mono">Algorithm: {algLabel(rec.alg)} · Seed {rec.seed}</p>
                    <div className="progress-track">
                      <div className="progress-fill" style={{ width: `${Math.round(rec.progress * 100)}%` }} />
                    </div>
                    <span className="mono text-xs">Step {rec.step} of {rec.total} ({Math.round(rec.progress * 100)}%)</span>
                  </div>
                </div>
              )}

              {/* Error / Cancelled Overlay */}
              {rec && !isStale && rec?.status === 'failed' && (
                <div className="die-overlay err-state">
                  <h4>PLACEMENT RUN FAILED</h4>
                  <p>{rec.error}</p>
                </div>
              )}
            </div>

            {/* RUDY Demand Colorbar Legend */}
            {showHeatmap && curRudy && (
              <div className="eda-heatmap-legend">
                <span className="legend-label">RUDY WIRE DEMAND:</span>
                <span className="mono text-xs text-accent">0.0 (low)</span>
                <div className="legend-bar-gradient" />
                <span className="mono text-xs text-warn">{curRudy.max_demand.toFixed(1)} (hotspot)</span>
                <span className="legend-score mono">
                  Congestion C_score: <strong className="text-accent">{curRudy.congestion_score.toFixed(3)}</strong>
                </span>
              </div>
            )}
          </main>

          {/* ────── Column 3: Metrics & Inspector ────── */}
          <aside className="eda-inspector-column">
            {/* Primary Placement Result Panel */}
            <section className="eda-panel">
              <div className="panel-title">
                <span>PLACEMENT RESULT</span>
                {rec && (
                  isStale ? (
                    <span className="badge badge-muted">stale config</span>
                  ) : (
                    <span className={`st ${stCls(rec.status)}`}>{rec.status}</span>
                  )
                )}
              </div>

              {/* Combined Objective Hero Display */}
              <div className="metric-hero-box">
                <div className="hero-top">
                  <span className="hero-label">COMBINED OBJECTIVE J(P)</span>
                  <span className="badge-lower-better">LOWER IS BETTER</span>
                </div>
                <div className="hero-value mono">
                  {rec?.status === 'completed' && !isStale && rec.objective != null ? fmt(rec.objective, 2) : '—'}
                </div>
                <div className="hero-formula mono text-xs">
                  J(P) = 1.0·HPWL + 1.0·Congestion
                </div>
              </div>

              {/* Secondary Metrics Grid */}
              <div className="metric-tile-grid">
                <div className="metric-tile">
                  <span className="tile-label">HPWL (WIRELENGTH)</span>
                  <strong className="mono">{rec?.status === 'completed' && !isStale ? fmt(rec.hpwl, 1) : '—'}</strong>
                </div>
                <div className="metric-tile">
                  <span className="tile-label">RUDY CONGESTION</span>
                  <strong className="mono">{rec?.status === 'completed' && !isStale ? fmt(rec.congestion, 3) : '—'}</strong>
                </div>
                <div className="metric-tile">
                  <span className="tile-label">ELAPSED RUNTIME</span>
                  <strong className="mono">{!isStale ? fmtMs(rec?.elapsedMs ?? null) : '—'}</strong>
                </div>
                <div className="metric-tile">
                  <span className="tile-label">PLACEMENT STEPS</span>
                  <strong className="mono">{rec && !isStale ? `${rec.step}/${rec.total}` : '—'}</strong>
                </div>
              </div>

              {/* Legality Verification Card */}
              <div className="legality-card">
                <div className="legality-header">
                  <span>LEGALITY VERIFICATION</span>
                  {rec?.status === 'completed' && !isStale ? (
                    issues.length === 0 ? (
                      <span className="badge badge-ok">✓ LEGAL PLACEMENT</span>
                    ) : (
                      <span className="badge badge-err">⚠ {issues.length} ISSUE(S)</span>
                    )
                  ) : (
                    <span className="badge badge-muted">AWAITING RUN</span>
                  )}
                </div>
                {rec?.status === 'completed' && !isStale && issues.length === 0 && (
                  <p className="legality-desc text-ok">
                    All {rec.placement.length} macros strictly contained within chip boundaries without overlapping.
                  </p>
                )}
                {rec?.status === 'completed' && !isStale && issues.length > 0 && (
                  <ul className="issues-list mono text-xs">
                    {issues.slice(0, 3).map((iss, i) => (
                      <li key={i}>{iss}</li>
                    ))}
                  </ul>
                )}
                {(!rec || rec.status !== 'completed' || isStale) && (
                  <p className="hint" style={{ fontSize: '11px', marginTop: '4px' }}>
                    {isStale
                      ? 'Awaiting run for current configuration.'
                      : 'Awaiting placement run to verify boundary constraints and macro overlap.'}
                  </p>
                )}
              </div>
            </section>

            {/* Interactive Macro Inspector */}
            <section className="eda-panel flex-1">
              <div className="panel-title">
                <span>MACRO INSPECTOR</span>
                {selectedMacro && (
                  <button className="btn btn-xs" onClick={() => setSelectedMacroId(null)}>✕ Deselect</button>
                )}
              </div>

              {selectedMacro ? (
                <div className="inspector-content">
                  <div className="inspector-head">
                    <span className="inspector-badge mono font-bold">M{selectedMacro.macro?.id}</span>
                    <span className="mono text-xs">{selectedMacro.macro?.w} × {selectedMacro.macro?.h} units</span>
                  </div>

                  <div className="inspector-fields">
                    <div className="insp-row">
                      <span>Coordinates (X, Y)</span>
                      <strong className="mono text-accent">
                        {selectedMacro.placed ? `(${selectedMacro.placed.x}, ${selectedMacro.placed.y})` : 'Unplaced'}
                      </strong>
                    </div>
                    <div className="insp-row">
                      <span>Dimensions (W × H)</span>
                      <strong className="mono">{selectedMacro.macro?.w} × {selectedMacro.macro?.h}</strong>
                    </div>
                    <div className="insp-row">
                      <span>Boundary Check</span>
                      <span className={`badge ${selectedMacro.isOutOfBounds ? 'badge-err' : 'badge-ok'}`}>
                        {selectedMacro.isOutOfBounds ? 'Out of bounds' : 'Within Canvas'}
                      </span>
                    </div>
                  </div>

                  <div className="insp-nets-box">
                    <span className="insp-subhead">CONNECTED NETS ({selectedMacro.connectedNets.length})</span>
                    <div className="insp-nets-list">
                      {selectedMacro.connectedNets.map((n) => (
                        <div key={n.id} className="insp-net-item mono text-xs">
                          <span className="text-accent">Net {n.id}</span>
                          <span>wt: {n.weight}</span>
                          <span className="text-muted">peers: [{n.macro_ids.filter((id) => id !== selectedMacroId).join(', ')}]</span>
                        </div>
                      ))}
                      {selectedMacro.connectedNets.length === 0 && (
                        <p className="hint">No nets attached to this macro.</p>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="inspector-placeholder">
                  <p className="hint">
                    Click any macro on the floorplan to inspect coordinates, dimensions, connected nets, and boundary legality.
                  </p>
                </div>
              )}
            </section>
          </aside>

          {/* ────── Bottom Section: Analysis, Trace & Comparison ────── */}
          <section className="eda-bottom-drawer">
            <div className="drawer-header">
              <div className="drawer-tabs">
                <button
                  className={`drawer-tab ${bottomTab === 'convergence' ? 'active' : ''}`}
                  onClick={() => setBottomTab('convergence')}
                >
                  Convergence Curve
                </button>
                <button
                  className={`drawer-tab ${bottomTab === 'comparison' ? 'active' : ''}`}
                  onClick={() => setBottomTab('comparison')}
                >
                  Algorithm Comparison ({currentFinished.length})
                </button>
                <button
                  className={`drawer-tab ${bottomTab === 'history' ? 'active' : ''}`}
                  onClick={() => { setBottomTab('history'); void loadHistory() }}
                >
                  Backend Run History
                </button>
                <button
                  className={`drawer-tab ${bottomTab === 'export' ? 'active' : ''}`}
                  onClick={() => setBottomTab('export')}
                >
                  Export Center
                </button>
              </div>

              {rec?.runId && !isStale && (
                <div className="mono text-xs text-muted">
                  Active Run: <span className="text-accent">{rec.runId.slice(0, 8)}</span> (Seed {rec.seed})
                </div>
              )}
            </div>

            <div className="drawer-body">
              {/* Convergence Tab */}
              {bottomTab === 'convergence' && (
                <div className="drawer-pane">
                  {!isStale && rec?.trace && rec.trace.length > 1 ? (
                    <ConvergenceChart trace={rec.trace} totalSteps={rec.total} />
                  ) : (
                    <p className="hint">
                      {isStale
                        ? `Configuration changed. Run ${algLabel(alg)} to generate a new convergence curve for the current netlist.`
                        : 'Run an optimization algorithm (Greedy, SA, or Random) to generate a step-by-step convergence curve.'}
                    </p>
                  )}
                </div>
              )}

              {/* Comparison Tab */}
              {bottomTab === 'comparison' && (
                <div className="drawer-pane">
                  <div className="compare-mini-cards">
                    {allKeys.map((k) => {
                      const r = runs[k]
                      const na = (RL_RE.test(k) && !rlKey) || !isAvail(k)
                      const isCurrent = isRunCurrent(r)
                      const isOk = isCurrent && r?.status === 'completed'
                      const isBest = isOk && (r.objective ?? r.hpwl) === bestObj && currentFinished.length > 1
                      return (
                        <div key={k} className={`compare-card ${isBest ? 'best' : ''} ${na ? 'na' : ''}`}>
                          <div className="cmp-head">
                            <strong className="mono">{algLabel(k)}</strong>
                            {na ? (
                              <span className="badge badge-muted">Unavailable</span>
                            ) : !isCurrent && r ? (
                              <span className="badge badge-muted">stale config</span>
                            ) : (
                              <span className={`st ${stCls(r?.status ?? 'idle')}`}>{r?.status ?? 'idle'}</span>
                            )}
                          </div>
                          <div className="cmp-metrics mono">
                            <div><span>Objective J(P):</span> <strong>{isOk ? fmt(r.objective) : '—'}</strong></div>
                            <div><span>HPWL:</span> <strong>{isOk ? fmt(r.hpwl) : '—'}</strong></div>
                            <div><span>Congestion:</span> <strong>{isOk ? fmt(r.congestion, 3) : '—'}</strong></div>
                            <div><span>Time:</span> <strong>{isCurrent ? fmtMs(r?.elapsedMs ?? null) : '—'}</strong></div>
                          </div>
                          {isBest && <span className="badge badge-ok font-bold">★ Lowest Objective</span>}
                        </div>
                      )
                    })}
                  </div>

                  <div className="table-wrap" style={{ marginTop: '12px' }}>
                    <table className="metrics">
                      <thead>
                        <tr>
                          <th>Algorithm</th>
                          <th>Status</th>
                          <th className="num">HPWL</th>
                          <th className="num">Congestion</th>
                          <th className="num">Objective J(P)</th>
                          <th className="num">vs Best</th>
                          <th className="num">Steps</th>
                          <th className="num">Elapsed</th>
                          <th>Run ID</th>
                        </tr>
                      </thead>
                      <tbody>
                        {allKeys.map((k) => {
                          const r = runs[k]
                          const na = (RL_RE.test(k) && !rlKey) || !isAvail(k)
                          const isCurrent = isRunCurrent(r)
                          const ok = isCurrent && r?.status === 'completed' && r.hpwl != null
                          const currentVal = r?.objective ?? r?.hpwl
                          const isBest = ok && currentVal === bestObj && currentFinished.length > 1
                          return (
                            <tr key={k} className={isBest ? 'best' : ''}>
                              <td className="font-bold">{algLabel(k)}</td>
                              <td>
                                {na ? (
                                  <span className="st st-na">unavailable</span>
                                ) : !isCurrent && r ? (
                                  <span className="badge badge-muted">stale config</span>
                                ) : (
                                  <span className={stCls(r?.status ?? 'idle')}>{r?.status ?? 'not run'}</span>
                                )}
                              </td>
                              <td className="num">{ok ? fmt(r.hpwl) : '—'}</td>
                              <td className="num">{ok ? fmt(r.congestion, 3) : '—'}</td>
                              <td className="num font-bold text-accent">{ok ? fmt(r.objective, 2) : '—'}</td>
                              <td className="num font-bold">
                                {ok && bestObj != null && currentVal != null && currentVal > 0
                                  ? (currentVal === bestObj ? 'Best' : `+${(((currentVal / bestObj) - 1) * 100).toFixed(1)}%`)
                                  : '—'}
                              </td>
                              <td className="num">{isCurrent && r ? `${r.step}/${r.total}` : '—'}</td>
                              <td className="num">{isCurrent ? fmtMs(r?.elapsedMs ?? null) : '—'}</td>
                              <td className="mono">{isCurrent ? (r?.runId?.slice(0, 8) ?? '—') : '—'}</td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* History Tab */}
              {bottomTab === 'history' && (
                <div className="drawer-pane">
                  {histErr && <p className="notice-error-box">{histErr}</p>}
                  {history && (history.length === 0 ? <p className="hint">No server placements found.</p> : (
                    <div className="table-wrap">
                      <table className="metrics">
                        <thead>
                          <tr>
                            <th>Run ID</th>
                            <th>Status</th>
                            <th className="num">Progress</th>
                            <th className="num">HPWL</th>
                            <th className="num">Congestion</th>
                            <th className="num">Objective</th>
                            <th className="num">Macros Placed</th>
                          </tr>
                        </thead>
                        <tbody>
                          {history.map((h) => (
                            <tr key={h.run_id}>
                              <td className="mono">{h.run_id.slice(0, 8)}</td>
                              <td><span className={stCls(h.status)}>{h.status}</span></td>
                              <td className="num">{Math.round(h.progress * 100)}%</td>
                              <td className="num">{fmt(h.current_hpwl)}</td>
                              <td className="num">{fmt(h.current_congestion, 3)}</td>
                              <td className="num font-bold text-accent">{fmt(h.current_objective, 2)}</td>
                              <td className="num">{h.placement.length}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ))}
                </div>
              )}

              {/* Export Tab */}
              {bottomTab === 'export' && (
                <div className="drawer-pane">
                  <p className="hint">Export verified placements, runtime metrics, or complete circuit netlists in standard exchange formats:</p>
                  <div className="btn-action-row" style={{ marginTop: '10px' }}>
                    <button className="btn" disabled={!hasRuns} onClick={exportJson}>Results · JSON</button>
                    <button className="btn" disabled={!exportable.length} onClick={exportPlacementsCsv}>Placements · CSV</button>
                    <button className="btn" disabled={!hasRuns} onClick={exportMetricsCsv}>Metrics · CSV</button>
                    <button className="btn" onClick={exportNetlist}>Circuit Netlist · JSON</button>
                  </div>
                  <span className="mono text-xs text-muted" style={{ display: 'block', marginTop: '10px' }}>
                    {exportable.length} completed run(s) ready for export.
                  </span>
                </div>
              )}
            </div>
          </section>
        </div>
      ) : (
        <ExperimentsView apiUrl={base} />
      )}
    </div>
  )
}
