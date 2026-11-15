import { useCallback, useEffect, useState } from 'react'

/* ───────────────────────── Types ───────────────────────── */
interface FigureItem {
  filename: string
  url: string
  title: string
}

interface ExpABenchmark {
  benchmark: string
  display_name?: string
  macros: number
  canvas?: string
  grid_size?: string
  grid_w?: number
  grid_h?: number
  random_mean?: number
  random_obj?: number
  greedy_mean?: number
  greedy_obj?: number
  sa_mean?: number
  sa_obj?: number
  rl_mean?: number
  rl_obj?: number
  best_algorithm: string
  ranking?: string
  rl_vs_random_pct?: number
  rl_improvement_pct?: number
}

interface ExpAData {
  experiment: string
  title?: string
  description?: string
  provenance_label?: string
  rows?: ExpABenchmark[]
  benchmarks?: ExpABenchmark[]
  figures?: FigureItem[]
}

interface ExpBTopology {
  topology: string
  display_name?: string
  macros?: number
  canvas?: string
  grid_size?: string
  random_mean?: number
  random_obj?: number
  greedy_mean?: number
  greedy_obj?: number
  sa_mean?: number
  sa_obj?: number
  rl_mean?: number
  rl_obj?: number
  best_algorithm: string
  ranking?: string
}

interface ExpBSensitivity {
  algorithm: string
  pct_change?: number
  topology_sensitivity_pct?: number
  note?: string
}

interface ExpBData {
  experiment: string
  title?: string
  description?: string
  provenance_label?: string
  controlled_setup?: string
  topologies: ExpBTopology[]
  sensitivities: ExpBSensitivity[]
  figures?: FigureItem[]
}

interface ExpDHoldout {
  condition: string
  display_name?: string
  netlist_id?: string
  macros: number
  canvas?: string
  grid_size?: string
  random_mean?: number
  random_obj?: number
  greedy_mean?: number
  greedy_obj?: number
  sa_mean?: number
  sa_obj?: number
  rl_mean?: number
  rl_obj?: number
  best_algorithm: string
  rl_per_seed?: number[]
  ranking: string
}

interface ExpDGap {
  holdout: string
  condition?: string
  absolute_gap?: number
  percentage_gap?: number
  inst_delta?: number
  inst_gap_pct?: number
  topo_delta?: number
  topo_gap_pct?: number
  interpretation: string
}

interface ExpDData {
  experiment: string
  title?: string
  description?: string
  provenance_label?: string
  controlled_setup?: string
  conditions?: ExpDHoldout[]
  holdout_conditions?: ExpDHoldout[]
  gaps?: ExpDGap[]
  generalization_gaps?: ExpDGap[]
  key_findings?: string[]
  figures?: FigureItem[]
}

interface ReportInfo {
  report_directory?: string
  html_available: boolean
  html_url?: string
  html_size_bytes?: number
  pdf_available: boolean
  pdf_url?: string
  pdf_status?: string
  figures_count?: number
  figures?: { name: string; available?: boolean; url: string }[]
}

interface ExperimentsViewProps {
  apiUrl: string
}

/* ───────────────────────── Helpers ───────────────────────── */
const fmt = (n: number | null | undefined, d = 2) =>
  n == null || Number.isNaN(n) ? '—' : n.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d })

const fmtPct = (n: number | null | undefined) => {
  if (n == null || Number.isNaN(n)) return '—'
  const sign = n > 0 ? '+' : ''
  return `${sign}${n.toFixed(2)}%`
}

/* ───────────────────────── Component ───────────────────────── */
export function ExperimentsView({ apiUrl }: ExperimentsViewProps) {
  const [expA, setExpA] = useState<ExpAData | null>(null)
  const [expB, setExpB] = useState<ExpBData | null>(null)
  const [expD, setExpD] = useState<ExpDData | null>(null)
  const [reports, setReports] = useState<ReportInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [generating, setGenerating] = useState(false)
  const [actionNotice, setActionNotice] = useState<{ kind: 'ok' | 'error'; text: string } | null>(null)
  const [subSection, setSubSection] = useState<'all' | 'a' | 'b' | 'd' | 'reports'>('all')

  const cleanBase = apiUrl.replace(/\/+$/, '')

  const loadData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [resA, resB, resD, resRep] = await Promise.all([
        fetch(`${cleanBase}/api/experiments/a`).then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status} fetching Experiment A`)
          return r.json() as Promise<ExpAData>
        }),
        fetch(`${cleanBase}/api/experiments/b`).then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status} fetching Experiment B`)
          return r.json() as Promise<ExpBData>
        }),
        fetch(`${cleanBase}/api/experiments/d`).then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status} fetching Experiment D`)
          return r.json() as Promise<ExpDData>
        }),
        fetch(`${cleanBase}/api/reports`).then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status} fetching Reports`)
          return r.json() as Promise<ReportInfo>
        }),
      ])
      setExpA(resA)
      setExpB(resB)
      setExpD(resD)
      setReports(resRep)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [cleanBase])

  useEffect(() => {
    void loadData()
  }, [loadData])

  const handleGenerateReport = async () => {
    setGenerating(true)
    setActionNotice(null)
    try {
      const res = await fetch(`${cleanBase}/api/reports/generate`, { method: 'POST' })
      const data = await res.json()
      if (!res.ok) {
        throw new Error(data.detail || data.message || `HTTP ${res.status}`)
      }
      setActionNotice({ kind: 'ok', text: data.message || 'Report regenerated successfully.' })
      const repRes = await fetch(`${cleanBase}/api/reports`)
      if (repRes.ok) {
        setReports(await repRes.json())
      }
    } catch (e) {
      setActionNotice({ kind: 'error', text: `Failed to generate report: ${e instanceof Error ? e.message : String(e)}` })
    } finally {
      setGenerating(false)
    }
  }

  const resolveUrl = (path: string) => {
    if (path.startsWith('http://') || path.startsWith('https://')) return path
    return `${cleanBase}${path.startsWith('/') ? '' : '/'}${path}`
  }

  const expARows = expA?.rows ?? expA?.benchmarks ?? []
  const expBTopologies = expB?.topologies ?? []
  const expBSens = expB?.sensitivities ?? []
  const expDConditions = expD?.conditions ?? expD?.holdout_conditions ?? []
  const expDGaps = expD?.generalization_gaps ?? expD?.gaps ?? []

  return (
    <div className="eda-experiments-container">
      {/* ───────── Overview Header Card ───────── */}
      <section className="eda-exp-hero-card">
        <div className="exp-hero-main">
          <div className="exp-hero-badge mono">EDA RESEARCH & EVALUATION LABORATORY</div>
          <h2>Macro Placement & Floorplanning Benchmarks</h2>
          <p>
            Rigorous empirical comparison of Combinatorial Heuristics (Greedy, Simulated Annealing)
            and Deep Reinforcement Learning (Maskable PPO with CNN spatial embeddings) across scaling, connectivity, and zero-shot generalization.
          </p>
        </div>

        {/* Statistical Overview Counters */}
        <div className="research-stat-counters">
          <div className="stat-counter-box">
            <span className="counter-val mono text-accent">3</span>
            <span className="counter-lbl">Experiments</span>
            <span className="counter-sub">Scaling · Topology · Transfer</span>
          </div>
          <div className="stat-counter-box">
            <span className="counter-val mono text-cyan">4</span>
            <span className="counter-lbl">Algorithms</span>
            <span className="counter-sub">Greedy · SA · RL · Random</span>
          </div>
          <div className="stat-counter-box">
            <span className="counter-val mono text-ok">3</span>
            <span className="counter-lbl">RL Multi-Seeds</span>
            <span className="counter-sub">Seeds 42, 43, 44</span>
          </div>
          <div className="stat-counter-box">
            <span className="counter-val mono text-warn">2</span>
            <span className="counter-lbl">Holdout Regimes</span>
            <span className="counter-sub">Unseen Inst · Clustered Topo</span>
          </div>
        </div>
      </section>

      {/* Sub-Navigation & Report Action Bar */}
      <div className="eda-exp-nav-bar">
        <div className="exp-subtabs">
          <button className={`exp-tab-btn ${subSection === 'all' ? 'active' : ''}`} onClick={() => setSubSection('all')}>
            All Experiments
          </button>
          <button className={`exp-tab-btn ${subSection === 'a' ? 'active' : ''}`} onClick={() => setSubSection('a')}>
            [Exp A] Size Scaling
          </button>
          <button className={`exp-tab-btn ${subSection === 'b' ? 'active' : ''}`} onClick={() => setSubSection('b')}>
            [Exp B] Connectivity
          </button>
          <button className={`exp-tab-btn ${subSection === 'd' ? 'active' : ''}`} onClick={() => setSubSection('d')}>
            [Exp D] Generalization
          </button>
          <button className={`exp-tab-btn ${subSection === 'reports' ? 'active' : ''}`} onClick={() => setSubSection('reports')}>
            Research Reports
          </button>
        </div>

        <div className="exp-report-actions">
          <button
            className="btn btn-primary btn-sm"
            disabled={generating}
            onClick={() => void handleGenerateReport()}
          >
            {generating ? 'Regenerating…' : '⟳ Generate Report'}
          </button>

          {reports?.html_available && reports.html_url && (
            <>
              <a className="btn btn-sm" href={resolveUrl(reports.html_url)} target="_blank" rel="noreferrer">
                ↗ View HTML
              </a>
              <a className="btn btn-sm" href={resolveUrl(reports.html_url)} download="macro_placement_report.html">
                ⬇ Download HTML
              </a>
            </>
          )}

          {reports?.pdf_available && reports.pdf_url ? (
            <a className="btn btn-sm" href={resolveUrl(reports.pdf_url)} download="macro_placement_report.pdf">
              ⬇ Download PDF
            </a>
          ) : (
            <span className="badge badge-muted" title={reports?.pdf_status || 'weasyprint required for PDF'}>
              PDF: {reports?.pdf_status || 'Not generated'}
            </span>
          )}

          <button className="btn btn-sm" onClick={() => void loadData()} title="Refresh API data">
            Refresh
          </button>
        </div>
      </div>

      {actionNotice && (
        <div className={`eda-alert eda-alert-${actionNotice.kind}`}>
          <span>{actionNotice.text}</span>
          <button className="btn-close" onClick={() => setActionNotice(null)}>✕</button>
        </div>
      )}

      {loading && !expA && (
        <div className="eda-loading-box">
          <div className="spinner-eda" />
          <span>Retrieving experimental data from API…</span>
        </div>
      )}

      {error && (
        <div className="eda-alert eda-alert-error">
          <p><strong>Failed to load experiments:</strong> {error}</p>
          <button className="btn btn-sm" onClick={() => void loadData()}>Retry</button>
        </div>
      )}

      {/* ───────── Key Research Findings ───────── */}
      {(subSection === 'all' || subSection === 'reports') && (
        <section className="eda-panel">
          <div className="panel-title">
            <span>CORE EMPIRICAL FINDINGS & METHODOLOGY</span>
            <span className="badge badge-accent mono">CANONICAL MULTI-SEED</span>
          </div>
          <div className="eda-findings-grid">
            <div className="finding-card">
              <div className="finding-num mono">01</div>
              <div className="finding-body">
                <h4>Greedy Heuristic Dominates Runtime and Objectives</h4>
                <p>
                  Deterministic Greedy placement consistently achieves the lowest objective across standard scaling benchmarks
                  (<strong>159.23</strong> on dense_10, <strong>561.60</strong> on scaling_20, <strong>1150.46</strong> on scaling_30) with sub-second execution.
                </p>
              </div>
            </div>

            <div className="finding-card">
              <div className="finding-num mono">02</div>
              <div className="finding-body">
                <h4>Simulated Annealing Excels on Clustered Topologies</h4>
                <p>
                  On the unseen clustered community topology, <strong>Simulated Annealing (280.87)</strong> outperforms Greedy (<strong>282.70</strong>)
                  and exhibits the highest sensitivity to net clustering (<strong>-29.34%</strong>). Greedy does not win every holdout.
                </p>
              </div>
            </div>

            <div className="finding-card">
              <div className="finding-num mono">03</div>
              <div className="finding-body">
                <h4>RL Validates Action Masking but Shows Generalization Gaps</h4>
                <p>
                  Maskable PPO achieves 100% legal placements and improves over Random by <strong>+8.98%</strong> to <strong>+28.95%</strong>, but suffers
                  out-of-distribution degradation: <strong>+31.67%</strong> on unseen instances and <strong>+37.28%</strong> on clustered topologies.
                </p>
              </div>
            </div>

            <div className="finding-card">
              <div className="finding-num mono">04</div>
              <div className="finding-body">
                <h4>Standardized Objective & Controlled Experimental Canvas</h4>
                <p>
                  All runs evaluate identical cost: <code>J(P) = 1.0·HPWL + 1.0·Congestion</code>.
                  Generalization and sensitivity tests isolate graph topology on a controlled 12-macro, 28×28 grid canvas.
                </p>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* ───────── Experiment A: Scaling ───────── */}
      {(subSection === 'all' || subSection === 'a') && expA && (
        <section className="eda-panel">
          <div className="panel-title">
            <div>
              <span>EXPERIMENT A — PROBLEM SIZE SCALING</span>
              <p className="hint">
                Algorithmic scaling from 10 macros (24×24) to 20 macros (36×36) and 30 macros (48×48).
              </p>
            </div>
            <div className="badge-cluster">
              <span className="badge badge-ok" title={expA.provenance_label || 'Canonical multi-seed Experiment A results (36 runs across 3 seeds).'}>
                Canonical Multi-Seed (36 runs)
              </span>
              <span className="badge badge-muted" title="Local CSV preserves 9 single-seed classical baseline runs">
                Local Artifacts (9 rows)
              </span>
            </div>
          </div>

          <div className="table-wrap">
            <table className="metrics">
              <thead>
                <tr>
                  <th>Benchmark</th>
                  <th>Canvas & Macros</th>
                  <th className="num">Random Mean</th>
                  <th className="num">Greedy Mean</th>
                  <th className="num">SA Mean</th>
                  <th className="num">RL (PPO) Mean</th>
                  <th>Best Algorithm</th>
                  <th className="num">RL Improvement vs Random</th>
                </tr>
              </thead>
              <tbody>
                {expARows.map((b) => {
                  const rnd = b.random_mean ?? b.random_obj
                  const grd = b.greedy_mean ?? b.greedy_obj
                  const sa = b.sa_mean ?? b.sa_obj
                  const rl = b.rl_mean ?? b.rl_obj
                  const imp = b.rl_improvement_pct ?? b.rl_vs_random_pct
                  const canvas = b.canvas ?? b.grid_size ?? (b.grid_w ? `${b.grid_w}×${b.grid_h}` : '')
                  return (
                    <tr key={b.benchmark}>
                      <td className="mono font-bold text-accent">{b.display_name || b.benchmark}</td>
                      <td>{b.macros} macros ({canvas})</td>
                      <td className="num">{fmt(rnd)}</td>
                      <td className="num font-bold text-ok">{fmt(grd)}</td>
                      <td className="num">{fmt(sa)}</td>
                      <td className="num font-bold text-accent">{fmt(rl)}</td>
                      <td><span className="badge badge-best">{b.best_algorithm}</span></td>
                      <td className="num font-bold text-ok">{fmtPct(imp)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="hint" style={{ marginTop: '8px' }}>
            <strong>RL Improvement</strong> = (Random Mean - RL Mean) / Random Mean × 100%.
            Maskable PPO achieves <strong>+28.95%</strong> improvement on dense_10, <strong>+8.98%</strong> on scaling_20, and <strong>+11.40%</strong> on scaling_30 over Random placement.
          </p>

          {expA.figures && expA.figures.length > 0 && (
            <div className="eda-figures-grid" style={{ marginTop: '16px' }}>
              {expA.figures.map((fig) => (
                <div key={fig.filename} className="eda-figure-card">
                  <div className="fig-head mono">{fig.title}</div>
                  <img src={resolveUrl(fig.url)} alt={fig.title} loading="lazy" />
                  <div className="fig-caption mono text-xs">Artifact: {fig.filename}</div>
                </div>
              ))}
            </div>
          )}
        </section>
      )}

      {/* ───────── Experiment B: Connectivity ───────── */}
      {(subSection === 'all' || subSection === 'b') && expB && (
        <section className="eda-panel">
          <div className="panel-title">
            <div>
              <span>EXPERIMENT B — CONNECTIVITY SENSITIVITY</span>
              <p className="hint">
                Evaluates algorithmic response to Random (Erdős-Rényi) vs. Clustered (Community) topologies ({expB.controlled_setup || '12 macros, 28×28 grid canvas, 14 nets'}).
              </p>
            </div>
            <span className="badge badge-archival" title={expB.provenance_label || 'Verified archival reconstruction'}>
              Verified Archival Reconstruction
            </span>
          </div>

          <div className="sub-grid-2">
            <div className="table-wrap">
              <h4>Objective by Topology (Random vs Clustered)</h4>
              <table className="metrics">
                <thead>
                  <tr>
                    <th>Topology</th>
                    <th className="num">Random</th>
                    <th className="num">Greedy</th>
                    <th className="num">SA</th>
                    <th className="num">RL (PPO)</th>
                    <th>Best</th>
                  </tr>
                </thead>
                <tbody>
                  {expBTopologies.map((t) => {
                    const rnd = t.random_mean ?? t.random_obj
                    const grd = t.greedy_mean ?? t.greedy_obj
                    const sa = t.sa_mean ?? t.sa_obj
                    const rl = t.rl_mean ?? t.rl_obj
                    return (
                      <tr key={t.topology}>
                        <td className="font-bold">{t.display_name || t.topology}</td>
                        <td className="num">{fmt(rnd)}</td>
                        <td className="num font-bold text-ok">{fmt(grd)}</td>
                        <td className="num">{fmt(sa)}</td>
                        <td className="num font-bold text-accent">{fmt(rl)}</td>
                        <td><span className="badge badge-best">{t.best_algorithm}</span></td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            <div className="table-wrap">
              <h4>Topology Sensitivity (Clustered vs Random)</h4>
              <table className="metrics">
                <thead>
                  <tr>
                    <th>Algorithm</th>
                    <th className="num">Sensitivity Δ</th>
                    <th>Analytical Observation</th>
                  </tr>
                </thead>
                <tbody>
                  {expBSens.map((s) => {
                    const pct = s.topology_sensitivity_pct ?? s.pct_change
                    return (
                      <tr key={s.algorithm}>
                        <td className="font-bold">{s.algorithm}</td>
                        <td className="num font-bold text-warn">{fmtPct(pct)}</td>
                        <td className="hint">{s.note || ''}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {expB.figures && expB.figures.length > 0 && (
            <div className="eda-figures-grid" style={{ marginTop: '16px' }}>
              {expB.figures.map((fig) => (
                <div key={fig.filename} className="eda-figure-card">
                  <div className="fig-head mono">{fig.title}</div>
                  <img src={resolveUrl(fig.url)} alt={fig.title} loading="lazy" />
                  <div className="fig-caption mono text-xs">Artifact: {fig.filename}</div>
                </div>
              ))}
            </div>
          )}
        </section>
      )}

      {/* ───────── Experiment D: Generalization ───────── */}
      {(subSection === 'all' || subSection === 'd') && expD && (
        <section className="eda-panel">
          <div className="panel-title">
            <div>
              <span>EXPERIMENT D — RL GENERALIZATION & ROBUSTNESS</span>
              <p className="hint">
                Zero-shot holdout transfer across training, unseen macro instances, and unseen clustered topologies ({expD.controlled_setup}).
              </p>
            </div>
            <span className="badge badge-archival" title={expD.provenance_label || 'Verified archival reconstruction'}>
              Verified Archival Reconstruction
            </span>
          </div>

          {/* Generalization Gap Highlight Cards */}
          <div className="gen-gap-cards-grid">
            <div className="gen-card">
              <span className="gen-title">TRAINING DISTRIBUTION BASELINE</span>
              <div className="gen-value mono text-accent">335.42</div>
              <span className="hint">exp_d_train_random (Random Topology, 12m, 28×28)</span>
            </div>
            <div className="gen-card">
              <span className="gen-title">UNSEEN MACRO INSTANCE TRANSFER</span>
              <div className="gen-value mono text-warn">441.64</div>
              <span className="badge badge-warn font-bold">+106.22 (+31.67%) Gap</span>
            </div>
            <div className="gen-card">
              <span className="gen-title">UNSEEN CLUSTERED TOPOLOGY TRANSFER</span>
              <div className="gen-value mono text-err">460.45</div>
              <span className="badge badge-err font-bold">+125.04 (+37.28%) Gap</span>
            </div>
          </div>

          <div className="table-wrap" style={{ marginTop: '14px' }}>
            <h4>Holdout Performance & Multi-Seed Breakdown</h4>
            <table className="metrics">
              <thead>
                <tr>
                  <th>Holdout Condition</th>
                  <th>Canvas</th>
                  <th className="num">Random</th>
                  <th className="num">Greedy</th>
                  <th className="num">SA</th>
                  <th className="num">RL (Mean)</th>
                  <th>RL Per-Seed [42, 43, 44]</th>
                  <th>Best Algorithm</th>
                  <th>Algorithm Ranking</th>
                </tr>
              </thead>
              <tbody>
                {expDConditions.map((h) => {
                  const rnd = h.random_mean ?? h.random_obj
                  const grd = h.greedy_mean ?? h.greedy_obj
                  const sa = h.sa_mean ?? h.sa_obj
                  const rl = h.rl_mean ?? h.rl_obj
                  const canvas = h.canvas ?? h.grid_size ?? '28×28'
                  return (
                    <tr key={h.condition}>
                      <td className="mono font-bold text-accent">{h.display_name || h.netlist_id || h.condition}</td>
                      <td>{h.macros}m ({canvas})</td>
                      <td className="num">{fmt(rnd)}</td>
                      <td className={`num font-bold ${h.best_algorithm === 'Greedy' ? 'text-ok' : ''}`}>
                        {fmt(grd)}
                      </td>
                      <td className={`num font-bold ${h.best_algorithm === 'Simulated Annealing' ? 'text-ok' : ''}`}>
                        {fmt(sa)}
                      </td>
                      <td className="num font-bold text-accent">{fmt(rl)}</td>
                      <td className="mono text-xs">
                        {h.rl_per_seed ? `[${h.rl_per_seed.map((s) => s.toFixed(2)).join(', ')}]` : '—'}
                      </td>
                      <td><span className="badge badge-best">{h.best_algorithm}</span></td>
                      <td className="mono text-xs font-bold">{h.ranking}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="hint" style={{ marginTop: '6px' }}>
            <strong>Holdout Finding:</strong> On unseen clustered topology, <strong>Simulated Annealing (280.87)</strong> achieves lower objective than <strong>Greedy (282.70)</strong>, demonstrating Greedy does not dominate every holdout.
          </p>

          <div className="sub-grid-2" style={{ marginTop: '16px' }}>
            <div className="table-wrap">
              <h4>Quantified Generalization Gaps</h4>
              <table className="metrics">
                <thead>
                  <tr>
                    <th>Holdout Condition</th>
                    <th className="num">Absolute Gap (Δ)</th>
                    <th className="num">Percentage Gap</th>
                    <th>Analytical Interpretation</th>
                  </tr>
                </thead>
                <tbody>
                  {expDGaps.map((g) => {
                    const delta = g.absolute_gap ?? g.inst_delta ?? g.topo_delta
                    const pct = g.percentage_gap ?? g.inst_gap_pct ?? g.topo_gap_pct
                    return (
                      <tr key={g.holdout || g.condition}>
                        <td className="font-bold">{g.holdout || g.condition}</td>
                        <td className="num font-bold text-warn">+{fmt(delta)}</td>
                        <td className="num font-bold text-warn">{fmtPct(pct)}</td>
                        <td className="hint">{g.interpretation}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            <div className="eda-takeaways-card">
              <h4>Key Generalization Takeaways</h4>
              <ul className="takeaways-list">
                {expD.key_findings ? (
                  expD.key_findings.map((f, idx) => (
                    <li key={idx}>
                      <span className="check-bullet">✔</span>
                      <div>{f}</div>
                    </li>
                  ))
                ) : (
                  <>
                    <li><span className="check-bullet">✔</span><div>RL maintains 100% legal placements across all unseen holdout evaluations via action masking.</div></li>
                    <li><span className="check-bullet">✔</span><div>RL outperforms Random baseline on all holdouts (unseen inst: 441.64 vs 535.44; unseen topo: 460.45 vs 568.86).</div></li>
                    <li><span className="check-bullet">✔</span><div>RL exhibits measurable out-of-distribution degradation: +31.67% (instance) and +37.28% (topology).</div></li>
                    <li><span className="check-bullet">✔</span><div>Simulated Annealing achieves the lowest objective on unseen clustered topology (280.87 vs Greedy 282.70).</div></li>
                  </>
                )}
              </ul>
            </div>
          </div>

          {expD.figures && expD.figures.length > 0 && (
            <div className="eda-figures-grid" style={{ marginTop: '16px' }}>
              {expD.figures.map((fig) => (
                <div key={fig.filename} className="eda-figure-card">
                  <div className="fig-head mono">{fig.title}</div>
                  <img src={resolveUrl(fig.url)} alt={fig.title} loading="lazy" />
                  <div className="fig-caption mono text-xs">Artifact: {fig.filename}</div>
                </div>
              ))}
            </div>
          )}
        </section>
      )}

      {/* ───────── Report Center ───────── */}
      {(subSection === 'all' || subSection === 'reports') && (
        <section className="eda-panel">
          <div className="panel-title">
            <span>RESEARCH REPORT CENTER</span>
            <span className="badge badge-accent mono">AUTOMATED ARTIFACT PIPELINE</span>
          </div>
          <div className="report-center-layout">
            <div className="report-center-info">
              <h4>Self-Contained Scientific HTML & PDF Reports</h4>
              <p>
                Generates a publication-grade synthesis report incorporating Experiments A, B, and D datasets,
                statistical confidence intervals, embedded SVG/PNG figure artifacts, and cryptographic SHA-256 provenance hashes.
              </p>
              <div className="report-artifact-status-list">
                <div className="status-item">
                  <span className={reports?.html_available ? 'text-ok' : 'text-muted'}>
                    {reports?.html_available ? '✓' : '○'}
                  </span>
                  <span>Standalone HTML Report (with embedded styles & figures)</span>
                  {reports?.html_size_bytes && (
                    <span className="mono text-xs text-muted">({(reports.html_size_bytes / 1024).toFixed(1)} KB)</span>
                  )}
                </div>
                <div className="status-item">
                  <span className={reports?.figures_count && reports.figures_count >= 4 ? 'text-ok' : 'text-muted'}>
                    {reports?.figures_count && reports.figures_count >= 4 ? '✓' : '○'}
                  </span>
                  <span>High-Resolution Matplotlib Figures ({reports?.figures_count ?? 4} Generated)</span>
                </div>
                <div className="status-item">
                  <span className={reports?.pdf_available ? 'text-ok' : 'text-muted'}>
                    {reports?.pdf_available ? '✓' : '○'}
                  </span>
                  <span>PDF Document ({reports?.pdf_status || 'Not generated locally; weasyprint required'})</span>
                </div>
              </div>
            </div>

            <div className="report-center-controls">
              <button
                className="btn btn-primary btn-run"
                disabled={generating}
                onClick={() => void handleGenerateReport()}
              >
                {generating ? 'Regenerating Report…' : '⟳ Run Report Generator'}
              </button>

              {reports?.html_available && reports.html_url && (
                <div className="report-links-stack">
                  <a className="btn btn-accent" href={resolveUrl(reports.html_url)} target="_blank" rel="noreferrer">
                    ↗ Open HTML Report in New Tab
                  </a>
                  <a className="btn" href={resolveUrl(reports.html_url)} download="macro_placement_report.html">
                    ⬇ Download HTML Report
                  </a>
                </div>
              )}
            </div>
          </div>
        </section>
      )}
    </div>
  )
}
