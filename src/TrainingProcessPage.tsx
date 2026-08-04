import {useEffect, useState} from 'react'
import type {CSSProperties} from 'react'
import {getGlobalModelDashboard, getRegistryRuns} from './api'
import type {GlobalModelDashboardData, RegistryRun} from './types'

const pct = (value: number | undefined) => value === undefined ? '—' : `${(value * 100).toFixed(1)} %`

export function ModelPerformanceDashboard({onClose}: {onClose: () => void}) {
  const [dashboard, setDashboard] = useState<GlobalModelDashboardData | null>(null)
  const [runs, setRuns] = useState<RegistryRun[]>([])
  useEffect(() => { void Promise.all([getGlobalModelDashboard(), getRegistryRuns()]).then(([data, registry]) => { setDashboard(data); setRuns(registry.runs) }).catch(() => undefined) }, [])
  const model = dashboard?.model ?? null
  const totals = dashboard?.dataset.totals
  const score = model?.validation_metrics.symmetric_score
  const precision = model?.validation_metrics.precision
  const recall = model?.validation_metrics.recall
  const labelled = totals?.confirmed_frames ?? 0
  const openErrors = totals?.critical_flags ?? 0
  const refinements = totals?.refinements ?? 0
  const totalRunFrames = Math.max(labelled, runs.length ? labelled + openErrors : 0)
  const labelingProgress = totalRunFrames ? Math.round((labelled / totalRunFrames) * 100) : 0
  return <div className="performance-dashboard">
    <header className="performance-header">
      <button onClick={onClose}>← Übersicht</button>
      <div><span className="eyebrow">ARIADNE · GESAMTDATENSATZ</span><h1>Modellstatus auf einen Blick</h1><p>Durchschnittliche Modellleistung über alle verfügbaren Trainingsdaten, ergänzt um den gemeinsamen Labeling- und Refinement-Stand.</p></div>
      <div className="performance-readonly"><i /> <span>Gesamtansicht über alle Daten · automatisch aktualisiert</span></div>
    </header>

    <section className="performance-kpis" aria-label="Kennzahlen">
      {[[score === undefined ? '—' : `${score.toFixed(1)} / 100`,'Modellgenauigkeit','Aktuelle Validierung','blue'],[labelled.toLocaleString('de-DE'),'Gelabelte Frames','Bestätigte Ground Truth','green'],[`${labelingProgress} %`,'Labeling-Fortschritt','bezogen auf bekannte Frames','orange'],[runs.length.toLocaleString('de-DE'),'Geprüfte Sequenzen','gespeicherte Video-Runs','blue'],[openErrors.toLocaleString('de-DE'),'Offene Fehler','gespeicherte Fehlerframes','red'],[refinements.toLocaleString('de-DE'),'Abgeschlossene Refinements','bestätigte Korrekturen','green']].map(([value,label,detail,tone]) => <article key={String(label)} className={`performance-kpi ${tone}`}><b>{value}</b><span>{label}</span><small>{detail}</small></article>)}
    </section>

    <section className="performance-grid top">
      <article className="performance-card quality"><div className="card-head"><div><span>GESAMTQUALITÄT</span><h2>Wie zuverlässig ist die Erkennung insgesamt?</h2></div><small>{model ? 'Gemeinsamer Modellstand' : 'Noch kein Modell'}</small></div>{model ? <><div className="metric-bars">{[['Trefferquote', score ?? 0,'blue'],['Erkannte Wegflächen', (precision ?? 0) * 100,'green'],['Vollständigkeit', (recall ?? 0) * 100,'orange'],['Gesamtqualität', (model.validation_metrics.dice ?? 0) * 100,'blue']].map(([label,value,tone]) => <div key={String(label)}><span>{label}</span><b>{Number(value).toFixed(1)} %</b><i><em className={String(tone)} style={{width:`${Math.max(0,Math.min(100,Number(value)))}%`}} /></i></div>)}</div><small className="chart-note">Die Werte fassen die Validierung des gemeinsamen Modells über den gesamten verfügbaren Datensatz zusammen.</small></> : <div className="empty">Sobald ein Modell trainiert wurde, erscheint hier seine aktuelle Qualität.</div>}</article>
      <article className="performance-card progress-card"><div className="card-head"><div><span>LABELING-FORTSCHRITT</span><h2>Datensatz-Status</h2></div></div><div className="progress-ring" style={{'--progress': `${labelingProgress * 3.6}deg`} as CSSProperties}><b>{labelingProgress} %</b><small>gelabelt</small></div><div className="progress-breakdown"><span><i className="green" />Gelabelt <b>{labelled}</b></span><span><i className="orange" />In Prüfung <b>{openErrors}</b></span><span><i className="muted" />Noch offen <b>—</b></span></div></article>
    </section>

    <section className="performance-grid lower">
      <article className="performance-card"><div className="card-head"><div><span>QUALITÄT NACH KLASSE</span><h2>Aktuell trainierte Klassen</h2></div></div><table className="performance-table"><thead><tr><th>Objektklasse</th><th>Accuracy</th><th>Gelabelt</th><th>Offene Fehler</th><th>Status</th></tr></thead><tbody><tr><td>Befahrbarer Weg</td><td>{score === undefined ? '—' : `${score.toFixed(1)} %`}</td><td>{labelled.toLocaleString('de-DE')}</td><td>{openErrors}</td><td><span className={score !== undefined && score >= 85 ? 'status-good' : 'status-watch'}>{score !== undefined && score >= 85 ? 'Sehr gut' : 'Prüfung nötig'}</span></td></tr></tbody></table><small className="chart-note">Weitere Objektklassen werden erst ausgewiesen, sobald ein eigenes Klassenmodell und validierte Metriken vorliegen.</small></article>
      <article className="performance-card"><div className="card-head"><div><span>FEHLERVERTEILUNG</span><h2>Aktuelle Fehlerquellen</h2></div></div><div className="error-list"><div><span>Gemeldete Fehlerframes</span><b>{openErrors}</b><i style={{width:'100%'}} /></div><div><span>Dokumentierte Refinements</span><b>{refinements}</b><i style={{width:`${Math.min(100, refinements / Math.max(1, refinements + openErrors) * 100)}%`}} /></div><div><span>Fehler mit Pinselmarkierung</span><b>—</b><i style={{width:'0%'}} /></div></div></article>
    </section>

    <section className="performance-grid lower"><article className="performance-card model-status"><div className="card-head"><div><span>AKTUELLER MODELLSTATUS</span><h2>{model ? 'Modell ist einsatzbereit' : 'Modell wird vorbereitet'}</h2></div><strong>{model ? 'AKTIV' : 'AUSSTEHEND'}</strong></div><dl><dt>Letzte Aktualisierung</dt><dd>{model ? new Date(model.created_at).toLocaleString('de-DE') : '—'}</dd><dt>Gelernte Beispiele</dt><dd>{model?.dataset.confirmed_frames.toLocaleString('de-DE') ?? '—'}</dd><dt>Geprüfte Beispiele</dt><dd>{model?.split.validation_frames.toLocaleString('de-DE') ?? '—'}</dd><dt>Verbesserungen einbezogen</dt><dd>{model?.dataset.refinements_included.toLocaleString('de-DE') ?? '—'}</dd></dl></article><article className="performance-card"><div className="card-head"><div><span>LETZTE AKTIVITÄTEN</span><h2>Was bereits passiert ist</h2></div></div><ol className="activity-list">{model && <li><i className="green" />Das aktuelle Modell wurde erfolgreich geprüft.</li>}{refinements > 0 && <li><i className="blue" />{refinements} Verbesserungen wurden dokumentiert.</li>}{openErrors > 0 && <li><i className="red" />{openErrors} Fehlerframes warten auf weitere Prüfung.</li>}{runs.length > 0 && <li><i className="orange" />{runs.length} Videoaufnahmen stehen im Projekt zur Verfügung.</li>}{!model && <li><i className="muted" />Noch kein abgeschlossener Modelllauf vorhanden.</li>}</ol></article></section>
  </div>
}

export default function TrainingProcessPage({onClose}: {onClose: () => void}) {
  const steps = ['Video', 'Labeling', 'Erstes KI-Training', 'KI-Ergebnisse', 'Refinement', 'Verbesserter Datensatz', 'Neues Training']
  return <div className="training-process-page">
    <header className="process-hero">
      <button onClick={onClose}>← Übersicht</button>
      <span className="eyebrow">ARIADNE · KI-TRANSPARENZ</span>
      <h1>So entsteht unsere Bilderkennungs-KI</h1>
      <p>Unser Prozess kombiniert menschliches Labeling mit kontinuierlichem KI-Training und Refinement, um die Erkennungsqualität Schritt für Schritt zu verbessern.</p>
      <div className="process-flow">{steps.map((step, index) => <div key={step} className={index === steps.length - 1 ? 'loop' : ''}><b>{String(index + 1).padStart(2, '0')}</b><span>{step}</span>{index < steps.length - 1 && <i>↓</i>}</div>)}</div>
    </header>
    <section className="process-section split"><div className="process-video-illustration"><span>VIDEOFRAME</span><i /><i /><i /><b>LABELS</b></div><div><span className="process-kicker">01 · LABELING</span><h2>Präzises Labeling der Videodaten</h2><p>Aus Videoframes entstehen überprüfbare, präzise Flächenlabels. Sie bilden die Grundlage des ersten Wegmodells.</p></div></section>
    <section className="process-section"><span className="process-kicker">REFINEMENT</span><h2>Kontinuierlicher menschlicher Feedback-Loop</h2><div className="refinement-flow">{[['◉','KI erkennt Bild'],['✓','Mensch überprüft Ergebnis'],['!','Fehlerhafte Frames markieren'],['▣','Fehler dokumentieren'],['＋','Labels ergänzen'],['↻','Neues Training']].map(([icon,title]) => <article key={title}><i>{icon}</i><b>{title}</b><span>Nachvollziehbarer Schritt im kontinuierlichen Lernkreislauf.</span></article>)}</div></section>
    <section className="process-section continuous"><span className="process-kicker">WARUM DAS WICHTIG IST</span><h2>Training ist ein Kreislauf, keine Einmalentscheidung</h2><div className="learning-loop">{['Training','Vorhersage','Prüfung','Refinement','Verbesserter Datensatz'].map(step => <span key={step}>{step}</span>)}</div></section>
  </div>
}
