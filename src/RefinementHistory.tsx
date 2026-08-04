import {useEffect, useMemo, useState} from 'react'
import {getRegistryRuns, listCriticalFlags, listOffPathIntervals} from './api'
import type {OffPathInterval, RegistryRun, TerrainMask} from './types'

type CriticalFlag = {
  video_id: string
  frame_index: number
  timestamp_ms: number
  severity: number
  note: string
  annotator: string
  created_at: string
  brush_mask?: TerrainMask
}

export default function RefinementHistory({initialRun, onClose}: {initialRun: RegistryRun | null; onClose: () => void}) {
  const [runs, setRuns] = useState<RegistryRun[]>([])
  const [selectedRunId, setSelectedRunId] = useState(initialRun?.run_id ?? '')
  const [frames, setFrames] = useState<CriticalFlag[]>([])
  const [ranges, setRanges] = useState<OffPathInterval[]>([])
  const [message, setMessage] = useState('')
  const selected = useMemo(() => runs.find(run => run.run_id === selectedRunId) ?? initialRun, [runs, selectedRunId, initialRun])

  useEffect(() => {
    void getRegistryRuns()
      .then(result => {
        setRuns(result.runs)
        if (!selectedRunId) setSelectedRunId(initialRun?.run_id ?? result.runs[0]?.run_id ?? '')
      })
      .catch(error => setMessage(error instanceof Error ? error.message : 'Runs konnten nicht geladen werden'))
  }, [initialRun?.run_id])

  useEffect(() => {
    if (!selected) {
      setFrames([])
      setRanges([])
      return
    }
    void Promise.all([listCriticalFlags(selected.mission_id, selected.video_id), listOffPathIntervals(selected.mission_id, selected.video_id)])
      .then(([flags, intervals]) => {
        setFrames(flags.items)
        setRanges(intervals)
      })
      .catch(error => setMessage(error instanceof Error ? error.message : 'Refinements konnten nicht geladen werden'))
  }, [selected?.run_id])

  return (
    <div className="global-model-page">
      <header className="global-model-header">
        <button onClick={onClose}>← Zurück</button>
        <div>
          <span className="eyebrow">QUALITY LOOP</span>
          <h1>Refinement-Historie</h1>
          <p>Gespeicherte Fehlerframes und Video-Fehlerbereiche dieses Runs. Das Originalvideo wird dabei nie verändert.</p>
        </div>
      </header>
      <section className="global-dataset-panel refinement-history-selector">
        <label>
          Run auswählen
          <select value={selectedRunId} onChange={event => setSelectedRunId(event.target.value)}>
            {runs.map(run => <option key={run.run_id} value={run.run_id}>{run.original_name} · {run.mission_name}</option>)}
          </select>
        </label>
        {selected && <small>{selected.run_id} · {selected.original_name}</small>}
        {message && <div className="labeling-message">{message}</div>}
      </section>
      {selected && (
        <section className="global-result-panel refinement-history-grid">
          <article>
            <div className="section-head"><h2>Fehlerframes</h2><p>{frames.length} direkt gespeicherte Frame-Referenzen</p></div>
            {frames.length ? <div className="refinement-history-list">{frames.map(frame => <div key={`${frame.video_id}-${frame.frame_index}`}><b>Frame {frame.frame_index + 1}</b><span>{(frame.timestamp_ms / 1000).toFixed(2)} s · Fehlerstufe {frame.severity}/5</span><small>{frame.note || 'Keine Notiz'}</small></div>)}</div> : <div className="empty">Noch keine Fehlerframes gespeichert.</div>}
          </article>
          <article>
            <div className="section-head"><h2>Videobereiche</h2><p>{ranges.length} gespeicherte Fehlerintervalle</p></div>
            {ranges.length ? <div className="refinement-history-list">{ranges.map(range => <div key={range.id}><b>{(range.start_ms / 1000).toFixed(2)} s – {(range.end_ms / 1000).toFixed(2)} s</b><span>Vom Weg abgekommen / keine befahrbare Fläche</span><small>{range.note || 'Keine Notiz'}</small></div>)}</div> : <div className="empty">Noch keine Videobereiche gespeichert.</div>}
          </article>
        </section>
      )}
    </div>
  )
}
