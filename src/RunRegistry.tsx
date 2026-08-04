import {useEffect, useState} from 'react'
import {getRegistryRuns, updateRegistryRun} from './api'
import {TERRAIN_CATEGORY_OPTIONS, terrainCategoryLabel} from './terrainCategories'
import type {RegistryListing, RegistryRun, RunStatus} from './types'

const megabytes = (bytes: number) => `${new Intl.NumberFormat('de-DE', {maximumFractionDigits: 1}).format(bytes / 1024 / 1024)} MB`

/** A.5: eine Wahrheitsquelle fuer die Kette Run → Frames → Masken → Training.
 *  Der Videoordner wird bei jedem Laden gescannt; neue Aufnahmen erscheinen
 *  automatisch als Run. */
export default function RunRegistry({onClose, onOpenRefinement = () => undefined}: {onClose: () => void; onOpenRefinement?: (run: RegistryRun) => void}) {
  const [listing, setListing] = useState<RegistryListing | null>(null)
  const [message, setMessage] = useState('')
  const [busy, setBusy] = useState('')
  const [notes, setNotes] = useState<Record<string, string>>({})

  const load = () =>
    getRegistryRuns()
      .then(result => {
        setListing(result)
        setNotes(Object.fromEntries(result.runs.map(run => [run.run_id, run.note])))
        if (result.scan.added) setMessage(`${result.scan.added} neue Aufnahme(n) als Run angelegt.`)
        if (result.scan.removed.length)
          setMessage(`${result.scan.removed.length} Run(s) entfernt, weil das Video nicht mehr im Ordner liegt.`)
      })
      .catch(error => setMessage(error instanceof Error ? error.message : 'Run-Registry konnte nicht geladen werden'))

  useEffect(() => {
    void load()
  }, [])

  const patch = async (
    run: RegistryRun,
    payload: {status?: RunStatus; terrain_category?: string | null; note?: string},
    confirmation: string,
  ) => {
    if (busy) return
    setBusy(run.run_id)
    try {
      await updateRegistryRun(run.mission_id, run.video_id, payload)
      await load()
      setMessage(confirmation)
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Run konnte nicht aktualisiert werden')
    } finally {
      setBusy('')
    }
  }

  if (!listing)
    return (
      <div className="global-model-page">
        <header className="global-model-header">
          <button onClick={onClose}>← Übersicht</button>
          <div>
            <span className="eyebrow">DATEN-INFRASTRUKTUR</span>
            <h1>Run-Registry</h1>
          </div>
        </header>
        <div className="empty">{message || 'Runs werden gescannt …'}</div>
      </div>
    )

  return (
    <div className="global-model-page">
      <header className="global-model-header">
        <button onClick={onClose}>← Übersicht</button>
        <div>
          <span className="eyebrow">DATEN-INFRASTRUKTUR</span>
          <h1>Run-Registry</h1>
          <p>Ein Run ist genau ein Originalvideo. Der Videoordner wird bei jedem Laden gescannt; neue Aufnahmen erscheinen automatisch.</p>
        </div>
      </header>

      <section className="global-dataset-panel">
        <div className="global-totals">
          <div>
            <span>Runs</span>
            <b>{listing.totals.runs}</b>
          </div>
          {listing.statuses.map(status => (
            <div key={status.value}>
              <span>{status.label}</span>
              <b>{listing.counts[status.value]}</b>
            </div>
          ))}
        </div>
        {listing.totals.missing_video_file > 0 && (
          <small className="terrain-hint">
            {listing.totals.missing_video_file} Run(s) ohne Videodatei im Ordner — die Aufnahme liegt nur noch als Eintrag vor.
          </small>
        )}
        <small className="terrain-hint">{listing.note}</small>
        {message && (
          <div className="labeling-message" role="status">
            {message}
          </div>
        )}
      </section>

      <section className="global-result-panel">
        <div className="section-head">
          <h2>Runs</h2>
          <p>
            Status, vorherrschender Untergrund und Notiz. Die Kategorie wird nach mission.json durchgeschrieben und gilt damit für alle
            Frames dieses Videos.
          </p>
        </div>
        {!listing.runs.length ? (
          <div className="empty">Im Videoordner liegt noch keine Aufnahme.</div>
        ) : (
          <div className="registry-list">
            {listing.runs.map(run => (
              <article key={run.run_id} className={run.status}>
                <div className="registry-identity">
                  <b>{run.original_name}</b>
                  <span>{run.mission_name}</span>
                  <small>
                    {run.run_id} · {run.video_available ? megabytes(run.size_bytes) : 'Videodatei fehlt'}
                  </small>
                </div>
                <label>
                  Status
                  <select
                    aria-label={`Status ${run.original_name}`}
                    value={run.status}
                    disabled={busy === run.run_id}
                    onChange={event =>
                      void patch(
                        run,
                        {status: event.target.value as RunStatus},
                        `${run.original_name}: Status auf „${listing.statuses.find(item => item.value === event.target.value)?.label}“ gesetzt.`,
                      )
                    }
                  >
                    {listing.statuses.map(status => (
                      <option key={status.value} value={status.value}>
                        {status.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Vorherrschender Untergrund
                  <select
                    aria-label={`Untergrund ${run.original_name}`}
                    value={run.terrain_category ?? ''}
                    disabled={busy === run.run_id}
                    onChange={event =>
                      void patch(
                        run,
                        {terrain_category: event.target.value || null},
                        `${run.original_name}: Untergrund auf „${terrainCategoryLabel(event.target.value || null)}“ gesetzt — gilt für alle Frames dieses Videos.`,
                      )
                    }
                  >
                    <option value="">Noch keine Kategorie</option>
                    {TERRAIN_CATEGORY_OPTIONS.map(option => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Notiz
                  <textarea
                    aria-label={`Notiz ${run.original_name}`}
                    value={notes[run.run_id] ?? ''}
                    maxLength={2000}
                    placeholder="Besonderheiten dieser Aufnahme"
                    onChange={event => setNotes(current => ({...current, [run.run_id]: event.target.value}))}
                  />
                  <button
                    disabled={busy === run.run_id || (notes[run.run_id] ?? '') === run.note}
                    onClick={() => void patch(run, {note: notes[run.run_id] ?? ''}, `${run.original_name}: Notiz gespeichert.`)}
                  >
                    Notiz speichern
                  </button>
                </label>
                <button type="button" className="registry-refinement" onClick={() => onOpenRefinement(run)}>
                  Refinement
                </button>
              </article>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
