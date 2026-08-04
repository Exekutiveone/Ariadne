import type {useCorridorPlanner} from './corridorPlanner'

const number = (value: number, digits = 2) => new Intl.NumberFormat('de-DE', {maximumFractionDigits: digits}).format(value)

/** Bedienfeld zur Korridorprüfung und Trajektorienplanung. Den Zustand hält
 *  `useCorridorPlanner`, damit das Overlay über dem Video dasselbe sieht. */
export default function CorridorReadout({
  planner,
  planning,
  onTogglePlanning,
  note,
  onNote,
  annotator,
}: {
  planner: ReturnType<typeof useCorridorPlanner>
  planning: boolean
  onTogglePlanning: (value: boolean) => void
  note: string
  onNote: (value: string) => void
  annotator: string
}) {
  const {
    check,
    stored,
    draft,
    saving,
    message,
    calibration,
    setCalibration,
    activeCorridor,
    setSelected,
    corridorProposal,
    unchangedFromProposal,
    adopt,
    clear,
    persist,
    discard,
  } = planner

  return (
    <div className="corridor-panel">
      <div className="section-head compact">
        <h3>Korridore und Trajektorie</h3>
        <p>
          Deterministische Geometrie auf der Maske — kein ML. Geprüft wird nur die Breite, nicht wie weit voraus der Korridor frei bleibt.
        </p>
      </div>

      <div className="corridor-calibration">
        <label>
          Fahrzeugbreite · {number(calibration.vehicle_width_m)} m
          <input
            aria-label="Fahrzeugbreite in Metern"
            type="range"
            min="0.4"
            max="3"
            step="0.05"
            value={calibration.vehicle_width_m}
            onChange={event => setCalibration({...calibration, vehicle_width_m: +event.target.value})}
          />
        </label>
        <label>
          Zuschlag · {number(calibration.clearance_m)} m
          <input
            aria-label="Sicherheitszuschlag in Metern"
            type="range"
            min="0"
            max="0.5"
            step="0.01"
            value={calibration.clearance_m}
            onChange={event => setCalibration({...calibration, clearance_m: +event.target.value})}
          />
        </label>
        <label>
          Bodenbreite unten · {number(calibration.ground_width_at_bottom_m)} m
          <input
            aria-label="Bodenbreite am unteren Bildrand in Metern"
            type="range"
            min="1"
            max="12"
            step="0.25"
            value={calibration.ground_width_at_bottom_m}
            onChange={event => setCalibration({...calibration, ground_width_at_bottom_m: +event.target.value})}
          />
          <small>
            Kalibrierung pro Kameraaufbau: wie viele Meter Boden die volle Bildbreite ganz unten abdeckt. Keine Messung aus dem Bild.
          </small>
        </label>
      </div>

      {!check ? (
        <div className="empty">Die Korridore kommen mit der Frame-Vorhersage; wähle einen Frame.</div>
      ) : (
        <>
          <div className="corridor-status-grid">
            {check.corridors.map(corridor => (
              <article
                key={corridor.corridor}
                className={`${corridor.status} ${corridor.corridor === activeCorridor ? 'active' : ''}`}
                onClick={() => setSelected(corridor.corridor)}
              >
                <b>{corridor.label}</b>
                <strong>{corridor.status_label.toUpperCase()}</strong>
                <span>{corridor.meaning}</span>
                <small>{corridor.reason}</small>
              </article>
            ))}
          </div>
          <small className="corridor-note">
            Korridor anklicken — hier oder direkt im Bild — um ihn auszuwählen. Der gewählte Korridor ist im Video hervorgehoben.
          </small>

          <div className="corridor-geometry">
            <div>
              <span>Fluchtpunkt</span>
              <b>
                x {number(check.decomposition.vanishing_point.x, 1)} · y {number(check.decomposition.vanishing_point.y, 1)}
              </b>
              <small>
                {check.decomposition.vanishing_point.source === 'path_edge_line_intersection'
                  ? `aus ${check.decomposition.vanishing_point.rows_used} Wegrandzeilen gefittet`
                  : 'Notlösung, kein Fit aus den Wegrändern'}
              </small>
            </div>
            <div>
              <span>Nicht ausgewertet</span>
              <b>{Math.round(check.decomposition.irrelevant_zone.image_fraction_skipped * 100)} %</b>
              <small>{check.decomposition.irrelevant_zone.reason}</small>
            </div>
            <div>
              <span>Streifenbreite</span>
              <b>{number(check.strip.required_width_m)} m</b>
              <small>unten {number(check.strip.required_width_px_at_bottom, 0)} px, zum Fluchtpunkt hin linear schmaler</small>
            </div>
          </div>

          <div className="trajectory-planner">
            <div className="section-head compact">
              <h3>Trajektorie planen</h3>
              <p>Der Vorschlag legt je Zeile die Mitte des breitesten befahrbaren Laufs. Ziehe die Punkte, um ihn zu verbessern.</p>
            </div>
            <label className="trajectory-toggle">
              <input type="checkbox" checked={planning} onChange={event => onTogglePlanning(event.target.checked)} />
              <span>Bearbeiten: Klick setzt einen Punkt, Rechtsklick entfernt ihn</span>
            </label>
            <div className="trajectory-actions">
              <button disabled={!corridorProposal.length} onClick={adopt}>
                Vorschlag übernehmen
              </button>
              <button disabled={!draft} onClick={clear}>
                Entwurf verwerfen
              </button>
            </div>
            <label>
              Notiz
              <textarea
                value={note}
                maxLength={1000}
                placeholder="Warum weicht dein Verlauf vom Vorschlag ab?"
                onChange={event => onNote(event.target.value)}
              />
            </label>
            <div className="trajectory-actions">
              <button className="primary" disabled={saving || !draft || draft.length < 2} onClick={() => void persist(note, annotator)}>
                {saving ? 'WIRD GESPEICHERT …' : stored ? 'Trajektorie aktualisieren' : 'Trajektorie speichern'}
              </button>
              <button disabled={saving || !stored} onClick={() => void discard()}>
                Gespeicherte löschen
              </button>
            </div>
            <div className="trajectory-state">
              <b>{draft ? `${draft.length} Punkte im Entwurf` : 'Kein Entwurf'}</b>
              <span>
                {stored
                  ? `Gespeichert: Revision ${stored.revision} · ${stored.origin === 'manual_edit' ? 'von Hand nachgebessert' : stored.origin === 'manual' ? 'komplett selbst gesetzt' : 'unverändert vom Modell'}`
                  : 'Für diesen Frame ist nichts gespeichert.'}
              </span>
              {draft && (
                <small>
                  {unchangedFromProposal
                    ? 'Entwurf entspricht dem Vorschlag — wird als „unverändert vom Modell" gespeichert.'
                    : 'Entwurf weicht vom Vorschlag ab — wird als Handarbeit gespeichert.'}
                </small>
              )}
            </div>
          </div>

          {message && (
            <div className="labeling-message" role="status">
              {message}
            </div>
          )}
          <small className="corridor-note">{check.limitations[check.limitations.length - 1]}</small>
        </>
      )}
    </div>
  )
}
