import {useEffect, useState} from 'react'
import {getCorridorCheck} from './api'
import type {CorridorCheck} from './types'

const STATUS_CLASS = {free: 'free', blocked: 'blocked', uncertain: 'uncertain'} as const
const number = (value: number, digits = 2) => new Intl.NumberFormat('de-DE', {maximumFractionDigits: digits}).format(value)

/** A.3/A.4: Status je Korridor fuer den aktuell gezeigten Frame. Rein
 *  geometrisch auf der Maske des globalen Wegmodells, kein eigenes ML. */
export default function CorridorReadout({missionId, videoId, frameIndex, enabled}: {missionId: string; videoId: string; frameIndex: number; enabled: boolean}) {
  const [check, setCheck] = useState<CorridorCheck | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [vehicleWidth, setVehicleWidth] = useState(1.2)
  const [clearance, setClearance] = useState(.1)
  const [groundWidth, setGroundWidth] = useState(4)

  useEffect(() => {
    if (!enabled || !missionId || !videoId) {setCheck(null); return}
    let cancelled = false
    setLoading(true)
    void getCorridorCheck(missionId, videoId, frameIndex, {vehicle_width_m: vehicleWidth, clearance_m: clearance, ground_width_at_bottom_m: groundWidth})
      .then(result => {if (!cancelled) {setCheck(result); setError('')}})
      .catch(problem => {if (!cancelled) {setCheck(null); setError(problem instanceof Error ? problem.message : 'Korridorprüfung fehlgeschlagen')}})
      .finally(() => {if (!cancelled) setLoading(false)})
    return () => {cancelled = true}
  }, [enabled, missionId, videoId, frameIndex, vehicleWidth, clearance, groundWidth])

  return <div className="corridor-panel">
    <div className="section-head compact"><h3>Korridore im Bildraum</h3><p>Deterministische Geometrie auf der Maske — kein ML. Geprüft wird nur die Breite, nicht wie weit voraus der Korridor frei bleibt.</p></div>
    <div className="corridor-calibration">
      <label>Fahrzeugbreite · {number(vehicleWidth)} m<input aria-label="Fahrzeugbreite in Metern" type="range" min="0.4" max="3" step="0.05" value={vehicleWidth} onChange={event => setVehicleWidth(+event.target.value)}/></label>
      <label>Zuschlag · {number(clearance)} m<input aria-label="Sicherheitszuschlag in Metern" type="range" min="0" max="0.5" step="0.01" value={clearance} onChange={event => setClearance(+event.target.value)}/></label>
      <label>Bodenbreite unten · {number(groundWidth)} m<input aria-label="Bodenbreite am unteren Bildrand in Metern" type="range" min="1" max="12" step="0.25" value={groundWidth} onChange={event => setGroundWidth(+event.target.value)}/><small>Kalibrierung pro Kameraaufbau: wie viele Meter Boden die volle Bildbreite ganz unten abdeckt. Keine Messung aus dem Bild.</small></label>
    </div>
    {!enabled ? <div className="empty">Wähle einen Frame, um die Korridore zu prüfen.</div>
      : error ? <div className="labeling-message" role="status">{error}</div>
      : !check ? <div className="empty">{loading ? 'Korridore werden geprüft …' : 'Noch keine Korridorprüfung.'}</div>
      : <>
        <div className="corridor-status-grid">{check.corridors.map(corridor => <article key={corridor.corridor} className={STATUS_CLASS[corridor.status]}>
          <b>{corridor.label}</b>
          <strong>{corridor.status_label.toUpperCase()}</strong>
          <span>{corridor.meaning}</span>
          <small>{corridor.reason}</small>
        </article>)}</div>
        <div className="corridor-geometry">
          <div><span>Fluchtpunkt</span><b>x {number(check.decomposition.vanishing_point.x, 1)} · y {number(check.decomposition.vanishing_point.y, 1)}</b><small>{check.decomposition.vanishing_point.source === 'path_edge_line_intersection' ? `aus ${check.decomposition.vanishing_point.rows_used} Wegrandzeilen gefittet` : 'Notlösung, kein Fit aus den Wegrändern'}</small></div>
          <div><span>Nicht ausgewertet</span><b>{Math.round(check.decomposition.irrelevant_zone.image_fraction_skipped * 100)} %</b><small>{check.decomposition.irrelevant_zone.reason}</small></div>
          <div><span>Streifenbreite</span><b>{number(check.strip.required_width_m)} m</b><small>unten {number(check.strip.required_width_px_at_bottom, 0)} px, zum Fluchtpunkt hin linear schmaler</small></div>
        </div>
        <small className="corridor-note">{check.limitations[check.limitations.length - 1]}</small>
      </>}
  </div>
}
