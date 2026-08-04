import {useCallback, useEffect, useMemo, useState} from 'react'
import {createTrajectory, deleteTrajectory, listTrajectories, updateTrajectory} from './api'
import type {CorridorCheck, NormalizedPoint, StoredTrajectory} from './types'

/** Anzahl Griffe, auf die der KI-Vorschlag zum Bearbeiten eingedampft wird.
 *  28 Stützstellen lassen sich nicht sinnvoll von Hand ziehen. */
export const TRAJECTORY_HANDLES = 9

export type CorridorCalibration = {vehicle_width_m: number; clearance_m: number; ground_width_at_bottom_m: number}

function thin(points: NormalizedPoint[], count = TRAJECTORY_HANDLES): NormalizedPoint[] {
  if (points.length <= count) return points.map(point => [...point] as NormalizedPoint)
  const picks = new Set<number>()
  for (let index = 0; index < count; index++) picks.add(Math.round((index * (points.length - 1)) / (count - 1)))
  return [...picks].sort((a, b) => a - b).map(index => [...points[index]] as NormalizedPoint)
}

const same = (a: NormalizedPoint[], b: NormalizedPoint[]) =>
  a.length === b.length && a.every((point, index) => Math.abs(point[0] - b[index][0]) < 1e-6 && Math.abs(point[1] - b[index][1]) < 1e-6)

/** Hält Korridorauswahl und die von Hand bearbeitete Trajektorie zusammen, damit
 *  Overlay und Bedienfeld denselben Zustand sehen.
 *
 *  Die Korridorprüfung wird bewusst **nicht** hier geholt: sie kommt mit der
 *  Vorhersage desselben Frames herein. Als eigener Aufruf lief die Inferenz
 *  zweimal über denselben Frame (gemessen 2 × 0,33 s je Frame-Wechsel). Die
 *  Kalibrierung lebt trotzdem hier, weil sie zur Bedienung gehört; der Aufrufer
 *  reicht sie an die Vorhersage weiter. */
export function useCorridorPlanner(
  missionId: string,
  videoId: string,
  frameIndex: number,
  timestampMs: number,
  check: CorridorCheck | null,
) {
  const [stored, setStored] = useState<StoredTrajectory | null>(null)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState('')
  const [selected, setSelected] = useState<string | null>(null)
  const [draft, setDraft] = useState<NormalizedPoint[] | null>(null)
  const [calibration, setCalibration] = useState<CorridorCalibration>({vehicle_width_m: 1.2, clearance_m: 0.1, ground_width_at_bottom_m: 4})
  const enabled = Boolean(missionId && videoId)

  // Gespeicherte Trajektorie des Frames laden; sie hat Vorrang vor dem Vorschlag.
  // Ein Frame kann mehrere Trajektorien tragen (siehe GroundTruthLabeler); dieses
  // Bedienfeld verfeinert gezielt EINEN Modellvorschlag und nimmt darum bewusst
  // nur die erste — fuer freies Mehrfach-Planen ist der Handlabeler da.
  useEffect(() => {
    if (!enabled || !missionId || !videoId) {
      setStored(null)
      setDraft(null)
      return
    }
    let cancelled = false
    void listTrajectories(missionId, videoId, frameIndex)
      .then(items => {
        if (cancelled) return
        const result = items[0] ?? null
        setStored(result)
        setDraft(result ? result.points.map(point => [...point] as NormalizedPoint) : null)
      })
      .catch(() => {
        if (!cancelled) {
          setStored(null)
          setDraft(null)
        }
      })
    return () => {
      cancelled = true
    }
  }, [enabled, missionId, videoId, frameIndex])

  const activeCorridor = selected
  const corridorProposal = useMemo(() => {
    if (!activeCorridor) return []
    const corridor = check?.corridors.find(item => item.corridor === activeCorridor)
    return corridor?.trajectory.points.length ? corridor.trajectory.points : []
  }, [check, activeCorridor])

  const toggleSelected = useCallback((corridor: string) => {
    setSelected(current => (current === corridor ? null : corridor))
    setDraft(null)
  }, [])

  const adopt = useCallback(() => {
    if (!corridorProposal.length) {
      setMessage('Für diesen Korridor gibt es keinen Vorschlag.')
      return
    }
    setDraft(thin(corridorProposal as NormalizedPoint[]))
    setMessage('Vorschlag übernommen — jetzt die Punkte ziehen, um ihn zu verbessern.')
  }, [corridorProposal])

  const clear = useCallback(() => {
    setDraft(null)
    setMessage('Entwurf verworfen.')
  }, [])

  const movePoint = useCallback((index: number, point: NormalizedPoint) => {
    setDraft(current => current && current.map((existing, position) => (position === index ? point : existing)))
  }, [])

  const addPoint = useCallback((point: NormalizedPoint) => {
    setDraft(current => {
      const next = current ? [...current, point] : [point]
      // Von unten nach oben sortiert: die Trajektorie laeuft vom Fahrzeug weg.
      return next.sort((a, b) => b[1] - a[1])
    })
  }, [])

  const removePoint = useCallback((index: number) => {
    setDraft(current => (current && current.length > 1 ? current.filter((_, position) => position !== index) : current))
  }, [])

  const unchangedFromProposal = Boolean(draft && corridorProposal.length && same(draft, thin(corridorProposal as NormalizedPoint[])))

  const persist = useCallback(
    async (note: string, annotator: string) => {
      if (!draft || draft.length < 2 || saving) {
        if (draft && draft.length < 2) setMessage('Eine Trajektorie braucht mindestens zwei Punkte.')
        return
      }
      setSaving(true)
      try {
        const payload = {
          timestamp_ms: timestampMs,
          points: draft,
          corridor: activeCorridor as 'mitte' | 'rechts' | 'links',
          origin: unchangedFromProposal ? ('model_proposal' as const) : ('manual_edit' as const),
          note,
          annotator: annotator.trim() || 'human',
        }
        const saved = stored
          ? await updateTrajectory(missionId, videoId, frameIndex, stored.id, payload)
          : await createTrajectory(missionId, videoId, frameIndex, payload)
        setStored(saved)
        setMessage(
          `Trajektorie für Frame ${frameIndex + 1} gespeichert (Revision ${saved.revision}, ${unchangedFromProposal ? 'unverändert vom Modell' : 'von Hand nachgebessert'}).`,
        )
      } catch (problem) {
        setMessage(problem instanceof Error ? problem.message : 'Trajektorie konnte nicht gespeichert werden')
      } finally {
        setSaving(false)
      }
    },
    [draft, saving, missionId, videoId, frameIndex, timestampMs, activeCorridor, unchangedFromProposal, stored],
  )

  const discard = useCallback(async () => {
    if (!stored || saving) return
    setSaving(true)
    try {
      await deleteTrajectory(missionId, videoId, frameIndex, stored.id)
      setStored(null)
      setDraft(null)
      setMessage(`Gespeicherte Trajektorie für Frame ${frameIndex + 1} gelöscht.`)
    } catch (problem) {
      setMessage(problem instanceof Error ? problem.message : 'Trajektorie konnte nicht gelöscht werden')
    } finally {
      setSaving(false)
    }
  }, [stored, saving, missionId, videoId, frameIndex])

  return {
    check,
    stored,
    draft,
    saving,
    message,
    calibration,
    setCalibration,
    activeCorridor,
    toggleSelected,
    corridorProposal,
    unchangedFromProposal,
    adopt,
    clear,
    movePoint,
    addPoint,
    removePoint,
    persist,
    discard,
    setMessage,
  }
}
