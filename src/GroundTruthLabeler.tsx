import {useEffect, useMemo, useRef, useState} from 'react'
import type {CSSProperties, PointerEvent as ReactPointerEvent, WheelEvent as ReactWheelEvent} from 'react'
import {
  createOffPathInterval,
  deleteCriticalFlag,
  deleteOffPathInterval,
  deleteTrajectory,
  getGroundTruth,
  getRoiProfile,
  getTrajectory,
  listCriticalFlags,
  listOffPathIntervals,
  saveCriticalFlag,
  saveRoiProfile,
  saveTrajectory,
  getLabelingVideos,
  getPathModel,
  getPathTrainingJob,
  listGroundTruth,
  predictPathFrame,
  runSegmentation,
  saveGroundTruth,
  savePathRefinement,
  startPathTrainingJob,
  trainPathModel,
  updateVideoFullyNotTraversable,
  updateVideoTerrainCategory,
} from './api'
import GradeLegend from './GradeLegend'
import {AI_BINARY_PALETTE, COMPARISON_LEGEND, COMPARISON_PALETTE, paintMaskCanvas, paletteFromGradeOntology, rleValueAt} from './masks'
import {shapeId, useLabelOntology} from './labelOntology'
import type {LabelShape} from './labelOntology'
import {TERRAIN_CATEGORY_OPTIONS, terrainCategoryLabel} from './terrainCategories'
import type {
  GroundTruthSummary,
  GroundTruthStatus,
  LabelingVideo,
  Mission,
  NormalizedPoint,
  OffPathInterval,
  PathModelResult,
  PathPrediction,
  PathTrainingJob,
  RoiProfile,
  StoredTrajectory,
  TerrainMask,
} from './types'

type Tool = 'add' | 'edit' | 'move' | 'pan'

/** Stabile Kennung einer Stelle ueber Frames hinweg. */
const newTrackingId = () => `t-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`

const samePoints = (a: NormalizedPoint[], b: NormalizedPoint[]) =>
  a.length === b.length && a.every((point, index) => Math.abs(point[0] - b[index][0]) < 1e-9 && Math.abs(point[1] - b[index][1]) < 1e-9)
type Drag =
  | {kind: 'vertex'; index: number; before: NormalizedPoint[]}
  | {kind: 'polygon'; origin: NormalizedPoint; before: NormalizedPoint[]}
  | {kind: 'pan'; clientX: number; clientY: number; before: {x: number; y: number}}

const emptySummary = (missionId: string): GroundTruthSummary => ({
  schema_version: '2.0',
  mission_id: missionId,
  ontology: {},
  counts: {total: 0, draft: 0, confirmed: 0, skipped: 0},
  items: [],
})

const copyPoints = (points: NormalizedPoint[]) => points.map(([x, y]) => [x, y] as NormalizedPoint)
const clamp = (value: number, minimum = 0, maximum = 1) => Math.max(minimum, Math.min(maximum, value))
const timestampFor = (frameIndex: number, fps: number) => Math.round((frameIndex / fps) * 1000)

// Liegt jetzt in masks.ts; hier weiterhin exportiert, weil der Labeler die
// Einstiegsstelle fuer den Refinement-Klick ist.
export {rleValueAt}

export function buildFrameSelection(total: number, mode: 'stride' | 'count', stride: number, count: number) {
  if (total <= 0) return []
  if (mode === 'stride') {
    const safeStride = Math.max(1, Math.round(stride))
    return Array.from({length: Math.ceil(total / safeStride)}, (_, index) => index * safeStride).filter(index => index < total)
  }
  const safeCount = Math.max(1, Math.min(total, Math.round(count)))
  if (safeCount === 1) return [0]
  return [...new Set(Array.from({length: safeCount}, (_, index) => Math.round((index * (total - 1)) / (safeCount - 1))))]
}

/** Mischt Frames aus mehreren Videos zu einer Zufallsreihenfolge fuer den
 *  Shuffle-Modus — je Video annaehernd gleich viele, gleichmaessig verteilt. */
export function buildShuffleSelection(videos: {video_id: string; total_frames: number}[], totalRequested: number) {
  if (!videos.length) return []
  const perVideo = Math.max(1, Math.round(totalRequested / videos.length))
  const pool = videos.flatMap(video =>
    buildFrameSelection(video.total_frames, 'count', 0, perVideo).map(frame_index => ({video_id: video.video_id, frame_index})),
  )
  // Fisher-Yates: jede Ziehung gleich wahrscheinlich, kein Bias durch sort().
  for (let index = pool.length - 1; index > 0; index--) {
    const swap = Math.floor(Math.random() * (index + 1))
    ;[pool[index], pool[swap]] = [pool[swap], pool[index]]
  }
  return pool
}

/** mm:ss fuer Zeitstempel in Millisekunden, gerundet auf die Sekunde. */
export function formatTimestamp(ms: number) {
  const totalSeconds = Math.round(ms / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}:${String(seconds).padStart(2, '0')}`
}

export function normalizedPointFromBounds(
  clientX: number,
  clientY: number,
  bounds: Pick<DOMRect, 'left' | 'top' | 'width' | 'height'>,
): NormalizedPoint {
  return [clamp((clientX - bounds.left) / Math.max(1, bounds.width)), clamp((clientY - bounds.top) / Math.max(1, bounds.height))]
}

export function polygonForNextFrame(points: NormalizedPoint[], direction: -1 | 1, enabled = true) {
  return enabled && direction === 1 && points.length >= 3 ? copyPoints(points) : null
}

export function pointerAction(tool: Tool, vertex: number | null, hasPolygon: boolean, insidePolygon: boolean) {
  if (vertex !== null) return 'vertex'
  if (tool === 'pan') return 'pan'
  if (tool === 'move' && hasPolygon && insidePolygon) return 'polygon'
  if (tool === 'add') return 'add'
  return 'none'
}

function pointInPolygon(point: NormalizedPoint, polygon: NormalizedPoint[]) {
  let inside = false
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
    const [xi, yi] = polygon[index]
    const [xj, yj] = polygon[previous]
    if (yi > point[1] !== yj > point[1] && point[0] < ((xj - xi) * (point[1] - yi)) / (yj - yi || 1e-9) + xi) inside = !inside
  }
  return inside
}

function nearestSegmentIndex(point: NormalizedPoint, polygon: NormalizedPoint[]) {
  let bestIndex = polygon.length - 1
  let bestDistance = Number.POSITIVE_INFINITY
  for (let index = 0; index < polygon.length; index++) {
    const start = polygon[index]
    const end = polygon[(index + 1) % polygon.length]
    const dx = end[0] - start[0]
    const dy = end[1] - start[1]
    const lengthSquared = dx * dx + dy * dy
    const projection = lengthSquared ? clamp(((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / lengthSquared) : 0
    const x = start[0] + projection * dx
    const y = start[1] + projection * dy
    const distance = (point[0] - x) ** 2 + (point[1] - y) ** 2
    if (distance < bestDistance) {
      bestDistance = distance
      bestIndex = index
    }
  }
  return bestIndex
}

export default function GroundTruthLabeler({
  mission,
  onClose,
  onProcessingComplete,
  onMissionUpdated,
}: {
  mission: Mission
  onClose: () => void
  onProcessingComplete: () => void | Promise<void>
  onMissionUpdated?: () => void | Promise<void>
}) {
  const [videos, setVideos] = useState<LabelingVideo[]>([])
  const [activeVideoId, setActiveVideoId] = useState('')
  const [loading, setLoading] = useState(true)
  const [selectionMode, setSelectionMode] = useState<'stride' | 'count' | 'shuffle'>('stride')
  const [stride, setStride] = useState(10)
  const [targetCount, setTargetCount] = useState(100)
  const [selectionPosition, setSelectionPosition] = useState(0)
  const [points, setPoints] = useState<NormalizedPoint[]>([])
  const [tool, setTool] = useState<Tool>('add')
  const [selectedVertex, setSelectedVertex] = useState<number | null>(null)
  const [past, setPast] = useState<NormalizedPoint[][]>([])
  const [future, setFuture] = useState<NormalizedPoint[][]>([])
  const [status, setStatus] = useState<GroundTruthStatus | 'new'>('new')
  const [revision, setRevision] = useState(0)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [modelTraining, setModelTraining] = useState(false)
  const [pathModel, setPathModel] = useState<PathModelResult | null>(null)
  const [showAiMask, setShowAiMask] = useState(true)
  const [aiOverlay, setAiOverlay] = useState<'grade' | 'comparison'>('grade')
  const [aiMaskOpacity, setAiMaskOpacity] = useState(0.38)
  const [pathPrediction, setPathPrediction] = useState<PathPrediction | null>(null)
  const [predictionLoading, setPredictionLoading] = useState(false)
  const [trainingProfile, setTrainingProfile] = useState<'quick' | 'overnight'>('overnight')
  const [trainingHours, setTrainingHours] = useState(8)
  const [trainingJob, setTrainingJob] = useState<PathTrainingJob | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [playbackSpeed, setPlaybackSpeed] = useState(1)
  const [refinementMode, setRefinementMode] = useState(false)
  const [refinementSelection, setRefinementSelection] = useState<{point: NormalizedPoint; kind: 'missed_label' | 'invented_path'} | null>(
    null,
  )
  const [refinementSaving, setRefinementSaving] = useState(false)
  const [message, setMessage] = useState('')
  const [notes, setNotes] = useState('')
  const [annotator, setAnnotator] = useState('Simon')
  const [videoTerrainDraft, setVideoTerrainDraft] = useState('')
  const [videoTerrainSaving, setVideoTerrainSaving] = useState(false)
  const [summary, setSummary] = useState<GroundTruthSummary>(() => emptySummary(mission.id))
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({x: 0, y: 0})
  const [maskOpacity, setMaskOpacity] = useState(0.3)
  const [fullFrameNotTraversable, setFullFrameNotTraversable] = useState(false)
  // Die Werkzeuge bearbeiten weiterhin genau eine Fläche (`points`). Fertige
  // Flächen wandern in diese Liste — so bleibt der eingespielte Zeichenfluss
  // erhalten und es lassen sich trotzdem beliebig viele Klassen je Frame setzen.
  const [shapes, setShapes] = useState<LabelShape[]>([])
  const [shapeClass, setShapeClass] = useState('traversable')
  const [shapeCertainty, setShapeCertainty] = useState<LabelShape['certainty']>('certain')
  const [shapeHardNegative, setShapeHardNegative] = useState(false)
  const [shapeNote, setShapeNote] = useState('')
  const [editingShapeId, setEditingShapeId] = useState<string | null>(null)
  // Trajektorie von Hand: braucht kein Modell und keinen Vorschlag.
  const [trajectory, setTrajectory] = useState<NormalizedPoint[]>([])
  const [storedTrajectory, setStoredTrajectory] = useState<StoredTrajectory | null>(null)
  const [trajectoryMode, setTrajectoryMode] = useState(false)
  const [trajectorySaving, setTrajectorySaving] = useState(false)
  const {classesOf, describe, error: ontologyError} = useLabelOntology()
  const [roiProfile, setRoiProfile] = useState<RoiProfile | null>(null)
  const [roiTop, setRoiTop] = useState(0)
  const [roiBottom, setRoiBottom] = useState(0)
  const [roiSaving, setRoiSaving] = useState(false)
  // Off-Path-Intervalle: waehrend des Anschauens markiert, wo KEIN einziger
  // Frame befahrbar ist. scrubMs folgt der freien Videoscrubbing-Position und
  // ist bewusst getrennt vom Polygon-Frame (timestampMs) — beim Anschauen
  // wird kontinuierlich gesucht, beim Labeln diskret Frame fuer Frame.
  const [intervalMode, setIntervalMode] = useState(false)
  const [scrubMs, setScrubMs] = useState(0)
  const [intervalStartMs, setIntervalStartMs] = useState<number | null>(null)
  const [intervalNote, setIntervalNote] = useState('')
  const [offPathIntervals, setOffPathIntervals] = useState<OffPathInterval[]>([])
  const [intervalSaving, setIntervalSaving] = useState(false)
  // Video-Komplettlabel: das ganze Video zeigt nur nicht befahrbaren Grund.
  const [fullyNotTraversableSaving, setFullyNotTraversableSaving] = useState(false)
  // Linear/Shuffle als echte Arbeitsmodi: shuffleSequence haelt {video_id,
  // frame_index}-Paare quer durch alle Videos der Mission.
  const [shuffleSequence, setShuffleSequence] = useState<{video_id: string; frame_index: number}[]>([])
  const [shuffleTotal, setShuffleTotal] = useState(100)
  // Kritisch-Flags direkt im Workflow statt nur im Modellzentrum.
  const [criticalFlags, setCriticalFlags] = useState<
    {video_id: string; frame_index: number; severity: number; note: string; annotator: string; created_at: string}[]
  >([])
  const [flagSeverity, setFlagSeverity] = useState(3)
  const [flagNote, setFlagNote] = useState('')
  const [flagSaving, setFlagSaving] = useState(false)
  // Suchbare Historie.
  const [historySearch, setHistorySearch] = useState('')
  const [historyStatusFilter, setHistoryStatusFilter] = useState<GroundTruthStatus | 'all'>('all')
  const videoRef = useRef<HTMLVideoElement>(null)
  const aiMaskCanvasRef = useRef<HTMLCanvasElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)
  const overlayRef = useRef<SVGSVGElement>(null)
  const dragRef = useRef<Drag | null>(null)
  // Identitaet der gerade bearbeiteten Stelle und ihr Zustand beim Uebernehmen.
  // Aus dem Vergleich entsteht "unveraendert uebernommen" vs. "angepasst".
  const trackingIdRef = useRef<string | null>(null)
  const carriedOriginRef = useRef<{points: NormalizedPoint[]; fromFrame: number} | null>(null)
  const carryRef = useRef<{
    videoId: string
    frameIndex: number
    points: NormalizedPoint[]
    trackingId: string
    fromFrame: number
    classId: string
  } | null>(null)

  // Im Shuffle-Modus bestimmt die Position in der gemischten Sequenz, welches
  // Video gerade aktiv ist — nicht die manuelle Dropdown-Auswahl. Beide
  // Herleitungen haengen an derselben Position und bleiben so synchron.
  const manualVideo = videos.find(video => video.video_id === activeVideoId) ?? videos[0]
  const shuffleEntryAt = (position: number) => (selectionMode === 'shuffle' ? shuffleSequence[position] : undefined)
  const activeVideo =
    selectionMode === 'shuffle'
      ? (videos.find(v => v.video_id === shuffleEntryAt(selectionPosition)?.video_id) ?? manualVideo)
      : manualVideo
  const selectedFrames = useMemo(() => {
    if (selectionMode === 'shuffle') return shuffleSequence.map(entry => entry.frame_index)
    return activeVideo ? buildFrameSelection(activeVideo.total_frames, selectionMode, stride, targetCount) : []
  }, [activeVideo, selectionMode, stride, targetCount, shuffleSequence])
  const frameIndex = selectedFrames[Math.min(selectionPosition, Math.max(0, selectedFrames.length - 1))] ?? 0
  const timestampMs = activeVideo ? timestampFor(frameIndex, activeVideo.fps) : 0
  const hasPolygon = points.length >= 3

  const regenerateShuffle = () => {
    setShuffleSequence(
      buildShuffleSelection(
        videos.map(video => ({video_id: video.video_id, total_frames: video.total_frames})),
        shuffleTotal,
      ),
    )
    setSelectionPosition(0)
  }

  useEffect(() => {
    setVideoTerrainDraft(activeVideo?.terrain_category ?? '')
  }, [activeVideo?.video_id, activeVideo?.terrain_category])

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    getLabelingVideos(mission.id)
      .then(result => {
        if (cancelled) return
        setVideos(result.videos)
        setActiveVideoId(result.videos[0]?.video_id ?? '')
      })
      .catch(error => setMessage(error instanceof Error ? error.message : 'Videos konnten nicht geladen werden'))
      .finally(() => !cancelled && setLoading(false))
    return () => {
      cancelled = true
    }
  }, [mission.id])

  useEffect(() => {
    void listGroundTruth(mission.id)
      .then(setSummary)
      .catch(error => setMessage(error instanceof Error ? error.message : 'Labelübersicht konnte nicht geladen werden'))
    void getPathModel(mission.id)
      .then(setPathModel)
      .catch(error => setMessage(error instanceof Error ? error.message : 'CPU-Wegmodell konnte nicht geladen werden'))
    void getPathTrainingJob(mission.id)
      .then(setTrainingJob)
      .catch(() => undefined)
  }, [mission.id])

  useEffect(() => {
    if (!trainingJob || !['queued', 'running'].includes(trainingJob.status)) return
    const timer = window.setInterval(() => {
      void getPathTrainingJob(mission.id)
        .then(job => {
          setTrainingJob(job)
          if (job?.status === 'completed') void getPathModel(mission.id).then(setPathModel)
        })
        .catch(() => undefined)
    }, 5000)
    return () => window.clearInterval(timer)
  }, [mission.id, trainingJob?.status, trainingJob?.job_id])

  useEffect(() => {
    if (!showAiMask || !pathModel || !activeVideo) {
      setPathPrediction(null)
      return
    }
    let cancelled = false
    setPredictionLoading(true)
    void predictPathFrame(mission.id, activeVideo.video_id, frameIndex)
      .then(prediction => {
        if (!cancelled) setPathPrediction(prediction)
      })
      .catch(error => {
        if (!cancelled) {
          setPathPrediction(null)
          setMessage(error instanceof Error ? error.message : 'KI-Wegmaske konnte nicht berechnet werden')
        }
      })
      .finally(() => {
        if (!cancelled) setPredictionLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [mission.id, activeVideo?.video_id, frameIndex, showAiMask, pathModel?.run_id])

  useEffect(() => {
    setRefinementSelection(null)
  }, [activeVideo?.video_id, frameIndex])

  // Trajektorie des Frames laden. Sie haengt nicht am Modell — hier wird von
  // Hand gezeichnet, im Modellzentrum zusaetzlich ein Vorschlag angeboten.
  useEffect(() => {
    if (!activeVideo) return
    let cancelled = false
    void getTrajectory(mission.id, activeVideo.video_id, frameIndex)
      .then(result => {
        if (cancelled) return
        setStoredTrajectory(result)
        setTrajectory(result ? (result.points.map(point => [...point]) as NormalizedPoint[]) : [])
      })
      .catch(() => {
        if (!cancelled) {
          setStoredTrajectory(null)
          setTrajectory([])
        }
      })
    return () => {
      cancelled = true
    }
  }, [mission.id, activeVideo?.video_id, frameIndex])

  // Profil des Videos laden; ohne gespeichertes Profil dienen die Erfahrungs-
  // werte des Backends als Startpunkt der Regler.
  useEffect(() => {
    if (!activeVideo) return
    let cancelled = false
    void getRoiProfile(mission.id, activeVideo.video_id)
      .then(result => {
        if (cancelled) return
        setRoiProfile(result)
        setRoiTop(result.top_ignore_fraction ?? result.suggested?.top_ignore_fraction ?? 0)
        setRoiBottom(result.bottom_ignore_fraction ?? result.suggested?.bottom_ignore_fraction ?? 0)
      })
      .catch(() => {
        if (!cancelled) setRoiProfile(null)
      })
    return () => {
      cancelled = true
    }
  }, [mission.id, activeVideo?.video_id])

  // Off-Path-Intervalle und Kritisch-Flags des aktiven Videos laden.
  useEffect(() => {
    if (!activeVideo) return
    let cancelled = false
    void listOffPathIntervals(mission.id, activeVideo.video_id)
      .then(result => {
        if (!cancelled) setOffPathIntervals(result)
      })
      .catch(() => {
        if (!cancelled) setOffPathIntervals([])
      })
    void listCriticalFlags(mission.id, activeVideo.video_id)
      .then(result => {
        if (!cancelled) setCriticalFlags(result.items)
      })
      .catch(() => {
        if (!cancelled) setCriticalFlags([])
      })
    return () => {
      cancelled = true
    }
  }, [mission.id, activeVideo?.video_id])

  useEffect(() => {
    const existing = criticalFlags.find(item => item.video_id === activeVideo?.video_id && item.frame_index === frameIndex)
    setFlagSeverity(existing?.severity ?? 3)
    setFlagNote(existing?.note ?? '')
  }, [criticalFlags, activeVideo?.video_id, frameIndex])

  const activeCriticalFlag = criticalFlags.find(item => item.video_id === activeVideo?.video_id && item.frame_index === frameIndex)

  const reportCriticalFlag = async () => {
    if (!activeVideo || flagSaving) return
    setFlagSaving(true)
    try {
      await saveCriticalFlag(mission.id, activeVideo.video_id, frameIndex, {
        severity: flagSeverity,
        note: flagNote.trim(),
        annotator: annotator.trim() || 'Simon',
      })
      setCriticalFlags(await listCriticalFlags(mission.id, activeVideo.video_id).then(result => result.items))
      setMessage(`Frame ${frameIndex + 1} als kritisch gemeldet (Stufe ${flagSeverity}).`)
    } catch (problem) {
      setMessage(problem instanceof Error ? problem.message : 'Meldung konnte nicht gespeichert werden')
    } finally {
      setFlagSaving(false)
    }
  }

  const removeCriticalFlag = async () => {
    if (!activeVideo || flagSaving) return
    setFlagSaving(true)
    try {
      await deleteCriticalFlag(mission.id, activeVideo.video_id, frameIndex)
      setCriticalFlags(current => current.filter(item => !(item.video_id === activeVideo.video_id && item.frame_index === frameIndex)))
      setMessage(`Kritisch-Meldung für Frame ${frameIndex + 1} entfernt.`)
    } catch (problem) {
      setMessage(problem instanceof Error ? problem.message : 'Meldung konnte nicht entfernt werden')
    } finally {
      setFlagSaving(false)
    }
  }

  // Intervall-Modus: der Nutzer scrubbt frei im Video (siehe controls={intervalMode}
  // am <video>-Element) statt Frame fuer Frame zu springen.
  const swappedInterval = (): [number, number] | null => {
    if (intervalStartMs === null) return null
    return intervalStartMs <= scrubMs ? [intervalStartMs, scrubMs] : [scrubMs, intervalStartMs]
  }

  const saveOffPathInterval = async () => {
    const bounds = swappedInterval()
    if (!activeVideo || !bounds || intervalSaving) return
    const [start, end] = bounds
    if (end - start < 200) {
      setMessage('Ein Intervall muss mindestens 200 ms lang sein — scrubbe weiter, bevor du speicherst.')
      return
    }
    setIntervalSaving(true)
    try {
      await createOffPathInterval(mission.id, activeVideo.video_id, {
        start_ms: start,
        end_ms: end,
        note: intervalNote,
        annotator: annotator.trim() || 'Simon',
      })
      setOffPathIntervals(await listOffPathIntervals(mission.id, activeVideo.video_id))
      setIntervalStartMs(null)
      setIntervalNote('')
      setMessage(
        `Off-Path-Intervall ${formatTimestamp(start)}–${formatTimestamp(end)} gespeichert. Geht als Vollnegativ-Frames ins Training.`,
      )
    } catch (problem) {
      setMessage(problem instanceof Error ? problem.message : 'Intervall konnte nicht gespeichert werden')
    } finally {
      setIntervalSaving(false)
    }
  }

  const removeOffPathInterval = async (intervalId: string) => {
    if (!activeVideo || intervalSaving) return
    setIntervalSaving(true)
    try {
      await deleteOffPathInterval(mission.id, activeVideo.video_id, intervalId)
      setOffPathIntervals(current => current.filter(item => item.id !== intervalId))
      setMessage('Intervall entfernt.')
    } catch (problem) {
      setMessage(problem instanceof Error ? problem.message : 'Intervall konnte nicht entfernt werden')
    } finally {
      setIntervalSaving(false)
    }
  }

  const toggleFullyNotTraversable = async () => {
    if (!activeVideo || fullyNotTraversableSaving) return
    const next = !activeVideo.fully_not_traversable
    setFullyNotTraversableSaving(true)
    try {
      await updateVideoFullyNotTraversable(mission.id, activeVideo.video_id, next)
      setVideos(current =>
        current.map(video => (video.video_id === activeVideo.video_id ? {...video, fully_not_traversable: next} : video)),
      )
      setMessage(
        next
          ? `${activeVideo.original_name} als komplett nicht befahrbar markiert — geht direkt als Trainingsbeispiel ein.`
          : `Komplettlabel für ${activeVideo.original_name} entfernt.`,
      )
      await onMissionUpdated?.()
    } catch (problem) {
      setMessage(problem instanceof Error ? problem.message : 'Komplettlabel konnte nicht gespeichert werden')
    } finally {
      setFullyNotTraversableSaving(false)
    }
  }

  const roiBands = (): LabelShape[] => {
    const bands: LabelShape[] = []
    const make = (kind: 'top' | 'bottom', fraction: number): LabelShape => {
      const [from, to] = kind === 'top' ? [0, fraction] : [1 - fraction, 1]
      return {
        id: `roi-${kind}`,
        class_id: 'roi_ignore',
        points: [
          [0, from],
          [1, from],
          [1, to],
          [0, to],
        ],
        certainty: 'certain',
        origin: 'manual',
        hard_negative: false,
        note: `${kind === 'top' ? 'Oberer' : 'Unterer'} Bildrand, ${Math.round(fraction * 100)} % ignoriert`,
      }
    }
    if (roiTop > 0) bands.push(make('top', roiTop))
    if (roiBottom > 0) bands.push(make('bottom', roiBottom))
    return bands
  }

  const applyRoiBands = () => {
    const bands = roiBands()
    if (!bands.length) return
    setShapes(current => [...current.filter(item => !item.id.startsWith('roi-')), ...bands])
    setDirty(true)
    setMessage(`Auswertungsbereich auf Frame ${frameIndex + 1} gelegt. Er gilt erst nach dem Speichern.`)
  }

  const persistRoiProfile = async () => {
    if (!activeVideo || roiSaving) return
    setRoiSaving(true)
    try {
      const saved = await saveRoiProfile(mission.id, activeVideo.video_id, {
        top_ignore_fraction: roiTop > 0 ? roiTop : null,
        bottom_ignore_fraction: roiBottom > 0 ? roiBottom : null,
        roi: roiBands(),
        note: '',
        annotator: annotator.trim() || 'Simon',
      })
      setRoiProfile(saved)
      setMessage(`ROI-Profil für ${activeVideo.original_name} gespeichert (Revision ${saved.revision}).`)
    } catch (problem) {
      setMessage(problem instanceof Error ? problem.message : 'ROI-Profil konnte nicht gespeichert werden')
    } finally {
      setRoiSaving(false)
    }
  }

  const persistTrajectory = async () => {
    if (!activeVideo || trajectorySaving) return
    if (trajectory.length < 2) {
      setMessage('Eine Trajektorie braucht mindestens zwei Punkte.')
      return
    }
    setTrajectorySaving(true)
    try {
      const saved = await saveTrajectory(mission.id, activeVideo.video_id, frameIndex, {
        timestamp_ms: timestampMs,
        points: trajectory,
        corridor: null,
        origin: 'manual',
        note: '',
        annotator: annotator.trim() || 'Simon',
      })
      setStoredTrajectory(saved)
      setMessage(`Trajektorie für Frame ${frameIndex + 1} gespeichert (Revision ${saved.revision}, von Hand gezeichnet).`)
    } catch (problem) {
      setMessage(problem instanceof Error ? problem.message : 'Trajektorie konnte nicht gespeichert werden')
    } finally {
      setTrajectorySaving(false)
    }
  }

  const discardTrajectory = async () => {
    if (!activeVideo || trajectorySaving) return
    setTrajectorySaving(true)
    try {
      if (storedTrajectory) await deleteTrajectory(mission.id, activeVideo.video_id, frameIndex)
      setStoredTrajectory(null)
      setTrajectory([])
      setMessage(`Trajektorie für Frame ${frameIndex + 1} entfernt.`)
    } catch (problem) {
      setMessage(problem instanceof Error ? problem.message : 'Trajektorie konnte nicht gelöscht werden')
    } finally {
      setTrajectorySaving(false)
    }
  }

  // Beim Refinement muss die Vergleichsmaske sichtbar sein, weil genau deren
  // rote und gelbe Flaechen angeklickt werden.
  const effectiveAiOverlay = refinementMode ? 'comparison' : aiOverlay
  const aiLayer = useMemo(() => {
    if (!pathPrediction) return {mask: null, palette: AI_BINARY_PALETTE, graded: false}
    if (effectiveAiOverlay === 'comparison') {
      return pathPrediction.evaluation
        ? {mask: pathPrediction.evaluation.comparison_mask, palette: COMPARISON_PALETTE, graded: false}
        : {mask: pathPrediction.mask, palette: AI_BINARY_PALETTE, graded: false}
    }
    return pathPrediction.grade_mask
      ? {mask: pathPrediction.grade_mask, palette: paletteFromGradeOntology(pathPrediction.grade_ontology), graded: true}
      : {mask: pathPrediction.mask, palette: AI_BINARY_PALETTE, graded: false}
  }, [pathPrediction, effectiveAiOverlay])

  useEffect(() => {
    const canvas = aiMaskCanvasRef.current
    if (canvas) paintMaskCanvas(canvas, showAiMask ? aiLayer.mask : null, aiLayer.palette)
  }, [aiLayer, showAiMask])

  useEffect(() => {
    if (!isPlaying || dirty || selectedFrames.length < 2) {
      if (isPlaying && (dirty || selectedFrames.length < 2)) setIsPlaying(false)
      return
    }
    const timer = window.setInterval(() => {
      setSelectionPosition(current => {
        if (current >= selectedFrames.length - 1) {
          setIsPlaying(false)
          return current
        }
        return current + 1
      })
    }, 900 / playbackSpeed)
    return () => window.clearInterval(timer)
  }, [isPlaying, dirty, selectedFrames.length, playbackSpeed])

  useEffect(() => {
    if (!activeVideo) return
    // Im Intervall-Modus scrubbt der Nutzer frei mit den nativen Videocontrols
    // (siehe controls={intervalMode} unten) — ein erzwungener Sprung auf den
    // Polygon-Frame wuerde dieses Scrubben staendig unterbrechen.
    if (!intervalMode) {
      const video = videoRef.current
      if (video) {
        video.pause()
        const seek = timestampMs / 1000
        if (Math.abs(video.currentTime - seek) > 0.0005) video.currentTime = seek
      }
    }
    let cancelled = false
    setLoading(true)
    setMessage('Frame wird geladen …')
    getGroundTruth(mission.id, activeVideo.video_id, frameIndex)
      .then(annotation => {
        if (cancelled) return
        const carried = carryRef.current
        // Gespeicherte Flächen landen in der Liste; die Werkzeuge bleiben frei
        // für die nächste. Zum Ändern wird eine Fläche gezielt zurückgeholt.
        const stored: LabelShape[] = [...(annotation?.polygons ?? []), ...(annotation?.roi ?? [])].map((item, index) => ({
          id: item.id || shapeId(item.class_id ?? 'traversable', index),
          class_id: item.class_id ?? 'traversable',
          points: copyPoints(item.points as NormalizedPoint[]),
          certainty: item.certainty ?? 'certain',
          origin: item.origin ?? 'manual',
          hard_negative: item.hard_negative ?? false,
          note: item.note ?? '',
          tracking_id: item.tracking_id ?? null,
          carried_from_frame: item.carried_from_frame ?? null,
          edit: item.edit ?? 'new',
        }))
        setShapes(stored)
        setEditingShapeId(null)
        if (!carryRef.current || carryRef.current.frameIndex !== frameIndex) {
          trackingIdRef.current = null
          carriedOriginRef.current = null
        }
        setShapeNote('')
        setShapeHardNegative(false)
        const polygon = stored.length === 1 ? stored[0].points : undefined
        const carriedIntoCurrentFrame = !annotation && carried?.videoId === activeVideo.video_id && carried.frameIndex === frameIndex
        if (polygon?.length) {
          // Genau eine Fläche: direkt zum Weiterbearbeiten öffnen, wie bisher.
          setShapes([])
          setShapeClass(stored[0].class_id)
          setShapeCertainty(stored[0].certainty)
          setShapeHardNegative(stored[0].hard_negative)
          setShapeNote(stored[0].note)
          setEditingShapeId(stored[0].id)
          setPoints(copyPoints(polygon))
          setTool('edit')
          setMessage('Gespeicherte Fläche geladen und vollständig editierbar.')
          setFullFrameNotTraversable(false)
        } else if (stored.length > 1) {
          setPoints([])
          setTool('add')
          setFullFrameNotTraversable(false)
          setMessage(`${stored.length} gespeicherte Flächen geladen. Zum Ändern eine aus der Liste wählen.`)
        } else if (
          annotation?.mask &&
          !annotation.polygons.length &&
          annotation.mask.rle.length === 2 &&
          annotation.mask.rle[0] === 2 &&
          annotation.mask.rle[1] === annotation.mask.width * annotation.mask.height
        ) {
          setPoints([])
          setTool('add')
          setFullFrameNotTraversable(true)
          setMessage('Gespeichertes Vollbild-Label geladen: Dieser Frame ist komplett nicht befahrbar.')
        } else if (carriedIntoCurrentFrame) {
          trackingIdRef.current = carried!.trackingId
          carriedOriginRef.current = {points: copyPoints(carried!.points), fromFrame: carried!.fromFrame}
          setShapeClass(carried!.classId)
          setPoints(copyPoints(carried!.points))
          setTool('edit')
          setFullFrameNotTraversable(false)
          // Erster Punkt gleich ausgewaehlt: Pfeiltasten koennen sofort
          // nachjustieren, ohne dass zuerst ein Punkt angeklickt werden muss.
          setSelectedVertex(0)
          setMessage(
            'Das Polygon des vorherigen Frames bleibt als neue Vorlage liegen. Punkt ausgewählt — Pfeiltasten justieren fein, oder ziehe direkt.',
          )
        } else {
          setPoints([])
          setTool('add')
          setFullFrameNotTraversable(false)
          setMessage(
            annotation?.status === 'skipped'
              ? 'Dieser Frame wurde als nicht relevant übersprungen.'
              : annotation?.mask
                ? 'Dieses ältere Rasterlabel enthält noch kein editierbares Polygon.'
                : 'Neuer Frame: Setze die Polygonpunkte direkt im Bild.',
          )
        }
        if (carried?.videoId === activeVideo.video_id && carried.frameIndex === frameIndex) carryRef.current = null
        setStatus(carriedIntoCurrentFrame ? 'draft' : (annotation?.status ?? 'new'))
        setRevision(annotation?.revision ?? 0)
        setNotes(annotation?.notes ?? '')
        setAnnotator(annotation?.annotator ?? 'Simon')
        setDirty(carriedIntoCurrentFrame)
        setPast(carriedIntoCurrentFrame ? [[]] : [])
        setFuture([])
        // Weitergetragene Fläche behält ihren vorausgewählten Punkt (siehe
        // oben) — jeder andere Fall startet ohne Auswahl.
        if (!carriedIntoCurrentFrame) setSelectedVertex(null)
      })
      .catch(error => setMessage(error instanceof Error ? error.message : 'Ground Truth konnte nicht geladen werden'))
      .finally(() => !cancelled && setLoading(false))
    return () => {
      cancelled = true
    }
  }, [mission.id, activeVideo?.video_id, frameIndex, timestampMs, intervalMode])

  const fullFrameMask = (): TerrainMask | null =>
    activeVideo ? {width: activeVideo.width, height: activeVideo.height, rle: [2, activeVideo.width * activeVideo.height]} : null

  const updatePoints = (next: NormalizedPoint[]) => {
    setPast(current => [...current.slice(-49), copyPoints(points)])
    setFuture([])
    setPoints(copyPoints(next))
    setDirty(true)
    if (status === 'confirmed' || status === 'skipped') setStatus('draft')
  }

  const undo = () => {
    const previous = past.at(-1)
    if (!previous) return
    setPast(current => current.slice(0, -1))
    setFuture(current => [copyPoints(points), ...current.slice(0, 49)])
    setPoints(copyPoints(previous))
    setDirty(true)
    setSelectedVertex(null)
  }

  const redo = () => {
    const next = future[0]
    if (!next) return
    setFuture(current => current.slice(1))
    setPast(current => [...current.slice(-49), copyPoints(points)])
    setPoints(copyPoints(next))
    setDirty(true)
    setSelectedVertex(null)
  }

  const pointFromEvent = (event: ReactPointerEvent<SVGSVGElement>): NormalizedPoint => {
    const bounds = event.currentTarget.getBoundingClientRect()
    return normalizedPointFromBounds(event.clientX, event.clientY, bounds)
  }

  const nearestVertex = (event: ReactPointerEvent<SVGSVGElement>, point: NormalizedPoint) => {
    const bounds = event.currentTarget.getBoundingClientRect()
    const thresholdX = 13 / Math.max(1, bounds.width)
    const thresholdY = 13 / Math.max(1, bounds.height)
    let match: number | null = null
    let best = 1
    points.forEach(([x, y], index) => {
      const distance = ((point[0] - x) / thresholdX) ** 2 + ((point[1] - y) / thresholdY) ** 2
      if (distance <= best) {
        best = distance
        match = index
      }
    })
    return match
  }

  const pointerDown = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (loading || saving) return
    const point = pointFromEvent(event)
    if (trajectoryMode) {
      // Von unten nach oben sortiert: die Bahn läuft vom Fahrzeug weg.
      setTrajectory(current => [...current, point].sort((a, b) => b[1] - a[1]))
      setMessage('Bahnpunkt gesetzt. Rechtsklick auf einen Punkt entfernt ihn wieder.')
      return
    }
    if (refinementMode) {
      const comparison = pathPrediction?.evaluation?.comparison_mask
      if (!comparison) {
        setMessage('Für diesen Frame ist keine farbige Modellbewertung verfügbar.')
        return
      }
      const value = rleValueAt(comparison, point)
      if (value !== 2 && value !== 3) {
        setRefinementSelection(null)
        setMessage('Klicke eine rote oder gelbe Fehlerfläche an.')
        return
      }
      setRefinementSelection({point, kind: value === 2 ? 'missed_label' : 'invented_path'})
      setMessage(
        value === 2
          ? 'Rote Fläche ausgewählt: Das Modell hat deinen bisherigen Weg hier übersehen.'
          : 'Gelbe Fläche ausgewählt: Das Modell hat hier zusätzlichen Weg erkannt.',
      )
      return
    }
    const vertex = nearestVertex(event, point)
    const action = pointerAction(tool, vertex, hasPolygon, hasPolygon && pointInPolygon(point, points))
    if (action === 'vertex' && vertex !== null) {
      setSelectedVertex(vertex)
      dragRef.current = {kind: 'vertex', index: vertex, before: copyPoints(points)}
    } else if (action === 'pan') {
      dragRef.current = {kind: 'pan', clientX: event.clientX, clientY: event.clientY, before: pan}
    } else if (action === 'polygon') {
      dragRef.current = {kind: 'polygon', origin: point, before: copyPoints(points)}
    } else if (action === 'add') {
      const next = hasPolygon
        ? [...points.slice(0, nearestSegmentIndex(point, points) + 1), point, ...points.slice(nearestSegmentIndex(point, points) + 1)]
        : [...points, point]
      updatePoints(next)
      setSelectedVertex(next.findIndex(item => item === point))
    }
    if (dragRef.current) event.currentTarget.setPointerCapture(event.pointerId)
  }

  const pointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
    const drag = dragRef.current
    if (!drag) return
    if (drag.kind === 'pan') {
      setPan({x: drag.before.x + event.clientX - drag.clientX, y: drag.before.y + event.clientY - drag.clientY})
      return
    }
    const point = pointFromEvent(event)
    if (drag.kind === 'vertex') {
      setPoints(current => current.map((item, index) => (index === drag.index ? point : item)))
    } else {
      const dx = point[0] - drag.origin[0]
      const dy = point[1] - drag.origin[1]
      const minX = Math.min(...drag.before.map(item => item[0]))
      const maxX = Math.max(...drag.before.map(item => item[0]))
      const minY = Math.min(...drag.before.map(item => item[1]))
      const maxY = Math.max(...drag.before.map(item => item[1]))
      const safeDx = clamp(dx, -minX, 1 - maxX)
      const safeDy = clamp(dy, -minY, 1 - maxY)
      setPoints(drag.before.map(([x, y]) => [x + safeDx, y + safeDy]))
    }
  }

  const pointerUp = (event: ReactPointerEvent<SVGSVGElement>) => {
    const drag = dragRef.current
    if (drag && drag.kind !== 'pan') {
      setPast(current => [...current.slice(-49), copyPoints(drag.before)])
      setFuture([])
      setDirty(true)
      if (status === 'confirmed' || status === 'skipped') setStatus('draft')
    }
    dragRef.current = null
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
  }

  const removeSelectedPoint = () => {
    if (selectedVertex === null) return
    const next = points.filter((_, index) => index !== selectedVertex)
    updatePoints(next)
    setSelectedVertex(null)
    if (next.length < 3) setTool('add')
  }

  const removeNearestPoint = (event: ReactPointerEvent<SVGSVGElement>) => {
    event.preventDefault()
    const index = nearestVertex(event, pointFromEvent(event))
    if (index === null) return
    setSelectedVertex(index)
    const next = points.filter((_, pointIndex) => pointIndex !== index)
    updatePoints(next)
    setSelectedVertex(null)
    if (next.length < 3) setTool('add')
  }

  const resetViewport = () => {
    setZoom(1)
    setPan({x: 0, y: 0})
  }
  const zoomAt = (nextZoom: number, clientX?: number, clientY?: number) => {
    const bounded = clamp(nextZoom, 0.5, 6)
    const bounds = stageRef.current?.getBoundingClientRect()
    const x = clientX !== undefined && bounds ? clientX - bounds.left : (bounds?.width ?? 0) / 2
    const y = clientY !== undefined && bounds ? clientY - bounds.top : (bounds?.height ?? 0) / 2
    const ratio = bounded / zoom
    setPan(current => ({x: x - (x - current.x) * ratio, y: y - (y - current.y) * ratio}))
    setZoom(bounded)
  }
  const wheel = (event: ReactWheelEvent<HTMLDivElement>) => {
    event.preventDefault()
    zoomAt(zoom * (event.deltaY < 0 ? 1.15 : 1 / 1.15), event.clientX, event.clientY)
  }

  const navigate = (direction: -1 | 1, allowDirty = false, carryForward = true) => {
    if (dirty && !allowDirty) {
      setMessage('Speichere die Änderung oder nutze Rückgängig, bevor du den Frame wechselst.')
      return false
    }
    const nextPosition = clamp(selectionPosition + direction, 0, Math.max(0, selectedFrames.length - 1))
    if (nextPosition === selectionPosition) return false
    // Im Shuffle-Modus liegt der naechste Frame meist in einem ANDEREN Video —
    // eine Flaeche dorthin weiterzutragen ergaebe geometrisch keinen Sinn.
    const effectiveCarryForward = carryForward && selectionMode !== 'shuffle'
    const carriedPoints = polygonForNextFrame(points, direction, effectiveCarryForward)
    // Die Spur-ID reist mit: dieselbe Stelle behaelt ueber Frames hinweg ihre
    // Identitaet, damit spaeter ableitbar ist, wie sie sich veraendert hat.
    carryRef.current =
      carriedPoints && activeVideo
        ? {
            videoId: activeVideo.video_id,
            frameIndex: selectedFrames[nextPosition],
            points: carriedPoints,
            trackingId: ensureTrackingId(),
            fromFrame: frameIndex,
            classId: shapeClass,
          }
        : null
    setSelectionPosition(nextPosition)
    return true
  }

  /** Ohne Umweg mehrere Positionen weiterspringen — kein Weitertragen, das
   *  ergäbe über eine Lücke hinweg keinen Sinn. */
  const jumpBy = (steps: number) => {
    if (dirty) {
      setMessage('Speichere die Änderung oder nutze Rückgängig, bevor du springst.')
      return
    }
    const next = clamp(selectionPosition + steps, 0, Math.max(0, selectedFrames.length - 1))
    if (next === selectionPosition) return
    carryRef.current = null
    setSelectionPosition(next)
  }

  const isResolved = (videoId: string | undefined, index: number) =>
    !!videoId && summary.items.some(item => item.video_id === videoId && item.frame_index === index && item.status === 'confirmed')

  /** Springt zum naechsten noch nicht bestaetigten Frame in der aktuellen
   *  Auswahl — videouebergreifend im Shuffle-Modus. */
  const jumpToUnresolved = (direction: 1 | -1) => {
    if (dirty) {
      setMessage('Speichere die Änderung oder nutze Rückgängig, bevor du springst.')
      return
    }
    for (let position = selectionPosition + direction; position >= 0 && position < selectedFrames.length; position += direction) {
      const videoId = selectionMode === 'shuffle' ? shuffleEntryAt(position)?.video_id : activeVideo?.video_id
      if (!isResolved(videoId, selectedFrames[position])) {
        carryRef.current = null
        setSelectionPosition(position)
        return
      }
    }
    setMessage(direction === 1 ? 'Kein weiterer ungeklärter Frame danach.' : 'Kein ungeklärter Frame davor.')
  }

  /** Spur-ID der aktuellen Fläche — erzeugt sie beim ersten Zugriff und merkt
   *  sie. Ohne das Merken bekäme dieselbe Fläche beim Speichern und beim
   *  Weitertragen zwei verschiedene IDs, und die Verkettung bräche auseinander. */
  const ensureTrackingId = () => {
    if (!trackingIdRef.current) trackingIdRef.current = newTrackingId()
    return trackingIdRef.current
  }

  const currentShape = (): LabelShape => {
    const carried = carriedOriginRef.current
    return {
      id: editingShapeId ?? shapeId(shapeClass, shapes.length),
      class_id: shapeClass,
      points: copyPoints(points),
      certainty: shapeCertainty,
      origin: 'manual',
      hard_negative: shapeHardNegative,
      note: shapeNote,
      tracking_id: ensureTrackingId(),
      carried_from_frame: carried?.fromFrame ?? null,
      // Unveraendert uebernommen oder nachgezogen? Genau die Stellen, an denen
      // von Hand nachgebessert wurde, sind die schwierigen im Video.
      edit: !carried ? 'new' : samePoints(carried.points, points) ? 'carried_unchanged' : 'carried_adjusted',
    }
  }

  /** Aktuelle Fläche in die Liste übernehmen und die Werkzeuge freimachen. */
  const commitShape = () => {
    if (!hasPolygon) {
      setMessage('Setze mindestens drei Punkte, bevor du die Fläche übernimmst.')
      return
    }
    const shape = currentShape()
    setShapes(current => [...current.filter(item => item.id !== shape.id), shape])
    setPoints([])
    setEditingShapeId(null)
    setShapeNote('')
    setShapeHardNegative(false)
    trackingIdRef.current = null
    carriedOriginRef.current = null
    setTool('add')
    setDirty(true)
    setMessage(`${describe(shape.class_id).label} übernommen. Nächste Fläche kann gezeichnet werden.`)
  }

  /** Eine übernommene Fläche zurück in die Werkzeuge holen. */
  const editShape = (shape: LabelShape) => {
    if (hasPolygon && !editingShapeId) {
      setMessage('Übernimm oder lösche zuerst die aktuelle Fläche.')
      return
    }
    setShapes(current => current.filter(item => item.id !== shape.id))
    setPoints(copyPoints(shape.points))
    setShapeClass(shape.class_id)
    setShapeCertainty(shape.certainty)
    setShapeHardNegative(shape.hard_negative)
    setShapeNote(shape.note)
    setEditingShapeId(shape.id)
    trackingIdRef.current = shape.tracking_id ?? null
    // Eine erneut geoeffnete, bereits gespeicherte Flaeche gilt als korrigiert,
    // sobald sie veraendert wird — nicht als neu gezeichnet.
    carriedOriginRef.current =
      shape.carried_from_frame != null ? {points: copyPoints(shape.points), fromFrame: shape.carried_from_frame} : null
    setTool('edit')
    setMessage(`${describe(shape.class_id).label} zum Bearbeiten geladen.`)
  }

  const removeShape = (id: string) => {
    setShapes(current => current.filter(item => item.id !== id))
    setDirty(true)
  }

  const persist = async (nextStatus: GroundTruthStatus) => {
    if (!activeVideo || saving) return null
    const willHaveShapes = shapes.length > 0 || hasPolygon
    if (nextStatus === 'confirmed' && !willHaveShapes && !fullFrameNotTraversable) {
      setMessage('Markiere mindestens eine Fläche oder wähle Vollbild-Nicht-befahrbar, bevor du bestätigst.')
      return null
    }
    setSaving(true)
    setMessage(
      nextStatus === 'skipped'
        ? 'Frame wird als nicht relevant gespeichert …'
        : fullFrameNotTraversable
          ? 'Vollbild-Label wird gespeichert …'
          : 'Polygon wird gespeichert …',
    )
    try {
      // Die gerade bearbeitete Fläche zählt mit, auch ohne vorheriges
      // "Fläche übernehmen" — sonst ginge sie beim Speichern still verloren.
      const collected = nextStatus === 'skipped' ? [] : [...shapes, ...(hasPolygon ? [currentShape()] : [])]
      const polygons = collected.filter(shape => describe(shape.class_id).layer !== 'roi')
      const roi = collected.filter(shape => describe(shape.class_id).layer === 'roi')
      const saved = await saveGroundTruth(mission.id, activeVideo.video_id, frameIndex, {
        timestamp_ms: timestampMs,
        mask: fullFrameNotTraversable ? (fullFrameMask() ?? undefined) : undefined,
        polygons: fullFrameNotTraversable ? [] : polygons,
        roi,
        frame_width: activeVideo.width,
        frame_height: activeVideo.height,
        status: nextStatus,
        annotator: annotator.trim() || 'Simon',
        notes,
        label_mode: selectionMode === 'shuffle' ? 'shuffle' : 'linear',
      })
      setStatus(saved.status)
      setRevision(saved.revision)
      setDirty(false)
      setPast([])
      setFuture([])
      setSummary(await listGroundTruth(mission.id))
      setMessage(
        nextStatus === 'confirmed'
          ? fullFrameNotTraversable
            ? `Frame ${frameIndex + 1} als Vollbild-Nicht-befahrbar bestätigt.`
            : `Frame ${frameIndex + 1} bestätigt.`
          : nextStatus === 'skipped'
            ? `Frame ${frameIndex + 1} übersprungen.`
            : `Entwurf für Frame ${frameIndex + 1} gespeichert.`,
      )
      return saved
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Ground Truth konnte nicht gespeichert werden')
      return null
    } finally {
      setSaving(false)
    }
  }

  const saveVideoTerrainCategory = async () => {
    if (!activeVideo || videoTerrainSaving) return
    const nextTerrainCategory = videoTerrainDraft.trim() || null
    if ((activeVideo.terrain_category ?? null) === nextTerrainCategory) {
      setMessage('Terrainkategorie ist bereits gespeichert.')
      return
    }
    setVideoTerrainSaving(true)
    setMessage(
      nextTerrainCategory
        ? `Terrainkategorie für ${activeVideo.original_name} wird gespeichert …`
        : `Terrainkategorie für ${activeVideo.original_name} wird entfernt …`,
    )
    try {
      const saved = await updateVideoTerrainCategory(mission.id, activeVideo.video_id, {terrain_category: nextTerrainCategory})
      setVideos(current =>
        current.map(video =>
          video.video_id === activeVideo.video_id ? {...video, terrain_category: saved.terrain_category ?? null} : video,
        ),
      )
      setVideoTerrainDraft(saved.terrain_category ?? '')
      setMessage(
        saved.terrain_category
          ? `Terrainkategorie für ${activeVideo.original_name} gespeichert: ${terrainCategoryLabel(saved.terrain_category)}`
          : `Terrainkategorie für ${activeVideo.original_name} entfernt.`,
      )
      await onMissionUpdated?.()
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Terrainkategorie konnte nicht gespeichert werden')
    } finally {
      setVideoTerrainSaving(false)
    }
  }

  const saveAndNext = async () => {
    if (await persist('confirmed')) navigate(1, true)
  }
  const skipFrame = async () => {
    if (dirty) {
      setMessage('Der Frame enthält ungespeicherte Änderungen. Speichere oder mache sie zuerst rückgängig.')
      return
    }
    if (await persist('skipped')) navigate(1, true, false)
  }

  const openSavedMask = (item: GroundTruthSummary['items'][number]) => {
    if (dirty) {
      setMessage('Speichere oder verwirf zuerst die aktuelle Änderung, bevor du eine gespeicherte Maske öffnest.')
      return
    }
    carryRef.current = null
    setSelectionMode('stride')
    setStride(1)
    setActiveVideoId(item.video_id)
    setSelectionPosition(item.frame_index)
    resetViewport()
  }

  const startProcessing = async () => {
    if (!summary.counts.confirmed || processing) {
      setMessage('Bestätige mindestens ein Polygon, bevor du die Auswertung startest.')
      return
    }
    setProcessing(true)
    setMessage('Markierungsrunde abgeschlossen. Die automatische Auswertung wird jetzt ausdrücklich gestartet …')
    try {
      await runSegmentation(mission.id)
      setMessage('Auswertung abgeschlossen. Ergebnis wird geöffnet …')
      await onProcessingComplete()
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Auswertung konnte nicht gestartet werden')
    } finally {
      setProcessing(false)
    }
  }

  const trainCpuPathModel = async () => {
    if (dirty || modelTraining) return
    if (summary.counts.confirmed < 10) {
      setMessage('Für das Training werden mindestens 10 bestätigte Polygonframes benötigt.')
      return
    }
    setModelTraining(true)
    setMessage(`CPU-Wegmodell wird aus ${summary.counts.confirmed} bestätigten Polygonframes trainiert …`)
    try {
      const result = await trainPathModel(mission.id)
      setPathModel(result)
      setMessage(
        `CPU-Training abgeschlossen: ${result.validation_metrics.symmetric_score.toFixed(2)} von 100 Punkten auf getrennten Validierungsframes.`,
      )
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'CPU-Wegmodell konnte nicht trainiert werden')
    } finally {
      setModelTraining(false)
    }
  }

  const confirmRefinement = async () => {
    if (!activeVideo || !refinementSelection || refinementSaving) return
    setRefinementSaving(true)
    try {
      const saved = await savePathRefinement(mission.id, activeVideo.video_id, frameIndex, {
        x: refinementSelection.point[0],
        y: refinementSelection.point[1],
        expected_kind: refinementSelection.kind,
        action: 'accept_model',
      })
      const refreshed = await predictPathFrame(mission.id, activeVideo.video_id, frameIndex)
      setPathPrediction(refreshed)
      setRefinementSelection(null)
      setMessage(
        `Refinement gespeichert. Dieser Frame enthält jetzt ${saved.refinement_count} Feedback-Korrektur${saved.refinement_count === 1 ? '' : 'en'} für das nächste Training.`,
      )
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Refinement konnte nicht gespeichert werden')
    } finally {
      setRefinementSaving(false)
    }
  }

  const startBackgroundTraining = async () => {
    if (dirty) {
      setMessage('Speichere oder verwirf zuerst die Änderungen am aktuellen Polygon.')
      return
    }
    if (summary.counts.confirmed < 10) {
      setMessage('Für das Training werden mindestens 10 bestätigte Polygonframes benötigt.')
      return
    }
    if (trainingJob && ['queued', 'running'].includes(trainingJob.status)) {
      setMessage('Für diese Mission läuft bereits ein Hintergrundtraining.')
      return
    }
    try {
      const duration = trainingProfile === 'quick' ? 0.25 : trainingHours
      const job = await startPathTrainingJob(mission.id, trainingProfile, duration)
      setTrainingJob(job)
      setMessage(job.message)
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Hintergrundtraining konnte nicht gestartet werden')
    }
  }

  useEffect(() => {
    const keydown = (event: KeyboardEvent) => {
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes((event.target as HTMLElement)?.tagName)) return
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
        event.preventDefault()
        if (event.shiftKey) redo()
        else undo()
        return
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'y') {
        event.preventDefault()
        redo()
        return
      }
      if (event.key === 'Delete' || event.key === 'Backspace') {
        event.preventDefault()
        removeSelectedPoint()
        return
      }
      // Punkt-Feintuning per Pfeiltaste geht vor Frame-Navigation — sonst
      // liesse sich ein ausgewählter Punkt nie mit den Pfeiltasten justieren.
      if (selectedVertex !== null && tool === 'edit' && ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(event.key)) {
        event.preventDefault()
        const step = (event.shiftKey ? 0.01 : 0.0015) / zoom
        const dx = event.key === 'ArrowLeft' ? -step : event.key === 'ArrowRight' ? step : 0
        const dy = event.key === 'ArrowUp' ? -step : event.key === 'ArrowDown' ? step : 0
        updatePoints(points.map((point, index) => (index === selectedVertex ? [clamp(point[0] + dx), clamp(point[1] + dy)] : point)))
        return
      }
      if (event.key === 'ArrowRight' && !dirty) {
        event.preventDefault()
        navigate(1)
        return
      }
      if (event.key === 'ArrowLeft' && !dirty) {
        event.preventDefault()
        navigate(-1)
        return
      }
      if (event.key.toLowerCase() === 'u') {
        event.preventDefault()
        jumpToUnresolved(event.shiftKey ? -1 : 1)
        return
      }
      if (event.key.toLowerCase() === 's' && !event.ctrlKey && !event.metaKey) {
        event.preventDefault()
        void persist('draft')
        return
      }
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault()
        void saveAndNext()
        return
      }
      if (event.key.toLowerCase() === 'f') {
        event.preventDefault()
        setFullFrameNotTraversable(current => !current)
        setDirty(true)
        if (status === 'confirmed' || status === 'skipped') setStatus('draft')
        return
      }
      if (event.key.toLowerCase() === 't') {
        event.preventDefault()
        setTrajectoryMode(current => !current)
        return
      }
      if (['1', '2', '3', '4'].includes(event.key)) {
        const chosen = classesOf('core')[Number(event.key) - 1]
        if (chosen) {
          event.preventDefault()
          setShapeClass(chosen.class_id)
        }
      }
    }
    window.addEventListener('keydown', keydown)
    return () => window.removeEventListener('keydown', keydown)
  })

  if (!activeVideo)
    return (
      <div className="labeling-page">
        <div className="labeling-header">
          <button onClick={onClose}>← Missionen</button>
          <h1>Ground Truth</h1>
        </div>
        <div className="alert error">{loading ? 'Originalvideos werden gelesen …' : message || 'Kein Video verfügbar.'}</div>
      </div>
    )

  const frameStyle = {
    aspectRatio: `${activeVideo.width} / ${activeVideo.height}`,
    width: activeVideo.width >= activeVideo.height ? '100%' : `min(100%, ${Math.max(28, (70 * activeVideo.width) / activeVideo.height)}vh)`,
  } as CSSProperties
  const transformStyle = {transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`} as CSSProperties
  const polygonPoints = points.map(point => point.join(',')).join(' ')
  // Fortschritt innerhalb der aktuell gewählten Auswahl — videoübergreifend
  // korrekt auch im Shuffle-Modus, weil isResolved je Position das richtige
  // Video nachschlägt statt pauschal activeVideo anzunehmen.
  const resolvedInSelection = selectedFrames.filter((index, position) =>
    isResolved(selectionMode === 'shuffle' ? shuffleEntryAt(position)?.video_id : activeVideo.video_id, index),
  ).length
  const progressFraction = selectedFrames.length ? resolvedInSelection / selectedFrames.length : 0
  // Durchsuchbare Historie: Suche greift auf Videoname, Bearbeiter und
  // Framenummer zu, damit ein bestimmter Eintrag auch bei vielen hundert
  // gespeicherten Masken schnell wiederzufinden ist.
  const filteredHistoryItems = summary.items.filter(item => {
    if (historyStatusFilter !== 'all' && item.status !== historyStatusFilter) return false
    if (!historySearch.trim()) return true
    const needle = historySearch.trim().toLowerCase()
    const videoName = videos.find(candidate => candidate.video_id === item.video_id)?.original_name ?? ''
    return (
      videoName.toLowerCase().includes(needle) ||
      item.annotator.toLowerCase().includes(needle) ||
      String(item.frame_index + 1).includes(needle)
    )
  })

  return (
    <div className="labeling-page">
      <header className="labeling-header">
        <button onClick={onClose}>← Missionen</button>
        <div>
          <span className="eyebrow">MANUELLE GROUND TRUTH · OPTIONALE KI-VORSCHAU</span>
          <h1>{mission.name}</h1>
          <p>Markiere den befahrbaren Bereich im Originalvideoframe; die KI-Maske ist nur ein getrenntes Vergleichs-Overlay.</p>
        </div>
        <div className="labeling-progress">
          <b>
            Frame {frameIndex + 1} von {activeVideo.total_frames.toLocaleString('de-DE')}
          </b>
          <span>
            Auswahl {selectionPosition + 1} von {selectedFrames.length.toLocaleString('de-DE')}
          </span>
        </div>
      </header>

      <div className="labeling-workspace">
        <section className="labeling-main">
          <div className="labeling-viewport-toolbar">
            <div>
              <button
                className={isPlaying ? 'play-button active' : 'play-button'}
                disabled={dirty || selectedFrames.length < 2}
                onClick={() => setIsPlaying(current => !current)}
                aria-label={isPlaying ? 'Gelabelte Frames pausieren' : 'Gelabelte Frames abspielen'}
              >
                {isPlaying ? '❚❚ Pause' : '▶ Gelabelte Frames abspielen'}
              </button>
              <label className="playback-speed-control">
                Tempo
                <select aria-label="Abspielgeschwindigkeit" value={playbackSpeed} onChange={event => setPlaybackSpeed(+event.target.value)}>
                  <option value="0.25">0,25×</option>
                  <option value="0.5">0,5×</option>
                  <option value="1">1×</option>
                  <option value="2">2×</option>
                  <option value="4">4×</option>
                </select>
              </label>
              <button onClick={() => zoomAt(zoom / 1.25)} aria-label="Herauszoomen">
                −
              </button>
              <b>{Math.round(zoom * 100)}%</b>
              <button onClick={() => zoomAt(zoom * 1.25)} aria-label="Hineinzoomen">
                +
              </button>
              <button onClick={resetViewport}>Ansicht zurücksetzen</button>
              <button
                className={intervalMode ? 'interval-mode-toggle active' : 'interval-mode-toggle'}
                onClick={() => {
                  setIntervalMode(current => !current)
                  setIntervalStartMs(null)
                }}
              >
                {intervalMode ? '✕ Intervall-Modus beenden' : 'Intervall-Modus: kein Weg im Video markieren'}
              </button>
            </div>
            <span>
              {intervalMode
                ? 'Video frei scrubbar · Start und Ende eines wegfreien Abschnitts markieren'
                : 'Mausrad zoomt · Hand verschiebt das Bild'}
            </span>
          </div>
          {activeCriticalFlag && (
            <div className={`critical-flag-banner severity-${activeCriticalFlag.severity}`} role="alert">
              <b>⚠ Als kritisch gemeldet · Stufe {activeCriticalFlag.severity}/5</b>
              {activeCriticalFlag.note && <span>{activeCriticalFlag.note}</span>}
              <button disabled={flagSaving} onClick={() => void removeCriticalFlag()}>
                Meldung entfernen
              </button>
            </div>
          )}
          <div className="labeling-stage-background">
            <div className="labeling-stage" ref={stageRef} style={frameStyle} onWheel={wheel}>
              <div className="labeling-transform" style={transformStyle}>
                <video
                  ref={videoRef}
                  src={`/api/v1/missions/${mission.id}/videos/${activeVideo.video_id}/content`}
                  preload="auto"
                  muted
                  playsInline
                  controls={intervalMode}
                  onTimeUpdate={event => {
                    if (intervalMode) setScrubMs(Math.round(event.currentTarget.currentTime * 1000))
                  }}
                  onLoadedMetadata={() => {
                    if (videoRef.current && !intervalMode) videoRef.current.currentTime = timestampMs / 1000
                  }}
                />
                <canvas ref={aiMaskCanvasRef} className="ai-path-mask-layer" style={{opacity: aiMaskOpacity}} aria-hidden="true" />
                <svg
                  ref={overlayRef}
                  viewBox="0 0 1 1"
                  preserveAspectRatio="none"
                  className={`polygon-overlay tool-${tool} ${fullFrameNotTraversable ? 'full-frame-mode' : ''} ${trajectoryMode ? 'trajectory-mode' : ''}`}
                  onPointerDown={pointerDown}
                  onPointerMove={pointerMove}
                  onPointerUp={pointerUp}
                  onPointerCancel={pointerUp}
                  onContextMenu={removeNearestPoint}
                  aria-label="Ground-Truth-Polygonfläche"
                >
                  {/* Bereits übernommene Flächen in ihrer Klassenfarbe. */}
                  {shapes.map(shape => (
                    <polygon
                      key={shape.id}
                      points={shape.points.map(([x, y]) => `${x},${y}`).join(' ')}
                      className={`committed-shape ${shape.certainty} ${shape.hard_negative ? 'hard-negative' : ''}`}
                      style={{fill: describe(shape.class_id).color, fillOpacity: maskOpacity, stroke: describe(shape.class_id).color}}
                      onClick={() => editShape(shape)}
                    >
                      <title>{`${describe(shape.class_id).label} — zum Bearbeiten anklicken`}</title>
                    </polygon>
                  ))}
                  {!fullFrameNotTraversable && points.length >= 3 && (
                    <polygon
                      points={polygonPoints}
                      className="ground-truth-polygon"
                      style={{fillOpacity: maskOpacity, fill: describe(shapeClass).color, stroke: describe(shapeClass).color}}
                    />
                  )}
                  {/* Von Hand gezeichnete Trajektorie. */}
                  {trajectory.length > 1 && (
                    <polyline
                      points={trajectory.map(([x, y]) => `${x},${y}`).join(' ')}
                      className="manual-trajectory"
                      vectorEffect="non-scaling-stroke"
                    />
                  )}
                  {trajectory.map(([x, y], index) => (
                    <ellipse
                      key={`trajectory-${index}`}
                      cx={x}
                      cy={y}
                      rx={0.011 / zoom}
                      ry={0.019 / zoom}
                      className="manual-trajectory-point"
                      onContextMenu={event => {
                        event.preventDefault()
                        event.stopPropagation()
                        setTrajectory(current => current.filter((_, position) => position !== index))
                      }}
                    />
                  ))}
                  {!fullFrameNotTraversable && points.length > 1 && points.length < 3 && (
                    <polyline points={polygonPoints} className="ground-truth-polyline" />
                  )}
                  {!fullFrameNotTraversable &&
                    points.map(([x, y], index) => (
                      <ellipse
                        key={index}
                        cx={x}
                        cy={y}
                        rx={0.008 / zoom}
                        ry={0.014 / zoom}
                        className={selectedVertex === index ? 'polygon-vertex selected' : 'polygon-vertex'}
                      />
                    ))}
                  {refinementSelection && (
                    <ellipse
                      cx={refinementSelection.point[0]}
                      cy={refinementSelection.point[1]}
                      rx={0.014 / zoom}
                      ry={0.024 / zoom}
                      className="refinement-marker"
                    />
                  )}
                </svg>
                {fullFrameNotTraversable && (
                  <div className="full-frame-flag">
                    <b>Ganzes Bild</b>
                    <span>als nicht befahrbar markiert</span>
                  </div>
                )}
              </div>
            </div>
          </div>
          <div className="labeling-navigation">
            <button disabled={selectionPosition === 0 || saving || dirty} onClick={() => navigate(-1)}>
              ← Vorheriger Frame
            </button>
            <button disabled={saving || dirty} onClick={() => void skipFrame()}>
              Frame überspringen
            </button>
            <button disabled={selectionPosition >= selectedFrames.length - 1 || saving || dirty} onClick={() => navigate(1)}>
              Nächster Frame →
            </button>
            <button disabled={dirty} title="Vorheriger ungeklärter Frame (U mit Shift)" onClick={() => jumpToUnresolved(-1)}>
              ⏮ Vorheriger ungeklärter
            </button>
            <button disabled={dirty} title="Nächster ungeklärter Frame (U)" onClick={() => jumpToUnresolved(1)}>
              Nächster ungeklärter ⏭
            </button>
            <button disabled={dirty || selectionPosition < 10} title="10 Positionen zurück" onClick={() => jumpBy(-10)}>
              ⏪ −10
            </button>
            <button
              disabled={dirty || selectionPosition >= selectedFrames.length - 10}
              title="10 Positionen vor"
              onClick={() => jumpBy(10)}
            >
              +10 ⏩
            </button>
          </div>

          {intervalMode && activeVideo && (
            <div className="off-path-interval-panel">
              <h3>Off-Path-Intervalle: kein Weg im Bild</h3>
              <small>
                Scrubbe im Video zum Anfang eines Abschnitts ohne jeden befahrbaren Frame, setze den Start, scrubbe zum Ende und speichere.
                Der Abschnitt geht als Vollnegativ-Frames ins Training ein — die KI wird bestraft, wenn sie dort Wegfläche erkennt.
              </small>
              <div className="off-path-timeline">
                <div className="off-path-timeline-track">
                  {offPathIntervals.map(interval => {
                    const durationMs = activeVideo.duration_seconds * 1000
                    const left = (interval.start_ms / durationMs) * 100
                    const width = ((interval.end_ms - interval.start_ms) / durationMs) * 100
                    return (
                      <div
                        key={interval.id}
                        className="off-path-interval-block"
                        style={{left: `${left}%`, width: `${Math.max(0.3, width)}%`}}
                        title={`${formatTimestamp(interval.start_ms)}–${formatTimestamp(interval.end_ms)}${interval.note ? ' · ' + interval.note : ''}`}
                      />
                    )
                  })}
                  {intervalStartMs !== null && (
                    <div
                      className="off-path-start-marker"
                      style={{left: `${(intervalStartMs / (activeVideo.duration_seconds * 1000)) * 100}%`}}
                    />
                  )}
                  <div className="off-path-scrub-marker" style={{left: `${(scrubMs / (activeVideo.duration_seconds * 1000)) * 100}%`}} />
                </div>
              </div>
              <div className="off-path-interval-controls">
                <span>Position: {formatTimestamp(scrubMs)}</span>
                {intervalStartMs === null ? (
                  <button onClick={() => setIntervalStartMs(scrubMs)}>Start hier setzen</button>
                ) : (
                  <>
                    <span>Start: {formatTimestamp(intervalStartMs)}</span>
                    <button onClick={() => setIntervalStartMs(null)}>Start verwerfen</button>
                    <input
                      value={intervalNote}
                      maxLength={500}
                      placeholder="Notiz (optional)"
                      onChange={event => setIntervalNote(event.target.value)}
                    />
                    <button className="primary" disabled={intervalSaving} onClick={() => void saveOffPathInterval()}>
                      {intervalSaving ? 'WIRD GESPEICHERT …' : 'Intervall speichern'}
                    </button>
                  </>
                )}
              </div>
              {offPathIntervals.length > 0 && (
                <ul className="off-path-interval-list">
                  {offPathIntervals.map(interval => (
                    <li key={interval.id}>
                      <b>
                        {formatTimestamp(interval.start_ms)}–{formatTimestamp(interval.end_ms)}
                      </b>
                      {interval.note && <span>{interval.note}</span>}
                      <button className="danger" disabled={intervalSaving} onClick={() => void removeOffPathInterval(interval.id)}>
                        Entfernen
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </section>

        <aside className="labeling-controls">
          <h2>Frame-Auswahl</h2>
          <label>
            Originalvideo
            <select
              value={activeVideo.video_id}
              disabled={dirty || selectionMode === 'shuffle'}
              onChange={event => {
                setActiveVideoId(event.target.value)
                setSelectionPosition(0)
                resetViewport()
              }}
            >
              {videos.map(video => (
                <option key={video.video_id} value={video.video_id}>
                  {video.original_name} · {terrainCategoryLabel(video.terrain_category)}
                </option>
              ))}
            </select>
            <small>
              {selectionMode === 'shuffle'
                ? 'Im Shuffle-Modus wechselt das Video automatisch mit jedem gemischten Frame.'
                : activeVideo.terrain_category
                  ? `Terrainkategorie: ${terrainCategoryLabel(activeVideo.terrain_category)}`
                  : 'Terrainkategorie noch nicht gewählt. Alle Frames dieses Videos übernehmen dieses Label.'}
            </small>
          </label>
          <label>
            Terrainkategorie
            <select
              aria-label="Terrainkategorie"
              value={videoTerrainDraft}
              disabled={videoTerrainSaving}
              onChange={event => setVideoTerrainDraft(event.target.value)}
            >
              <option value="">Noch keine Kategorie</option>
              {TERRAIN_CATEGORY_OPTIONS.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            <small>Die Kategorie gilt für alle Frames dieses Videos und kann hier nachträglich geändert werden.</small>
          </label>
          <div className="terrain-category-actions">
            <button
              disabled={videoTerrainSaving || videoTerrainDraft.trim() === (activeVideo.terrain_category ?? '')}
              onClick={() => void saveVideoTerrainCategory()}
            >
              {videoTerrainSaving ? 'KATEGORIE WIRD GESPEICHERT …' : 'Kategorie speichern'}
            </button>
            <small>
              {videoTerrainDraft.trim() === (activeVideo.terrain_category ?? '')
                ? 'Keine ungespeicherten Änderungen an der Videokategorie.'
                : 'Videokategorie geändert – zum Übernehmen speichern.'}
            </small>
          </div>
          <div className="fully-not-traversable-panel">
            <button
              className={activeVideo.fully_not_traversable ? 'fully-not-traversable-toggle active' : 'fully-not-traversable-toggle'}
              disabled={fullyNotTraversableSaving}
              onClick={() => void toggleFullyNotTraversable()}
            >
              {fullyNotTraversableSaving
                ? 'WIRD GESPEICHERT …'
                : activeVideo.fully_not_traversable
                  ? '✓ Ganzes Video: komplett nicht befahrbar'
                  : 'Ganzes Video als komplett nicht befahrbar markieren'}
            </button>
            <small>
              Für Videos, die von Anfang bis Ende keinen befahrbaren Grund zeigen. Kein Polygon nötig — das ganze Video geht direkt als
              Trainingsbeispiel ein.
            </small>
          </div>
          <div className="sampling-tabs">
            <button
              className={selectionMode === 'stride' ? 'active' : ''}
              disabled={dirty}
              onClick={() => {
                setSelectionMode('stride')
                setSelectionPosition(0)
              }}
            >
              Schrittweite
            </button>
            <button
              className={selectionMode === 'count' ? 'active' : ''}
              disabled={dirty}
              onClick={() => {
                setSelectionMode('count')
                setSelectionPosition(0)
              }}
            >
              Anzahl Frames
            </button>
            <button
              className={selectionMode === 'shuffle' ? 'active' : ''}
              disabled={dirty}
              onClick={() => {
                setSelectionMode('shuffle')
                if (!shuffleSequence.length)
                  setShuffleSequence(
                    buildShuffleSelection(
                      videos.map(video => ({video_id: video.video_id, total_frames: video.total_frames})),
                      shuffleTotal,
                    ),
                  )
                setSelectionPosition(0)
              }}
            >
              Shuffle
            </button>
          </div>
          {selectionMode === 'shuffle' ? (
            <div className="shuffle-mode-panel">
              <label>
                Anzahl gemischter Frames
                <input
                  aria-label="Anzahl gemischter Frames"
                  type="number"
                  min="1"
                  max="2000"
                  step="1"
                  value={shuffleTotal}
                  disabled={dirty}
                  onChange={event => {
                    const value = Number(event.target.value)
                    if (Number.isFinite(value) && value >= 1) setShuffleTotal(Math.min(2000, Math.round(value)))
                  }}
                />
              </label>
              <button disabled={dirty} onClick={regenerateShuffle}>
                🔀 Neu mischen
              </button>
              <small>
                Zieht Frames quer über alle {videos.length} Videos der Mission und ordnet sie zufällig — gut geeignet für unabhängige
                Review-Bewertung ohne zeitliche Nähe zwischen den Frames. Wird als eigener Arbeitsmodus mit jedem Label gespeichert.
              </small>
            </div>
          ) : selectionMode === 'stride' ? (
            <>
              <label>
                Jeden n-ten Frame
                <select
                  value={[1, 5, 10, 20, 50].includes(stride) ? stride : 'custom'}
                  disabled={dirty}
                  onChange={event => {
                    if (event.target.value !== 'custom') setStride(+event.target.value)
                    setSelectionPosition(0)
                  }}
                >
                  <option value="1">jeden Frame</option>
                  <option value="5">jeden 5. Frame</option>
                  <option value="10">jeden 10. Frame</option>
                  <option value="20">jeden 20. Frame</option>
                  <option value="50">jeden 50. Frame</option>
                  <option value="custom">Benutzerdefiniert …</option>
                </select>
              </label>
              {![1, 5, 10, 20, 50].includes(stride) && (
                <label>
                  Eigene Schrittweite
                  <input
                    aria-label="Eigene Schrittweite"
                    type="number"
                    min="1"
                    max={activeVideo.total_frames}
                    step="1"
                    value={stride}
                    disabled={dirty}
                    onChange={event => {
                      const value = Number(event.target.value)
                      if (Number.isFinite(value) && value >= 1) setStride(Math.min(activeVideo.total_frames, Math.round(value)))
                      setSelectionPosition(0)
                    }}
                  />
                  <small>Zum Beispiel 3, 7, 25 oder 100.</small>
                </label>
              )}
            </>
          ) : (
            <label>
              Anzahl zu labelnder Frames
              <input
                aria-label="Anzahl zu labelnder Frames"
                type="number"
                min="1"
                max={activeVideo.total_frames}
                step="1"
                value={targetCount}
                disabled={dirty}
                onChange={event => {
                  const value = Number(event.target.value)
                  if (Number.isFinite(value) && value >= 1) setTargetCount(Math.min(activeVideo.total_frames, Math.round(value)))
                  setSelectionPosition(0)
                }}
              />
              <small>Die Frames werden gleichmäßig über das gesamte Video verteilt.</small>
            </label>
          )}
          <div className="sampling-summary">
            <b>{selectedFrames.length.toLocaleString('de-DE')} Frames ausgewählt</b>
            <span>
              {summary.counts.confirmed} bestätigt · {summary.counts.draft} Entwürfe · {summary.counts.skipped} übersprungen
            </span>
            <span className={`label-mode-badge ${selectionMode === 'shuffle' ? 'shuffle' : 'linear'}`}>
              Arbeitsmodus: {selectionMode === 'shuffle' ? 'Shuffle' : 'Linear'}
            </span>
          </div>
          <div
            className="progress-mini-map"
            title={`${resolvedInSelection} von ${selectedFrames.length} Frames in dieser Auswahl bestätigt`}
          >
            <div className="progress-mini-map-track">
              <div className="progress-mini-map-fill" style={{width: `${Math.round(progressFraction * 100)}%`}} />
            </div>
            <small>
              {resolvedInSelection} / {selectedFrames.length} bestätigt ({Math.round(progressFraction * 100)} %)
            </small>
          </div>

          <hr />
          <h2>Polygon bearbeiten</h2>
          <label>
            Masken-Deckkraft · {Math.round(maskOpacity * 100)} %
            <input
              aria-label="Masken-Deckkraft"
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={maskOpacity}
              onChange={event => setMaskOpacity(+event.target.value)}
            />
          </label>
          <div className="ai-mask-controls">
            <label className="toggle-row">
              <input type="checkbox" checked={showAiMask} disabled={!pathModel} onChange={event => setShowAiMask(event.target.checked)} />
              <span>KI-Wegmaske anzeigen</span>
            </label>
            <label>
              KI-Masken-Deckkraft · {Math.round(aiMaskOpacity * 100)} %
              <input
                aria-label="KI-Masken-Deckkraft"
                type="range"
                min="0.05"
                max="0.9"
                step="0.05"
                value={aiMaskOpacity}
                disabled={!showAiMask || !pathModel}
                onChange={event => setAiMaskOpacity(+event.target.value)}
              />
            </label>
            {showAiMask && pathPrediction && (
              <div className="ai-overlay-tabs" role="group" aria-label="Darstellung der KI-Maske">
                <button
                  className={effectiveAiOverlay === 'grade' ? 'active' : ''}
                  disabled={refinementMode || !pathPrediction.grade_mask}
                  onClick={() => setAiOverlay('grade')}
                >
                  Abstufung
                </button>
                <button className={effectiveAiOverlay === 'comparison' ? 'active' : ''} onClick={() => setAiOverlay('comparison')}>
                  {pathPrediction.evaluation ? 'Vergleich' : 'Einfarbig'}
                </button>
              </div>
            )}
            <small>
              {predictionLoading
                ? 'KI-Maske wird für diesen Frame berechnet …'
                : !pathPrediction
                  ? pathModel
                    ? 'KI-Maske einschalten, um die Erkennung auf diesem Frame zu sehen.'
                    : 'Trainiere zuerst ein Wegmodell, um dessen Erkennung direkt einzublenden.'
                  : aiLayer.graded
                    ? `Abgestufte KI-Einschätzung · ${(pathPrediction.path_fraction * 100).toFixed(1)} % des Bildes als Weg erkannt`
                    : pathPrediction.evaluation
                      ? `Bewertung dieses Frames: ${pathPrediction.evaluation.metrics.symmetric_score.toFixed(1)} / 100 · Grün korrekt · Rot übersehen · Gelb fälschlich erkannt`
                      : `Türkis: KI-erkannt · ${(pathPrediction.path_fraction * 100).toFixed(1)} % des Bildes · kein Ground-Truth-Vergleich für diesen Frame`}
            </small>
            {showAiMask &&
              pathPrediction &&
              (aiLayer.graded ? (
                <GradeLegend ontology={pathPrediction.grade_ontology} mask={pathPrediction.grade_mask} grading={pathPrediction.grading} />
              ) : (
                pathPrediction.evaluation && (
                  <div className="grade-legend">
                    <b>Vergleich mit deinem Label</b>
                    <ul>
                      {COMPARISON_LEGEND.map(entry => (
                        <li key={entry.value}>
                          <i style={{background: entry.color, borderColor: entry.color}} aria-hidden="true" />
                          <span>{entry.label}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )
              ))}
            {refinementMode && <small className="refinement-hint">Für das Refinement wird die Vergleichsmaske angezeigt.</small>}
            {pathPrediction?.evaluation && (
              <div className="refinement-controls">
                <button
                  className={refinementMode ? 'active' : ''}
                  onClick={() => {
                    setRefinementMode(current => !current)
                    setRefinementSelection(null)
                  }}
                >
                  {refinementMode ? 'Refinement beenden' : 'Refinement starten'}
                </button>
                <span>{pathPrediction.evaluation.refinement_count} gespeicherte Korrekturen</span>
                {refinementMode && <small>Klicke eine rote oder gelbe zusammenhängende Fläche an.</small>}
                {refinementSelection && (
                  <button className="confirm-refinement" disabled={refinementSaving} onClick={() => void confirmRefinement()}>
                    {refinementSaving ? 'WIRD GESPEICHERT …' : 'KI HATTE HIER RECHT'}
                  </button>
                )}
              </div>
            )}
          </div>
          <div className="label-class-panel">
            <h2>Klasse der Fläche</h2>
            {ontologyError && <small className="terrain-hint">{ontologyError} — es stehen nur die vier Kernklassen zur Wahl.</small>}
            {(['core', 'obstacle', 'zone', 'roi'] as const).map(layer => {
              const classes = classesOf(layer)
              if (!classes.length) return null
              const titles: Record<string, string> = {
                core: 'Untergrund — genau eine je Fläche',
                obstacle: 'Hindernis — erklärt, warum etwas nicht fahrbar ist',
                zone: 'Problemzone — macht eine Fläche unsicher',
                roi: 'Auswertungsbereich — schneidet nichts weg',
              }
              return (
                <div key={layer} className="label-class-group">
                  <small>{titles[layer]}</small>
                  <div className="label-class-chips">
                    {classes.map(item => (
                      <button
                        key={item.class_id}
                        className={shapeClass === item.class_id ? 'active' : ''}
                        style={{
                          borderColor: item.color,
                          color: shapeClass === item.class_id ? '#11150e' : item.color,
                          background: shapeClass === item.class_id ? item.color : 'transparent',
                        }}
                        title={item.description}
                        onClick={() => setShapeClass(item.class_id)}
                      >
                        {item.label}
                      </button>
                    ))}
                  </div>
                </div>
              )
            })}
            <label>
              Sicherheit
              <select
                aria-label="Sicherheit der Fläche"
                value={shapeCertainty}
                onChange={event => setShapeCertainty(event.target.value as LabelShape['certainty'])}
              >
                <option value="certain">Sicher</option>
                <option value="uncertain">Unsicher</option>
                <option value="partially_occluded">Teilweise verdeckt</option>
              </select>
            </label>
            <label className="hard-negative-toggle">
              <input type="checkbox" checked={shapeHardNegative} onChange={event => setShapeHardNegative(event.target.checked)} />
              <span>Hartes Negativbeispiel</span>
            </label>
            <small>
              Für Flächen, die aussehen wie das Gegenteil dessen, was sie sind — Schatten, der wie ein Hindernis wirkt, Lichtfleck, der wie
              freie Fläche aussieht.
            </small>
            <label>
              Notiz zur Fläche
              <input value={shapeNote} maxLength={500} placeholder="Optional" onChange={event => setShapeNote(event.target.value)} />
            </label>
            <button className="commit-shape" disabled={!hasPolygon} onClick={commitShape}>
              {editingShapeId ? 'Änderung übernehmen' : 'Fläche übernehmen'}
            </button>
            {shapes.length > 0 && (
              <div className="committed-shape-list">
                <b>{shapes.length} übernommene Fläche(n)</b>
                {shapes.map(shape => (
                  <div key={shape.id}>
                    <i style={{background: describe(shape.class_id).color}} />
                    <span>{describe(shape.class_id).label}</span>
                    <small>
                      {shape.certainty === 'certain' ? '' : shape.certainty === 'uncertain' ? 'unsicher · ' : 'verdeckt · '}
                      {shape.hard_negative ? 'hartes Negativ · ' : ''}
                      {shape.points.length} Punkte
                    </small>
                    <button onClick={() => editShape(shape)}>Bearbeiten</button>
                    <button className="danger" onClick={() => removeShape(shape.id)}>
                      Entfernen
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="roi-profile-panel">
            <h2>Auswertungsbereich</h2>
            <small>
              Schneidet das Bild nicht zu — es sagt nur, welche Pixel im Training zählen. Das Original bleibt vollständig erhalten.
            </small>
            <div className="roi-band-controls">
              <label>
                Oben ignorieren · {Math.round(roiTop * 100)} %
                <input
                  aria-label="Oberes Band ignorieren"
                  type="range"
                  min="0"
                  max="0.6"
                  step="0.01"
                  value={roiTop}
                  onChange={event => setRoiTop(+event.target.value)}
                />
              </label>
              <label>
                Unten ignorieren · {Math.round(roiBottom * 100)} %
                <input
                  aria-label="Unteres Band ignorieren"
                  type="range"
                  min="0"
                  max="0.4"
                  step="0.01"
                  value={roiBottom}
                  onChange={event => setRoiBottom(+event.target.value)}
                />
              </label>
            </div>
            <div className="roi-actions">
              <button disabled={roiTop <= 0 && roiBottom <= 0} onClick={applyRoiBands}>
                Bänder auf diesen Frame legen
              </button>
              <button disabled={roiSaving} onClick={() => void persistRoiProfile()}>
                {roiSaving ? 'PROFIL WIRD GESPEICHERT …' : 'Als Videoprofil speichern'}
              </button>
            </div>
            <small>
              {roiProfile && roiProfile.revision > 0
                ? `Videoprofil gespeichert (Revision ${roiProfile.revision}): oben ${Math.round((roiProfile.top_ignore_fraction ?? 0) * 100)} %, unten ${Math.round((roiProfile.bottom_ignore_fraction ?? 0) * 100)} %. Es gilt als Vorschlag — was zählt, steht am Frame.`
                : 'Für dieses Video ist noch kein Profil gespeichert. Steht die Kamera fest, lohnt es sich: dann gilt derselbe Bereich in jedem Frame.'}
            </small>
          </div>

          <div className="manual-trajectory-panel">
            <h2>Trajektorie von Hand</h2>
            <label className="trajectory-toggle">
              <input type="checkbox" checked={trajectoryMode} onChange={event => setTrajectoryMode(event.target.checked)} />
              <span>Bahn zeichnen: Klick setzt einen Punkt, Rechtsklick entfernt ihn</span>
            </label>
            <small>Braucht kein trainiertes Modell. Im KI-Modellzentrum gibt es zusätzlich einen Vorschlag zum Nachbessern.</small>
            <div className="trajectory-actions">
              <button className="primary" disabled={trajectorySaving || trajectory.length < 2} onClick={() => void persistTrajectory()}>
                {trajectorySaving ? 'WIRD GESPEICHERT …' : storedTrajectory ? 'Trajektorie aktualisieren' : 'Trajektorie speichern'}
              </button>
              <button disabled={trajectorySaving || (!trajectory.length && !storedTrajectory)} onClick={() => void discardTrajectory()}>
                Trajektorie löschen
              </button>
            </div>
            <div className="trajectory-state">
              <b>{trajectory.length} Bahnpunkte</b>
              <span>
                {storedTrajectory
                  ? `Gespeichert: Revision ${storedTrajectory.revision} · ${storedTrajectory.origin === 'manual' ? 'von Hand' : storedTrajectory.origin === 'manual_edit' ? 'nachgebessert' : 'Modellvorschlag'}`
                  : 'Für diesen Frame ist nichts gespeichert.'}
              </span>
            </div>
          </div>

          <div className="polygon-tool-grid">
            <button className={tool === 'add' ? 'active' : ''} onClick={() => setTool('add')}>
              ＋ Punkt hinzufügen
            </button>
            <button className={tool === 'edit' ? 'active' : ''} disabled={!points.length} onClick={() => setTool('edit')}>
              Punkt auswählen
            </button>
            <button className={tool === 'move' ? 'active' : ''} disabled={!hasPolygon} onClick={() => setTool('move')}>
              Polygon verschieben
            </button>
            <button className={tool === 'pan' ? 'active' : ''} onClick={() => setTool('pan')}>
              ✋ Bild verschieben
            </button>
          </div>
          {tool === 'add' && (
            <small>
              {hasPolygon
                ? 'Klicke auf eine Polygonkante, um dort einen weiteren Punkt einzufügen.'
                : 'Klicke mindestens drei Punkte im befahrbaren Bereich an.'}
            </small>
          )}
          <div className="full-frame-toggle">
            <button
              className={fullFrameNotTraversable ? 'active' : ''}
              onClick={() => {
                setFullFrameNotTraversable(current => !current)
                setDirty(true)
                if (status === 'confirmed' || status === 'skipped') setStatus('draft')
              }}
            >
              Ganzes Bild nicht befahrbar
            </button>
            <small>Für Frames ohne befahrbaren Bereich. Das speichert eine Vollbild-Ground-Truth statt eines Polygons.</small>
          </div>
          <div className="polygon-edit-actions">
            <button disabled={selectedVertex === null} onClick={removeSelectedPoint}>
              Ausgewählten Punkt entfernen
            </button>
            <button disabled={!hasPolygon || tool !== 'add'} onClick={() => setTool('edit')}>
              Polygon schließen
            </button>
          </div>
          <div className="history-actions">
            <button disabled={!past.length} onClick={undo}>
              ↶ Undo
            </button>
            <button disabled={!future.length} onClick={redo}>
              ↷ Redo
            </button>
            <button
              className="danger"
              disabled={!points.length}
              onClick={() => {
                updatePoints([])
                setTool('add')
                setSelectedVertex(null)
              }}
            >
              Aktuelles Polygon löschen
            </button>
          </div>
          <small className="automatic-carry-note">
            Beim nächsten Frame bleibt das aktuelle Polygon automatisch als editierbare Vorlage sichtbar.
          </small>

          <hr />
          <div className={`label-record-state ${status} ${dirty ? 'dirty' : ''}`}>
            <b>
              {dirty
                ? 'Ungespeicherte Änderung'
                : status === 'confirmed'
                  ? `Bestätigt · Revision ${revision}`
                  : status === 'draft'
                    ? `Entwurf · Revision ${revision}`
                    : status === 'skipped'
                      ? 'Als nicht relevant übersprungen'
                      : 'Noch nicht markiert'}
            </b>
            <span>{fullFrameNotTraversable ? 'Ganzes Bild als nicht befahrbar' : `${points.length} Polygonpunkte`}</span>
            <span className={`label-mode-badge ${selectionMode === 'shuffle' ? 'shuffle' : 'linear'}`}>
              {selectionMode === 'shuffle' ? 'Shuffle' : 'Linear'}
            </span>
          </div>

          <div className="critical-flag-panel">
            <h3>Kritisch-Markierung</h3>
            <small>Für Frames mit Fehleinschätzungsrisiko — direkt sichtbar im Workflow statt nur im Modellzentrum.</small>
            <label>
              Schweregrad · {flagSeverity}/5
              <input
                aria-label="Schweregrad der Kritisch-Meldung"
                type="range"
                min="1"
                max="5"
                step="1"
                value={flagSeverity}
                onChange={event => setFlagSeverity(+event.target.value)}
              />
            </label>
            <input value={flagNote} maxLength={500} placeholder="Grund (optional)" onChange={event => setFlagNote(event.target.value)} />
            <div className="critical-flag-actions">
              <button disabled={flagSaving} onClick={() => void reportCriticalFlag()}>
                {activeCriticalFlag ? 'Meldung aktualisieren' : 'Als kritisch melden'}
              </button>
              {activeCriticalFlag && (
                <button className="danger" disabled={flagSaving} onClick={() => void removeCriticalFlag()}>
                  Entfernen
                </button>
              )}
            </div>
            {criticalFlags.length > 0 && (
              <small>
                {criticalFlags.length} kritisch gemeldete{criticalFlags.length === 1 ? 'r Frame' : ' Frames'} in diesem Video.
              </small>
            )}
          </div>

          <details className="keyboard-shortcut-legend">
            <summary>Tastaturkürzel</summary>
            <ul>
              <li>
                <kbd>←</kbd>/<kbd>→</kbd> Frame vor/zurück
              </li>
              <li>
                <kbd>U</kbd> / <kbd>Shift</kbd>+<kbd>U</kbd> nächster/vorheriger ungeklärter Frame
              </li>
              <li>Pfeiltasten bei ausgewähltem Punkt (Werkzeug „Punkt auswählen“) — feines Nachjustieren, mit Shift größere Schritte</li>
              <li>
                <kbd>S</kbd> Entwurf speichern
              </li>
              <li>
                <kbd>Enter</kbd> bestätigen &amp; nächster Frame
              </li>
              <li>
                <kbd>F</kbd> Ganzes Bild nicht befahrbar umschalten
              </li>
              <li>
                <kbd>T</kbd> Trajektorie-Modus umschalten
              </li>
              <li>
                <kbd>1</kbd>–<kbd>4</kbd> Kernklasse wählen
              </li>
              <li>
                <kbd>Entf</kbd> ausgewählten Punkt entfernen
              </li>
              <li>
                <kbd>Strg</kbd>+<kbd>Z</kbd> / <kbd>Strg</kbd>+<kbd>Shift</kbd>+<kbd>Z</kbd> Undo/Redo
              </li>
            </ul>
          </details>

          <label>
            Bearbeiter
            <input
              value={annotator}
              maxLength={80}
              onChange={event => {
                setAnnotator(event.target.value)
                setDirty(true)
              }}
            />
          </label>
          <label>
            Notiz
            <textarea
              value={notes}
              maxLength={1000}
              onChange={event => {
                setNotes(event.target.value)
                setDirty(true)
              }}
              placeholder="Optionaler Hinweis zu diesem Frame"
            />
          </label>
          {message && (
            <div className="labeling-message" role="status">
              {message}
            </div>
          )}
          <div className="label-save-actions">
            <button disabled={saving} onClick={() => void persist('draft')}>
              Entwurf speichern
            </button>
            <button disabled={saving || (!hasPolygon && !fullFrameNotTraversable)} onClick={() => void persist('confirmed')}>
              {fullFrameNotTraversable ? 'Vollbild bestätigen' : 'Polygon bestätigen'}
            </button>
            <button
              className="primary"
              disabled={saving || (!hasPolygon && !fullFrameNotTraversable) || selectionPosition >= selectedFrames.length - 1}
              onClick={() => void saveAndNext()}
            >
              {fullFrameNotTraversable ? 'Vollbild bestätigen &amp; nächster Frame' : 'Bestätigen &amp; nächster Frame'}
            </button>
          </div>
        </aside>
      </div>

      <section className="saved-mask-library">
        <div className="saved-mask-library-head">
          <div>
            <span className="eyebrow">PERSISTENTE GROUND TRUTH</span>
            <h2>Gespeicherte Polygonmasken</h2>
            <p>Öffne einen Eintrag, um seine Maske wieder über dem Originalvideo anzuzeigen und weiterzubearbeiten.</p>
          </div>
          <b>{summary.counts.total} Einträge</b>
        </div>
        <div className="history-filter-bar">
          <input
            aria-label="Historie durchsuchen"
            value={historySearch}
            placeholder="Suche nach Video, Bearbeiter oder Framenummer …"
            onChange={event => setHistorySearch(event.target.value)}
          />
          <select
            aria-label="Status filtern"
            value={historyStatusFilter}
            onChange={event => setHistoryStatusFilter(event.target.value as GroundTruthStatus | 'all')}
          >
            <option value="all">Alle Status</option>
            <option value="confirmed">Bestätigt</option>
            <option value="draft">Entwurf</option>
            <option value="skipped">Übersprungen</option>
          </select>
          <small>
            {filteredHistoryItems.length} von {summary.items.length} Einträgen
          </small>
        </div>
        {summary.items.length ? (
          filteredHistoryItems.length ? (
            <div className="saved-mask-list">
              {filteredHistoryItems.map(item => {
                const video = videos.find(candidate => candidate.video_id === item.video_id)
                return (
                  <button
                    key={`${item.video_id}-${item.frame_index}`}
                    className={item.video_id === activeVideo.video_id && item.frame_index === frameIndex ? 'active' : ''}
                    onClick={() => openSavedMask(item)}
                  >
                    <span className={`mask-status ${item.status}`}>
                      {item.status === 'confirmed' ? 'BESTÄTIGT' : item.status === 'skipped' ? 'ÜBERSPRUNGEN' : 'ENTWURF'}
                    </span>
                    <b>Frame {item.frame_index + 1}</b>
                    <span>{video?.original_name ?? item.video_id}</span>
                    <small>
                      {item.statistics.polygon_count ?? 0} Polygon · {item.statistics.point_count ?? 0} Punkte · Revision {item.revision}
                    </small>
                    <span className={`label-mode-badge ${item.label_mode === 'shuffle' ? 'shuffle' : 'linear'}`}>
                      {item.label_mode === 'shuffle' ? 'Shuffle' : 'Linear'}
                    </span>
                  </button>
                )
              })}
            </div>
          ) : (
            <div className="empty saved-mask-empty">Keine Einträge für diese Suche/diesen Filter.</div>
          )
        ) : (
          <div className="empty saved-mask-empty">
            Noch keine Polygonmasken gespeichert. Sobald du einen Entwurf speicherst oder ein Polygon bestätigst, erscheint es hier.
          </div>
        )}
      </section>

      <section className="path-model-panel">
        <div className="path-model-head">
          <div>
            <span className="eyebrow">LOKALES CPU-TRAINING</span>
            <h2>Wegerkennung aus deinen Labels</h2>
            <p>
              Innerhalb deiner Polygone lernt das Modell „Weg“, außerhalb „kein Weg“. Übersehene und fälschlich erfundene Wegfläche kosten
              gleich viele Punkte.
            </p>
          </div>
          <button disabled={dirty || modelTraining || summary.counts.confirmed < 10} onClick={() => void trainCpuPathModel()}>
            {modelTraining ? 'CPU-TRAINING LÄUFT …' : `WEG-KI AUF ${summary.counts.confirmed} LABELFRAMES TRAINIEREN`}
          </button>
        </div>
        <div className="background-training-panel">
          <div>
            <b>Training im Hintergrund</b>
            <span>Der Browser darf geschlossen werden. Der lokale Server und dieser Rechner müssen eingeschaltet bleiben.</span>
          </div>
          <label>
            Profil
            <select
              aria-label="Trainingsprofil"
              value={trainingProfile}
              disabled={trainingJob?.status === 'running' || trainingJob?.status === 'queued'}
              onChange={event => setTrainingProfile(event.target.value as 'quick' | 'overnight')}
            >
              <option value="quick">Schnelltest · 1 Kandidat</option>
              <option value="overnight">Über Nacht · Varianten vergleichen</option>
            </select>
          </label>
          {trainingProfile === 'overnight' && (
            <label>
              Dauer
              <select
                aria-label="Trainingsdauer"
                value={trainingHours}
                disabled={trainingJob?.status === 'running' || trainingJob?.status === 'queued'}
                onChange={event => setTrainingHours(+event.target.value)}
              >
                <option value="1">1 Stunde</option>
                <option value="4">4 Stunden</option>
                <option value="8">8 Stunden</option>
                <option value="12">12 Stunden</option>
              </select>
            </label>
          )}
          <button
            disabled={dirty || summary.counts.confirmed < 10 || trainingJob?.status === 'running' || trainingJob?.status === 'queued'}
            onClick={() => void startBackgroundTraining()}
          >
            {trainingProfile === 'overnight' ? 'NACHTTRAINING STARTEN' : 'SCHNELLTEST IM HINTERGRUND'}
          </button>
        </div>
        {trainingJob && (
          <div className={`training-job-status ${trainingJob.status}`} role="status">
            <b>
              {trainingJob.status === 'completed'
                ? 'Training abgeschlossen'
                : trainingJob.status === 'running' || trainingJob.status === 'queued'
                  ? 'Training läuft'
                  : 'Training unterbrochen'}
            </b>
            <span>{trainingJob.message}</span>
            <small>
              {trainingJob.candidates_completed} von {trainingJob.maximum_candidates} Kandidaten geprüft
              {trainingJob.best_validation_score !== null ? ` · bester Score ${trainingJob.best_validation_score.toFixed(2)} / 100` : ''}
            </small>
          </div>
        )}
        {pathModel ? (
          <>
            <div className="active-model-version">
              <div>
                <span>Aktives Modell</span>
                <b>{pathModel.run_id}</b>
                <small>Erstellt: {new Date(pathModel.created_at).toLocaleString('de-DE')}</small>
              </div>
              {trainingJob?.status === 'completed' && trainingJob.initial_run_id && (
                <strong className={trainingJob.best_run_id !== trainingJob.initial_run_id ? 'changed' : 'unchanged'}>
                  {trainingJob.best_run_id !== trainingJob.initial_run_id ? '✓ MODELL AKTUALISIERT' : 'BISHERIGES MODELL BLEIBT BESSER'}
                </strong>
              )}
            </div>
            <div className="path-model-metrics">
              <div>
                <span>Validierungsscore</span>
                <b>{pathModel.validation_metrics.symmetric_score.toFixed(2)} / 100</b>
              </div>
              <div>
                <span>Punktabzug</span>
                <b>{pathModel.validation_metrics.symmetric_penalty_points.toFixed(2)}</b>
              </div>
              <div>
                <span>Weg übersehen</span>
                <b>{Math.round(pathModel.validation_metrics.missed_label_fraction * 1000) / 10} %</b>
              </div>
              <div>
                <span>Weg erfunden</span>
                <b>{Math.round(pathModel.validation_metrics.invented_path_fraction * 1000) / 10} %</b>
              </div>
              <div>
                <span>IoU</span>
                <b>{Math.round(pathModel.validation_metrics.iou * 1000) / 1000}</b>
              </div>
              <div>
                <span>CPU-Laufzeit</span>
                <b>{pathModel.runtime_seconds.toFixed(1)} s</b>
              </div>
            </div>
            <div className="path-model-split">
              <b>
                {pathModel.ground_truth.confirmed_frames} Ground-Truth-Frames aus {pathModel.ground_truth.videos} Videos
              </b>
              <span>
                {pathModel.split.train_frames} Training · {pathModel.split.validation_frames} getrennte Validierung ·{' '}
                {pathModel.split.training_pixels_sampled.toLocaleString('de-DE')} Trainingspixel
              </span>
              <small>Kein Frame befindet sich gleichzeitig in Training und Validierung. Keine Cloud und keine GPU verwendet.</small>
            </div>
            <div className="path-model-evidence">
              {pathModel.evidence.map(item => (
                <figure key={`${item.video_id}-${item.frame_index}`}>
                  <img src={item.image_url} alt={`${item.kind} Validierungsframe ${item.frame_index + 1}`} />
                  <figcaption>
                    <b>{item.kind === 'best' ? 'Bester' : item.kind === 'worst' ? 'Schwierigster' : 'Mittlerer'} Validierungsframe</b>
                    <span>
                      Frame {item.frame_index + 1} · Score {item.metrics.symmetric_score.toFixed(1)}
                    </span>
                    <small>Grün korrekt · Rot übersehen · Gelb fälschlich erkannt</small>
                  </figcaption>
                </figure>
              ))}
            </div>
          </>
        ) : (
          <div className="empty path-model-empty">Noch kein CPU-Wegmodell trainiert. Deine Polygonlabels bleiben dabei unverändert.</div>
        )}
      </section>

      <section className="labeling-finish">
        <div>
          <b>Markierungsrunde abschließen</b>
          <span>
            Erst diese explizite Aktion startet die vollständige Segmentierung, Befahrbarkeit und Trajektorienberechnung. Die optionale
            türkisfarbene Wegmaske ist nur eine Modellvorschau und verändert deine Ground-Truth-Polygone nicht.
          </span>
        </div>
        <button disabled={processing || summary.counts.confirmed === 0 || dirty} onClick={() => void startProcessing()}>
          {processing ? 'AUSWERTUNG LÄUFT …' : 'MARKIERUNG ABSCHLIESSEN & AUSWERTUNG STARTEN'}
        </button>
      </section>
    </div>
  )
}
