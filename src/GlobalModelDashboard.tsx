import {useEffect, useMemo, useRef, useState} from 'react'
import type {PointerEvent as ReactPointerEvent} from 'react'
import {
  createOffPathInterval,
  deleteCriticalFlag,
  deleteOffPathInterval,
  getGlobalModelDashboard,
  getGlobalVideoAnalysisResult,
  getGlobalVideoAnalysisStatus,
  getLabelingVideos,
  listCriticalFlags,
  listOffPathIntervals,
  predictGlobalPathFrame,
  saveCriticalFlag,
  startGlobalVideoAnalysis,
  trainGlobalPathModel,
} from './api'
import GradeLegend from './GradeLegend'
import {AI_BINARY_PALETTE, COMPARISON_PALETTE, decodeRleValues, encodeRleValues, paintMaskCanvas, paletteFromGradeOntology} from './masks'
import CorridorOverlay from './CorridorOverlay'
import CorridorReadout from './CorridorReadout'
import {useCorridorPlanner} from './corridorPlanner'
import TerrainModelPanel from './TerrainModelPanel'
import {terrainCategoryLabel} from './terrainCategories'
import type {
  GlobalModelDashboardData,
  GlobalVideoAnalysisResult,
  GlobalVideoAnalysisStatus,
  GradeOntology,
  Grading,
  LabelingVideo,
  OffPathInterval,
  PathPrediction,
  TerrainMask,
} from './types'

const duration = (seconds: number | null) =>
  seconds === null
    ? 'wird berechnet …'
    : seconds < 60
      ? `${Math.ceil(seconds)} s`
      : `${Math.floor(seconds / 60)} min ${Math.ceil(seconds % 60)} s`

type ReviewKind = 'incorrect' | 'major'
type ReviewSnapshot = {
  id: string
  videoId: string
  frameIndex: number
  timestampMs: number
  kind: ReviewKind
  saved: boolean
}

const reviewKindLabel: Record<ReviewKind, string> = {
  incorrect: 'Nicht korrekt markiert',
  major: 'Erheblicher Fehler',
}
const playbackSpeeds = [0.25, 0.5, 1, 2, 4]
const playbackSpeedLabel = (value: number) => `${String(value).replace('.', ',')}×`

export default function GlobalModelDashboard({onClose, onOpenRefinement = () => undefined}: {onClose: () => void; onOpenRefinement?: () => void}) {
  const [data, setData] = useState<GlobalModelDashboardData | null>(null)
  const [training, setTraining] = useState(false)
  const [message, setMessage] = useState('')
  const [selectedMissionId, setSelectedMissionId] = useState('')
  const [videos, setVideos] = useState<LabelingVideo[]>([])
  const [selectedVideoId, setSelectedVideoId] = useState('')
  const [frameIndex, setFrameIndex] = useState(0)
  const [prediction, setPrediction] = useState<PathPrediction | null>(null)
  const [predictionLoading, setPredictionLoading] = useState(false)
  const [maskOpacity, setMaskOpacity] = useState(0.48)
  const [showAiMask, setShowAiMask] = useState(true)
  const [showLabelMask, setShowLabelMask] = useState(true)
  const [showGrades, setShowGrades] = useState(true)
  const [showFeedbackBrush, setShowFeedbackBrush] = useState(false)
  const [flagSeverity, setFlagSeverity] = useState(4)
  const [flagNote, setFlagNote] = useState('')
  const [flagBrushSize, setFlagBrushSize] = useState(20)
  const [flagMask, setFlagMask] = useState<number[] | null>(null)
  const [criticalFlags, setCriticalFlags] = useState<
    {
      video_id: string
      frame_index: number
      timestamp_ms: number
      severity: number
      note: string
      annotator: string
      created_at: string
      brush_mask?: TerrainMask
    }[]
  >([])
  const [flagSaving, setFlagSaving] = useState(false)
  const [reviewSnapshots, setReviewSnapshots] = useState<ReviewSnapshot[]>([])
  const [reviewSaving, setReviewSaving] = useState(false)
  const [errorIntervalStartMs, setErrorIntervalStartMs] = useState<number | null>(null)
  const [errorIntervalSaving, setErrorIntervalSaving] = useState(false)
  const [savedVideoRanges, setSavedVideoRanges] = useState<OffPathInterval[]>([])
  const [playing, setPlaying] = useState(false)
  const [reversePlaying, setReversePlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [analysisStatus, setAnalysisStatus] = useState<GlobalVideoAnalysisStatus | null>(null)
  const [analysisResult, setAnalysisResult] = useState<GlobalVideoAnalysisResult | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const brushCanvasRef = useRef<HTMLCanvasElement>(null)
  const paintCursorRef = useRef<{x: number; y: number} | null>(null)

  useEffect(() => {
    void getGlobalModelDashboard()
      .then(setData)
      .catch(error => setMessage(error instanceof Error ? error.message : 'Modellzentrum konnte nicht geladen werden'))
  }, [])

  useEffect(() => {
    const first = data?.dataset.missions.find(item => item.confirmed_frames > 0)
    if (first && !selectedMissionId) setSelectedMissionId(first.mission_id)
  }, [data, selectedMissionId])

  useEffect(() => {
    if (!selectedMissionId) return
    setVideos([])
    setSelectedVideoId('')
    setFrameIndex(0)
    setPlaying(false)
    void getLabelingVideos(selectedMissionId)
      .then(result => {
        setVideos(result.videos)
        setSelectedVideoId(result.videos[0]?.video_id ?? '')
      })
      .catch(error => setMessage(error instanceof Error ? error.message : 'Videos konnten nicht geladen werden'))
  }, [selectedMissionId])

  const activeVideo = videos.find(item => item.video_id === selectedVideoId) ?? videos[0]
  const playbackActive = playing || reversePlaying
  const [planning, setPlanning] = useState(false)
  const [trajectoryNote, setTrajectoryNote] = useState('')
  // Die Korridore kommen aus derselben Antwort wie die Maske — ein zweiter
  // Aufruf haette dieselbe Inferenz noch einmal ueber den Frame laufen lassen.
  const planner = useCorridorPlanner(
    selectedMissionId,
    activeVideo?.video_id ?? '',
    frameIndex,
    activeVideo ? Math.round((frameIndex / activeVideo.fps) * 1000) : 0,
    prediction?.corridors ?? null,
  )

  useEffect(() => {
    if (!selectedMissionId || !activeVideo) {
      setCriticalFlags([])
      return
    }
    let cancelled = false
    void listCriticalFlags(selectedMissionId, activeVideo.video_id)
      .then(result => {
        if (!cancelled) setCriticalFlags(result.items)
      })
      .catch(error => {
        if (!cancelled) setMessage(error instanceof Error ? error.message : 'Meldungen konnten nicht geladen werden')
      })
    return () => {
      cancelled = true
    }
  }, [selectedMissionId, activeVideo?.video_id])

  useEffect(() => {
    if (!selectedMissionId || !activeVideo) {
      setSavedVideoRanges([])
      return
    }
    let cancelled = false
    void listOffPathIntervals(selectedMissionId, activeVideo.video_id)
      .then(items => {
        if (!cancelled) setSavedVideoRanges(items)
      })
      .catch(error => {
        if (!cancelled) setMessage(error instanceof Error ? error.message : 'Videobereiche konnten nicht geladen werden')
      })
    return () => {
      cancelled = true
    }
  }, [selectedMissionId, activeVideo?.video_id])

  useEffect(() => {
    if (!data?.model || !selectedMissionId || !activeVideo) return
    let cancelled = false
    setAnalysisStatus(null)
    setAnalysisResult(null)
    setPrediction(null)
    setPlaying(false)
    void getGlobalVideoAnalysisStatus(selectedMissionId, activeVideo.video_id)
      .then(status => {
        if (cancelled) return
        setAnalysisStatus(status)
        if (status?.status === 'completed')
          void getGlobalVideoAnalysisResult(selectedMissionId, activeVideo.video_id).then(result => {
            if (!cancelled) setAnalysisResult(result)
          })
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [data?.model?.run_id, selectedMissionId, activeVideo?.video_id])

  useEffect(() => {
    if (!analysisStatus || !['queued', 'running'].includes(analysisStatus.status) || !activeVideo) return
    const timer = window.setInterval(
      () =>
        void getGlobalVideoAnalysisStatus(selectedMissionId, activeVideo.video_id)
          .then(status => {
            setAnalysisStatus(status)
            if (status?.status === 'completed')
              void getGlobalVideoAnalysisResult(selectedMissionId, activeVideo.video_id).then(setAnalysisResult)
          })
          .catch(() => undefined),
      1500,
    )
    return () => window.clearInterval(timer)
  }, [analysisStatus?.status, selectedMissionId, activeVideo?.video_id])

  // Einzelframe-Vorschau wird auch bei fertiger Videoanalyse geholt, solange
  // deren gespeicherte Frames noch keine Abstufung enthalten (Analysen vor
  // Phase 3).
  const needsGradePreview = showGrades && !analysisResult?.frames[frameIndex]?.grade_mask
  useEffect(() => {
    if (!data?.model || !selectedMissionId || !activeVideo) {
      setPrediction(null)
      return
    }
    // Korridore und Trajektorienvorschlag kommen ausschliesslich aus dieser
    // Live-Antwort — die Batch-Analyse speichert sie nicht je
    // Frame. Deshalb bleibt der Aufruf bei jedem Frame noetig, sobald nicht
    // gerade abgespielt wird, unabhaengig davon, ob die Abstufung bereits aus
    // der Batch-Analyse vorliegt. Nur waehrend der Wiedergabe wird verzichtet,
    // um die API nicht pro Frame zu treffen.
    if (playbackActive) {
      setPrediction(null)
      return
    }
    let cancelled = false
    setPredictionLoading(true)
    void predictGlobalPathFrame(selectedMissionId, activeVideo.video_id, frameIndex, planner.calibration)
      .then(result => {
        if (!cancelled) setPrediction(result)
      })
      .catch(error => {
        if (!cancelled) setMessage(error instanceof Error ? error.message : 'Frameanalyse fehlgeschlagen')
      })
      .finally(() => {
        if (!cancelled) setPredictionLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [
    data?.model?.run_id,
    selectedMissionId,
    activeVideo?.video_id,
    frameIndex,
    playbackActive,
    // Aendert sich die Kalibrierung, muessen die Korridore neu berechnet werden;
    // sie haengen an derselben Antwort wie die Maske.
    planner.calibration,
  ])

  useEffect(() => {
    if (!videoRef.current) return
    videoRef.current.playbackRate = speed
  }, [speed, activeVideo?.video_id])

  const setPlaybackSpeed = (nextSpeed: number) => {
    setSpeed(nextSpeed)
    // Die Analyseframes sind bereits vorbereitet und folgen der Videouhr.
    // Direkt setzen vermeidet einen sichtbaren Taktwechsel beim Umschalten.
    if (videoRef.current) videoRef.current.playbackRate = nextSpeed
  }

  useEffect(() => {
    if (!playing || !activeVideo) return
    let animation = 0
    const synchronize = () => {
      const video = videoRef.current
      if (video) setFrameIndex(Math.min(activeVideo.total_frames - 1, Math.floor(video.currentTime * activeVideo.fps)))
      animation = window.requestAnimationFrame(synchronize)
    }
    animation = window.requestAnimationFrame(synchronize)
    return () => window.cancelAnimationFrame(animation)
  }, [playing, activeVideo?.video_id, activeVideo?.fps, activeVideo?.total_frames])

  useEffect(() => {
    if (!reversePlaying || !activeVideo) return
    let animation = 0
    let previous = performance.now()
    let carry = 0
    const rewind = (now: number) => {
      carry += ((now - previous) * speed * activeVideo.fps) / 1000
      previous = now
      const step = Math.floor(carry)
      if (step > 0) {
        carry -= step
        setFrameIndex(current => {
          const next = Math.max(0, current - step)
          if (videoRef.current) videoRef.current.currentTime = next / activeVideo.fps
          if (next === 0) setReversePlaying(false)
          return next
        })
      }
      animation = window.requestAnimationFrame(rewind)
    }
    animation = window.requestAnimationFrame(rewind)
    return () => window.cancelAnimationFrame(animation)
  }, [reversePlaying, speed, activeVideo?.video_id, activeVideo?.fps])

  useEffect(() => {
    if (!activeVideo) {
      setFlagMask(null)
      setFlagNote('')
      setFlagSeverity(4)
      return
    }
    const existing = criticalFlags.find(item => item.video_id === activeVideo.video_id && item.frame_index === frameIndex)
    setFlagSeverity(existing?.severity ?? 4)
    setFlagNote(existing?.note ?? '')
    if (existing?.brush_mask) setFlagMask(decodeRleValues(existing.brush_mask))
    else setFlagMask(new Array(activeVideo.width * activeVideo.height).fill(0))
    paintCursorRef.current = null
  }, [activeVideo?.video_id, frameIndex, criticalFlags])

  const analyzedFrame = analysisResult?.frames[frameIndex] ?? null
  const gradePreview = showGrades && !analyzedFrame?.grade_mask ? prediction : null
  const displayed: {
    mask: TerrainMask
    grade_mask?: TerrainMask
    grade_ontology?: GradeOntology
    grading?: Grading
    evaluation?: {metrics: {symmetric_score: number}; comparison_mask: TerrainMask}
    path_fraction: number
    model_run_id: string
  } | null = gradePreview?.grade_mask
    ? gradePreview
    : analyzedFrame
      ? {
          mask: analyzedFrame.mask,
          grade_mask: analyzedFrame.grade_mask,
          // Threshold-abhaengig, aber gleich fuer den ganzen Lauf — kommt vom
          // Analyseergebnis, nicht vom Einzelframe (siehe GlobalVideoAnalysisResult).
          grade_ontology: analysisResult!.grade_ontology,
          grading: analysisResult!.grading,
          evaluation: analyzedFrame.evaluation,
          path_fraction: analyzedFrame.path_fraction,
          model_run_id: analysisResult!.model_run_id,
        }
      : prediction

  // Abstufung, wenn die Antwort sie mitliefert; sonst die bisherige
  // Vergleichs- beziehungsweise Einfarbdarstellung.
  const layer = useMemo(() => {
    if (!displayed) return {mask: null as TerrainMask | null, palette: AI_BINARY_PALETTE, graded: false}
    if (showGrades && displayed.grade_mask) {
      return {mask: displayed.grade_mask, palette: paletteFromGradeOntology(displayed.grade_ontology), graded: true}
    }
    if (displayed.evaluation && showAiMask && showLabelMask) {
      return {mask: displayed.evaluation.comparison_mask, palette: COMPARISON_PALETTE, graded: false}
    }
    if (displayed.evaluation && showLabelMask && !showAiMask) {
      return {
        mask: displayed.evaluation.comparison_mask,
        palette: {0: [0, 0, 0, 0], 1: [58, 214, 92, 255], 2: [58, 214, 92, 255], 3: [0, 0, 0, 0]} as typeof COMPARISON_PALETTE,
        graded: false,
      }
    }
    if (displayed.evaluation && showAiMask) {
      return {
        mask: displayed.evaluation.comparison_mask,
        palette: {0: [0, 0, 0, 0], 1: [64, 220, 235, 255], 2: [0, 0, 0, 0], 3: [64, 220, 235, 255]} as typeof COMPARISON_PALETTE,
        graded: false,
      }
    }
    return showAiMask
      ? {mask: displayed.mask, palette: AI_BINARY_PALETTE, graded: false}
      : {mask: null as TerrainMask | null, palette: AI_BINARY_PALETTE, graded: false}
  }, [displayed, showAiMask, showLabelMask, showGrades])

  useEffect(() => {
    const canvas = canvasRef.current
    if (canvas) paintMaskCanvas(canvas, layer.mask, layer.palette)
  }, [layer])

  const flagMaskRle = useMemo(() => {
    if (!activeVideo || !flagMask) return null
    return encodeRleValues(flagMask, activeVideo.width, activeVideo.height)
  }, [activeVideo?.width, activeVideo?.height, flagMask])

  useEffect(() => {
    const canvas = brushCanvasRef.current
    if (!canvas || !flagMaskRle) {
      if (canvas) {
        const context = canvas.getContext('2d')
        if (context) context.clearRect(0, 0, canvas.width, canvas.height)
      }
      return
    }
    paintMaskCanvas(canvas, flagMaskRle, {0: [0, 0, 0, 0], 1: [234, 76, 64, 130]})
  }, [flagMaskRle])

  const activeCriticalFlag = criticalFlags.find(item => item.video_id === activeVideo?.video_id && item.frame_index === frameIndex) ?? null
  const currentReviewSnapshots = useMemo(
    () => reviewSnapshots.filter(snapshot => snapshot.videoId === activeVideo?.video_id),
    [reviewSnapshots, activeVideo?.video_id],
  )
  const reviewFrameRange = useMemo(() => {
    if (!currentReviewSnapshots.length) return null
    const frames = currentReviewSnapshots.map(snapshot => snapshot.frameIndex)
    return {first: Math.min(...frames), last: Math.max(...frames)}
  }, [currentReviewSnapshots])
  const pausedFrameReference = useMemo(() => {
    if (!activeVideo) return null
    const timestampMs = Math.round((frameIndex / activeVideo.fps) * 1000)
    const review = currentReviewSnapshots.find(snapshot => snapshot.frameIndex === frameIndex)
    return {timestampMs, review}
  }, [activeVideo?.video_id, activeVideo?.fps, frameIndex, currentReviewSnapshots])

  const saveCurrentReviewFrame = async () => {
    if (!selectedMissionId || !activeVideo || !analysisResult || showFeedbackBrush || reviewSaving) return
    const currentFrame = Math.max(
      0,
      Math.min(activeVideo.total_frames - 1, Math.floor((videoRef.current?.currentTime ?? frameIndex / activeVideo.fps) * activeVideo.fps)),
    )
    const existingReview = currentReviewSnapshots.find(snapshot => snapshot.frameIndex === currentFrame)
    const kind = existingReview?.kind ?? 'incorrect'
    videoRef.current?.pause()
    setPlaying(false)
    setFrameIndex(currentFrame)
    setReviewSnapshots(current => {
      if (current.some(snapshot => snapshot.videoId === activeVideo.video_id && snapshot.frameIndex === currentFrame)) return current
      return [
        ...current,
        {
          id: `${activeVideo.video_id}-${currentFrame}`,
          videoId: activeVideo.video_id,
          frameIndex: currentFrame,
          timestampMs: Math.round((currentFrame / activeVideo.fps) * 1000),
          kind,
          saved: false,
        },
      ]
    })
    setReviewSaving(true)
    try {
      const existing = criticalFlags.find(flag => flag.video_id === activeVideo.video_id && flag.frame_index === currentFrame)
      const quickReviewNote = `Schnellreview: ${reviewKindLabel[kind]}.`
      const note = [existing?.note, quickReviewNote].filter((value, index, values) => value && values.indexOf(value) === index).join(' ')
      await saveCriticalFlag(selectedMissionId, activeVideo.video_id, currentFrame, {
        severity: Math.max(existing?.severity ?? 0, kind === 'major' ? 5 : 3),
        brush_mask: existing?.brush_mask,
        note,
        annotator: 'human',
      })
      const refreshed = await listCriticalFlags(selectedMissionId, activeVideo.video_id)
      setCriticalFlags(refreshed.items)
      setReviewSnapshots(current =>
        current.map(snapshot =>
          snapshot.videoId === activeVideo.video_id && snapshot.frameIndex === currentFrame ? {...snapshot, saved: true} : snapshot,
        ),
      )
      setMessage(`Falscher Frame ${currentFrame + 1} direkt als Trainingsfeedback gespeichert.`)
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Falscher Frame konnte nicht gespeichert werden')
    } finally {
      setReviewSaving(false)
    }
  }

  const videoTimestampMs = () =>
    activeVideo ? Math.round((videoRef.current?.currentTime ?? frameIndex / activeVideo.fps) * 1000) : 0

  const toggleVideoErrorRecording = async () => {
    if (!activeVideo || !analysisResult || showFeedbackBrush || errorIntervalSaving) return
    const timestampMs = videoTimestampMs()
    if (errorIntervalStartMs === null) {
      setErrorIntervalStartMs(timestampMs)
      if (videoRef.current?.paused) {
        setReversePlaying(false)
        try {
          await videoRef.current.play()
        } catch (error) {
          setErrorIntervalStartMs(null)
          setMessage(error instanceof Error ? error.message : 'Video konnte nicht gestartet werden')
          return
        }
      }
      setMessage(`Fehlerintervall gestartet bei ${(timestampMs / 1000).toFixed(1)} s. Mit „Video beenden“ speichern.`)
      return
    }
    const startMs = Math.min(errorIntervalStartMs, timestampMs)
    const endMs = Math.max(errorIntervalStartMs, timestampMs)
    if (endMs - startMs < 200) {
      setMessage('Das Fehlerintervall muss mindestens 0,2 Sekunden lang sein.')
      return
    }
    videoRef.current?.pause()
    setPlaying(false)
    setErrorIntervalSaving(true)
    try {
      const savedRange = await createOffPathInterval(selectedMissionId, activeVideo.video_id, {
        start_ms: startMs,
        end_ms: endMs,
        note: 'Quick Review: KI-Ausgabe in diesem Abschnitt fehlerhaft.',
        annotator: 'human',
      })
      setSavedVideoRanges(current => [...current, savedRange])
      setErrorIntervalStartMs(null)
      setMessage(`Fehlerintervall ${(startMs / 1000).toFixed(1)}–${(endMs / 1000).toFixed(1)} s gespeichert.`)
    } catch (error) {
      setErrorIntervalStartMs(null)
      setMessage(error instanceof Error ? error.message : 'Fehlerintervall konnte nicht gespeichert werden')
    } finally {
      setErrorIntervalSaving(false)
    }
  }

  const setReviewKind = (id: string, kind: ReviewKind) =>
    setReviewSnapshots(current => current.map(snapshot => (snapshot.id === id ? {...snapshot, kind, saved: false} : snapshot)))

  const removeReviewSnapshot = async (snapshot: ReviewSnapshot) => {
    if (!selectedMissionId || !activeVideo || reviewSaving) return
    if (!snapshot.saved) {
      setReviewSnapshots(current => current.filter(item => item.id !== snapshot.id))
      return
    }
    setReviewSaving(true)
    try {
      await deleteCriticalFlag(selectedMissionId, activeVideo.video_id, snapshot.frameIndex)
      setReviewSnapshots(current => current.filter(item => item.id !== snapshot.id))
      setCriticalFlags(current => current.filter(flag => !(flag.video_id === activeVideo.video_id && flag.frame_index === snapshot.frameIndex)))
      setMessage(`Fehlerframe ${snapshot.frameIndex + 1} aus den Trainingsdaten entfernt.`)
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Fehlerframe konnte nicht gelöscht werden')
    } finally {
      setReviewSaving(false)
    }
  }

  const removeSavedVideoRange = async (range: OffPathInterval) => {
    if (!selectedMissionId || !activeVideo || errorIntervalSaving) return
    setErrorIntervalSaving(true)
    try {
      await deleteOffPathInterval(selectedMissionId, activeVideo.video_id, range.id)
      setSavedVideoRanges(current => current.filter(item => item.id !== range.id))
      setMessage(`Videobereich ${(range.start_ms / 1000).toFixed(1)}–${(range.end_ms / 1000).toFixed(1)} s aus den Trainingsdaten entfernt.`)
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Videobereich konnte nicht gelöscht werden')
    } finally {
      setErrorIntervalSaving(false)
    }
  }

  const saveReviewSnapshots = async () => {
    if (!selectedMissionId || !activeVideo || reviewSaving) return
    const pending = currentReviewSnapshots.filter(snapshot => !snapshot.saved)
    if (!pending.length) return
    setReviewSaving(true)
    try {
      for (const snapshot of pending) {
        const existing = criticalFlags.find(
          flag => flag.video_id === activeVideo.video_id && flag.frame_index === snapshot.frameIndex,
        )
        const quickReviewNote = `Schnellreview: ${reviewKindLabel[snapshot.kind]}.`
        const note = [existing?.note, quickReviewNote].filter((value, index, values) => value && values.indexOf(value) === index).join(' ')
        await saveCriticalFlag(selectedMissionId, activeVideo.video_id, snapshot.frameIndex, {
          severity: Math.max(existing?.severity ?? 0, snapshot.kind === 'major' ? 5 : 3),
          brush_mask: existing?.brush_mask,
          note,
          annotator: 'human',
        })
      }
      const refreshed = await listCriticalFlags(selectedMissionId, activeVideo.video_id)
      setCriticalFlags(refreshed.items)
      setReviewSnapshots(current =>
        current.map(snapshot =>
          snapshot.videoId === activeVideo.video_id && pending.some(item => item.id === snapshot.id) ? {...snapshot, saved: true} : snapshot,
        ),
      )
      setMessage(`${pending.length} Review-Frame(s) als Trainingsfeedback gespeichert.`)
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Review-Frames konnten nicht gespeichert werden')
    } finally {
      setReviewSaving(false)
    }
  }

  const setBrushPixel = (values: number[], x: number, y: number) => {
    if (!activeVideo) return values
    const width = activeVideo.width
    const height = activeVideo.height
    const radius = Math.max(1, flagBrushSize)
    const minY = Math.max(0, Math.floor(y - radius))
    const maxY = Math.min(height - 1, Math.ceil(y + radius))
    const minX = Math.max(0, Math.floor(x - radius))
    const maxX = Math.min(width - 1, Math.ceil(x + radius))
    for (let py = minY; py <= maxY; py++) {
      for (let px = minX; px <= maxX; px++) {
        const dx = px - x
        const dy = py - y
        if (dx * dx + dy * dy <= radius * radius) values[py * width + px] = 1
      }
    }
    return values
  }

  const paintBrushStroke = (from: {x: number; y: number}, to: {x: number; y: number}) => {
    if (!activeVideo || !flagMask) return
    const next = flagMask.slice()
    const distance = Math.max(1, Math.hypot(to.x - from.x, to.y - from.y))
    const steps = Math.max(1, Math.ceil(distance / Math.max(2, flagBrushSize / 2)))
    for (let index = 0; index <= steps; index++) {
      const t = index / steps
      setBrushPixel(next, from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t)
    }
    setFlagMask(next)
  }

  const feedbackPoint = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect()
    if (!activeVideo || bounds.width <= 0 || bounds.height <= 0) return null
    const x = Math.max(0, Math.min(activeVideo.width - 1, ((event.clientX - bounds.left) * activeVideo.width) / bounds.width))
    const y = Math.max(0, Math.min(activeVideo.height - 1, ((event.clientY - bounds.top) * activeVideo.height) / bounds.height))
    return {x, y}
  }

  const beginFlagDrawing = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!showFeedbackBrush || !activeVideo) return
    const point = feedbackPoint(event)
    if (!point) return
    videoRef.current?.pause()
    setPlaying(false)
    paintCursorRef.current = point
    if (flagMask) setFlagMask(setBrushPixel(flagMask.slice(), point.x, point.y))
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  const moveFlagDrawing = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!showFeedbackBrush || !paintCursorRef.current || !activeVideo) return
    const point = feedbackPoint(event)
    if (!point) return
    paintBrushStroke(paintCursorRef.current, point)
    paintCursorRef.current = point
  }

  const endFlagDrawing = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    paintCursorRef.current = null
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
  }

  const clearFlagMask = () => {
    if (!activeVideo) return
    setFlagMask(new Array(activeVideo.width * activeVideo.height).fill(0))
  }

  const saveCriticalFrameFlag = async () => {
    if (!selectedMissionId || !activeVideo || flagSaving) return
    setFlagSaving(true)
    try {
      const brushMask = flagMask?.some(value => value === 1) ? (flagMaskRle ?? undefined) : undefined
      await saveCriticalFlag(selectedMissionId, activeVideo.video_id, frameIndex, {
        severity: flagSeverity,
        brush_mask: brushMask ?? undefined,
        note: flagNote.trim(),
        annotator: 'human',
      })
      const refreshed = await listCriticalFlags(selectedMissionId, activeVideo.video_id)
      setCriticalFlags(refreshed.items)
      setMessage(`Frame ${frameIndex + 1} als Meldung gespeichert (Stufe ${flagSeverity}).`)
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Meldung konnte nicht gespeichert werden')
    } finally {
      setFlagSaving(false)
    }
  }

  const removeCriticalFrameFlag = async () => {
    if (!selectedMissionId || !activeVideo || flagSaving) return
    setFlagSaving(true)
    try {
      await deleteCriticalFlag(selectedMissionId, activeVideo.video_id, frameIndex)
      const refreshed = await listCriticalFlags(selectedMissionId, activeVideo.video_id)
      setCriticalFlags(refreshed.items)
      setFlagMask(new Array(activeVideo.width * activeVideo.height).fill(0))
      setFlagNote('')
      setFlagSeverity(4)
      setMessage(`Meldung fuer Frame ${frameIndex + 1} geloescht.`)
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Meldung konnte nicht geloescht werden')
    } finally {
      setFlagSaving(false)
    }
  }

  const train = async () => {
    if (training) return
    setTraining(true)
    setMessage('Globales CPU-Modell wird aus allen bestätigten Labels und Refinements trainiert …')
    try {
      const model = await trainGlobalPathModel()
      const refreshed = await getGlobalModelDashboard()
      setData(refreshed)
      setMessage(`Globales Modell ${model.run_id} trainiert: ${model.validation_metrics.symmetric_score.toFixed(2)} / 100.`)
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Globales Modell konnte nicht trainiert werden')
    } finally {
      setTraining(false)
    }
  }

  const analyzeVideo = async () => {
    if (!activeVideo) return
    try {
      setAnalysisResult(null)
      setPrediction(null)
      setPlaying(false)
      const status = await startGlobalVideoAnalysis(selectedMissionId, activeVideo.video_id)
      setAnalysisStatus(status)
      setMessage(status.message)
      if (status.status === 'completed') setAnalysisResult(await getGlobalVideoAnalysisResult(selectedMissionId, activeVideo.video_id))
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Videoanalyse konnte nicht gestartet werden')
    }
  }

  const seekFrame = (next: number) => {
    if (!activeVideo) return
    const bounded = Math.max(0, Math.min(activeVideo.total_frames - 1, next))
    videoRef.current?.pause()
    setReversePlaying(false)
    if (videoRef.current) videoRef.current.currentTime = bounded / activeVideo.fps
    setPlaying(false)
    setFrameIndex(bounded)
  }

  const seekTime = (seconds: number) => {
    if (!activeVideo) return
    // Der Regler bewegt sich in 10-ms-Schritten; die Maske bleibt auf dem
    // zugehörigen Quellframe eingerastet, damit Video und Analyse synchron sind.
    seekFrame(Math.round(seconds * activeVideo.fps))
  }

  const toggleForwardPlayback = () => {
    const video = videoRef.current
    if (!video || !analysisResult) return
    setReversePlaying(false)
    if (video.paused) void video.play()
    else video.pause()
  }

  const toggleReversePlayback = () => {
    const video = videoRef.current
    if (!video || !analysisResult) return
    if (reversePlaying) {
      setReversePlaying(false)
      return
    }
    video.pause()
    setReversePlaying(true)
  }

  if (!data)
    return (
      <div className="global-model-page">
        <header className="global-model-header">
          <button onClick={onClose}>← Übersicht</button>
          <div>
            <span className="eyebrow">MISSIONSÜBERGREIFENDE WEGERKENNUNG</span>
            <h1>KI-Modellzentrum</h1>
          </div>
        </header>
        <div className="empty">{message || 'Datensätze werden geladen …'}</div>
      </div>
    )
  const {dataset, model} = data
  return (
    <div className="global-model-page">
      <header className="global-model-header">
        <button onClick={onClose}>← Übersicht</button>
        <div>
          <span className="eyebrow">MISSIONSÜBERGREIFENDE WEGERKENNUNG</span>
          <h1>KI-Modellzentrum</h1>
          <p>Ein gemeinsames CPU-Basismodell aus allen bisher bestätigten Weglabels — dazu die videobasierte Terrainklassifizierung.</p>
        </div>
        <button className="global-refinement-nav" onClick={onOpenRefinement}>REFINEMENTS →</button>
      </header>
      <details className="global-dashboard-section">
        <summary>
          <span>Trainingsdaten und globales Training</span>
          <small>{dataset.totals.missions} Missionen · {dataset.totals.confirmed_frames} Label-Frames</small>
        </summary>
      <section className="global-dataset-panel">
        <div className="global-dataset-head">
          <div>
            <h2>Verfügbare Trainingsdaten</h2>
            <p>Alle Missionen mit bestätigten Polygonframes werden automatisch einbezogen.</p>
          </div>
          <button disabled={training || dataset.totals.missions < 2 || dataset.totals.confirmed_frames < 10} onClick={() => void train()}>
            {training ? 'GLOBALES TRAINING LÄUFT …' : 'AUF ALLEN LABELS TRAINIEREN'}
          </button>
        </div>
        <div className="global-totals">
          <div>
            <span>Missionen</span>
            <b>{dataset.totals.missions}</b>
          </div>
          <div>
            <span>Label-Frames</span>
            <b>{dataset.totals.confirmed_frames}</b>
          </div>
          <div>
            <span>Videos</span>
            <b>{dataset.totals.videos}</b>
          </div>
          <div>
            <span>Refinements</span>
            <b>{dataset.totals.refinements}</b>
          </div>
          <div>
            <span>Fehlermeldungen</span>
            <b>{dataset.totals.critical_flags}</b>
          </div>
        </div>
        <div className="global-mission-list">
          {dataset.missions.map(mission => (
            <article key={mission.mission_id} className={mission.confirmed_frames ? 'included' : ''}>
              <i />
              <div>
                <b>{mission.name}</b>
                <span>{mission.mission_id}</span>
              </div>
              <strong>{mission.confirmed_frames} Labels</strong>
              <small>
                {mission.videos} Videos · {mission.refinements} Refinements · {mission.critical_flags} Meldungen
              </small>
            </article>
          ))}
        </div>
        {message && (
          <div className="labeling-message" role="status">
            {message}
          </div>
        )}
      </section>
      </details>
      <details className="global-dashboard-section">
        <summary>
          <span>Aktives globales Modell</span>
          <small>{model ? `Validierung ${model.validation_metrics.symmetric_score.toFixed(1)} / 100` : 'Noch kein Modell'}</small>
        </summary>
      <section className="global-result-panel">
        <div className="section-head">
          <h2>Aktives globales Modell</h2>
          <p>Es wird separat gespeichert und überschreibt keine missionsspezifischen Modelle.</p>
        </div>
        {model ? (
          <>
            <div className="active-model-version">
              <div>
                <span>Modelllauf</span>
                <b>{model.run_id}</b>
                <small>{new Date(model.created_at).toLocaleString('de-DE')}</small>
              </div>
              <strong className="changed">GLOBAL · {model.dataset.missions.length} MISSIONEN</strong>
            </div>
            <div className="global-totals model">
              <div>
                <span>Validierung</span>
                <b>{model.validation_metrics.symmetric_score.toFixed(2)} / 100</b>
              </div>
              <div>
                <span>Training</span>
                <b>{model.split.train_frames} Frames</b>
              </div>
              <div>
                <span>Validierung</span>
                <b>{model.split.validation_frames} Frames</b>
              </div>
              <div>
                <span>Laufzeit</span>
                <b>{model.runtime_seconds.toFixed(1)} s</b>
              </div>
              <div>
                <span>Fehlermeldungen</span>
                <b>{model.dataset.critical_flags_included}</b>
              </div>
            </div>
            <div className="path-model-evidence">
              {model.evidence.map(item => (
                <figure key={`${item.kind}-${item.video_id}-${item.frame_index}`}>
                  <img src={item.image_url} alt={`${item.kind} globaler Validierungsframe`} />
                  <figcaption>
                    <b>{item.kind === 'best' ? 'Bester' : item.kind === 'worst' ? 'Schwierigster' : 'Mittlerer'} Frame</b>
                    <span>
                      Frame {item.frame_index + 1} · {item.metrics.symmetric_score.toFixed(1)} / 100
                    </span>
                    <small>Grün korrekt · Rot übersehen · Gelb fälschlich erkannt</small>
                  </figcaption>
                </figure>
              ))}
            </div>
          </>
        ) : (
          <div className="empty">
            Noch kein globales Modell trainiert. Die vorhandenen missionsspezifischen Modelle bleiben unverändert.
          </div>
        )}
      </section>
      </details>
      <details className="global-dashboard-section">
        <summary>
          <span>Videoanalyse und Wiedergabe</span>
          <small>{analysisResult ? 'Analyse bereit' : analysisStatus?.status === 'running' ? 'Analyse läuft' : 'Video wählen'}</small>
        </summary>
      <section className="global-analysis-player">
        <div className="section-head">
          <h2>Globale Videoanalyse und Wiedergabe</h2>
          <p>Das Modell berechnet zuerst alle Frames. Danach läuft das Originalvideo flüssig mit der vorberechneten Wegmaske.</p>
        </div>
        {!model ? (
          <div className="empty">Trainiere zuerst ein globales Modell, um den Analyseplayer zu aktivieren.</div>
        ) : (
          <>
            <div className="global-video-setup">
              <label>
                Mission
                <select value={selectedMissionId} onChange={event => setSelectedMissionId(event.target.value)}>
                  {dataset.missions
                    .filter(item => item.confirmed_frames > 0)
                    .map(item => (
                      <option key={item.mission_id} value={item.mission_id}>
                        {item.name} · {item.confirmed_frames} Labels
                      </option>
                    ))}
                </select>
              </label>
              <label>
                Originalvideo
                <select
                  value={activeVideo?.video_id ?? ''}
                  onChange={event => {
                    setSelectedVideoId(event.target.value)
                    setFrameIndex(0)
                    setPlaying(false)
                  }}
                >
                  {videos.map(video => (
                    <option key={video.video_id} value={video.video_id}>
                      {video.original_name} · {terrainCategoryLabel(video.terrain_category)}
                    </option>
                  ))}
                </select>
                <small style={{display: 'block', marginTop: 5}}>
                  {activeVideo
                    ? `Terrainkategorie: ${terrainCategoryLabel(activeVideo.terrain_category)} – alle Frames dieses Videos tragen dieses Label.`
                    : 'Terrainkategorie nicht gesetzt.'}
                </small>
              </label>
              <button
                disabled={!activeVideo || analysisStatus?.status === 'running' || analysisStatus?.status === 'queued'}
                onClick={() => void analyzeVideo()}
              >
                {analysisResult
                  ? needsGradePreview
                    ? 'ANALYSE ERNEUERN (ABSTUFUNG NACHRÜSTEN)'
                    : 'ANALYSE BEREITS FERTIG'
                  : analysisStatus?.status === 'running' || analysisStatus?.status === 'queued'
                    ? 'VIDEO WIRD ANALYSIERT …'
                    : 'VIDEO VOLLSTÄNDIG ANALYSIEREN'}
              </button>
            </div>
            {analysisStatus && (
              <div className={`global-analysis-progress ${analysisStatus.status}`}>
                <div>
                  <b>{analysisStatus.message}</b>
                  <span>
                    {analysisStatus.processed_frames.toLocaleString('de-DE')} von {analysisStatus.total_frames.toLocaleString('de-DE')}{' '}
                    Frames · {Math.round(analysisStatus.progress * 100)} %
                  </span>
                  <small>
                    Verstrichen: {duration(analysisStatus.elapsed_seconds)} · Restzeit: {duration(analysisStatus.eta_seconds)}
                  </small>
                </div>
                <div className="progress-track">
                  <i style={{width: `${analysisStatus.progress * 100}%`}} />
                </div>
              </div>
            )}
            <div className={`global-player-grid ${analysisResult ? 'ready' : 'waiting'}`}>
              <div className="global-video-column">
                <div
                  className="global-video-stage"
                  style={activeVideo ? {aspectRatio: `${activeVideo.width}/${activeVideo.height}`} : undefined}
                >
                  {activeVideo && (
                    <video
                      ref={videoRef}
                      src={`/api/v1/missions/${selectedMissionId}/videos/${activeVideo.video_id}/content`}
                      muted
                      playsInline
                      preload="metadata"
                      onTimeUpdate={event => {
                        if (activeVideo)
                          setFrameIndex(
                            Math.min(activeVideo.total_frames - 1, Math.floor(event.currentTarget.currentTime * activeVideo.fps)),
                          )
                      }}
                      onPlay={() => setPlaying(true)}
                      onPause={() => setPlaying(false)}
                      onEnded={() => setPlaying(false)}
                      title={
                        showFeedbackBrush
                          ? 'Fehler-Pinsel ist aktiv'
                          : analysisResult
                            ? 'Nutze die Tasten unter dem Video, um einen Fehlerframe direkt zu speichern oder einen Fehlerbereich zu erfassen.'
                            : 'Nach Abschluss der Videoanalyse können Review-Frames erfasst werden'
                      }
                    />
                  )}
                  <canvas ref={canvasRef} style={{opacity: layer.mask ? maskOpacity : 0}} />
                  <canvas
                    ref={brushCanvasRef}
                    className={`global-feedback-canvas ${showFeedbackBrush ? 'active' : ''}`}
                    style={{
                      opacity: activeCriticalFlag || showFeedbackBrush ? 0.92 : 0.68,
                      pointerEvents: showFeedbackBrush ? 'auto' : 'none',
                    }}
                    onPointerDown={beginFlagDrawing}
                    onPointerMove={moveFlagDrawing}
                    onPointerUp={endFlagDrawing}
                    onPointerLeave={endFlagDrawing}
                  />
                  <CorridorOverlay
                    check={planner.check}
                    activeCorridor={planner.activeCorridor}
                    onSelectCorridor={planner.toggleSelected}
                    proposal={planner.corridorProposal}
                    draft={planner.draft}
                    planning={planning}
                    aspect={activeVideo ? activeVideo.width / activeVideo.height : 16 / 9}
                    onMovePoint={planner.movePoint}
                    onAddPoint={planner.addPoint}
                    onRemovePoint={planner.removePoint}
                  />
                  {!analysisResult && !displayed && (
                    <div className="global-player-lock">
                      <b>
                        {analysisStatus?.status === 'running'
                          ? `${Math.round(analysisStatus.progress * 100)} % analysiert`
                          : 'Video noch nicht analysiert'}
                      </b>
                      <span>Nach Abschluss kann das Video normal mit synchroner KI-Maske abgespielt werden.</span>
                    </div>
                  )}
                  {predictionLoading && !analysisResult && <div className="global-prediction-loading">VORSCHAU WIRD BERECHNET …</div>}
                </div>
                <div className="global-transport">
                  <button disabled={!analysisResult || frameIndex <= 0} onClick={() => seekFrame(frameIndex - 1)}>
                    ←
                  </button>
                  <div className="global-play-directions" role="group" aria-label="Videowiedergabe">
                    <button
                      type="button"
                      disabled={!analysisResult || frameIndex <= 0}
                      aria-pressed={reversePlaying}
                      onClick={toggleReversePlayback}
                    >
                      {reversePlaying ? '❚❚ ZURÜCK' : '◀ ZURÜCK'}
                    </button>
                    <button type="button" disabled={!analysisResult} aria-pressed={playing && !reversePlaying} onClick={toggleForwardPlayback}>
                      {playing && !reversePlaying ? '❚❚ PAUSE' : '▶ VORWÄRTS'}
                    </button>
                  </div>
                  <div className="global-speed-boost" role="group" aria-label="Wiedergabegeschwindigkeit">
                    <span>Tempo</span>
                    <input
                      aria-label="Wiedergabegeschwindigkeit"
                      type="range"
                      min="0"
                      max={playbackSpeeds.length - 1}
                      step="1"
                      value={Math.max(0, playbackSpeeds.indexOf(speed))}
                      disabled={!analysisResult}
                      onChange={event => setPlaybackSpeed(playbackSpeeds[+event.target.value])}
                    />
                    <output>{playbackSpeedLabel(speed)}</output>
                  </div>
                  <input
                    aria-label="Präzise Zeitposition"
                    type="range"
                    min="0"
                    max={Math.max(0, activeVideo ? (activeVideo.total_frames - 1) / activeVideo.fps : 0)}
                    step="0.01"
                    value={activeVideo ? frameIndex / activeVideo.fps : 0}
                    disabled={!analysisResult}
                    onChange={event => seekTime(+event.target.value)}
                  />
                  <span>
                    {(activeVideo ? frameIndex / activeVideo.fps : 0).toFixed(2)} s · Frame {frameIndex + 1} / {(activeVideo?.total_frames ?? 0).toLocaleString('de-DE')}
                  </span>
                  <button
                    disabled={!analysisResult || !activeVideo || frameIndex >= activeVideo.total_frames - 1}
                    onClick={() => seekFrame(frameIndex + 1)}
                  >
                    →
                  </button>
                </div>
                <div className="global-paused-actions" aria-label="Video- und Frame-Erfassung">
                  <button
                    type="button"
                    disabled={!analysisResult || errorIntervalSaving || reversePlaying}
                      onClick={() => void toggleVideoErrorRecording()}
                  >
                    {errorIntervalStartMs === null ? 'VIDEO STARTEN' : 'VIDEO BEENDEN & SPEICHERN'}
                  </button>
                  <button type="button" disabled={!analysisResult || showFeedbackBrush || reviewSaving || playing || reversePlaying} onClick={() => void saveCurrentReviewFrame()}>
                    FALSCHEN FRAME DIREKT SPEICHERN
                  </button>
                </div>
              </div>
              <aside className="global-player-controls">
                <section className="global-review-gallery" aria-label="Gespeicherte Review-Frames">
                  <div className="global-review-gallery-head">
                    <div>
                      <span>QUICK REVIEW</span>
                      <b>Frame-Referenzen</b>
                    </div>
                    <small>{currentReviewSnapshots.length} erfasst</small>
                  </div>
                  <p>
                    Einzelne falsche Frames werden mit der rechten Taste sofort gespeichert. „Video starten“ erfasst dagegen erst beim anschließenden „Video beenden & speichern“ den Bereich zwischen Start- und Endframe. Es werden keine Bildkopien oder Videos abgelegt.
                  </p>
                  {!playing && !reversePlaying && pausedFrameReference && (
                    <div className={`global-review-current-stamp ${pausedFrameReference.review ? 'marked' : ''}`}>
                      <span>AKTUELL PAUSIERT</span>
                      <b>Frame {frameIndex + 1} · {(pausedFrameReference.timestampMs / 1000).toFixed(2)} s</b>
                      <small>
                        {pausedFrameReference.review
                          ? pausedFrameReference.review.saved
                            ? 'Diese Referenz ist dauerhaft gespeichert.'
                            : 'Vorgemerkt, aber noch nicht dauerhaft gespeichert.'
                          : 'Noch nicht vorgemerkt oder gespeichert.'}
                      </small>
                    </div>
                  )}
                  {reviewFrameRange && (
                    <div className="global-review-range">
                      Referenzbereich: Frame {reviewFrameRange.first + 1} bis {reviewFrameRange.last + 1}
                    </div>
                  )}
                  {errorIntervalStartMs !== null && (
                    <div className="global-review-interval-active">
                      <i /> Fehlerintervall läuft seit {(errorIntervalStartMs / 1000).toFixed(1)} s · „Video beenden & speichern“ zum Abschließen
                    </div>
                  )}
                  {currentReviewSnapshots.length ? (
                    <>
                      <div className="global-review-snapshot-list">
                        {currentReviewSnapshots.map(snapshot => (
                          <article key={snapshot.id} className={snapshot.saved ? 'saved' : ''}>
                            <button
                              type="button"
                              className="global-review-thumbnail"
                              onClick={() => seekFrame(snapshot.frameIndex)}
                              title={`Frame ${snapshot.frameIndex + 1} öffnen`}
                            >
                              <span>ÖFFNEN<br />FRAME {snapshot.frameIndex + 1}</span>
                            </button>
                            <div>
                              <b>Frame {snapshot.frameIndex + 1}</b>
                              <small>{(snapshot.timestampMs / 1000).toFixed(1)} s {snapshot.saved ? '· gespeichert' : '· offen'}</small>
                              <div className="global-review-kind">
                                {(Object.keys(reviewKindLabel) as ReviewKind[]).map(kind => (
                                  <button
                                    key={kind}
                                    type="button"
                                    className={snapshot.kind === kind ? 'active' : ''}
                                    disabled={snapshot.saved}
                                    onClick={() => setReviewKind(snapshot.id, kind)}
                                  >
                                    {kind === 'incorrect' ? 'Nicht korrekt' : 'Erheblich'}
                                  </button>
                                ))}
                              </div>
                            </div>
                            <button
                              type="button"
                              className="global-review-remove"
                              disabled={reviewSaving}
                              onClick={() => void removeReviewSnapshot(snapshot)}
                              aria-label={`Frame ${snapshot.frameIndex + 1} aus Review entfernen`}
                            >
                              ×
                            </button>
                          </article>
                        ))}
                      </div>
                      <button
                        type="button"
                        className="global-review-save"
                        disabled={reviewSaving || !currentReviewSnapshots.some(snapshot => !snapshot.saved)}
                        onClick={() => void saveReviewSnapshots()}
                      >
                        {reviewSaving ? 'REFERENZEN WERDEN GESPEICHERT …' : 'AUSWAHL DAUERHAFT SPEICHERN'}
                      </button>
                    </>
                  ) : (
                    <div className="global-review-empty">Noch keine Frames erfasst.</div>
                  )}
                  {activeVideo && (
                    <details className="global-review-video">
                      <summary>Originalvideo hier ansehen</summary>
                      <video
                        controls
                        muted
                        playsInline
                        preload="metadata"
                        src={`/api/v1/missions/${selectedMissionId}/videos/${activeVideo.video_id}/content`}
                      />
                    </details>
                  )}
                  {savedVideoRanges.length > 0 && (
                    <section className="global-saved-video-ranges" aria-label="Gespeicherte Videobereiche">
                      <b>Gespeicherte Videobereiche</b>
                      {savedVideoRanges.map(range => (
                        <div key={range.id}>
                          <button type="button" onClick={() => seekFrame(Math.round((range.start_ms / 1000) * (activeVideo?.fps ?? 1)))}>
                            {`${(range.start_ms / 1000).toFixed(1)} s – ${(range.end_ms / 1000).toFixed(1)} s`}
                          </button>
                          <button
                            type="button"
                            disabled={errorIntervalSaving}
                            onClick={() => void removeSavedVideoRange(range)}
                            aria-label={`Videobereich ${(range.start_ms / 1000).toFixed(1)} bis ${(range.end_ms / 1000).toFixed(1)} Sekunden löschen`}
                          >
                            Löschen
                          </button>
                        </div>
                      ))}
                    </section>
                  )}
                </section>
                <details className="global-control-section">
                  <summary>Flächenidentifizierung und Masken</summary>
                  <div className="global-control-content">
                <div className="global-mask-switches">
                  <label>
                    <input type="checkbox" checked={showGrades} onChange={event => setShowGrades(event.target.checked)} />
                    <span>Abstufung anzeigen</span>
                  </label>
                  <label>
                    <input
                      type="checkbox"
                      checked={showAiMask}
                      disabled={!analysisResult || layer.graded}
                      onChange={event => setShowAiMask(event.target.checked)}
                    />
                    <span>KI-Maske anzeigen</span>
                  </label>
                  <label>
                    <input
                      type="checkbox"
                      checked={showLabelMask}
                      disabled={!analysisResult || layer.graded}
                      onChange={event => setShowLabelMask(event.target.checked)}
                    />
                    <span>Label-Maske anzeigen</span>
                  </label>
                  <label>
                    Masken-Deckkraft · {Math.round(maskOpacity * 100)} %
                    <input
                      type="range"
                      min="0.05"
                      max="0.9"
                      step="0.05"
                      value={maskOpacity}
                      disabled={!analysisResult || (!showAiMask && !showLabelMask)}
                      onChange={event => setMaskOpacity(+event.target.value)}
                    />
                  </label>
                </div>
                  </div>
                </details>
                <details className="global-control-section">
                  <summary>Korridore und Trajektorien</summary>
                  <div className="global-control-content">
                <CorridorReadout
                  planner={planner}
                  planning={planning}
                  onTogglePlanning={setPlanning}
                  note={trajectoryNote}
                  onNote={setTrajectoryNote}
                  annotator="Simon"
                />
                  </div>
                </details>
                <details className="global-control-section">
                  <summary>Fehlerfeedback</summary>
                  <div className="global-control-content">
                <div className="global-feedback-panel">
                  <div className="section-head compact">
                    <h3>Fehler melden</h3>
                    <p>Frame markieren, Schweregrad setzen und gemalte Fehlerfläche speichern.</p>
                  </div>
                  <label>
                    <input type="checkbox" checked={showFeedbackBrush} onChange={event => setShowFeedbackBrush(event.target.checked)} />
                    <span>Fehler-Pinsel aktivieren</span>
                  </label>
                  <div className="severity-picker">
                    {[1, 2, 3, 4, 5].map(value => (
                      <button key={value} className={flagSeverity === value ? 'active' : ''} onClick={() => setFlagSeverity(value)}>
                        Stufe {value}
                      </button>
                    ))}
                  </div>
                  <label>
                    Pinselfläche · {flagBrushSize} px
                    <input
                      type="range"
                      min="4"
                      max="80"
                      step="1"
                      value={flagBrushSize}
                      onChange={event => setFlagBrushSize(+event.target.value)}
                    />
                  </label>
                  <label>
                    Fehlernotiz
                    <textarea
                      value={flagNote}
                      maxLength={1000}
                      placeholder="Optional: warum ist der Frame falsch?"
                      onChange={event => setFlagNote(event.target.value)}
                    />
                  </label>
                  <div className="feedback-actions">
                    <button onClick={clearFlagMask}>Maske leeren</button>
                    <button className="primary" disabled={flagSaving || !activeVideo} onClick={() => void saveCriticalFrameFlag()}>
                      {activeCriticalFlag ? 'Meldung aktualisieren' : 'Frame melden'}
                    </button>
                    <button disabled={flagSaving || !activeCriticalFlag} onClick={() => void removeCriticalFrameFlag()}>
                      Meldung löschen
                    </button>
                  </div>
                  <small>Stufe 1 = leichter Fehler, Stufe 5 = fast komplett falsch. Gemalte Bereiche werden mitgespeichert.</small>
                  {activeCriticalFlag && (
                    <small>
                      Bereits gemeldet: Stufe {activeCriticalFlag.severity} ·{' '}
                      {new Date(activeCriticalFlag.created_at).toLocaleString('de-DE')}
                    </small>
                  )}
                </div>
                  </div>
                </details>
                <details className="global-control-section">
                  <summary>Wiedergabe und Frame-Details</summary>
                  <div className="global-control-content">
                <label>
                  Abspieltempo
                  <select value={speed} disabled={!analysisResult} onChange={event => setPlaybackSpeed(+event.target.value)}>
                    <option value="0.25">0,25×</option>
                    <option value="0.5">0,5×</option>
                    <option value="1">1×</option>
                    <option value="2">2×</option>
                    <option value="4">4×</option>
                  </select>
                </label>
                {layer.graded && displayed && (
                  <GradeLegend ontology={displayed.grade_ontology} mask={displayed.grade_mask} grading={displayed.grading} />
                )}
                {displayed && (
                  <div className="global-frame-result">
                    <b>
                      {layer.graded
                        ? `${(displayed.path_fraction * 100).toFixed(1)} % Weg`
                        : displayed.evaluation
                          ? `${displayed.evaluation.metrics.symmetric_score.toFixed(1)} / 100`
                          : `${(displayed.path_fraction * 100).toFixed(1)} % Weg`}
                    </b>
                    <span>
                      {layer.graded
                        ? 'Abgestufte KI-Einschätzung für diesen Frame'
                        : !showAiMask && !showLabelMask
                          ? 'Originalvideo ohne Masken'
                          : displayed.evaluation
                            ? 'Gelabelter Frame: Grün korrekt · Rot übersehen · Gelb fälschlich'
                            : 'Türkis: vorberechnete globale KI-Wegmaske'}
                    </span>
                    <small>Modell {displayed.model_run_id}</small>
                  </div>
                )}
                {showGrades && analysisResult && !analyzedFrame?.grade_mask && (
                  <small className="grade-note">
                    Diese Analyse ist älter als die durchgehende Abstufung: Während der Wiedergabe erscheint nur die einfarbige
                    vorberechnete Maske, die Abstufung wird lediglich im Pausenzustand einzeln nachberechnet. „ANALYSE ERNEUERN“ oben
                    berechnet das Video einmalig neu — danach läuft die Abstufung durchgehend mit, auch bei laufender Wiedergabe.
                  </small>
                )}
                {analysisResult && (
                  <div className="global-analysis-complete">
                    <b>Wiedergabe bereit</b>
                    <span>
                      {analysisResult.analyzed_frames.toLocaleString('de-DE')} Frames in {duration(analysisResult.runtime_seconds)}{' '}
                      analysiert.
                    </span>
                  </div>
                  )}
                  </div>
                </details>
              </aside>
            </div>
          </>
        )}
      </section>
      </details>
      <details className="global-dashboard-section">
        <summary>
          <span>Terrainmodell</span>
          <small>Untergrundklassen und Video-Vorhersage</small>
        </summary>
        <TerrainModelPanel />
      </details>
    </div>
  )
}
