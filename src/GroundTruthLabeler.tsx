import {useEffect, useMemo, useRef, useState} from 'react'
import type {CSSProperties, PointerEvent as ReactPointerEvent, WheelEvent as ReactWheelEvent} from 'react'
import {
  getGroundTruth,
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
  updateVideoTerrainCategory,
} from './api'
import GradeLegend from './GradeLegend'
import {AI_BINARY_PALETTE, COMPARISON_LEGEND, COMPARISON_PALETTE, paintMaskCanvas, paletteFromGradeOntology, rleValueAt} from './masks'
import {TERRAIN_CATEGORY_OPTIONS, terrainCategoryLabel} from './terrainCategories'
import type {
  GroundTruthSummary,
  GroundTruthStatus,
  LabelingVideo,
  Mission,
  NormalizedPoint,
  PathModelResult,
  PathPrediction,
  PathTrainingJob,
  TerrainMask,
} from './types'

type Tool = 'add' | 'edit' | 'move' | 'pan'
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
  const [selectionMode, setSelectionMode] = useState<'stride' | 'count'>('stride')
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
  const videoRef = useRef<HTMLVideoElement>(null)
  const aiMaskCanvasRef = useRef<HTMLCanvasElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)
  const overlayRef = useRef<SVGSVGElement>(null)
  const dragRef = useRef<Drag | null>(null)
  const carryRef = useRef<{videoId: string; frameIndex: number; points: NormalizedPoint[]} | null>(null)

  const activeVideo = videos.find(video => video.video_id === activeVideoId) ?? videos[0]
  const selectedFrames = useMemo(
    () => (activeVideo ? buildFrameSelection(activeVideo.total_frames, selectionMode, stride, targetCount) : []),
    [activeVideo, selectionMode, stride, targetCount],
  )
  const frameIndex = selectedFrames[Math.min(selectionPosition, Math.max(0, selectedFrames.length - 1))] ?? 0
  const timestampMs = activeVideo ? timestampFor(frameIndex, activeVideo.fps) : 0
  const hasPolygon = points.length >= 3

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
    const video = videoRef.current
    if (video) {
      video.pause()
      const seek = timestampMs / 1000
      if (Math.abs(video.currentTime - seek) > 0.0005) video.currentTime = seek
    }
    let cancelled = false
    setLoading(true)
    setMessage('Frame wird geladen …')
    getGroundTruth(mission.id, activeVideo.video_id, frameIndex)
      .then(annotation => {
        if (cancelled) return
        const carried = carryRef.current
        const polygon = annotation?.polygons?.[0]?.points
        const carriedIntoCurrentFrame = !annotation && carried?.videoId === activeVideo.video_id && carried.frameIndex === frameIndex
        if (polygon?.length) {
          setPoints(copyPoints(polygon))
          setTool('edit')
          setMessage('Gespeichertes Polygon geladen und vollständig editierbar.')
          setFullFrameNotTraversable(false)
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
          setPoints(copyPoints(carried!.points))
          setTool('edit')
          setFullFrameNotTraversable(false)
          setMessage(
            'Das Polygon des vorherigen Frames bleibt als neue Vorlage liegen. Verschiebe die Punkte oder lösche es für diesen Frame.',
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
        setSelectedVertex(null)
      })
      .catch(error => setMessage(error instanceof Error ? error.message : 'Ground Truth konnte nicht geladen werden'))
      .finally(() => !cancelled && setLoading(false))
    return () => {
      cancelled = true
    }
  }, [mission.id, activeVideo?.video_id, frameIndex, timestampMs])

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
    const carriedPoints = polygonForNextFrame(points, direction, carryForward)
    carryRef.current =
      carriedPoints && activeVideo ? {videoId: activeVideo.video_id, frameIndex: selectedFrames[nextPosition], points: carriedPoints} : null
    setSelectionPosition(nextPosition)
    return true
  }

  const persist = async (nextStatus: GroundTruthStatus) => {
    if (!activeVideo || saving) return null
    if (nextStatus === 'confirmed' && !hasPolygon && !fullFrameNotTraversable) {
      setMessage('Setze mindestens drei Punkte oder wähle Vollbild-Nicht-befahrbar, bevor du bestätigst.')
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
      const polygons =
        nextStatus === 'skipped' || (!hasPolygon && !fullFrameNotTraversable)
          ? []
          : [{id: 'path-1', class_id: 'traversable' as const, points: copyPoints(points)}]
      const saved = await saveGroundTruth(mission.id, activeVideo.video_id, frameIndex, {
        timestamp_ms: timestampMs,
        mask: fullFrameNotTraversable ? (fullFrameMask() ?? undefined) : undefined,
        polygons: fullFrameNotTraversable ? [] : polygons,
        status: nextStatus,
        annotator: annotator.trim() || 'Simon',
        notes,
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
      } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'y') {
        event.preventDefault()
        redo()
      } else if (event.key === 'Delete' || event.key === 'Backspace') {
        event.preventDefault()
        removeSelectedPoint()
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
            </div>
            <span>Mausrad zoomt · Hand verschiebt das Bild</span>
          </div>
          <div className="labeling-stage-background">
            <div className="labeling-stage" ref={stageRef} style={frameStyle} onWheel={wheel}>
              <div className="labeling-transform" style={transformStyle}>
                <video
                  ref={videoRef}
                  src={`/api/v1/missions/${mission.id}/videos/${activeVideo.video_id}/content`}
                  preload="auto"
                  muted
                  playsInline
                  onLoadedMetadata={() => {
                    if (videoRef.current) videoRef.current.currentTime = timestampMs / 1000
                  }}
                />
                <canvas ref={aiMaskCanvasRef} className="ai-path-mask-layer" style={{opacity: aiMaskOpacity}} aria-hidden="true" />
                <svg
                  ref={overlayRef}
                  viewBox="0 0 1 1"
                  preserveAspectRatio="none"
                  className={`polygon-overlay tool-${tool} ${fullFrameNotTraversable ? 'full-frame-mode' : ''}`}
                  onPointerDown={pointerDown}
                  onPointerMove={pointerMove}
                  onPointerUp={pointerUp}
                  onPointerCancel={pointerUp}
                  onContextMenu={removeNearestPoint}
                  aria-label="Ground-Truth-Polygonfläche"
                >
                  {!fullFrameNotTraversable && points.length >= 3 && (
                    <polygon points={polygonPoints} className="ground-truth-polygon" style={{fillOpacity: maskOpacity}} />
                  )}
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
          </div>
        </section>

        <aside className="labeling-controls">
          <h2>Frame-Auswahl</h2>
          <label>
            Originalvideo
            <select
              value={activeVideo.video_id}
              disabled={dirty}
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
              {activeVideo.terrain_category
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
          </div>
          {selectionMode === 'stride' ? (
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
          </div>
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
        {summary.items.length ? (
          <div className="saved-mask-list">
            {summary.items.map(item => {
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
                </button>
              )
            })}
          </div>
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
