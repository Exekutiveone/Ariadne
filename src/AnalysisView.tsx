import {useEffect, useMemo, useRef, useState} from 'react'
import type {CSSProperties, MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent} from 'react'
import {CircleMarker, MapContainer, Polyline, TileLayer} from 'react-leaflet'
import {getGroundTruth, listGroundTruth, saveGroundTruth} from './api'
import type {
  Analysis,
  GroundTruthStatus,
  GroundTruthSummary,
  GroundTruthValue,
  Mission,
  OverlayMode,
  Reconstruction,
  Segmentation,
  SegmentationFrame,
  TerrainFrameEvaluation,
  TerrainMask,
  TraversabilityClass,
} from './types'

const n = (value: number, digits = 1) => new Intl.NumberFormat('de-DE', {maximumFractionDigits: digits}).format(value)
const clock = (seconds: number) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`
const percent = (value: number) => `${Math.round(Math.max(0, Math.min(1, value)) * 100)} %`

const modes: {id: OverlayMode; label: string}[] = [
  {id: 'original', label: 'Original'},
  {id: 'ground', label: 'Boden'},
  {id: 'traversability', label: 'Befahrbarkeit'},
  {id: 'labels', label: 'Eigene Labels'},
]

const groundTruthStyles: Record<GroundTruthValue, {label: string; color: string; rgba: [number, number, number, number]}> = {
  0: {label: 'Radierer / nicht markiert', color: '#ffffff', rgba: [0, 0, 0, 0]},
  1: {label: 'Befahrbar', color: '#55d96f', rgba: [85, 217, 111, 245]},
  2: {label: 'Nicht befahrbar', color: '#e05b52', rgba: [224, 91, 82, 245]},
  3: {label: 'Nicht bewertbar', color: '#737c78', rgba: [115, 124, 120, 245]},
}

const terrainFallback: Record<TraversabilityClass, {label: string; color: string}> = {
  likely_traversable: {label: 'Wahrscheinlich befahrbar', color: '#55d96f'},
  limited: {label: 'Eingeschränkt oder unsicher', color: '#e7c84d'},
  not_traversable: {label: 'Wahrscheinlich nicht befahrbar', color: '#e05b52'},
  unknown: {label: 'Nicht bewertbar', color: '#737c78'},
}

const terrainOrder: TraversabilityClass[] = ['likely_traversable', 'limited', 'not_traversable', 'unknown']
const riskPriority: Record<TraversabilityClass, number> = {likely_traversable: 0, limited: 1, not_traversable: 2, unknown: 3}
const reasonLabels: Record<string, string> = {
  connected_ground: 'zusammenhängende Bodenfläche',
  clearance_above_required: 'freie Breite oberhalb des Mindestwerts',
  current_frame_support: 'im aktuellen Videoframe belegt',
  limited_clearance_or_surface_uncertainty: 'begrenzte Breite oder unsicherer Untergrund',
  visible_obstacle_or_step_evidence: 'sichtbares Hindernis oder mögliche Stufe',
  insufficient_visible_ground_evidence: 'zu wenig sichtbare Bodenevidenz',
  confidence_gate: 'Konfidenzgrenze nicht erreicht',
  insufficient_connected_free_ground: 'zu wenig zusammenhängende freie Fläche',
  insufficient_clearance: 'freie Breite nicht ausreichend',
  low_visibility: 'eingeschränkte Sichtbarkeit',
  unstable_motion: 'zeitlich instabile Schätzung',
  frame_colour_texture_connectivity: 'Farbe, Textur und Zusammenhang im aktuellen Frame',
  representative_class: 'repräsentative Bewertungsklasse',
  low_confidence_review: 'niedrige Konfidenz zur manuellen Prüfung',
  bottleneck_review: 'mögliche Engstelle zur manuellen Prüfung',
  no_connected_free_corridor: 'kein zusammenhängender freier Korridor',
  width_below_required: 'freie Breite unter dem konfigurierten Mindestwert',
  limited_visibility: 'eingeschränkte Sichtbarkeit',
  metric_scale_estimated: 'metrische Bildskala nur geschätzt',
  connected_clear_ground: 'zusammenhängender freier Boden',
  temporally_stable_corridor: 'über mehrere Frames stabiler Korridor',
  visible_obstacle_evidence: 'sichtbare Hindernisevidenz',
  substantial_non_assessable_area: 'großer nicht bewertbarer Bildanteil',
  insufficient_green_surface_support: 'Korridor im aktuellen Frame nicht ausreichend grün gestützt',
}

const factorLabels: [keyof TerrainFrameEvaluation['factors'], string][] = [
  ['free_width_score', 'Freie Breite'],
  ['obstacle_clearance_score', 'Hindernisfreiheit'],
  ['connectivity_score', 'Zusammenhang'],
  ['smoothness_score', 'Ebenheit'],
  ['bottleneck_clearance_score', 'Engstellenfreiheit'],
  ['visibility_score', 'Sichtbarkeit'],
  ['calibration_score', 'Kalibrierung'],
  ['temporal_stability_score', 'Zeitliche Stabilität'],
]

type VideoWithFrameCallback = HTMLVideoElement & {
  requestVideoFrameCallback?: (callback: (now: number, metadata: {mediaTime: number}) => void) => number
  cancelVideoFrameCallback?: (handle: number) => void
}

function frameAtOrBefore(frames: SegmentationFrame[] | undefined, timeMs: number, toleranceMs: number) {
  if (!frames?.length) return undefined
  let low = 0
  let high = frames.length - 1
  let answer = -1
  while (low <= high) {
    const middle = (low + high) >> 1
    if (frames[middle].timestamp_ms <= timeMs) {
      answer = middle
      low = middle + 1
    } else high = middle - 1
  }
  const frame = answer >= 0 ? frames[answer] : undefined
  return frame && timeMs - frame.timestamp_ms <= toleranceMs ? frame : undefined
}

function pointInPolygon(x: number, y: number, polygon: [number, number][]) {
  let inside = false
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
    const [xi, yi] = polygon[index]
    const [xj, yj] = polygon[previous]
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

function tracePolygon(context: CanvasRenderingContext2D, polygon: [number, number][], width: number, height: number) {
  if (polygon.length < 3) return false
  context.beginPath()
  context.moveTo(polygon[0][0] * width, polygon[0][1] * height)
  for (let index = 1; index < polygon.length; index++) context.lineTo(polygon[index][0] * width, polygon[index][1] * height)
  context.closePath()
  return true
}

function renderRleMask(
  context: CanvasRenderingContext2D,
  mask: TerrainMask,
  targetWidth: number,
  targetHeight: number,
  opacity: number,
  palette: Record<number, [number, number, number, number]>,
) {
  if (mask.width < 1 || mask.height < 1 || !mask.rle.length) return
  const source = document.createElement('canvas')
  source.width = mask.width
  source.height = mask.height
  const sourceContext = source.getContext('2d')
  if (!sourceContext) return
  const image = sourceContext.createImageData(mask.width, mask.height)
  const pixelCount = mask.width * mask.height
  let pixel = 0
  for (let index = 0; index + 1 < mask.rle.length && pixel < pixelCount; index += 2) {
    const colour = palette[mask.rle[index]] ?? [0, 0, 0, 0]
    const end = Math.min(pixelCount, pixel + Math.max(0, Math.floor(mask.rle[index + 1])))
    for (; pixel < end; pixel++) {
      const offset = pixel * 4
      image.data[offset] = colour[0]
      image.data[offset + 1] = colour[1]
      image.data[offset + 2] = colour[2]
      image.data[offset + 3] = colour[3]
    }
  }
  sourceContext.putImageData(image, 0, 0)
  context.save()
  context.globalAlpha = opacity
  context.imageSmoothingEnabled = false
  context.drawImage(source, 0, 0, targetWidth, targetHeight)
  context.restore()
}

function decodeRleValues(mask: TerrainMask) {
  const values = new Array<number>(mask.width * mask.height).fill(0)
  let pixel = 0
  for (let index = 0; index + 1 < mask.rle.length && pixel < values.length; index += 2) {
    const value = mask.rle[index]
    const end = Math.min(values.length, pixel + Math.max(0, Math.floor(mask.rle[index + 1])))
    values.fill(value, pixel, end)
    pixel = end
  }
  return values
}

function encodeGroundTruthValues(values: number[], width: number, height: number): TerrainMask {
  const size = width * height
  const normalized = values.length === size ? values : new Array<number>(size).fill(0)
  const rle: number[] = []
  if (size) {
    let previous = normalized[0] ?? 0
    let count = 1
    for (let index = 1; index < size; index++) {
      const value = normalized[index] ?? 0
      if (value === previous) count++
      else {
        rle.push(previous, count)
        previous = value
        count = 1
      }
    }
    rle.push(previous, count)
  }
  return {width, height, rle}
}

function clipToEvaluatedDriveArea(
  context: CanvasRenderingContext2D,
  mask: TerrainMask,
  targetWidth: number,
  targetHeight: number,
) {
  const cellWidth = targetWidth / mask.width
  const cellHeight = targetHeight / mask.height
  const pixelCount = mask.width * mask.height
  let pixel = 0
  context.beginPath()
  for (let index = 0; index + 1 < mask.rle.length && pixel < pixelCount; index += 2) {
    const value = mask.rle[index]
    let remaining = Math.min(pixelCount - pixel, Math.max(0, Math.floor(mask.rle[index + 1])))
    while (remaining > 0) {
      const row = Math.floor(pixel / mask.width)
      const column = pixel % mask.width
      const span = Math.min(remaining, mask.width - column)
      if (value === 1 || value === 2) {
        context.rect(column * cellWidth, row * cellHeight, span * cellWidth + .02, cellHeight + .02)
      }
      pixel += span
      remaining -= span
    }
  }
  context.clip()
}

function drawOverlay(args: {
  canvas: HTMLCanvasElement
  mode: OverlayMode
  frame?: SegmentationFrame
  terrain?: TerrainFrameEvaluation
  opacity: number
  selectedRegionId: string | null
  showCorridor: boolean
  annotationMask?: TerrainMask
  annotationPolygon: [number, number][]
  annotationValue: GroundTruthValue
  showAiSuggestion: boolean
  manualPolygons: [number, number][][]
  manualOpacity: number
  showManualLabels: boolean
}) {
  const {canvas, mode, frame, terrain, opacity, selectedRegionId, showCorridor, annotationMask, annotationPolygon, annotationValue, showAiSuggestion, manualPolygons, manualOpacity, showManualLabels} = args
  const bounds = canvas.getBoundingClientRect()
  if (!bounds.width || !bounds.height) return
  const ratio = Math.min(2, window.devicePixelRatio || 1)
  const pixelWidth = Math.max(1, Math.round(bounds.width * ratio))
  const pixelHeight = Math.max(1, Math.round(bounds.height * ratio))
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth
    canvas.height = pixelHeight
  }
  const context = canvas.getContext('2d')
  if (!context) return
  context.setTransform(ratio, 0, 0, ratio, 0, 0)
  context.clearRect(0, 0, bounds.width, bounds.height)
  const drawManualPolygons = () => {
    for (const polygon of manualPolygons) {
      if (!tracePolygon(context, polygon, bounds.width, bounds.height)) continue
      context.save()
      context.globalAlpha = manualOpacity
      context.fillStyle = '#55d96f'
      context.fill()
      context.restore()
      context.strokeStyle = '#ffffff'
      context.lineWidth = 3
      context.setLineDash([8, 5])
      context.stroke()
      context.setLineDash([])
    }
  }
  if (mode === 'labels') {drawManualPolygons(); return}
  if (mode === 'original' || !frame) return

  if (!terrain) {if (showManualLabels) drawManualPolygons(); return}
  if (mode === 'annotation') {
    if (showAiSuggestion) {
      renderRleMask(context, terrain.traversability.mask, bounds.width, bounds.height, Math.min(.22, opacity), {
        0: [115, 124, 120, 150],
        1: [85, 217, 111, 150],
        2: [231, 200, 77, 150],
        3: [224, 91, 82, 150],
      })
    }
    if (annotationMask) {
      renderRleMask(context, annotationMask, bounds.width, bounds.height, Math.max(.48, opacity), {
        0: groundTruthStyles[0].rgba,
        1: groundTruthStyles[1].rgba,
        2: groundTruthStyles[2].rgba,
        3: groundTruthStyles[3].rgba,
      })
    }
    if (annotationPolygon.length) {
      context.beginPath()
      context.moveTo(annotationPolygon[0][0] * bounds.width, annotationPolygon[0][1] * bounds.height)
      for (let index = 1; index < annotationPolygon.length; index++) context.lineTo(annotationPolygon[index][0] * bounds.width, annotationPolygon[index][1] * bounds.height)
      context.strokeStyle = groundTruthStyles[annotationValue].color
      context.lineWidth = 3
      context.setLineDash([7, 5])
      context.stroke()
      context.setLineDash([])
      for (const [x, y] of annotationPolygon) {
        context.beginPath()
        context.arc(x * bounds.width, y * bounds.height, 4, 0, Math.PI * 2)
        context.fillStyle = '#ffffff'
        context.fill()
      }
    }
    return
  }
  if (mode === 'ground') {
    renderRleMask(context, terrain.ground.mask, bounds.width, bounds.height, opacity, {
      0: [0, 0, 0, 0],
      1: [83, 190, 210, 255],
    })
    if (showManualLabels) drawManualPolygons()
    return
  }

  renderRleMask(context, terrain.traversability.mask, bounds.width, bounds.height, opacity, {
    0: [115, 124, 120, 245],
    1: [85, 217, 111, 245],
    2: [231, 200, 77, 245],
    3: [224, 91, 82, 245],
  })

  const selectedRegion = terrain.traversability.regions.find(region => region.region_id === selectedRegionId)
  if (selectedRegion && tracePolygon(context, selectedRegion.polygon, bounds.width, bounds.height)) {
    context.strokeStyle = '#ffffff'
    context.lineWidth = 4
    context.setLineDash([8, 5])
    context.stroke()
    context.setLineDash([])
  }

  const corridor = terrain.corridor
  const sourceMatches = Math.abs(corridor.source_frame_timestamp_ms - frame.timestamp_ms) <= 1
  if (!showCorridor || !sourceMatches || corridor.status === 'unavailable') {
    if (showManualLabels) drawManualPolygons()
    return
  }
  context.save()
  clipToEvaluatedDriveArea(context, terrain.traversability.mask, bounds.width, bounds.height)
  const colour = corridor.status === 'available' ? '#67e6f1' : '#f1cf58'
  if (tracePolygon(context, corridor.polygon, bounds.width, bounds.height)) {
    context.save()
    context.globalAlpha = Math.min(0.34, opacity)
    context.fillStyle = colour
    context.fill()
    context.restore()
    context.strokeStyle = colour
    context.lineWidth = 2
    if (corridor.status === 'uncertain') context.setLineDash([8, 6])
    context.stroke()
    context.setLineDash([])
  }
  if (corridor.centerline.length >= 2) {
    context.beginPath()
    context.moveTo(corridor.centerline[0][0] * bounds.width, corridor.centerline[0][1] * bounds.height)
    for (let index = 1; index < corridor.centerline.length; index++) context.lineTo(corridor.centerline[index][0] * bounds.width, corridor.centerline[index][1] * bounds.height)
    context.save()
    context.strokeStyle = colour
    context.lineWidth = 4
    context.shadowColor = '#07100b'
    context.shadowBlur = 5
    if (corridor.status === 'uncertain') context.setLineDash([9, 7])
    context.stroke()
    context.restore()
  }
  context.restore()
  if (showManualLabels) drawManualPolygons()
}

function confidenceWord(value: number) {
  return value >= 0.75 ? 'hohe Evidenz' : value >= 0.5 ? 'mittlere Evidenz' : 'geringe Evidenz'
}

function reasonText(reason: string) {
  return reasonLabels[reason] ?? reason.replaceAll('_', ' ')
}

export default function AnalysisView({mission, data, reconstruction, segmentation, onClose}: {mission: Mission; data: Analysis; reconstruction: Reconstruction; segmentation: Segmentation; onClose: () => void}) {
  const [active, setActive] = useState(reconstruction.traversals[0]?.video_id ?? '')
  const [time, setTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [aspect, setAspect] = useState(16 / 9)
  const [playing, setPlaying] = useState(false)
  const [mode, setMode] = useState<OverlayMode>('original')
  const [opacity, setOpacity] = useState(0.42)
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null)
  const [showCorridor, setShowCorridor] = useState(true)
  const [speed, setSpeed] = useState(1)
  const [overlayRevision, setOverlayRevision] = useState(0)
  const [annotationValues, setAnnotationValues] = useState<number[]>([])
  const [annotationValue, setAnnotationValue] = useState<GroundTruthValue>(1)
  const [annotationTool, setAnnotationTool] = useState<'brush' | 'polygon'>('polygon')
  const [brushSize, setBrushSize] = useState(4)
  const [annotationPolygon, setAnnotationPolygon] = useState<[number, number][]>([])
  const [annotationStatus, setAnnotationStatus] = useState<GroundTruthStatus | 'new'>('new')
  const [annotationRevision, setAnnotationRevision] = useState(0)
  const [annotationDirty, setAnnotationDirty] = useState(false)
  const [annotationLoading, setAnnotationLoading] = useState(false)
  const [annotationSaving, setAnnotationSaving] = useState(false)
  const [annotationMessage, setAnnotationMessage] = useState('')
  const [annotationNotes, setAnnotationNotes] = useState('')
  const [annotator, setAnnotator] = useState('Simon')
  const [annotationSummary, setAnnotationSummary] = useState<GroundTruthSummary | null>(null)
  const [showAiSuggestion, setShowAiSuggestion] = useState(true)
  const [showManualLabels, setShowManualLabels] = useState(false)
  const [manualOpacity, setManualOpacity] = useState(.3)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const drawingRef = useRef(false)
  const undoStackRef = useRef<number[][]>([])

  const traversal = reconstruction.traversals.find(item => item.video_id === active) ?? reconstruction.traversals[0]
  const result = traversal ? segmentation.videos.find(item => item.video_id === traversal.video_id) : undefined
  const coords = traversal?.geojson.coordinates.map(([lng, lat]) => ({lat, lng})) ?? []
  const frame = useMemo(() => frameAtOrBefore(result?.frames, time * 1000, (result?.analysis_interval_ms ?? 250) * 1.35), [result, time])
  const terrain = frame?.terrain && Math.abs(frame.terrain.source_frame_timestamp_ms - frame.timestamp_ms) <= 1 ? frame.terrain : undefined
  const selectedRegion = terrain?.traversability.regions.find(item => item.region_id === selectedRegionId)
  const progress = duration ? Math.min(1, time / duration) : 0
  const mapProgress = traversal?.direction === 'B_TO_A' ? 1 - progress : progress
  const playPoint = coords.length ? coords[Math.min(coords.length - 1, Math.round(mapProgress * (coords.length - 1)))] : undefined
  const terrainFrames = result?.frames.filter(item => item.terrain) ?? []
  const evidenceFrames = terrainFrames.filter(item => item.terrain?.evidence.representative)
  const vehicle = segmentation.vehicle_configuration
  const overallClass = terrain?.traversability.overall_class ?? 'unknown'
  const overallStyle = segmentation.terrain_ontology?.[overallClass] ?? terrainFallback[overallClass]
  const annotationWidth = terrain?.traversability.mask.width ?? 96
  const annotationHeight = terrain?.traversability.mask.height ?? Math.max(8, Math.round(annotationWidth / Math.max(.1, aspect)))
  const annotationMask = useMemo(
    () => encodeGroundTruthValues(annotationValues, annotationWidth, annotationHeight),
    [annotationValues, annotationWidth, annotationHeight],
  )
  const annotationCounts = useMemo(() => annotationValues.reduce((counts, value) => {
    if (value >= 0 && value <= 3) counts[value]++
    return counts
  }, [0, 0, 0, 0]), [annotationValues])
  const annotationLabelledFraction = annotationValues.length ? 1 - annotationCounts[0] / annotationValues.length : 0
  const manualLabels = useMemo(() => (annotationSummary?.items ?? []).filter(item => item.status !== 'skipped' && item.polygons?.length), [annotationSummary])
  const nearestManualLabel = useMemo(() => {
    if (!manualLabels.length) return undefined
    const timestamp = time * 1000
    const nearest = manualLabels.reduce((best, item) => Math.abs(item.timestamp_ms - timestamp) < Math.abs(best.timestamp_ms - timestamp) ? item : best)
    return Math.abs(nearest.timestamp_ms - timestamp) <= 180 ? nearest : undefined
  }, [manualLabels, time])
  const manualPolygons = useMemo(() => (nearestManualLabel?.polygons ?? []).map(polygon => polygon.points), [nearestManualLabel])

  useEffect(() => {
    const video = videoRef.current as VideoWithFrameCallback | null
    if (!video?.requestVideoFrameCallback) return
    let stopped = false
    let handle = 0
    let lastUpdate = -1
    const update = (_now: number, metadata: {mediaTime: number}) => {
      if (metadata.mediaTime - lastUpdate >= 1 / 20 || metadata.mediaTime < lastUpdate) {
        lastUpdate = metadata.mediaTime
        setTime(metadata.mediaTime)
      }
      if (!stopped) handle = video.requestVideoFrameCallback!(update)
    }
    handle = video.requestVideoFrameCallback(update)
    return () => {
      stopped = true
      if (video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(handle)
    }
  }, [active])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(() => setOverlayRevision(value => value + 1))
    observer.observe(canvas)
    return () => observer.disconnect()
  }, [active])

  useEffect(() => {
    let cancelled = false
    setAnnotationSummary(null)
    void listGroundTruth(mission.id, active, true)
      .then(summary => {if (!cancelled) setAnnotationSummary(summary)})
      .catch(() => {if (!cancelled) setAnnotationSummary(null)})
    return () => {cancelled = true}
  }, [mission.id, active])

  useEffect(() => {
    let cancelled = false
    setAnnotationPolygon([])
    undoStackRef.current = []
    setAnnotationMessage('')
    if (!frame || !terrain) {
      setAnnotationValues([])
      setAnnotationStatus('new')
      setAnnotationRevision(0)
      setAnnotationDirty(false)
      return () => {cancelled = true}
    }
    setAnnotationLoading(true)
    void getGroundTruth(mission.id, active, frame.frame_index)
      .then(annotation => {
        if (cancelled) return
        if (annotation) {
          setAnnotationValues(annotation.mask ? decodeRleValues(annotation.mask) : [])
          setAnnotationStatus(annotation.status)
          setAnnotationRevision(annotation.revision)
          setAnnotationNotes(annotation.notes)
          setAnnotator(annotation.annotator)
        } else {
          setAnnotationValues(new Array(terrain.traversability.mask.width * terrain.traversability.mask.height).fill(0))
          setAnnotationStatus('new')
          setAnnotationRevision(0)
          setAnnotationNotes('')
        }
        setAnnotationDirty(false)
      })
      .catch(error => {
        if (!cancelled) setAnnotationMessage(error instanceof Error ? error.message : 'Ground Truth konnte nicht geladen werden')
      })
      .finally(() => {if (!cancelled) setAnnotationLoading(false)})
    return () => {cancelled = true}
  }, [mission.id, active, frame?.frame_index, terrain?.source_frame_hash])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    drawOverlay({canvas, mode, frame, terrain, opacity, selectedRegionId, showCorridor, annotationMask, annotationPolygon, annotationValue, showAiSuggestion, manualPolygons, manualOpacity, showManualLabels})
  }, [mode, frame, terrain, opacity, selectedRegionId, showCorridor, annotationMask, annotationPolygon, annotationValue, showAiSuggestion, manualPolygons, manualOpacity, showManualLabels, overlayRevision])

  const choose = (videoId: string) => {
    if (mode === 'annotation' && annotationDirty) {
      setAnnotationMessage('Speichere oder bestätige zuerst die aktuelle Markierung.')
      return
    }
    videoRef.current?.pause()
    setActive(videoId)
    setTime(0)
    setPlaying(false)
    setSelectedRegionId(null)
  }
  const seek = (value: number) => {
    const video = videoRef.current
    if (video) video.currentTime = value
    setTime(value)
  }
  const seekManualLabel = (frameIndex: number) => {
    const label = manualLabels.find(item => item.frame_index === frameIndex)
    if (!label) return
    videoRef.current?.pause()
    setPlaying(false)
    seek(label.timestamp_ms / 1000)
  }
  const toggle = () => {
    const video = videoRef.current
    if (!video) return
    video.paused ? void video.play() : video.pause()
  }
  const showEvidence = (target: SegmentationFrame) => {
    videoRef.current?.pause()
    setMode('traversability')
    setSelectedRegionId(null)
    seek(target.timestamp_ms / 1000)
  }
  const stepAnalysisFrame = (direction: -1 | 1, allowDirty = false) => {
    if (mode === 'annotation' && annotationDirty && !allowDirty) {
      setAnnotationMessage('Speichere oder bestätige zuerst die aktuelle Markierung.')
      return
    }
    if (!result?.frames.length) return
    const currentIndex = frame ? result.frames.indexOf(frame) : 0
    const next = result.frames[Math.max(0, Math.min(result.frames.length - 1, currentIndex + direction))]
    if (next) seek(next.timestamp_ms / 1000)
  }
  const rememberAnnotation = () => {
    undoStackRef.current = [...undoStackRef.current.slice(-19), [...annotationValues]]
  }
  const annotationPoint = (event: ReactPointerEvent<HTMLCanvasElement>): [number, number] => {
    const bounds = event.currentTarget.getBoundingClientRect()
    return [
      Math.max(0, Math.min(1, (event.clientX - bounds.left) / Math.max(1, bounds.width))),
      Math.max(0, Math.min(1, (event.clientY - bounds.top) / Math.max(1, bounds.height))),
    ]
  }
  const paintAnnotation = (point: [number, number]) => {
    const centerX = Math.round(point[0] * (annotationWidth - 1))
    const centerY = Math.round(point[1] * (annotationHeight - 1))
    setAnnotationValues(current => {
      const next = current.length === annotationWidth * annotationHeight ? [...current] : new Array(annotationWidth * annotationHeight).fill(0)
      for (let y = Math.max(0, centerY - brushSize); y <= Math.min(annotationHeight - 1, centerY + brushSize); y++) {
        for (let x = Math.max(0, centerX - brushSize); x <= Math.min(annotationWidth - 1, centerX + brushSize); x++) {
          if ((x - centerX) ** 2 + (y - centerY) ** 2 <= brushSize ** 2) next[y * annotationWidth + x] = annotationValue
        }
      }
      return next
    })
    setAnnotationDirty(true)
    setAnnotationStatus(old => old === 'confirmed' ? 'draft' : old)
  }
  const handleAnnotationPointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (mode !== 'annotation' || !terrain || annotationLoading) return
    videoRef.current?.pause()
    const point = annotationPoint(event)
    if (annotationTool === 'polygon') {
      setAnnotationPolygon(current => [...current, point])
      return
    }
    rememberAnnotation()
    drawingRef.current = true
    event.currentTarget.setPointerCapture(event.pointerId)
    paintAnnotation(point)
  }
  const handleAnnotationPointerMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (mode === 'annotation' && annotationTool === 'brush' && drawingRef.current) paintAnnotation(annotationPoint(event))
  }
  const handleAnnotationPointerUp = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    drawingRef.current = false
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
  }
  const closeAnnotationPolygon = () => {
    if (annotationPolygon.length < 3) {
      setAnnotationMessage('Setze mindestens drei Polygonpunkte.')
      return
    }
    rememberAnnotation()
    const raster = document.createElement('canvas')
    raster.width = annotationWidth
    raster.height = annotationHeight
    const context = raster.getContext('2d')
    if (!context) return
    context.beginPath()
    context.moveTo(annotationPolygon[0][0] * annotationWidth, annotationPolygon[0][1] * annotationHeight)
    for (let index = 1; index < annotationPolygon.length; index++) context.lineTo(annotationPolygon[index][0] * annotationWidth, annotationPolygon[index][1] * annotationHeight)
    context.closePath()
    context.fillStyle = '#ffffff'
    context.fill()
    const pixels = context.getImageData(0, 0, annotationWidth, annotationHeight).data
    setAnnotationValues(current => {
      const next = current.length === annotationWidth * annotationHeight ? [...current] : new Array(annotationWidth * annotationHeight).fill(0)
      for (let index = 0; index < next.length; index++) if (pixels[index * 4 + 3] > 0) next[index] = annotationValue
      return next
    })
    setAnnotationPolygon([])
    setAnnotationDirty(true)
    setAnnotationStatus(old => old === 'confirmed' ? 'draft' : old)
    setAnnotationMessage('Polygon übernommen.')
  }
  const undoAnnotation = () => {
    const previous = undoStackRef.current.at(-1)
    if (!previous) return
    undoStackRef.current = undoStackRef.current.slice(0, -1)
    setAnnotationValues(previous)
    setAnnotationPolygon([])
    setAnnotationDirty(true)
    setAnnotationMessage('Letzte Änderung rückgängig gemacht.')
  }
  const useAiSuggestion = () => {
    if (!terrain) return
    rememberAnnotation()
    const machine = decodeRleValues(terrain.traversability.mask)
    setAnnotationValues(machine.map(value => value === 1 ? 1 : value === 3 ? 2 : value === 0 ? 3 : 0))
    setAnnotationDirty(true)
    setAnnotationStatus('draft')
    setAnnotationPolygon([])
    setAnnotationMessage('KI-Vorschlag übernommen. Gelb bleibt absichtlich unmarkiert und muss von dir entschieden werden.')
  }
  const clearAnnotation = () => {
    rememberAnnotation()
    setAnnotationValues(new Array(annotationWidth * annotationHeight).fill(0))
    setAnnotationPolygon([])
    setAnnotationDirty(true)
    setAnnotationStatus('draft')
    setAnnotationMessage('Maske geleert. Unmarkierte Pixel werden im Training ignoriert.')
  }
  const persistAnnotation = async (status: GroundTruthStatus, advance: boolean) => {
    if (!frame || !terrain || annotationSaving) return
    setAnnotationSaving(true)
    setAnnotationMessage('Ground Truth wird gespeichert …')
    try {
      const saved = await saveGroundTruth(mission.id, active, frame.frame_index, {
        timestamp_ms: frame.timestamp_ms,
        source_frame_hash: terrain.source_frame_hash,
        mask: annotationMask,
        status,
        annotator: annotator.trim() || 'Simon',
        notes: annotationNotes,
      })
      setAnnotationStatus(saved.status)
      setAnnotationRevision(saved.revision)
      setAnnotationDirty(false)
      setAnnotationMessage(status === 'confirmed' ? `Frame ${frame.frame_index} als Ground Truth bestätigt.` : `Entwurf für Frame ${frame.frame_index} gespeichert.`)
      const summary = await listGroundTruth(mission.id, active)
      setAnnotationSummary(summary)
      if (advance) stepAnalysisFrame(1, true)
    } catch (error) {
      setAnnotationMessage(error instanceof Error ? error.message : 'Ground Truth konnte nicht gespeichert werden')
    } finally {
      setAnnotationSaving(false)
    }
  }
  const handleOverlayClick = (event: ReactMouseEvent<HTMLCanvasElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect()
    const x = (event.clientX - bounds.left) / Math.max(1, bounds.width)
    const y = (event.clientY - bounds.top) / Math.max(1, bounds.height)
    if (mode === 'traversability' && terrain) {
      const region = [...terrain.traversability.regions]
        .sort((left, right) => riskPriority[right.class_id] - riskPriority[left.class_id])
        .find(item => pointInPolygon(x, y, item.polygon))
      setSelectedRegionId(region?.region_id ?? null)
    }
  }

  useEffect(() => {
    const keydown = (event: KeyboardEvent) => {
      if (mode !== 'annotation' || ['INPUT', 'TEXTAREA', 'SELECT'].includes((event.target as HTMLElement)?.tagName)) return
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
        event.preventDefault()
        undoAnnotation()
      } else if (event.key === '1') setAnnotationValue(1)
      else if (event.key === '2') setAnnotationValue(2)
      else if (event.key === '3') setAnnotationValue(3)
      else if (event.key === '0' || event.key.toLowerCase() === 'e') setAnnotationValue(0)
      else if (event.key.toLowerCase() === 'b') setAnnotationTool('brush')
      else if (event.key.toLowerCase() === 'p') setAnnotationTool('polygon')
      else if (event.key === 'ArrowLeft') {event.preventDefault(); stepAnalysisFrame(-1)}
      else if (event.key === 'ArrowRight') {event.preventDefault(); stepAnalysisFrame(1)}
    }
    window.addEventListener('keydown', keydown)
    return () => window.removeEventListener('keydown', keydown)
  })

  if (!traversal || !result || !coords.length || !playPoint) {
    return <div className="analysis-page"><div className="alert error">Für diese Mission fehlen Video-, Routen- oder Goal-4-Daten.</div><button onClick={onClose}>← Zurück</button></div>
  }

  return <div className="analysis-page">
    <div className="analysis-top">
      <button onClick={onClose}>← Zurück zum Upload</button>
      <div><span className="eyebrow">GOAL 4 · AUTOMATISCHE AUSWERTUNG</span><h1>{mission.name}</h1></div>
      <a href={`/api/v1/missions/${mission.id}/segmentation/report`} target="_blank" rel="noreferrer">Analysebericht ↗</a>
    </div>
    <div className="metric-grid">
      <Metric label="Bestätigte Ground Truths" value={String(annotationSummary?.counts.confirmed ?? 0)}/>
      <Metric label="Gespeicherte Entwürfe" value={String(annotationSummary?.counts.draft ?? 0)}/>
      <Metric label="Terrainframes im Video" value={String(terrainFrames.length)}/>
      <Metric label="Aktuell markiert" value={percent(annotationLabelledFraction)}/>
      <Metric label="Evidenzframes" value={String(evidenceFrames.length)}/>
      <Metric label="Benötigte ARGUS-Breite" value={vehicle ? `${n(vehicle.required_width_m, 2)} m` : 'nicht konfiguriert'}/>
    </div>
    <div className="player-tabs">{reconstruction.traversals.map((item, index) => <button key={item.video_id} className={item.video_id === traversal.video_id ? 'active' : ''} onClick={() => choose(item.video_id)}>VIDEO {index + 1}<small>{item.direction === 'A_TO_B' ? 'A → B' : 'B → A'}</small></button>)}</div>
    <section className="goal4-player">
      <div className="player-main">
        <div className="video-stage" style={{aspectRatio: String(aspect), width: `min(100%, ${Math.max(0.1, aspect) * 72}vh)`}}>
          <video
            ref={videoRef}
            src={`/api/v1/missions/${mission.id}/videos/${traversal.video_id}/content`}
            preload="metadata"
            playsInline
            onLoadedMetadata={event => {
              setDuration(event.currentTarget.duration)
              setAspect(event.currentTarget.videoWidth / Math.max(1, event.currentTarget.videoHeight))
              event.currentTarget.playbackRate = speed
            }}
            onTimeUpdate={event => setTime(event.currentTarget.currentTime)}
            onSeeked={event => setTime(event.currentTarget.currentTime)}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onEnded={() => setPlaying(false)}
          />
          <canvas ref={canvasRef} className={`detection-layer ${mode === 'original' ? 'original' : ''} ${mode === 'annotation' ? 'annotation-active' : ''}`} onClick={handleOverlayClick} onPointerDown={handleAnnotationPointerDown} onPointerMove={handleAnnotationPointerMove} onPointerUp={handleAnnotationPointerUp} onPointerCancel={handleAnnotationPointerUp} onDoubleClick={() => mode === 'annotation' && annotationTool === 'polygon' && closeAnnotationPolygon()} role="img" aria-label="Synchrones Analyse- und Ground-Truth-Overlay"/>
          {(mode === 'ground' || mode === 'traversability' || mode === 'annotation') && !terrain && <div className="unassessable-overlay"><b>Nicht bewertbar</b><span>Kein zeitnah aus diesem Videoframe berechnetes Terrain-Ergebnis.</span></div>}
          <div className="video-badge">
            <i className={playing ? 'live' : ''}/>{playing ? 'PLAYBACK' : 'PAUSIERT'} ·{' '}
            {mode === 'original' && 'ORIGINALVIDEO · KEIN OVERLAY'}
            {mode === 'ground' && (terrain ? `BODEN ${percent(terrain.ground.visible_ratio)} · KI-EVIDENZ ${percent(terrain.ground.confidence)}` : 'BODEN NICHT BEWERTBAR')}
            {mode === 'traversability' && (terrain ? `${overallStyle.label.toUpperCase()} · KI-EVIDENZ ${percent(terrain.traversability.overall_confidence)}` : 'NICHT BEWERTBAR')}
            {mode === 'labels' && (nearestManualLabel ? `EIGENES LABEL · FRAME ${nearestManualLabel.frame_index + 1} · ${nearestManualLabel.status.toUpperCase()}` : 'KEIN EIGENES LABEL AN DIESEM FRAME')}
            {mode === 'annotation' && `${annotationStatus === 'confirmed' ? 'GROUND TRUTH BESTÄTIGT' : annotationDirty ? 'UNGESPEICHERTE ÄNDERUNG' : annotationStatus === 'draft' ? 'ENTWURF' : 'NEUER FRAME'} · ${percent(annotationLabelledFraction)} MARKIERT`}
            {frame && ` · FRAME ${frame.frame_index} · Δ ${Math.max(0, Math.round(time * 1000 - frame.timestamp_ms))} ms`}
          </div>
        </div>
        <div className="transport">
          <button onClick={toggle} aria-label={playing ? 'Video pausieren' : 'Video abspielen'}>{playing ? 'Ⅱ' : '▶'}</button>
          <span>{clock(time)}</span>
          <input aria-label="Videoposition" type="range" min="0" max={duration || 1} step="0.01" value={time} onChange={event => seek(+event.target.value)}/>
          <span>{clock(duration)}</span>
          <select aria-label="Geschwindigkeit" value={speed} onChange={event => {const value = +event.target.value; setSpeed(value); if (videoRef.current) videoRef.current.playbackRate = value}}>{[0.5, 1, 1.5, 2].map(value => <option key={value} value={value}>{value}×</option>)}</select>
        </div>
      </div>
      <aside className="player-controls">
        <h2>Darstellung</h2>
        <div className="mode-switch">{modes.map(item => <button key={item.id} className={mode === item.id ? 'active' : ''} aria-pressed={mode === item.id} onClick={() => {setMode(item.id); setSelectedRegionId(null); if (item.id === 'annotation') videoRef.current?.pause()}}>{item.label}</button>)}</div>
        {(mode === 'ground' || mode === 'traversability') && <label>KI-Overlay-Deckkraft · {Math.round(opacity * 100)} %<input type="range" min="0.1" max="0.9" step="0.05" value={opacity} onChange={event => setOpacity(+event.target.value)}/></label>}

        {(mode === 'ground' || mode === 'traversability') && <label className="corridor-toggle"><input type="checkbox" checked={showManualLabels} onChange={event => setShowManualLabels(event.target.checked)}/>Eigene Labels zusätzlich überlagern</label>}
        {(mode === 'labels' || showManualLabels) && <label>Eigene-Label-Deckkraft · {Math.round(manualOpacity * 100)} %<input aria-label="Eigene-Label-Deckkraft" type="range" min="0.05" max="0.9" step="0.05" value={manualOpacity} onChange={event => setManualOpacity(+event.target.value)}/></label>}

        {mode === 'original' && <div className="detection-detail"><b>Unverändertes Originalvideo</b><span>Keine Analysemaske und keine Fahrbewertung eingeblendet.</span></div>}

        {mode === 'labels' && <div className="manual-label-viewer">
          <h2>Eigene Ground Truth</h2>
          {manualLabels.length ? <><label>Gespeichertes Label auswählen<select aria-label="Gespeichertes eigenes Label" value={nearestManualLabel?.frame_index ?? ''} onChange={event => seekManualLabel(+event.target.value)}><option value="">Label auswählen …</option>{manualLabels.map(item => <option key={item.frame_index} value={item.frame_index}>Frame {item.frame_index + 1} · {clock(item.timestamp_ms / 1000)} · {item.status === 'confirmed' ? 'bestätigt' : 'Entwurf'}</option>)}</select></label>{nearestManualLabel ? <div className="detection-detail"><b>Manuelle Polygonmaske</b><span>Frame {nearestManualLabel.frame_index + 1} · {clock(nearestManualLabel.timestamp_ms / 1000)}</span><span>{nearestManualLabel.statistics.polygon_count ?? nearestManualLabel.polygons?.length ?? 0} Polygon · {nearestManualLabel.statistics.point_count ?? 0} Punkte</span><small>Die weiße Kontur unterscheidet dein Label von der automatischen Maske.</small></div> : <div className="terrain-summary unknown"><b>Zwischen zwei Labels</b><span>Wähle ein gespeichertes Label aus oder spiele bis zu einem gelabelten Frame.</span></div>}</> : <div className="terrain-summary unknown"><b>Noch keine eigenen Labels</b><span>Erstelle zuerst Polygonmasken im manuellen Labeling-Modus.</span></div>}
        </div>}

        {mode === 'annotation' && <div className="ground-truth-editor">
          <div className={`ground-truth-state ${annotationStatus} ${annotationDirty ? 'dirty' : ''}`}><b>{annotationLoading ? 'Maske wird geladen …' : annotationDirty ? 'Ungespeicherte Änderung' : annotationStatus === 'confirmed' ? `Bestätigte Ground Truth · Revision ${annotationRevision}` : annotationStatus === 'draft' ? `Gespeicherter Entwurf · Revision ${annotationRevision}` : 'Noch nicht markiert'}</b><span>{frame ? `Frame ${frame.frame_index} · ${clock(frame.timestamp_ms / 1000)}` : 'Kein Analyseframe'}</span></div>
          <h2>1 · Klasse wählen</h2>
          <div className="ground-truth-labels">{([1, 2, 3, 0] as GroundTruthValue[]).map(value => <button key={value} className={annotationValue === value ? 'active' : ''} style={{'--label-color': groundTruthStyles[value].color} as CSSProperties} onClick={() => setAnnotationValue(value)}><i/>{groundTruthStyles[value].label}<small>{value === 1 ? 'Taste 1' : value === 2 ? 'Taste 2' : value === 3 ? 'Taste 3' : 'Taste 0/E'}</small></button>)}</div>
          <h2>2 · Werkzeug</h2>
          <div className="annotation-tool-switch"><button className={annotationTool === 'polygon' ? 'active' : ''} onClick={() => setAnnotationTool('polygon')}>Polygon · P</button><button className={annotationTool === 'brush' ? 'active' : ''} onClick={() => setAnnotationTool('brush')}>Pinsel · B</button></div>
          {annotationTool === 'polygon' ? <div className="polygon-controls"><span>{annotationPolygon.length} Punkte · in das Video klicken</span><button disabled={annotationPolygon.length < 3} onClick={closeAnnotationPolygon}>Polygon schließen</button><button disabled={!annotationPolygon.length} onClick={() => setAnnotationPolygon([])}>Punkte abbrechen</button></div> : <label>Pinselgröße · {brushSize} Rasterpixel<input type="range" min="1" max="14" step="1" value={brushSize} onChange={event => setBrushSize(+event.target.value)}/></label>}
          <div className="annotation-actions"><button disabled={!undoStackRef.current.length} onClick={undoAnnotation}>↶ Rückgängig</button><button disabled={!terrain} onClick={useAiSuggestion}>KI-Vorschlag übernehmen</button><button onClick={clearAnnotation}>Alles leeren</button></div>
          <label className="corridor-toggle"><input type="checkbox" checked={showAiSuggestion} onChange={event => setShowAiSuggestion(event.target.checked)}/>KI-Maske schwach im Hintergrund zeigen</label>
          <div className="annotation-counts"><span style={{borderColor: groundTruthStyles[1].color}}>Grün <b>{percent(annotationCounts[1] / Math.max(1, annotationValues.length))}</b></span><span style={{borderColor: groundTruthStyles[2].color}}>Rot <b>{percent(annotationCounts[2] / Math.max(1, annotationValues.length))}</b></span><span style={{borderColor: groundTruthStyles[3].color}}>Grau <b>{percent(annotationCounts[3] / Math.max(1, annotationValues.length))}</b></span><span>Unmarkiert <b>{percent(annotationCounts[0] / Math.max(1, annotationValues.length))}</b></span></div>
          <label>Annotator<input value={annotator} maxLength={80} onChange={event => setAnnotator(event.target.value)}/></label>
          <label>Notiz zum Frame<textarea value={annotationNotes} maxLength={1000} placeholder="Optional: warum ist der Bereich befahrbar oder unsicher?" onChange={event => setAnnotationNotes(event.target.value)}/></label>
          {annotationMessage && <div className="annotation-message" role="status">{annotationMessage}</div>}
          <div className="annotation-save"><button disabled={annotationSaving || !terrain} onClick={() => void persistAnnotation('draft', false)}>Entwurf speichern</button><button disabled={annotationSaving || !terrain || annotationCounts[1] + annotationCounts[2] + annotationCounts[3] === 0} onClick={() => void persistAnnotation('confirmed', false)}>Bestätigen</button><button className="primary" disabled={annotationSaving || !terrain || annotationCounts[1] + annotationCounts[2] + annotationCounts[3] === 0} onClick={() => void persistAnnotation('confirmed', true)}>Bestätigen &amp; nächster Frame →</button></div>
          <small className="annotation-shortcuts">Schnell: 1 Grün · 2 Rot · 3 Grau · 0 Radierer · P Polygon · B Pinsel · Strg+Z · ←/→ Frame</small>
        </div>}

        {mode === 'ground' && <>
          <h2>Bodenmaske</h2>
          {terrain ? <div className="terrain-summary ground"><b>Aus aktuellem Videoframe berechnet</b><span>Sichtbare Bodenfläche {percent(terrain.ground.visible_ratio)}</span><span>KI-Evidenz {percent(terrain.ground.confidence)} · {confidenceWord(terrain.ground.confidence)}</span><span>Quelle: {terrain.ground.source === 'current_video_frame_inference' ? 'aktuelles Videobild' : terrain.ground.source}</span><small>Frame-Hash {terrain.source_frame_hash}</small></div> : <div className="terrain-summary unknown"><b>Nicht bewertbar</b><span>Für den aktuellen Zeitpunkt liegt keine Bodenmaske vor.</span></div>}
        </>}

        {mode === 'traversability' && <>
          <h2>Befahrbarkeitsklassen</h2>
          <div className="terrain-legend">{terrainOrder.map(classId => {const item = segmentation.terrain_ontology?.[classId] ?? terrainFallback[classId]; return <div key={classId}><i style={{background: item.color}}/><span>{item.label}</span><b>{terrain ? percent(terrain.traversability.class_coverage[classId] ?? 0) : '–'}</b></div>})}</div>
          <label className="corridor-toggle"><input type="checkbox" checked={showCorridor} onChange={event => setShowCorridor(event.target.checked)}/>Fahrkorridor und Mittellinie anzeigen</label>
          {terrain ? <>
            <div className={`terrain-summary ${overallClass}`}><b>{overallStyle.label}</b><span>KI-Evidenz {percent(terrain.traversability.overall_confidence)} · {confidenceWord(terrain.traversability.overall_confidence)}</span><span>Grauanteil {percent(terrain.quality.unknown_ratio)}</span></div>
            <div className={`corridor-status ${terrain.corridor.status}`}><b>{terrain.corridor.status === 'available' ? 'Korridorvorschlag verfügbar' : terrain.corridor.status === 'uncertain' ? 'Korridorvorschlag unsicher' : 'Kein belastbarer Korridor'}</b><span>Konfidenz {percent(terrain.corridor.confidence)}</span><span>{terrain.corridor.minimum_width_m == null ? `Mindestbreite nur relativ: ${n(terrain.corridor.minimum_width_ratio, 2)}×` : `Geschätzte Mindestbreite ${n(terrain.corridor.minimum_width_m, 2)} m`}</span><span>Grüne Bildstützung {percent(terrain.corridor.green_support_fraction ?? 0)}</span><span>{terrain.corridor.stable_frames} stabile Frames · Sprung {n(terrain.corridor.stability_px, 1)} px</span>{terrain.corridor.reasons.map(item => <small key={item}>{reasonText(item)}</small>)}</div>
            {vehicle && <div className="vehicle-readout"><b>ARGUS-Konfiguration</b><span>Fahrzeug {n(vehicle.width_m, 2)} m + je {n(vehicle.safety_margin_per_side_m, 2)} m Rand</span><span>Erforderlich {n(vehicle.required_width_m, 2)} m</span>{vehicle.source !== 'environment' && vehicle.source !== 'configured' && <small>Dokumentierte Arbeitsannahme – Breite und Kameraskala vor Einsatz am realen Fahrzeug bestätigen.</small>}</div>}
            {selectedRegion && <div className="detection-detail"><b>{(segmentation.terrain_ontology?.[selectedRegion.class_id] ?? terrainFallback[selectedRegion.class_id]).label}</b><span>Region {selectedRegion.region_id}</span><span>KI-Evidenz {percent(selectedRegion.confidence)}</span>{selectedRegion.reasons.map(item => <small key={item}>{reasonText(item)}</small>)}</div>}
            <h2 className="factor-heading">Bewertungsfaktoren</h2>
            <div className="factor-bars">{factorLabels.map(([key, label]) => <Factor key={key} label={label} value={terrain.factors[key]}/>)}</div>
            <div className="quality-readout"><span>Unschärferisiko <b>{percent(terrain.quality.blur_score)}</b></span><span>Belichtung nutzbar <b>{percent(terrain.quality.exposure_score)}</b></span><span>Motion-Inlier <b>{terrain.quality.motion_inliers}</b></span></div>
          </> : <div className="terrain-summary unknown"><b>Nicht bewertbar</b><span>Unsichere oder fehlende Daten werden niemals automatisch grün markiert.</span></div>}
        </>}
      </aside>
    </section>

    <div className="safety-boundary" role="note"><b>Keine Fahrfreigabe</b><span>Diese Darstellung ist ausschließlich eine KI-gestützte Geländeeinschätzung. Verdeckte oder unsichere Bereiche gelten nicht automatisch als befahrbar. Einsatzentscheidung und Vor-Ort-Prüfung bleiben erforderlich.</span></div>

    <section className="terrain-evidence">
      <div className="section-head"><h2>Repräsentative Evidenzframes</h2><p>Jede Karte stammt aus einem berechneten Videoframe. Anklicken pausiert das Video und springt exakt zur zugehörigen Maske.</p></div>
      <div className="evidence-frame-nav"><button onClick={() => stepAnalysisFrame(-1)}>← Vorheriger Analyseframe</button><span>{frame ? `Frame ${frame.frame_index} · ${clock(frame.timestamp_ms / 1000)}` : 'Kein Analyseframe gewählt'}</span><button onClick={() => stepAnalysisFrame(1)}>Nächster Analyseframe →</button></div>
      {evidenceFrames.length ? <div className="terrain-evidence-grid">{evidenceFrames.map(item => {const evaluation = item.terrain!; const classId = evaluation.traversability.overall_class; const style = segmentation.terrain_ontology?.[classId] ?? terrainFallback[classId]; const imageUrl = evaluation.evidence.overlay_url ?? evaluation.evidence.image_url; return <button key={`${item.video_id}-${item.frame_index}`} className={frame?.frame_index === item.frame_index ? 'active' : ''} onClick={() => showEvidence(item)}>{imageUrl ? <img src={imageUrl} alt={`Evidenzframe ${item.frame_index}`}/> : <div className="evidence-placeholder" style={{borderColor: style.color}}>Kein Vorschaubild</div>}<span><i style={{background: style.color}}/>{style.label}</span><b>Frame {item.frame_index} · {clock(item.timestamp_ms / 1000)}</b><small>KI-Evidenz {percent(evaluation.traversability.overall_confidence)}</small>{evaluation.evidence.reasons.slice(0, 2).map(reason => <small key={reason}>{reasonText(reason)}</small>)}</button>})}</div> : <div className="empty terrain-empty">Für dieses Video sind noch keine repräsentativen Terrain-Evidenzframes hinterlegt. Ältere Analyseläufe bleiben abspielbar, müssen für die neue Bodenanalyse aber erneut verarbeitet werden.</div>}
    </section>

    <section className="sync-map"><div className="section-head"><h2>Synchronisierte Strecke</h2><p>Die Position folgt der echten Videozeit; Gegenrichtung B → A wird auf der Karte automatisch umgekehrt.</p></div><MapContainer key={traversal.video_id} center={coords[0]} zoom={18} className="map"><TileLayer attribution="&copy; OpenStreetMap" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"/><Polyline positions={coords} pathOptions={{color: '#65c9d5', weight: 6}}/><CircleMarker center={coords[0]} radius={8} pathOptions={{color: '#fff', fillColor: '#d7f26b', fillOpacity: 1}}/><CircleMarker center={coords.at(-1)!} radius={8} pathOptions={{color: '#fff', fillColor: '#d7f26b', fillOpacity: 1}}/><CircleMarker center={playPoint} radius={11} pathOptions={{color: '#fff', fillColor: '#4de0ed', fillOpacity: 1}}/></MapContainer></section>
    <div className="truth-strip"><b>Ground-Truth-Regeln</b><span>Nur deine gespeicherten Markierungen werden Trainingslabels.</span><span>Unmarkierte Pixel werden ignoriert, nicht als Hindernis interpretiert.</span><span>KI-Vorschläge bleiben Entwürfe, bis du sie bestätigst.</span><span>Grau bedeutet ausdrücklich nicht bewertbar.</span><span>{data.keyframes.length} ursprüngliche Evidenz-Keyframes bleiben erhalten.</span></div>
  </div>
}

function Metric({label, value}: {label: string; value: string}) {
  return <div className="metric"><span>{label}</span><b>{value}</b></div>
}

function Factor({label, value}: {label: string; value: number}) {
  return <div><span>{label}</span><i><b style={{width: percent(value)}}/></i><small>{percent(value)}</small></div>
}
