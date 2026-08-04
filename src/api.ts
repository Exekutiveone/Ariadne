import type {
  Analysis,
  CorridorId,
  GlobalModelDashboardData,
  GlobalPathModelResult,
  GlobalVideoAnalysisResult,
  GlobalVideoAnalysisStatus,
  GroundTruthAnnotation,
  GroundTruthPolygon,
  GroundTruthStatus,
  GroundTruthSummary,
  LabelingVideoManifest,
  LabelOntology,
  LabelTracks,
  Mission,
  OffPathInterval,
  RoiProfile,
  PathModelResult,
  PathPrediction,
  PathTrainingJob,
  Point,
  Reconstruction,
  RegistryListing,
  RegistryRun,
  RunStatus,
  Segmentation,
  StoredTrajectory,
  TerrainDashboardData,
  TerrainFramePrediction,
  TerrainMask,
  TerrainModelResult,
  TerrainPredictionRun,
  VideoInput,
} from './types'
export type Survey = {
  name: string
  start: Point
  end: Point
  route: Point[]
  movement_start?: string
  movement_end?: string
  pauses: {start_seconds: number; end_seconds: number; note: string}[]
  notes: string
}
export function uploadMission(survey: Survey, videos: VideoInput[], onProgress: (value: number) => void): Promise<Mission> {
  return new Promise((resolve, reject) => {
    const data = new FormData()
    data.append('survey', JSON.stringify(survey))
    data.append(
      'video_metadata',
      JSON.stringify(
        videos.map(({direction, orientation, terrainCategory}) => ({
          direction,
          orientation,
          terrain_category: terrainCategory || undefined,
        })),
      ),
    )
    videos.forEach(v => data.append('videos', v.file))
    const xhr = new XMLHttpRequest()
    xhr.open('POST', '/api/v1/missions')
    xhr.upload.onprogress = e => e.lengthComputable && onProgress(Math.round((e.loaded / e.total) * 100))
    xhr.onload = () => {
      let body: any = {}
      try {
        body = JSON.parse(xhr.responseText)
      } catch {}
      if (xhr.status === 201) resolve(body)
      else reject(new Error(body.detail || 'Upload fehlgeschlagen'))
    }
    xhr.onerror = () => reject(new Error('Server nicht erreichbar. Die Mission wurde nicht verändert.'))
    xhr.send(data)
  })
}
export async function updateVideoTerrainCategory(
  missionId: string,
  videoId: string,
  payload: {terrain_category: string | null},
): Promise<{id: string; original_name: string; terrain_category?: string | null; fully_not_traversable?: boolean}> {
  const r = await fetch(`/api/v1/missions/${missionId}/videos/${videoId}`, {
    method: 'PATCH',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  })
  let body: any = {}
  try {
    body = await r.json()
  } catch {}
  if (!r.ok) throw new Error(body.detail || 'Terrainkategorie konnte nicht gespeichert werden')
  return body
}
export async function updateVideoFullyNotTraversable(
  missionId: string,
  videoId: string,
  fullyNotTraversable: boolean,
): Promise<{id: string; original_name: string; terrain_category?: string | null; fully_not_traversable?: boolean}> {
  // Nur dieses eine Feld im Body: das Backend aendert ausschliesslich
  // mitgeschickte Felder, eine reine Terrainkategorie bleibt unberuehrt.
  const r = await fetch(`/api/v1/missions/${missionId}/videos/${videoId}`, {
    method: 'PATCH',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({fully_not_traversable: fullyNotTraversable}),
  })
  let body: any = {}
  try {
    body = await r.json()
  } catch {}
  if (!r.ok) throw new Error(body.detail || 'Video-Komplettlabel konnte nicht gespeichert werden')
  return body
}
export async function listOffPathIntervals(missionId: string, videoId: string): Promise<OffPathInterval[]> {
  const r = await fetch(`/api/v1/missions/${missionId}/off-path-intervals/${videoId}`)
  if (!r.ok) throw new Error('Off-Path-Intervalle konnten nicht geladen werden')
  return r.json()
}
export async function createOffPathInterval(
  missionId: string,
  videoId: string,
  payload: {start_ms: number; end_ms: number; note: string; annotator: string},
): Promise<OffPathInterval> {
  const r = await fetch(`/api/v1/missions/${missionId}/off-path-intervals/${videoId}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  })
  let body: any = {}
  try {
    body = await r.json()
  } catch {}
  if (!r.ok) throw new Error(body.detail || 'Intervall konnte nicht gespeichert werden')
  return body
}
export async function deleteOffPathInterval(missionId: string, videoId: string, intervalId: string): Promise<void> {
  const r = await fetch(`/api/v1/missions/${missionId}/off-path-intervals/${videoId}/${intervalId}`, {method: 'DELETE'})
  if (!r.ok && r.status !== 404) throw new Error('Intervall konnte nicht gelöscht werden')
}
export async function getTrajectory(missionId: string, videoId: string, frameIndex: number): Promise<StoredTrajectory | null> {
  const r = await fetch(`/api/v1/missions/${missionId}/trajectories/${videoId}/${frameIndex}`)
  if (r.status === 404) return null
  if (!r.ok) throw new Error('Trajektorie konnte nicht geladen werden')
  return r.json()
}
export async function saveTrajectory(
  missionId: string,
  videoId: string,
  frameIndex: number,
  payload: {
    timestamp_ms: number
    points: number[][]
    corridor: CorridorId | null
    origin: 'model_proposal' | 'manual_edit' | 'manual'
    note: string
    annotator: string
  },
): Promise<StoredTrajectory> {
  const r = await fetch(`/api/v1/missions/${missionId}/trajectories/${videoId}/${frameIndex}`, {
    method: 'PUT',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  })
  let body: any = {}
  try {
    body = await r.json()
  } catch {}
  if (!r.ok) throw new Error(body.detail || 'Trajektorie konnte nicht gespeichert werden')
  return body
}
export async function deleteTrajectory(missionId: string, videoId: string, frameIndex: number): Promise<void> {
  const r = await fetch(`/api/v1/missions/${missionId}/trajectories/${videoId}/${frameIndex}`, {method: 'DELETE'})
  if (!r.ok && r.status !== 404) throw new Error('Trajektorie konnte nicht gelöscht werden')
}
export async function getRoiProfile(missionId: string, videoId: string): Promise<RoiProfile> {
  const r = await fetch(`/api/v1/missions/${missionId}/roi-profile/${videoId}`)
  if (!r.ok) throw new Error('ROI-Profil konnte nicht geladen werden')
  return r.json()
}
export async function saveRoiProfile(
  missionId: string,
  videoId: string,
  payload: {
    top_ignore_fraction: number | null
    bottom_ignore_fraction: number | null
    roi: GroundTruthPolygon[]
    note: string
    annotator: string
  },
): Promise<RoiProfile> {
  const r = await fetch(`/api/v1/missions/${missionId}/roi-profile/${videoId}`, {
    method: 'PUT',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  })
  let body: any = {}
  try {
    body = await r.json()
  } catch {}
  if (!r.ok) throw new Error(body.detail || 'ROI-Profil konnte nicht gespeichert werden')
  return body
}
export async function getLabelTracks(missionId: string, videoId: string): Promise<LabelTracks> {
  const r = await fetch(`/api/v1/missions/${missionId}/labels/tracks/${videoId}`)
  if (!r.ok) throw new Error('Spuren konnten nicht geladen werden')
  return r.json()
}
export async function getLabelOntology(): Promise<LabelOntology> {
  const r = await fetch('/api/v1/label-ontology')
  if (!r.ok) throw new Error('Labelklassen konnten nicht geladen werden')
  return r.json()
}
export async function getRegistryRuns(): Promise<RegistryListing> {
  const r = await fetch('/api/v1/registry/runs')
  if (!r.ok) throw new Error('Run-Registry konnte nicht geladen werden')
  return r.json()
}
export async function updateRegistryRun(
  missionId: string,
  videoId: string,
  payload: {status?: RunStatus; terrain_category?: string | null; note?: string},
): Promise<RegistryRun> {
  const r = await fetch(`/api/v1/registry/runs/${missionId}/${videoId}`, {
    method: 'PATCH',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  })
  let body: any = {}
  try {
    body = await r.json()
  } catch {}
  if (!r.ok) throw new Error(body.detail || 'Run konnte nicht aktualisiert werden')
  return body
}
export async function getTerrainDashboard(): Promise<TerrainDashboardData> {
  const r = await fetch('/api/v1/terrain-model/dashboard')
  if (!r.ok) throw new Error('Terrainmodell konnte nicht geladen werden')
  return r.json()
}
export async function trainTerrainModel(payload: {frame_stride: number; confidence_threshold: number}): Promise<TerrainModelResult> {
  const r = await fetch('/api/v1/terrain-model/train', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  })
  let body: any = {}
  try {
    body = await r.json()
  } catch {}
  if (!r.ok) throw new Error(body.detail || 'Terrainmodell konnte nicht trainiert werden')
  return body
}
export async function predictTerrainVideo(
  missionId: string,
  videoId: string,
  payload: {frame_stride: number; confidence_threshold: number},
): Promise<TerrainPredictionRun> {
  const r = await fetch(`/api/v1/terrain-model/predict-video/${missionId}/${videoId}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  })
  let body: any = {}
  try {
    body = await r.json()
  } catch {}
  if (!r.ok) throw new Error(body.detail || 'Video konnte nicht klassifiziert werden')
  return body
}
export async function predictTerrainFrame(
  missionId: string,
  videoId: string,
  frameIndex: number,
): Promise<TerrainFramePrediction & {model_run_id: string; confidence_threshold: number; video_terrain_category: string | null}> {
  const r = await fetch(`/api/v1/terrain-model/predict/${missionId}/${videoId}/${frameIndex}`)
  let body: any = {}
  try {
    body = await r.json()
  } catch {}
  if (!r.ok) throw new Error(body.detail || 'Frame konnte nicht klassifiziert werden')
  return body
}
export async function listMissions(): Promise<Mission[]> {
  const r = await fetch('/api/v1/missions')
  if (!r.ok) throw new Error('Missionen konnten nicht geladen werden')
  return r.json()
}
export async function getAnalysis(id: string): Promise<Analysis> {
  const r = await fetch(`/api/v1/missions/${id}/analysis`)
  if (!r.ok)
    throw new Error(r.status === 404 ? 'Für diese Mission liegt noch keine Auswertung vor.' : 'Auswertung konnte nicht geladen werden')
  return r.json()
}
export async function getReconstruction(id: string): Promise<Reconstruction> {
  const r = await fetch(`/api/v1/missions/${id}/reconstruction`)
  if (!r.ok)
    throw new Error(
      r.status === 404 ? 'Für diese Mission liegt noch keine videobasierte Route vor.' : 'Rekonstruktion konnte nicht geladen werden',
    )
  return r.json()
}
export async function getSegmentation(id: string): Promise<Segmentation> {
  const r = await fetch(`/api/v1/missions/${id}/segmentation`)
  if (!r.ok)
    throw new Error(
      r.status === 404 ? 'Für diese Mission liegt noch keine Goal-4-Analyse vor.' : 'Goal-4-Analyse konnte nicht geladen werden',
    )
  return r.json()
}
export async function listGroundTruth(missionId: string, videoId?: string, includeGeometry = false): Promise<GroundTruthSummary> {
  const params = new URLSearchParams()
  if (videoId) params.set('video_id', videoId)
  if (includeGeometry) params.set('include_geometry', 'true')
  const query = params.size ? `?${params}` : ''
  const r = await fetch(`/api/v1/missions/${missionId}/ground-truth${query}`)
  if (!r.ok) throw new Error('Ground-Truth-Übersicht konnte nicht geladen werden')
  return r.json()
}
export async function getGroundTruth(missionId: string, videoId: string, frameIndex: number): Promise<GroundTruthAnnotation | null> {
  const r = await fetch(`/api/v1/missions/${missionId}/ground-truth/${videoId}/${frameIndex}`)
  if (r.status === 404) return null
  if (!r.ok) throw new Error('Ground Truth konnte nicht geladen werden')
  return r.json()
}
export async function saveGroundTruth(
  missionId: string,
  videoId: string,
  frameIndex: number,
  payload: {
    timestamp_ms: number
    source_frame_hash?: string
    mask?: TerrainMask
    polygons?: GroundTruthPolygon[]
    roi?: GroundTruthPolygon[]
    frame_width?: number
    frame_height?: number
    status: GroundTruthStatus
    annotator: string
    notes: string
    label_mode?: 'linear' | 'shuffle'
  },
): Promise<GroundTruthAnnotation> {
  const r = await fetch(`/api/v1/missions/${missionId}/ground-truth/${videoId}/${frameIndex}`, {
    method: 'PUT',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  })
  let body: any = {}
  try {
    body = await r.json()
  } catch {}
  if (!r.ok) throw new Error(body.detail || 'Ground Truth konnte nicht gespeichert werden')
  return body
}
export async function deleteGroundTruth(missionId: string, videoId: string, frameIndex: number): Promise<void> {
  const r = await fetch(`/api/v1/missions/${missionId}/ground-truth/${videoId}/${frameIndex}`, {method: 'DELETE'})
  if (!r.ok && r.status !== 404) throw new Error('Ground Truth konnte nicht gelöscht werden')
}
export async function listCriticalFlags(
  missionId: string,
  videoId?: string,
): Promise<{
  schema_version: string
  mission_id: string
  kind: string
  counts: {total: number}
  items: {
    video_id: string
    frame_index: number
    timestamp_ms: number
    severity: number
    note: string
    annotator: string
    created_at: string
    brush_mask?: TerrainMask
  }[]
}> {
  const params = new URLSearchParams()
  if (videoId) params.set('video_id', videoId)
  const query = params.size ? `?${params}` : ''
  const r = await fetch(`/api/v1/missions/${missionId}/critical-flags${query}`)
  if (!r.ok) throw new Error('Meldungen konnten nicht geladen werden')
  return r.json()
}
export async function saveCriticalFlag(
  missionId: string,
  videoId: string,
  frameIndex: number,
  payload: {severity: number; brush_mask?: TerrainMask; note: string; annotator: string},
): Promise<{
  schema_version: string
  kind: string
  meaning: string
  mission_id: string
  video_id: string
  frame_index: number
  timestamp_ms: number
  severity: number
  note: string
  annotator: string
  created_at: string
  brush_mask?: TerrainMask
}> {
  const r = await fetch(`/api/v1/missions/${missionId}/critical-flags/${videoId}/${frameIndex}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  })
  let body: any = {}
  try {
    body = await r.json()
  } catch {}
  if (!r.ok) throw new Error(body.detail || 'Meldung konnte nicht gespeichert werden')
  return body
}
export async function deleteCriticalFlag(missionId: string, videoId: string, frameIndex: number): Promise<void> {
  const r = await fetch(`/api/v1/missions/${missionId}/critical-flags/${videoId}/${frameIndex}`, {method: 'DELETE'})
  if (!r.ok && r.status !== 404) throw new Error('Meldung konnte nicht gelöscht werden')
}
export async function getLabelingVideos(missionId: string): Promise<LabelingVideoManifest> {
  const r = await fetch(`/api/v1/missions/${missionId}/labeling/videos`)
  if (!r.ok) throw new Error('Videoframes konnten nicht für das Labeling vorbereitet werden')
  return r.json()
}
export async function runSegmentation(missionId: string): Promise<Segmentation> {
  const r = await fetch(`/api/v1/missions/${missionId}/segmentation`, {method: 'POST'})
  let body: any = {}
  try {
    body = await r.json()
  } catch {}
  if (!r.ok) throw new Error(body.detail || 'Auswertung konnte nicht gestartet werden')
  return body
}
export async function getPathModel(missionId: string): Promise<PathModelResult | null> {
  const r = await fetch(`/api/v1/missions/${missionId}/path-model`)
  if (r.status === 404) return null
  if (!r.ok) throw new Error('CPU-Wegmodell konnte nicht geladen werden')
  return r.json()
}
export async function trainPathModel(missionId: string): Promise<PathModelResult> {
  const r = await fetch(`/api/v1/missions/${missionId}/path-model/train`, {method: 'POST'})
  let body: any = {}
  try {
    body = await r.json()
  } catch {}
  if (!r.ok) throw new Error(body.detail || 'CPU-Wegmodell konnte nicht trainiert werden')
  return body
}
export async function predictPathFrame(missionId: string, videoId: string, frameIndex: number): Promise<PathPrediction> {
  const r = await fetch(`/api/v1/missions/${missionId}/path-model/predict/${videoId}/${frameIndex}`)
  let body: any = {}
  try {
    body = await r.json()
  } catch {}
  if (!r.ok) throw new Error(body.detail || 'KI-Wegmaske konnte nicht berechnet werden')
  return body
}
export async function savePathRefinement(
  missionId: string,
  videoId: string,
  frameIndex: number,
  payload: {x: number; y: number; expected_kind: 'missed_label' | 'invented_path'; action: 'accept_model'},
): Promise<{id: string; refinement_count: number}> {
  const r = await fetch(`/api/v1/missions/${missionId}/path-model/refinements/${videoId}/${frameIndex}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  })
  let body: any = {}
  try {
    body = await r.json()
  } catch {}
  if (!r.ok) throw new Error(body.detail || 'Refinement konnte nicht gespeichert werden')
  return body
}
export async function startPathTrainingJob(
  missionId: string,
  profile: 'quick' | 'overnight',
  durationHours: number,
): Promise<PathTrainingJob> {
  const params = new URLSearchParams({profile, duration_hours: String(durationHours)})
  const r = await fetch(`/api/v1/missions/${missionId}/path-model/train-background?${params}`, {method: 'POST'})
  let body: any = {}
  try {
    body = await r.json()
  } catch {}
  if (!r.ok) throw new Error(body.detail || 'Hintergrundtraining konnte nicht gestartet werden')
  return body
}
export async function getPathTrainingJob(missionId: string): Promise<PathTrainingJob | null> {
  const r = await fetch(`/api/v1/missions/${missionId}/path-model/train-background`)
  if (r.status === 404) return null
  if (!r.ok) throw new Error('Trainingsstatus konnte nicht geladen werden')
  return r.json()
}
export async function getGlobalModelDashboard(): Promise<GlobalModelDashboardData> {
  const r = await fetch('/api/v1/path-model/global/dashboard')
  if (!r.ok) throw new Error('Globales Modellzentrum konnte nicht geladen werden')
  return r.json()
}
export async function trainGlobalPathModel(): Promise<GlobalPathModelResult> {
  const r = await fetch('/api/v1/path-model/global/train', {method: 'POST'})
  let body: any = {}
  try {
    body = await r.json()
  } catch {}
  if (!r.ok) throw new Error(body.detail || 'Globales Wegmodell konnte nicht trainiert werden')
  return body
}
export async function predictGlobalPathFrame(
  missionId: string,
  videoId: string,
  frameIndex: number,
  calibration: {vehicle_width_m: number; clearance_m: number; ground_width_at_bottom_m: number},
): Promise<PathPrediction> {
  // Die Korridorprüfung hängt an derselben Antwort: als eigener Aufruf lief die
  // Inferenz zweimal über denselben Frame.
  const params = new URLSearchParams({
    vehicle_width_m: String(calibration.vehicle_width_m),
    clearance_m: String(calibration.clearance_m),
    ground_width_at_bottom_m: String(calibration.ground_width_at_bottom_m),
  })
  const r = await fetch(`/api/v1/path-model/global/predict/${missionId}/${videoId}/${frameIndex}?${params}`)
  let body: any = {}
  try {
    body = await r.json()
  } catch {}
  if (!r.ok) throw new Error(body.detail || 'Globale Frameanalyse konnte nicht berechnet werden')
  return body
}
export async function startGlobalVideoAnalysis(missionId: string, videoId: string): Promise<GlobalVideoAnalysisStatus> {
  const r = await fetch(`/api/v1/path-model/global/analyze-video/${missionId}/${videoId}`, {method: 'POST'})
  let body: any = {}
  try {
    body = await r.json()
  } catch {}
  if (!r.ok) throw new Error(body.detail || 'Globale Videoanalyse konnte nicht gestartet werden')
  return body
}
export async function getGlobalVideoAnalysisStatus(missionId: string, videoId: string): Promise<GlobalVideoAnalysisStatus | null> {
  const r = await fetch(`/api/v1/path-model/global/analyze-video/${missionId}/${videoId}/status`)
  if (r.status === 404) return null
  if (!r.ok) throw new Error('Status der globalen Videoanalyse konnte nicht geladen werden')
  return r.json()
}
export async function getGlobalVideoAnalysisResult(missionId: string, videoId: string): Promise<GlobalVideoAnalysisResult> {
  const r = await fetch(`/api/v1/path-model/global/analyze-video/${missionId}/${videoId}/result`)
  let body: any = {}
  try {
    body = await r.json()
  } catch {}
  if (!r.ok) throw new Error(body.detail || 'Ergebnis der globalen Videoanalyse ist noch nicht verfügbar')
  return body
}
