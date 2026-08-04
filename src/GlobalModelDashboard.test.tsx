import {fireEvent, render, screen, waitFor} from '@testing-library/react'
import {expect, test, vi} from 'vitest'
import {GRADE_ONTOLOGY_FALLBACK} from './masks'

const metrics = {
  tp: 10,
  tn: 10,
  fp: 1,
  fn: 1,
  missed_label_fraction: 0.09,
  invented_path_fraction: 0.09,
  symmetric_penalty_points: 9,
  symmetric_score: 91.47,
  iou: 0.796,
  dice: 0.9,
  precision: 0.9,
  recall: 0.9,
}

const model = {
  schema_version: '1.0',
  scope: 'global_cross_mission',
  run_id: 'global-run-1',
  created_at: '2026-08-03T17:41:13Z',
  model: {
    id: 'ariadne-cpu-path-rff',
    type: 'rff',
    hardware: 'CPU',
    cloud_used: false,
    input_width: 160,
    feature_count: 22,
    random_features: 64,
    threshold: 0.3,
  },
  dataset: {
    missions: [{mission_id: 'mission-1', name: 'Mission 1', confirmed_frames: 251, train_frames: 200, validation_frames: 51}],
    confirmed_frames: 251,
    videos: 4,
    refinements_included: 20,
    critical_flags_included: 3,
  },
  split: {strategy: 's', train_frames: 200, validation_frames: 51, training_pixels_sampled: 100, same_frame_in_train_and_validation: false},
  train_metrics: metrics,
  validation_metrics: metrics,
  evidence: [],
  runtime_seconds: 31,
  limitations: [],
}

// Fertige Videoanalyse OHNE grade_mask — der Zustand echter Analysen vor Phase 3.
const analysisResult = {
  schema_version: '1.0',
  model_run_id: 'global-run-1',
  mission_id: 'mission-1',
  video_id: 'video-1',
  fps: 30,
  total_frames: 3,
  width: 1920,
  height: 1080,
  analyzed_frames: 3,
  runtime_seconds: 12,
  frames: [0, 1, 2].map(index => ({
    frame_index: index,
    timestamp_ms: index * 33,
    mask: {width: 2, height: 2, rle: [1, 4]},
    path_fraction: 0.4,
  })),
}

vi.mock('./api', () => ({
  getGlobalModelDashboard: async () => ({
    dataset: {
      missions: [{mission_id: 'mission-1', name: 'Mission 1', confirmed_frames: 251, videos: 4, refinements: 20, critical_flags: 3}],
      totals: {missions: 1, confirmed_frames: 251, videos: 4, refinements: 20, critical_flags: 3},
    },
    model,
  }),
  getLabelingVideos: async () => ({
    mission_id: 'mission-1',
    source: 'original_video_metadata_only',
    automatic_processing_started: false,
    videos: [
      {video_id: 'video-1', original_name: 'Waldweg.mp4', fps: 30, total_frames: 3, width: 1920, height: 1080, duration_seconds: 0.1},
    ],
  }),
  updateVideoTerrainCategory: vi.fn(),
  listCriticalFlags: async () => ({
    schema_version: '1.0',
    mission_id: 'mission-1',
    kind: 'no_path_false_detection',
    counts: {total: 1},
    items: [
      {
        video_id: 'video-1',
        frame_index: 0,
        timestamp_ms: 0,
        severity: 4,
        note: 'wrong',
        annotator: 'human',
        created_at: '2026-08-03T17:41:13Z',
      },
    ],
  }),
  saveCriticalFlag: vi.fn(),
  deleteCriticalFlag: vi.fn(),
  listOffPathIntervals: async () => [],
  createOffPathInterval: vi.fn(),
  deleteOffPathInterval: vi.fn(),
  getGlobalVideoAnalysisStatus: async () => ({
    job_id: 'j',
    status: 'completed',
    model_run_id: 'global-run-1',
    mission_id: 'mission-1',
    video_id: 'video-1',
    pid: 0,
    started_at: '',
    finished_at: '',
    processed_frames: 3,
    total_frames: 3,
    progress: 1,
    elapsed_seconds: 12,
    eta_seconds: 0,
    message: 'fertig',
  }),
  getGlobalVideoAnalysisResult: async () => analysisResult,
  predictGlobalPathFrame: async () => ({
    schema_version: '1.0',
    model_run_id: 'global-run-1',
    video_id: 'video-1',
    frame_index: 0,
    timestamp_ms: 0,
    mask: {width: 2, height: 2, rle: [1, 4]},
    grade_mask: {width: 2, height: 2, rle: [1, 1, 2, 1, 4, 1, 5, 1]},
    grade_ontology: GRADE_ONTOLOGY_FALLBACK,
    grading: {
      margin: 'm',
      threshold: 0.3,
      bands: {safe_min_margin: 0.6, good_min_margin: 0.25, risky_min_margin: -0.2},
      problem_min_area_fraction: 0.002,
      problem_neighbourhood_px: 9,
      problem_clip_px: 25,
      smoothing: '3x3',
      note: 'KI-Einschätzung der Befahrbarkeit, keine sicherheitsrelevante Fahrfreigabe.',
    },
    path_fraction: 0.42,
    mean_separation: 0.2,
    confidence_note: '',
    source: 'global',
    // Die Korridorpruefung haengt an derselben Antwort, statt eine zweite
    // Inferenz ueber denselben Frame zu kosten.
    corridors: corridorCheck,
  }),
  startGlobalVideoAnalysis: vi.fn(),
  trainGlobalPathModel: vi.fn(),
  // Der Terrain-Abschnitt haengt am selben Modul; ohne Modell zeigt er nur seine Leerzustaende.
  getTerrainDashboard: async () => ({
    dataset: {
      videos: [],
      classes: [],
      totals: {categorized_videos: 0, uncategorized_videos: 0, classes: 0, missions: 0},
      label_source: 'video_terrain_category_inherited_by_all_frames',
    },
    model: null,
    runs: {active_run_id: null, training_runs: [], prediction_runs: []},
  }),
  trainTerrainModel: vi.fn(),
  predictTerrainVideo: vi.fn(),
  listTrajectories: async () => [],
  createTrajectory: (...args: unknown[]) => createTrajectory(...(args as [])),
  updateTrajectory: vi.fn(),
  deleteTrajectory: vi.fn(),
}))

const geometry = (x: number) => ({
  center: [
    [x, 0.35],
    [x, 0.68],
    [x, 1],
  ],
  left: [
    [x - 0.02, 0.35],
    [x - 0.1, 0.68],
    [x - 0.16, 1],
  ],
  right: [
    [x + 0.02, 0.35],
    [x + 0.1, 0.68],
    [x + 0.16, 1],
  ],
})

const corridorCheck = {
  schema_version: '2.0',
  kind: 'parallel_near_field_corridor_check',
  mask_size: {width: 160, height: 120},
  region: {
    kind: 'fixed_near_field_band',
    first_evaluated_row: 50,
    rows_skipped: 50,
    image_fraction_skipped: 0.42,
    reason: 'Nur das untere sichtbare Nahfeld wird auf Fahrzeugbreite geprüft.',
    evaluated_rows: 78,
    first_evaluated_row_normalized: 0.42,
  },
  proposed_trajectory: {
    corridor: 'mitte',
    label: 'Mitte',
    status: 'free',
    status_label: 'frei',
    points: [
      [0.5, 0.35],
      [0.5, 0.68],
      [0.5, 1],
    ],
    source: 'widest_drivable_run_center_in_parallel_near_field_band',
    note: 'Vorschlag im Korridor Mitte',
  },
  strip: {
    vehicle_width_m: 1.2,
    clearance_m: 0.1,
    required_width_m: 1.3,
    ground_width_at_bottom_m: 4,
    required_width_px_at_bottom: 52,
    scaling: 'konstante parallele Streifen im sichtbaren Nahfeld',
    search_band_factor: 1.5,
  },
  corridors: [
    {
      corridor: 'mitte',
      label: 'Mitte',
      meaning: 'Standard bei schmalen Wald- und Feldwegen',
      status: 'free',
      status_label: 'frei',
      reason: 'In allen 78 ausgewerteten Zeilen passt ein freier Streifen in den Korridor.',
      rows: {evaluated: 78, free: 78, uncertain: 0, blocked: 0},
      bottom_center_x: 80,
      geometry: geometry(0.5),
      trajectory: {
        points: [
          [0.5, 0.35],
          [0.5, 0.68],
          [0.5, 1],
        ],
        rows: 78,
        source: 'widest_drivable_run_center_in_parallel_near_field_band',
      },
    },
    {
      corridor: 'rechts',
      label: 'Rechts',
      meaning: 'Rechtsfahrgebot',
      status: 'blocked',
      status_label: 'blockiert',
      reason: 'In 40 von 78 Zeilen passt kein freier Streifen in den Korridor.',
      rows: {evaluated: 78, free: 38, uncertain: 0, blocked: 40},
      bottom_center_x: 132,
      geometry: geometry(0.82),
      trajectory: {
        points: [
          [0.82, 0.68],
          [0.82, 1],
        ],
        rows: 38,
        source: 'widest_drivable_run_center_in_parallel_near_field_band',
      },
    },
    {
      corridor: 'links',
      label: 'Links',
      meaning: 'Ausweichoption',
      status: 'uncertain',
      status_label: 'unsicher',
      reason: 'In 12 von 78 Zeilen ist der Streifen nicht sicher frei.',
      rows: {evaluated: 78, free: 66, uncertain: 12, blocked: 0},
      bottom_center_x: 28,
      geometry: geometry(0.18),
      trajectory: {
        points: [
          [0.18, 0.35],
          [0.18, 1],
        ],
        rows: 66,
        source: 'widest_drivable_run_center_in_parallel_near_field_band',
      },
    },
  ],
  graded_input: true,
  limitations: [
    'Nur Breitenprüfung.',
    'Kalibrierung pro Kameraaufbau.',
    'Deterministische Geometrie auf einer vorhergesagten Maske — keine sicherheitsrelevante Fahrfreigabe.',
  ],
  model_run_id: 'global-run-1',
  mission_id: 'mission-1',
  video_id: 'video-1',
  frame_index: 0,
  timestamp_ms: 0,
  path_fraction: 0.42,
  source: 'deterministic_geometry_on_global_model_mask',
}

const createTrajectory = vi.fn(async () => ({
  id: 'traj-1',
  schema_version: '1.0',
  mission_id: 'mission-1',
  video_id: 'video-1',
  frame_index: 0,
  timestamp_ms: 0,
  points: [
    [0.5, 0.35],
    [0.5, 1],
  ],
  corridor: 'mitte',
  origin: 'model_proposal',
  note: '',
  annotator: 'Simon',
  coordinate_space: 'normalized_to_original_frame',
  revision: 1,
  created_at: '',
  updated_at: '',
}))

import GlobalModelDashboard from './GlobalModelDashboard'

test('grades an already analyzed video from the live prediction and says so', async () => {
  render(<GlobalModelDashboard onClose={() => undefined} />)

  expect(await screen.findByText('Sicher befahrbar')).toBeInTheDocument()
  expect(screen.getByText('Problemzone / Hindernis')).toBeInTheDocument()
  // Der Vorbehalt steht sowohl an der Abstufung als auch an der Korridorpruefung.
  expect(screen.getAllByText(/keine sicherheitsrelevante Fahrfreigabe/).length).toBeGreaterThan(0)
  // Ehrlicher Hinweis, dass die gespeicherte Analyse die Abstufung noch nicht enthaelt,
  // und dass "ANALYSE ERNEUERN" das nachholt.
  expect(screen.getByText(/ist älter als die durchgehende Abstufung/)).toBeInTheDocument()
  expect(screen.getByRole('button', {name: 'ANALYSE ERNEUERN (ABSTUFUNG NACHRÜSTEN)'})).toBeInTheDocument()
})

test('falls back to the precomputed binary mask when grading is switched off', async () => {
  render(<GlobalModelDashboard onClose={() => undefined} />)

  // Erst wenn die gespeicherte Analyse geladen ist, sind die Maskenschalter aktiv.
  expect(await screen.findByText('Wiedergabe bereit')).toBeInTheDocument()
  const speedSlider = screen.getByRole('slider', {name: 'Wiedergabegeschwindigkeit'})
  fireEvent.change(speedSlider, {target: {value: '4'}})
  expect(speedSlider).toHaveValue('4')
  fireEvent.click(screen.getByRole('checkbox', {name: 'Abstufung anzeigen'}))

  expect(screen.queryByText('Sicher befahrbar')).not.toBeInTheDocument()
  expect(screen.getByRole('checkbox', {name: 'KI-Maske anzeigen'})).toBeEnabled()
  expect(screen.getByText(/vorberechnete globale KI-Wegmaske/)).toBeInTheDocument()
})

test('saves a paused incorrect frame immediately in the quick-review gallery', async () => {
  render(<GlobalModelDashboard onClose={() => undefined} />)

  expect(await screen.findByText('Wiedergabe bereit')).toBeInTheDocument()
  expect(screen.getByText(/Noch nicht vorgemerkt oder gespeichert/)).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', {name: 'FALSCHEN FRAME DIREKT SPEICHERN'}))

  expect(await screen.findByText('1 erfasst')).toBeInTheDocument()
  expect(screen.getByText(/Referenzbereich: Frame 1 bis 1/)).toBeInTheDocument()
  expect(await screen.findByText(/Diese Referenz ist dauerhaft gespeichert/)).toBeInTheDocument()
})

test('reports the three image-space corridors with their status', async () => {
  render(<GlobalModelDashboard onClose={() => undefined} />)

  expect(await screen.findByText('BLOCKIERT')).toBeInTheDocument()
  expect(screen.getByText('Korridore und Trajektorie')).toBeInTheDocument()
  expect(screen.getByText('FREI')).toBeInTheDocument()
  expect(screen.getByText('UNSICHER')).toBeInTheDocument()
  expect(screen.getByText('Ausgewertetes Nahfeld')).toBeInTheDocument()
  expect(screen.getByText(/Nur das untere sichtbare Nahfeld wird auf Fahrzeugbreite geprüft/)).toBeInTheDocument()
})

test('shows no corridor until one is explicitly selected', async () => {
  const {container} = render(<GlobalModelDashboard onClose={() => undefined} />)
  await screen.findByText('BLOCKIERT')

  expect(container.querySelector('.corridor-overlay')).not.toBeInTheDocument()
  fireEvent.click(screen.getByText('Mitte'))
  const overlay = container.querySelector('.corridor-overlay')!
  expect(overlay).toBeInTheDocument()
  expect(overlay.querySelectorAll('.corridor-wedge')).toHaveLength(1)
  expect(overlay.querySelector('.corridor-proposal')).toBeInTheDocument()
  expect(overlay.querySelector('.corridor-wedge.free.active')).toBeInTheDocument()
})

test('selecting another corridor replaces the visible corridor', async () => {
  const {container} = render(<GlobalModelDashboard onClose={() => undefined} />)
  await screen.findByText('BLOCKIERT')

  fireEvent.click(screen.getByText('Links'))

  expect(container.querySelector('.corridor-wedge.uncertain.active')).toBeInTheDocument()
  expect(container.querySelector('.corridor-wedge.free.active')).not.toBeInTheDocument()
  expect(container.querySelectorAll('.corridor-wedge')).toHaveLength(1)
})

test('adopting the proposal yields draggable handles that can be saved', async () => {
  const {container} = render(<GlobalModelDashboard onClose={() => undefined} />)
  await screen.findByText('BLOCKIERT')
  expect(container.querySelector('.corridor-draft')).not.toBeInTheDocument()

  fireEvent.click(screen.getByText('Mitte'))
  fireEvent.click(screen.getByRole('button', {name: 'Vorschlag übernehmen'}))

  expect(container.querySelector('.corridor-draft')).toBeInTheDocument()
  expect(container.querySelectorAll('.corridor-handle').length).toBeGreaterThan(1)
  expect(screen.getByText('3 Punkte im Entwurf')).toBeInTheDocument()
  // Unveraendert uebernommen: das wird als Modellvorschlag festgehalten, nicht als Handarbeit.
  expect(screen.getByText(/wird als „unverändert vom Modell" gespeichert/)).toBeInTheDocument()

  fireEvent.click(screen.getByRole('button', {name: 'Trajektorie speichern'}))
  await waitFor(() => expect(createTrajectory).toHaveBeenCalled())
  const payload = (createTrajectory.mock.calls[0] as unknown[])[3] as {origin: string; corridor: string; points: number[][]}
  expect(payload).toMatchObject({origin: 'model_proposal', corridor: 'mitte'})
  expect(payload.points).toHaveLength(3)
})
