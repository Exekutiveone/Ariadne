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
  symmetric_score: 91,
  iou: 0.83,
  dice: 0.9,
  precision: 0.9,
  recall: 0.9,
}

const pathModel = {
  schema_version: '1.0',
  run_id: 'path-run-1',
  mission_id: 'mission-1',
  created_at: '2026-08-03T00:00:00Z',
  model: {
    id: 'ariadne-cpu-path-rff',
    type: 'rff',
    hardware: 'CPU',
    cloud_used: false,
    input_width: 160,
    feature_count: 22,
    random_features: 64,
    threshold: 0.3,
    postprocessing: '3x3',
  },
  ground_truth: {positive: 'p', negative: 'n', confirmed_frames: 12, videos: 1},
  split: {strategy: 's', train_frames: 10, validation_frames: 2, training_pixels_sampled: 100, same_frame_in_train_and_validation: false},
  scoring: {rule: 'r', threshold_selection: metrics},
  train_metrics: metrics,
  validation_metrics: metrics,
  evidence: [],
  runtime_seconds: 1,
  limitations: [],
}

const grading = {
  margin: 'm = (score - threshold) / max(1e-6, 1 - threshold)',
  threshold: 0.3,
  bands: {safe_min_margin: 0.6, good_min_margin: 0.25, risky_min_margin: -0.2},
  problem_min_area_fraction: 0.002,
  problem_neighbourhood_px: 9,
  problem_clip_px: 25,
  smoothing: '3x3',
  note: 'KI-Einschätzung der Befahrbarkeit, keine sicherheitsrelevante Fahrfreigabe.',
}

// Frame mit Ground Truth: liefert Abstufung UND Vergleichsmaske.
const prediction = {
  schema_version: '1.0',
  model_run_id: 'path-run-1',
  video_id: 'video-1',
  frame_index: 0,
  timestamp_ms: 0,
  mask: {width: 2, height: 2, rle: [1, 4]},
  grade_mask: {width: 2, height: 2, rle: [1, 1, 2, 1, 4, 1, 5, 1]},
  grade_ontology: GRADE_ONTOLOGY_FALLBACK,
  grading,
  path_fraction: 0.42,
  mean_separation: 0.2,
  confidence_note: '',
  source: 'cpu',
  evaluation: {
    annotation_status: 'confirmed',
    metrics,
    comparison_mask: {width: 2, height: 2, rle: [1, 2, 2, 1, 3, 1]},
    legend: {},
    refinement_count: 2,
  },
}

vi.mock('./api', () => ({
  getProblemReasons: async () => ({items: []}),
  createProblemReason: async () => ({value: 'custom_test', label: 'Test', uses: 0}),
  getLabelOntology: async () => ({
    schema_version: '3.0',
    unlabelled: {key: 'unlabelled', value: 0, label: 'Nicht markiert', color: '#00000000'},
    layers: [
      {
        layer: 'core',
        label: 'Kernklasse',
        exclusive: true,
        classes: [
          {class_id: 'traversable', layer: 'core', label: 'Befahrbarer Boden', color: '#55d96f', value: 1, description: ''},
          {class_id: 'restricted', layer: 'core', label: 'Eingeschränkt befahrbar', color: '#e4c264', value: 4, description: ''},
          {class_id: 'not_traversable', layer: 'core', label: 'Nicht befahrbar', color: '#e05b52', value: 2, description: ''},
          {class_id: 'unknown', layer: 'core', label: 'Nicht bewertbar / verdeckt', color: '#737c78', value: 3, description: ''},
        ],
      },
      {
        layer: 'obstacle',
        label: 'Hindernis',
        exclusive: false,
        classes: [{class_id: 'tree', layer: 'obstacle', label: 'Baum', color: '#2f7d4f', value: null, description: ''}],
      },
      {
        layer: 'zone',
        label: 'Problemzone',
        exclusive: false,
        classes: [{class_id: 'mud', layer: 'zone', label: 'Matsch', color: '#7a5c3d', value: null, description: ''}],
      },
      {
        layer: 'roi',
        label: 'Auswertungsbereich',
        exclusive: false,
        classes: [
          {class_id: 'roi_ignore', layer: 'roi', label: 'Nicht interessiert / ignorieren', color: '#3a4149', value: null, description: ''},
        ],
      },
    ],
    certainty: [
      {value: 'certain', label: 'Sicher'},
      {value: 'uncertain', label: 'Unsicher'},
      {value: 'partially_occluded', label: 'Teilweise verdeckt'},
    ],
    origin: [{value: 'manual', label: 'Von Hand gesetzt'}],
    notes: [],
  }),
  getRoiProfile: async () => ({
    schema_version: '1.0',
    video_id: 'video-1',
    top_ignore_fraction: null,
    bottom_ignore_fraction: null,
    roi: [],
    note: '',
    revision: 0,
    suggested: {top_ignore_fraction: 0.2, bottom_ignore_fraction: 0.1},
    applies_as: 'suggestion_only_frame_labels_decide',
  }),
  saveRoiProfile: vi.fn(),
  listTrajectories: async () => [],
  createTrajectory: vi.fn(),
  updateTrajectory: vi.fn(),
  deleteTrajectory: vi.fn(),
  getLabelingVideos: async () => ({
    mission_id: 'mission-1',
    source: 'original_video_metadata_only',
    automatic_processing_started: false,
    videos: [
      {video_id: 'video-1', original_name: 'Waldweg.mp4', fps: 30, total_frames: 100, width: 1920, height: 1080, duration_seconds: 3},
    ],
  }),
  getGroundTruth: async () => null,
  listGroundTruth: async () => ({
    schema_version: '2.0',
    mission_id: 'mission-1',
    ontology: {},
    counts: {total: 0, draft: 0, confirmed: 0, skipped: 0},
    items: [],
  }),
  getPathModel: async () => pathModel,
  getPathTrainingJob: async () => null,
  predictPathFrame: async () => prediction,
  saveGroundTruth: vi.fn(),
  runSegmentation: vi.fn(),
  savePathRefinement: vi.fn(),
  updateVideoTerrainCategory: vi.fn(),
  updateVideoFullyNotTraversable: vi.fn(),
  startPathTrainingJob: vi.fn(),
  trainPathModel: vi.fn(),
  listOffPathIntervals: vi.fn(async () => []),
  createOffPathInterval: vi.fn(),
  deleteOffPathInterval: vi.fn(),
  listCriticalFlags: vi.fn(async () => ({items: []})),
  saveCriticalFlag: vi.fn(),
  deleteCriticalFlag: vi.fn(),
}))

import GroundTruthLabeler from './GroundTruthLabeler'

const mount = () =>
  render(
    <GroundTruthLabeler
      mission={{id: 'mission-1', name: 'Mission 1', videos: []} as any}
      onClose={() => undefined}
      onProcessingComplete={() => undefined}
    />,
  )

test('shows the graded AI mask with all six classes and the safety note by default', async () => {
  mount()

  expect(await screen.findByRole('button', {name: 'Abstufung'}, {timeout: 5_000})).toHaveClass('active')
  for (const label of [
    'Sicher befahrbar',
    'Gut befahrbar',
    'Knapp befahrbar',
    'Potenziell befahrbar, mit Risiko',
    'Problemzone / Hindernis',
  ]) {
    expect(screen.getByText(label)).toBeInTheDocument()
  }
  expect(screen.getByText(/keine sicherheitsrelevante Fahrfreigabe/)).toBeInTheDocument()
  expect(screen.getByText(/Grenzen auf dem normierten Schwellenabstand/)).toBeInTheDocument()
  expect(screen.getByText(/Abgestufte KI-Einschätzung/)).toBeInTheDocument()
})

test('switches to the comparison mask and back without losing either legend', async () => {
  mount()

  fireEvent.click(await screen.findByRole('button', {name: 'Vergleich'}, {timeout: 5_000}))

  expect(screen.getByText('Übersehene Wegfläche')).toBeInTheDocument()
  expect(screen.getByText('Fälschlich erkannter Weg')).toBeInTheDocument()
  expect(screen.queryByText('Knapp befahrbar')).not.toBeInTheDocument()

  fireEvent.click(screen.getByRole('button', {name: 'Abstufung'}))
  expect(screen.getByText('Knapp befahrbar')).toBeInTheDocument()
})

test('forces the comparison mask while refinement is active so error areas stay clickable', async () => {
  mount()

  fireEvent.click(await screen.findByRole('button', {name: 'Refinement starten'}))

  await waitFor(() => expect(screen.getByRole('button', {name: 'Vergleich'})).toHaveClass('active'))
  expect(screen.getByRole('button', {name: 'Abstufung'})).toBeDisabled()
  expect(screen.getByText('Für das Refinement wird die Vergleichsmaske angezeigt.')).toBeInTheDocument()

  fireEvent.click(screen.getByRole('button', {name: 'Refinement beenden'}))
  await waitFor(() => expect(screen.getByRole('button', {name: 'Abstufung'})).toBeEnabled())
})
