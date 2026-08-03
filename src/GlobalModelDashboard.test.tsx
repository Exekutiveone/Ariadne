import {fireEvent, render, screen} from '@testing-library/react'
import {expect, test, vi} from 'vitest'
import {GRADE_ONTOLOGY_FALLBACK} from './masks'

const metrics = {tp: 10, tn: 10, fp: 1, fn: 1, missed_label_fraction: .09, invented_path_fraction: .09, symmetric_penalty_points: 9, symmetric_score: 91.47, iou: .796, dice: .9, precision: .9, recall: .9}

const model = {
  schema_version: '1.0', scope: 'global_cross_mission', run_id: 'global-run-1', created_at: '2026-08-03T17:41:13Z',
  model: {id: 'ariadne-cpu-path-rff', type: 'rff', hardware: 'CPU', cloud_used: false, input_width: 160, feature_count: 22, random_features: 64, threshold: .3},
  dataset: {missions: [{mission_id: 'mission-1', name: 'Mission 1', confirmed_frames: 251, train_frames: 200, validation_frames: 51}], confirmed_frames: 251, videos: 4, refinements_included: 20},
  split: {strategy: 's', train_frames: 200, validation_frames: 51, training_pixels_sampled: 100, same_frame_in_train_and_validation: false},
  train_metrics: metrics, validation_metrics: metrics, evidence: [], runtime_seconds: 31, limitations: [],
}

// Fertige Videoanalyse OHNE grade_mask — der Zustand echter Analysen vor Phase 3.
const analysisResult = {
  schema_version: '1.0', model_run_id: 'global-run-1', mission_id: 'mission-1', video_id: 'video-1',
  fps: 30, total_frames: 3, width: 1920, height: 1080, analyzed_frames: 3, runtime_seconds: 12,
  frames: [0, 1, 2].map(index => ({frame_index: index, timestamp_ms: index * 33, mask: {width: 2, height: 2, rle: [1, 4]}, path_fraction: .4})),
}

vi.mock('./api', () => ({
  getGlobalModelDashboard: async () => ({
    dataset: {missions: [{mission_id: 'mission-1', name: 'Mission 1', confirmed_frames: 251, videos: 4, refinements: 20}], totals: {missions: 1, confirmed_frames: 251, videos: 4, refinements: 20}},
    model,
  }),
  getLabelingVideos: async () => ({mission_id: 'mission-1', source: 'original_video_metadata_only', automatic_processing_started: false, videos: [{video_id: 'video-1', original_name: 'Waldweg.mp4', fps: 30, total_frames: 3, width: 1920, height: 1080, duration_seconds: .1}]}),
  getGlobalVideoAnalysisStatus: async () => ({job_id: 'j', status: 'completed', model_run_id: 'global-run-1', mission_id: 'mission-1', video_id: 'video-1', pid: 0, started_at: '', finished_at: '', processed_frames: 3, total_frames: 3, progress: 1, elapsed_seconds: 12, eta_seconds: 0, message: 'fertig'}),
  getGlobalVideoAnalysisResult: async () => analysisResult,
  predictGlobalPathFrame: async () => ({
    schema_version: '1.0', model_run_id: 'global-run-1', video_id: 'video-1', frame_index: 0, timestamp_ms: 0,
    mask: {width: 2, height: 2, rle: [1, 4]},
    grade_mask: {width: 2, height: 2, rle: [1, 1, 2, 1, 4, 1, 5, 1]},
    grade_ontology: GRADE_ONTOLOGY_FALLBACK,
    grading: {margin: 'm', threshold: .3, bands: {safe_min_margin: .6, good_min_margin: .25, risky_min_margin: -.2}, problem_min_area_fraction: .002, problem_neighbourhood_px: 9, problem_clip_px: 25, smoothing: '3x3', note: 'KI-Einschätzung der Befahrbarkeit, keine sicherheitsrelevante Fahrfreigabe.'},
    path_fraction: .42, mean_separation: .2, confidence_note: '', source: 'global',
  }),
  startGlobalVideoAnalysis: vi.fn(),
  trainGlobalPathModel: vi.fn(),
}))

import GlobalModelDashboard from './GlobalModelDashboard'

test('grades an already analyzed video from the live prediction and says so', async () => {
  render(<GlobalModelDashboard onClose={() => undefined}/>)

  expect(await screen.findByText('Sicher befahrbar')).toBeInTheDocument()
  expect(screen.getByText('Problemzone / Hindernis')).toBeInTheDocument()
  expect(screen.getByText(/keine sicherheitsrelevante Fahrfreigabe/)).toBeInTheDocument()
  // Ehrlicher Hinweis, dass die gespeicherte Analyse die Abstufung noch nicht enthaelt.
  expect(screen.getByText(/enthält noch keine Abstufung/)).toBeInTheDocument()
})

test('falls back to the precomputed binary mask when grading is switched off', async () => {
  render(<GlobalModelDashboard onClose={() => undefined}/>)

  // Erst wenn die gespeicherte Analyse geladen ist, sind die Maskenschalter aktiv.
  expect(await screen.findByText('Wiedergabe bereit')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('checkbox', {name: 'Abstufung anzeigen'}))

  expect(screen.queryByText('Sicher befahrbar')).not.toBeInTheDocument()
  expect(screen.getByRole('checkbox', {name: 'KI-Maske anzeigen'})).toBeEnabled()
  expect(screen.getByText(/vorberechnete globale KI-Wegmaske/)).toBeInTheDocument()
})
