import {fireEvent, render, screen, waitFor} from '@testing-library/react'
import {expect, test, vi} from 'vitest'
import type {TerrainDashboardData, TerrainMetrics, TerrainPredictionRun} from './types'

const metrics = (overrides: Partial<TerrainMetrics> = {}): TerrainMetrics => ({
  frames: 120, accuracy: .95, balanced_accuracy: .94, mean_confidence: .88,
  uncertain_frames: 12, uncertain_fraction: .1, accuracy_on_confident: .99,
  confusion_matrix: [[58, 2], [4, 56]],
  per_class: [
    {terrain_category: 'schotterweg', support: 60, precision: .935, recall: .967, f1: .951},
    {terrain_category: 'walduntergrund', support: 60, precision: .966, recall: .933, f1: .949},
  ],
  ...overrides,
})

const dashboard: TerrainDashboardData = {
  dataset: {
    videos: [
      {mission_id: 'm-1', mission_name: 'Schotterlauf', video_id: 'v-1', original_name: 'Quer.MOV', terrain_category: 'schotterweg'},
      {mission_id: 'm-2', mission_name: 'Waldlauf', video_id: 'v-2', original_name: 'IMG_9742.MOV', terrain_category: 'walduntergrund'},
      {mission_id: 'm-2', mission_name: 'Waldlauf', video_id: 'v-3', original_name: 'Ohne.MOV', terrain_category: null},
    ],
    classes: [{terrain_category: 'schotterweg', videos: 4, missions: 1}, {terrain_category: 'walduntergrund', videos: 2, missions: 1}],
    totals: {categorized_videos: 6, uncategorized_videos: 1, classes: 2, missions: 2},
    label_source: 'video_terrain_category_inherited_by_all_frames',
  },
  model: {
    schema_version: '1.0', scope: 'video_terrain_classification', kind: 'training',
    run_id: 'terrain-20260804T081500Z-ab12cd34', created_at: '2026-08-04T08:15:00Z',
    model: {id: 'ariadne-cpu-terrain-rff', type: 'rff', hardware: 'CPU', cloud_used: false, input_width: 160, grid: 3, feature_count: 234, random_features: 256, softmax_scale: 4, confidence_threshold: .6},
    classes: ['schotterweg', 'walduntergrund'],
    dataset: {frame_stride: 15, label_source: 'video_terrain_category_inherited_by_all_frames', categorized_videos: 6, uncategorized_videos: 1, videos: [], frames: 360},
    split: {
      strategy: 'grouped_by_video_id', random_frame_split_used: false, same_video_in_multiple_parts: false,
      train: {videos: 3, frames: 240, classes: ['schotterweg', 'walduntergrund'], video_ids: ['v-1', 'v-4', 'v-2'], all_classes: ['schotterweg', 'walduntergrund']},
      validation: {videos: 2, frames: 120, classes: ['schotterweg', 'walduntergrund'], video_ids: ['v-5', 'v-6'], all_classes: ['schotterweg', 'walduntergrund']},
      test: null,
      notes: ['Kein Testteil gebildet: dafür werden mindestens drei Videos je Terrainkategorie benötigt.'],
    },
    calibration: {softmax_scale: 4, selected_on: 'validation_frames_only', negative_log_likelihood: .08},
    train_metrics: metrics(), validation_metrics: metrics(), test_metrics: null,
    runtime_seconds: 44.2,
    limitations: ['Dies ist eine KI-gestützte Einschätzung und keine sicherheitsrelevante Fahrfreigabe.'],
  },
  runs: {
    active_run_id: 'terrain-20260804T081500Z-ab12cd34',
    training_runs: [{run_id: 'terrain-20260804T081500Z-ab12cd34', kind: 'training', created_at: '2026-08-04T08:15:00Z', active: true, classes: ['schotterweg', 'walduntergrund'], frame_stride: 15, confidence_threshold: .6, validation_accuracy: .95, test_accuracy: null, runtime_seconds: 44.2}],
    prediction_runs: [],
  },
}

const predictionRun: TerrainPredictionRun = {
  schema_version: '1.0', kind: 'prediction', run_id: 'terrain-predict-20260804T090000Z-99887766',
  created_at: '2026-08-04T09:00:00Z', model_run_id: 'terrain-20260804T081500Z-ab12cd34',
  mission_id: 'm-1', video_id: 'v-1', original_name: 'Quer.MOV', video_terrain_category: 'schotterweg',
  frame_stride: 15, confidence_threshold: .6, classes: ['schotterweg', 'walduntergrund'],
  summary: {frames: 4, uncertain_frames: 1, uncertain_fraction: .25, mean_confidence: .81, dominant_category: 'schotterweg', counts: {schotterweg: 3, walduntergrund: 1}, matches_video_category: .75},
  frames: [
    {frame_index: 0, timestamp_ms: 0, predicted_category: 'schotterweg', top_category: 'schotterweg', confidence: .97, uncertain: false, scores: {schotterweg: .97, walduntergrund: .03}},
    {frame_index: 15, timestamp_ms: 500, predicted_category: 'schotterweg', top_category: 'schotterweg', confidence: .91, uncertain: false, scores: {schotterweg: .91, walduntergrund: .09}},
    {frame_index: 30, timestamp_ms: 65000, predicted_category: null, top_category: 'walduntergrund', confidence: .52, uncertain: true, scores: {schotterweg: .48, walduntergrund: .52}},
    {frame_index: 45, timestamp_ms: 1500, predicted_category: 'schotterweg', top_category: 'schotterweg', confidence: .84, uncertain: false, scores: {schotterweg: .84, walduntergrund: .16}},
  ],
  runtime_seconds: 9.1,
  limitations: [],
}

const trainTerrainModel = vi.fn(async () => dashboard.model!)
const predictTerrainVideo = vi.fn(async () => predictionRun)

vi.mock('./api', () => ({
  getTerrainDashboard: async () => dashboard,
  trainTerrainModel: (...args: unknown[]) => trainTerrainModel(...(args as [])),
  predictTerrainVideo: (...args: unknown[]) => predictTerrainVideo(...(args as [])),
}))

import TerrainModelPanel from './TerrainModelPanel'

test('shows the video-level split and states that no test part was formed', async () => {
  render(<TerrainModelPanel/>)

  expect(await screen.findByRole('heading', {name: 'Aktives Terrainmodell'})).toBeInTheDocument()
  expect(screen.getAllByText('terrain-20260804T081500Z-ab12cd34').length).toBeGreaterThan(0)
  // Der Split ist nach Video gruppiert; kein Video darf in zwei Teilmengen liegen.
  expect(screen.getByText('Training')).toBeInTheDocument()
  expect(screen.getByText(/240 Frames · lernt die Zuordnung/)).toBeInTheDocument()
  expect(screen.getByText(/120 Frames · wählt die Konfidenzkalibrierung/)).toBeInTheDocument()
  expect(screen.getByText('nicht gebildet')).toBeInTheDocument()
  expect(screen.getByText(/Kein Testteil gebildet/)).toBeInTheDocument()
  expect(screen.getByText(/keine sicherheitsrelevante Fahrfreigabe/)).toBeInTheDocument()
})

test('names the terrain classes in German and counts the uncategorized videos', async () => {
  render(<TerrainModelPanel/>)

  expect(await screen.findAllByText('Schotterweg')).not.toHaveLength(0)
  expect(screen.getAllByText('Walduntergrund').length).toBeGreaterThan(0)
  expect(screen.getByText(/1 Video\(s\) ohne Terrainkategorie/)).toBeInTheDocument()
})

test('passes the configured stride and threshold into the training run', async () => {
  render(<TerrainModelPanel/>)
  await screen.findByRole('heading', {name: 'Aktives Terrainmodell'})

  fireEvent.change(screen.getByRole('spinbutton', {name: 'Schrittweite der Frames'}), {target: {value: '40'}})
  fireEvent.click(screen.getByRole('button', {name: 'TERRAINMODELL TRAINIEREN'}))

  await waitFor(() => expect(trainTerrainModel).toHaveBeenCalledWith({frame_stride: 40, confidence_threshold: .6}))
})

test('marks frames below the confidence threshold as uncertain instead of assigning a class', async () => {
  render(<TerrainModelPanel/>)
  await screen.findByRole('heading', {name: 'Aktives Terrainmodell'})

  fireEvent.click(screen.getByRole('button', {name: 'VIDEO KLASSIFIZIEREN'}))

  // Grosszuegiger Timeout: die Klassifizierung durchlaeuft zwei Ladezyklen und
  // die Vorgabe von 1000 ms reisst unter Last gelegentlich.
  expect(await screen.findByText(/1 unsichere Frames unterhalb 60,0 %/, undefined, {timeout: 4000})).toBeInTheDocument()
  expect(screen.getByText('Frame 31')).toBeInTheDocument()
  expect(screen.getByText(/unsicher, am ehesten Walduntergrund \(52,0 %\)/)).toBeInTheDocument()
  // Die sicheren Frames tauchen in der Unsicher-Liste nicht auf.
  expect(screen.queryByText('Frame 1')).not.toBeInTheDocument()
  expect(screen.getByText(/Vorhersagelauf terrain-predict-20260804T090000Z-99887766 gespeichert/)).toBeInTheDocument()
})
