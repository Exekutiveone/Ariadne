import {fireEvent, render, screen} from '@testing-library/react'
import {expect, test, vi} from 'vitest'

vi.mock('react-leaflet', () => ({
  MapContainer: ({children}: any) => <div>{children}</div>,
  TileLayer: () => null,
  CircleMarker: () => null,
  Polyline: () => null,
}))
vi.mock('./api', () => ({
  getGroundTruth: async () => null,
  listGroundTruth: async () => ({
    schema_version: '2.0',
    mission_id: 'mission-1',
    ontology: {},
    counts: {total: 1, draft: 0, confirmed: 1, skipped: 0},
    items: [
      {
        video_id: 'video-1',
        frame_index: 0,
        timestamp_ms: 0,
        source_frame_hash: 'a'.repeat(64),
        status: 'confirmed',
        annotator: 'Simon',
        revision: 1,
        updated_at: '2026-08-03T00:00:00Z',
        statistics: {polygon_count: 1, point_count: 4, classes: {traversable: 1}},
        polygons: [
          {
            id: 'path-1',
            class_id: 'traversable',
            points: [
              [0.1, 0.9],
              [0.35, 0.4],
              [0.65, 0.4],
              [0.9, 0.9],
            ],
          },
        ],
      },
    ],
  }),
  saveGroundTruth: vi.fn(),
  predictPathFrame: async () => ({
    schema_version: '1.0',
    model_run_id: 'path-run-1',
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
    source: 'cpu',
  }),
}))

import {GRADE_ONTOLOGY_FALLBACK} from './masks'

import AnalysisView from './AnalysisView'

const terrain = {
  source_video_id: 'video-1',
  source_frame_index: 0,
  source_frame_timestamp_ms: 0,
  source_frame_hash: 'frame-hash',
  ground: {
    mask: {width: 1, height: 1, rle: [1, 1]},
    regions: [],
    confidence: 0.72,
    visible_ratio: 0.48,
    source: 'current_video_frame_inference',
  },
  traversability: {
    mask: {width: 1, height: 1, rle: [2, 1]},
    regions: [],
    class_coverage: {likely_traversable: 0, limited: 1, not_traversable: 0, unknown: 0},
    overall_class: 'limited',
    overall_confidence: 0.66,
  },
  corridor: {
    status: 'uncertain',
    polygon: [],
    centerline: [],
    confidence: 0.61,
    minimum_width_ratio: 0.9,
    minimum_width_m: 0.68,
    stability_px: 2,
    stable_frames: 3,
    green_support_fraction: 0.5,
    source_frame_timestamp_ms: 0,
    temporally_smoothed: true,
    reasons: ['metric_scale_estimated'],
  },
  factors: {
    free_width_score: 0.7,
    obstacle_clearance_score: 0.8,
    connectivity_score: 0.9,
    smoothness_score: 0.6,
    bottleneck_clearance_score: 0.7,
    visibility_score: 0.8,
    calibration_score: 0.45,
    temporal_stability_score: 0.9,
  },
  quality: {blur_score: 0.1, exposure_score: 0.9, motion_inliers: 42, unknown_ratio: 0},
  evidence: {representative: true, reasons: ['temporally_stable_corridor'], overlay_url: '/evidence.jpg'},
}

const mount = () =>
  render(
    <AnalysisView
      mission={{id: 'mission-1', name: 'Mission 1', status: 'READY_FOR_GOAL_2', created_at: '', route: [], videos: []} as any}
      data={{keyframes: []} as any}
      reconstruction={
        {
          traversals: [
            {
              video_id: 'video-1',
              video_name: 'Video 1',
              direction: 'A_TO_B',
              duration_seconds: 10,
              tracked_fraction: 1,
              median_matches: 10,
              median_inliers: 10,
              geojson: {
                type: 'LineString',
                coordinates: [
                  [9.28, 48.73],
                  [9.27, 48.74],
                ],
              },
            },
          ],
        } as any
      }
      segmentation={
        {
          schema_version: '3.0',
          run_id: 'run-1',
          mission_id: 'mission-1',
          model: {adapter: 'ForestInstanceAdapter', model_id: 'forest', version: '3', hardware: 'CPU'},
          configuration: {
            analysis_hz: 4,
            input_width: 640,
            min_area_ratio: 0.004,
            confirmation_hits: 3,
            max_track_gap_frames: 2,
            confidence_meaning: 'proxy',
          },
          ontology: {},
          terrain_ontology: {
            likely_traversable: {label: 'Wahrscheinlich befahrbar', color: '#55d96f', value: 1},
            limited: {label: 'Eingeschränkt oder unsicher', color: '#e7c84d', value: 2},
            not_traversable: {label: 'Wahrscheinlich nicht befahrbar', color: '#e05b52', value: 3},
            unknown: {label: 'Nicht bewertbar', color: '#737c78', value: 0},
          },
          vehicle_configuration: {
            width_m: 0.35,
            safety_margin_per_side_m: 0.2,
            required_width_m: 0.75,
            source: 'documented_default_assumption',
          },
          videos: [
            {
              video_id: 'video-1',
              video_name: 'Video 1',
              duration_seconds: 10,
              fps: 30,
              width: 640,
              height: 360,
              analysis_interval_ms: 250,
              frames: [
                {
                  video_id: 'video-1',
                  video_name: 'Video 1',
                  frame_index: 0,
                  timestamp_ms: 0,
                  quality: {sharpness: 100, motion_inliers: 42},
                  detections: [],
                  terrain,
                },
              ],
              tracks: [],
              counts: {visible_individuals_latest_frame: 0, confirmed_unique_per_video: {tree: 0, shrub: 0}},
              metrics: {
                analyzed_frames: 1,
                raw_detections: 0,
                tracks: 0,
                confirmed_tree_instances: 0,
                confirmed_shrub_instances: 0,
                average_track_length_frames: 0,
                short_track_fraction: 0,
                empty_frame_fraction: 1,
                median_motion_inliers: 42,
                inference_seconds: 1,
              },
            },
          ],
          counts: {confirmed_unique_per_video_sum: {tree: 0, shrub: 0}, mission_unique: null, mission_unique_reason: ''},
          metrics: {
            runtime_seconds: 1,
            analyzed_frames: 1,
            raw_detections: 0,
            tracks: 0,
            confirmed_tree_instances: 0,
            confirmed_shrub_instances: 0,
            empty_frame_fraction: 1,
            average_track_length_frames: 0,
          },
          truth_rules: {ground_truth_available: false, species_inference: false, navigation_grade: false, individual_definition: ''},
        } as any
      }
      onClose={() => undefined}
    />,
  )

test('offers automatic and own-label overlays while keeping the safety boundary visible', async () => {
  mount()

  for (const label of ['Original', 'Boden', 'Befahrbarkeit', 'KI-Abstufung', 'Eigene Labels']) {
    expect(screen.getByRole('button', {name: label})).toBeInTheDocument()
  }
  expect(screen.queryByRole('button', {name: 'Ground Truth'})).not.toBeInTheDocument()
  expect(screen.getByText('Keine Fahrfreigabe')).toBeInTheDocument()
  expect(screen.getByText('Repräsentative Evidenzframes')).toBeInTheDocument()

  fireEvent.click(screen.getByRole('button', {name: 'Befahrbarkeit'}))
  expect(screen.getByText(/Dokumentierte Arbeitsannahme/)).toBeInTheDocument()
  expect(screen.getByRole('checkbox', {name: 'Eigene Labels zusätzlich überlagern'})).toBeInTheDocument()

  fireEvent.click(screen.getByRole('button', {name: 'Original'}))
  expect(screen.getByText('Unverändertes Originalvideo')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', {name: 'Boden'}))
  expect(screen.getByText('Aus aktuellem Videoframe berechnet')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', {name: 'Eigene Labels'}))
  expect(await screen.findByRole('combobox', {name: 'Gespeichertes eigenes Label'})).toBeInTheDocument()
  expect(screen.getByText('Manuelle Polygonmaske')).toBeInTheDocument()
  expect(screen.getByRole('slider', {name: 'Eigene-Label-Deckkraft'})).toHaveValue('0.3')
})

test('shows the graded AI overlay as its own player mode with legend and safety note', async () => {
  mount()

  fireEvent.click(screen.getByRole('button', {name: 'KI-Abstufung'}))

  expect(await screen.findByText('Sicher befahrbar')).toBeInTheDocument()
  for (const label of ['Gut befahrbar', 'Knapp befahrbar', 'Potenziell befahrbar, mit Risiko', 'Problemzone / Hindernis']) {
    expect(screen.getByText(label)).toBeInTheDocument()
  }
  expect(screen.getByText(/keine sicherheitsrelevante Fahrfreigabe/)).toBeInTheDocument()
  expect(screen.getByText(/Modell path-run-1/)).toBeInTheDocument()
  // Der Vorbehalt muss auch in der Statuszeile ueber dem Video stehen.
  expect(screen.getByText(/KI-ABSTUFUNG · 42 % WEG · KEINE FAHRFREIGABE/)).toBeInTheDocument()
})
