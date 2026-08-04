import {fireEvent, render, screen, waitFor} from '@testing-library/react'
import {beforeEach, expect, test, vi} from 'vitest'

const updateVideoTerrainCategory = vi.hoisted(() =>
  vi.fn(async () => ({id: 'video-1', original_name: 'Waldweg.mp4', terrain_category: 'walduntergrund'})),
)

vi.mock('./api', () => ({
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
  getTrajectory: async () => null,
  saveTrajectory: vi.fn(),
  deleteTrajectory: vi.fn(),
  getLabelingVideos: async () => ({
    mission_id: 'mission-1',
    source: 'original_video_metadata_only',
    automatic_processing_started: false,
    videos: [
      {
        video_id: 'video-1',
        original_name: 'Waldweg.mp4',
        fps: 30,
        total_frames: 2400,
        width: 1920,
        height: 1080,
        duration_seconds: 80,
        terrain_category: 'schotterweg',
      },
    ],
  }),
  getGroundTruth: async () => null,
  listGroundTruth: async () => ({
    schema_version: '2.0',
    mission_id: 'mission-1',
    ontology: {},
    counts: {total: 1, draft: 1, confirmed: 0, skipped: 0},
    items: [
      {
        video_id: 'video-1',
        frame_index: 10,
        timestamp_ms: 333,
        source_frame_hash: 'a'.repeat(64),
        status: 'draft',
        annotator: 'Simon',
        revision: 1,
        updated_at: '2026-08-03T00:00:00Z',
        statistics: {polygon_count: 1, point_count: 4, classes: {traversable: 1}},
      },
    ],
  }),
  saveGroundTruth: vi.fn(),
  updateVideoTerrainCategory,
  updateVideoFullyNotTraversable: vi.fn(async () => ({fully_not_traversable: true})),
  runSegmentation: vi.fn(),
  getPathModel: async () => null,
  getPathTrainingJob: async () => null,
  predictPathFrame: vi.fn(),
  savePathRefinement: vi.fn(),
  startPathTrainingJob: vi.fn(),
  trainPathModel: vi.fn(),
  listOffPathIntervals: vi.fn(async () => []),
  createOffPathInterval: vi.fn(),
  deleteOffPathInterval: vi.fn(),
  listCriticalFlags: vi.fn(async () => ({items: []})),
  saveCriticalFlag: vi.fn(),
  deleteCriticalFlag: vi.fn(),
}))

import GroundTruthLabeler, {
  buildFrameSelection,
  clampPan,
  normalizedPointFromBounds,
  pointerAction,
  polygonForNextFrame,
  rleValueAt,
} from './GroundTruthLabeler'

beforeEach(() => {
  updateVideoTerrainCategory.mockClear()
})

test('builds selectable frame sets by stride or requested count', () => {
  expect(buildFrameSelection(25, 'stride', 10, 100)).toEqual([0, 10, 20])
  expect(buildFrameSelection(25, 'count', 10, 5)).toEqual([0, 6, 12, 18, 24])
})

test('keeps polygon coordinates stable when the viewport is zoomed and panned', () => {
  expect(normalizedPointFromBounds(250, 400, {left: 0, top: 0, width: 1000, height: 800})).toEqual([0.25, 0.5])
  expect(normalizedPointFromBounds(600, 850, {left: 100, top: 50, width: 2000, height: 1600})).toEqual([0.25, 0.5])
})

test('never lets the image be panned into showing empty space beyond its own edges', () => {
  // Bei Originalgroesse (Zoom 1) gibt es nichts zu verschieben.
  expect(clampPan({x: 40, y: -40}, 1, 300, 500)).toEqual({x: 0, y: 0})
  // Herausgezogen (positiv) clamped auf 0 - der linke/obere Rand darf nicht ueber den Rahmenrand hinaus.
  expect(clampPan({x: 999, y: 999}, 1.5, 300, 500)).toEqual({x: 0, y: 0})
  // Hineingezogen (negativ) clamped auf width*(1-zoom) - der rechte/untere Rand darf nicht sichtbar werden.
  expect(clampPan({x: -999, y: -999}, 1.5, 300, 500)).toEqual({x: -150, y: -250})
  // Innerhalb der Grenzen bleibt der Wert unveraendert.
  expect(clampPan({x: -60, y: -90}, 1.5, 300, 500)).toEqual({x: -60, y: -90})
})

test('carries polygons only forward and always prioritizes vertex dragging', () => {
  const polygon = [
    [0.1, 0.8],
    [0.5, 0.3],
    [0.9, 0.8],
  ] as [number, number][]
  expect(polygonForNextFrame(polygon, 1)).toEqual(polygon)
  expect(polygonForNextFrame(polygon, -1)).toBeNull()
  expect(pointerAction('pan', 1, true, true)).toBe('vertex')
  expect(pointerAction('add', 1, true, true)).toBe('vertex')
  expect(pointerAction('move', 1, true, true)).toBe('vertex')
})

test('finds a clicked error class inside an RLE comparison mask', () => {
  const mask = {width: 2, height: 2, rle: [0, 1, 2, 1, 3, 1, 1, 1]}
  expect(rleValueAt(mask, [0.75, 0.25])).toBe(2)
  expect(rleValueAt(mask, [0.25, 0.75])).toBe(3)
})

test('shows a pure manual polygon workflow with zoom, editing and explicit processing', async () => {
  render(
    <GroundTruthLabeler
      mission={{id: 'mission-1', name: 'Mission 1', videos: []} as any}
      onClose={() => undefined}
      onProcessingComplete={() => undefined}
    />,
  )

  await waitFor(() => expect(screen.getByText('Frame 1 von 2.400')).toBeInTheDocument())
  expect(screen.getByText('Auswahl 1 von 240')).toBeInTheDocument()
  expect(screen.getByRole('button', {name: 'Hineinzoomen'})).toBeInTheDocument()
  expect(screen.getByRole('button', {name: 'Herauszoomen'})).toBeInTheDocument()
  expect(screen.getByRole('button', {name: 'Gelabelte Frames abspielen'})).toBeEnabled()
  expect(screen.getByRole('combobox', {name: 'Abspielgeschwindigkeit'})).toHaveValue('1')
  const opacity = screen.getByRole('slider', {name: 'Masken-Deckkraft'})
  expect(opacity).toHaveValue('0.3')
  fireEvent.change(opacity, {target: {value: '0.7'}})
  expect(screen.getByText('Masken-Deckkraft · 70 %')).toBeInTheDocument()
  expect(screen.getByRole('button', {name: /Punkt hinzufügen/})).toBeInTheDocument()
  expect(screen.getByRole('button', {name: /Polygon verschieben/})).toBeInTheDocument()
  expect(screen.getByRole('button', {name: 'Frame überspringen'})).toBeInTheDocument()
  expect(screen.getByRole('button', {name: /MARKIERUNG ABSCHLIESSEN/})).toBeDisabled()
  expect(screen.getByText('Gespeicherte Polygonmasken')).toBeInTheDocument()
  expect(screen.getByRole('button', {name: /Frame 11/})).toBeInTheDocument()
  expect(screen.queryByText('Befahrbarkeit')).not.toBeInTheDocument()
  expect(screen.queryByText('Trajektorie')).not.toBeInTheDocument()
  expect(screen.queryByText('KI-Vorschlag übernehmen')).not.toBeInTheDocument()
  expect(screen.getByRole('button', {name: /WEG-KI AUF 0 LABELFRAMES TRAINIEREN/})).toBeDisabled()
  expect(screen.getByText('KI-Wegmaske anzeigen')).toBeInTheDocument()
  expect(screen.getByRole('combobox', {name: 'Trainingsprofil'})).toHaveValue('overnight')
  expect(screen.getByRole('button', {name: 'NACHTTRAINING STARTEN'})).toBeDisabled()
})

test('allows editing the terrain category of an existing video from the inventory', async () => {
  const onMissionUpdated = vi.fn()
  render(
    <GroundTruthLabeler
      mission={{id: 'mission-1', name: 'Mission 1', videos: []} as any}
      onClose={() => undefined}
      onProcessingComplete={() => undefined}
      onMissionUpdated={onMissionUpdated}
    />,
  )

  const categorySelect = await screen.findByRole('combobox', {name: 'Terrainkategorie'})
  await waitFor(() => expect(categorySelect).toHaveValue('schotterweg'))
  fireEvent.change(categorySelect, {target: {value: 'walduntergrund'}})
  fireEvent.click(screen.getByRole('button', {name: 'Kategorie speichern'}))

  await waitFor(() => expect(updateVideoTerrainCategory).toHaveBeenCalledWith('mission-1', 'video-1', {terrain_category: 'walduntergrund'}))
  await waitFor(() => expect(onMissionUpdated).toHaveBeenCalled())
  expect(screen.getByText(/Terrainkategorie für Waldweg\.mp4 gespeichert/i)).toBeInTheDocument()
})

test('offers every ontology layer and keeps obstacles apart from the core classes', async () => {
  render(
    <GroundTruthLabeler
      mission={{id: 'mission-1', name: 'Mission 1', videos: []} as any}
      onClose={() => undefined}
      onProcessingComplete={() => undefined}
    />,
  )
  await screen.findByRole('button', {name: 'Befahrbarer Boden'})

  // Vier Ebenen, jede mit eigener Ueberschrift — Hindernisse ersetzen keine
  // Kernklasse, sie erklaeren sie.
  expect(screen.getByText(/Untergrund — genau eine je Fläche/)).toBeInTheDocument()
  expect(screen.getByText(/Hindernis — erklärt, warum etwas nicht fahrbar ist/)).toBeInTheDocument()
  expect(screen.getByText(/Problemzone — macht eine Fläche unsicher/)).toBeInTheDocument()
  expect(screen.getByText(/Auswertungsbereich — schneidet nichts weg/)).toBeInTheDocument()
  expect(screen.getByRole('button', {name: 'Baum'})).toBeInTheDocument()
  expect(screen.getByRole('button', {name: 'Nicht interessiert / ignorieren'})).toBeInTheDocument()
})

test('a shape cannot be committed before it is a polygon', async () => {
  render(
    <GroundTruthLabeler
      mission={{id: 'mission-1', name: 'Mission 1', videos: []} as any}
      onClose={() => undefined}
      onProcessingComplete={() => undefined}
    />,
  )
  await screen.findByRole('button', {name: 'Befahrbarer Boden'})

  expect(screen.getByRole('button', {name: 'Fläche übernehmen'})).toBeDisabled()
})

test('drawing a trajectory by hand needs no model and no proposal', async () => {
  render(
    <GroundTruthLabeler
      mission={{id: 'mission-1', name: 'Mission 1', videos: []} as any}
      onClose={() => undefined}
      onProcessingComplete={() => undefined}
    />,
  )
  await screen.findByRole('button', {name: 'Befahrbarer Boden'})

  expect(screen.getByText('Trajektorie von Hand')).toBeInTheDocument()
  expect(screen.getByText(/Braucht kein trainiertes Modell/)).toBeInTheDocument()
  // Ohne Punkte ist Speichern gesperrt: eine Bahn braucht mindestens zwei.
  expect(screen.getByRole('button', {name: 'Trajektorie speichern'})).toBeDisabled()
  expect(screen.getByText('0 Bahnpunkte')).toBeInTheDocument()
  expect(screen.getByText('Für diesen Frame ist nichts gespeichert.')).toBeInTheDocument()
})
