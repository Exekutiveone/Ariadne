import {render, screen} from '@testing-library/react'
import {vi, test, expect} from 'vitest'
vi.mock('react-leaflet', () => ({
  MapContainer: ({children}: any) => <div>{children}</div>,
  TileLayer: () => null,
  Marker: ({children}: any) => <>{children}</>,
  Polyline: () => null,
  Tooltip: ({children}: any) => <>{children}</>,
  useMapEvents: () => null,
}))
vi.mock('./api', () => ({
  listMissions: async () => [],
  uploadMission: vi.fn(),
  updateVideoTerrainCategory: vi.fn(),
  getGlobalModelDashboard: async () => ({
    dataset: {missions: [], totals: {missions: 0, confirmed_frames: 0, videos: 0, refinements: 0, critical_flags: 0}},
    model: null,
  }),
  getGlobalVideoAnalysisResult: vi.fn(),
  getGlobalVideoAnalysisStatus: async () => null,
  startGlobalVideoAnalysis: vi.fn(),
  predictGlobalPathFrame: vi.fn(),
  trainGlobalPathModel: vi.fn(),
  getGroundTruth: async () => null,
  listGroundTruth: async () => ({counts: {total: 0, draft: 0, confirmed: 0, skipped: 0}, items: [], ontology: {}}),
  saveGroundTruth: vi.fn(),
  getPathModel: async () => null,
  getPathTrainingJob: async () => null,
  predictPathFrame: vi.fn(),
  savePathRefinement: vi.fn(),
  startPathTrainingJob: vi.fn(),
  trainPathModel: vi.fn(),
  getLabelingVideos: vi.fn(),
  runSegmentation: vi.fn(),
  listCriticalFlags: vi.fn(),
  saveCriticalFlag: vi.fn(),
  deleteCriticalFlag: vi.fn(),
}))
import App from './App'
test('renders Goal 1 upload workflow with visible requirements', () => {
  render(<App />)
  expect(screen.getByText('Survey-Mission')).toBeInTheDocument()
  expect(screen.getByText('Originalvideos')).toBeInTheDocument()
  expect(screen.queryByText(/Start A/)).not.toBeInTheDocument()
  expect(screen.queryByText(/Route A/)).not.toBeInTheDocument()
  expect(screen.getByLabelText('Speichervoraussetzungen')).toHaveTextContent('Missionsname')
  expect(screen.getByRole('button', {name: /MISSION PERSISTENT/})).toBeEnabled()
  expect(screen.getByRole('button', {name: 'KI-MODELLZENTRUM'})).toBeEnabled()
})
