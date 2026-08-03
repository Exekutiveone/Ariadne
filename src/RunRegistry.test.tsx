import {fireEvent, render, screen, waitFor} from '@testing-library/react'
import {expect, test, vi} from 'vitest'
import type {RegistryListing} from './types'

const listing: RegistryListing = {
  schema_version: '1.0', database: 'C:/data/registry.sqlite',
  scan: {added: 1, removed: [], total: 2},
  statuses: [
    {value: 'unlabeled', label: 'Ungelabelt'},
    {value: 'queued_for_labeling', label: 'Zum Labeln vorgemerkt'},
    {value: 'labeled', label: 'Gelabelt'},
    {value: 'training_ready', label: 'Trainingsbereit'},
  ],
  counts: {unlabeled: 1, queued_for_labeling: 0, labeled: 1, training_ready: 0},
  totals: {runs: 2, with_terrain_category: 1, missing_video_file: 1},
  terrain_categories: [{terrain_category: 'walduntergrund', runs: 1}],
  runs: [
    {run_id: 'm-1/v-1', mission_id: 'm-1', video_id: 'v-1', mission_name: 'Waldlauf', original_name: 'IMG_9742.MOV', video_available: true, size_bytes: 524288000, status: 'labeled', status_label: 'Gelabelt', terrain_category: 'walduntergrund', note: 'Sonnenflecken', discovered_at: '2026-08-04T08:00:00Z', updated_at: '2026-08-04T09:00:00Z'},
    {run_id: 'm-1/v-2', mission_id: 'm-1', video_id: 'v-2', mission_name: 'Waldlauf', original_name: 'IMG_9743.MOV', video_available: false, size_bytes: 0, status: 'unlabeled', status_label: 'Ungelabelt', terrain_category: null, note: '', discovered_at: '2026-08-04T08:00:00Z', updated_at: '2026-08-04T08:00:00Z'},
  ],
  note: 'Status und Notiz sind Handarbeit und stehen nur in dieser Datenbank.',
}

const updateRegistryRun = vi.fn(async () => listing.runs[0])
vi.mock('./api', () => ({
  getRegistryRuns: async () => listing,
  updateRegistryRun: (...args: unknown[]) => updateRegistryRun(...(args as [])),
}))

import RunRegistry from './RunRegistry'

test('lists every run with status, surface and note', async () => {
  render(<RunRegistry onClose={() => undefined}/>)

  expect(await screen.findByText('IMG_9742.MOV')).toBeInTheDocument()
  expect(screen.getByText('IMG_9743.MOV')).toBeInTheDocument()
  expect(screen.getByDisplayValue('Sonnenflecken')).toBeInTheDocument()
  expect(screen.getByRole('combobox', {name: 'Untergrund IMG_9742.MOV'})).toHaveValue('walduntergrund')
  expect(screen.getByRole('combobox', {name: 'Status IMG_9743.MOV'})).toHaveValue('unlabeled')
})

test('says when a new recording was picked up by the scan', async () => {
  render(<RunRegistry onClose={() => undefined}/>)
  expect(await screen.findByText('1 neue Aufnahme(n) als Run angelegt.')).toBeInTheDocument()
})

test('flags runs whose video file is gone instead of hiding them', async () => {
  render(<RunRegistry onClose={() => undefined}/>)

  expect(await screen.findByText(/1 Run\(s\) ohne Videodatei im Ordner/)).toBeInTheDocument()
  expect(screen.getByText(/m-1\/v-2 · Videodatei fehlt/)).toBeInTheDocument()
})

test('writing a surface through says that it applies to every frame of the video', async () => {
  render(<RunRegistry onClose={() => undefined}/>)
  await screen.findByText('IMG_9743.MOV')

  fireEvent.change(screen.getByRole('combobox', {name: 'Untergrund IMG_9743.MOV'}), {target: {value: 'schotterweg'}})

  await waitFor(() => expect(updateRegistryRun).toHaveBeenCalledWith('m-1', 'v-2', {terrain_category: 'schotterweg'}))
  expect(await screen.findByText(/gilt für alle Frames dieses Videos/)).toBeInTheDocument()
})

test('clearing a surface sends null rather than an empty string', async () => {
  render(<RunRegistry onClose={() => undefined}/>)
  await screen.findByText('IMG_9742.MOV')

  fireEvent.change(screen.getByRole('combobox', {name: 'Untergrund IMG_9742.MOV'}), {target: {value: ''}})

  await waitFor(() => expect(updateRegistryRun).toHaveBeenCalledWith('m-1', 'v-1', {terrain_category: null}))
})
