import {expect, test} from 'vitest'
import {
  AI_BINARY_PALETTE,
  GRADE_ONTOLOGY_FALLBACK,
  decodeRleValues,
  encodeRleValues,
  gradeEntries,
  hexToRgba,
  maskShares,
  paletteFromGradeOntology,
  rleValueAt,
} from './masks'

test('parses six- and eight-digit hex colours including full transparency', () => {
  expect(hexToRgba('#1e8c46')).toEqual([30, 140, 70, 255])
  expect(hexToRgba('#00000000')).toEqual([0, 0, 0, 0])
  expect(hexToRgba('#55d96f', 128)).toEqual([85, 217, 111, 128])
})

test('builds a palette keyed by class value from the backend ontology', () => {
  const palette = paletteFromGradeOntology(GRADE_ONTOLOGY_FALLBACK)

  expect(palette[1]).toEqual([30, 140, 70, 255])
  expect(palette[5]).toEqual([224, 91, 82, 255])
  // Klasse 0 ist Umgebung und muss unsichtbar bleiben.
  expect(palette[0][3]).toBe(0)
})

test('orders legend entries by class value', () => {
  expect(gradeEntries().map(entry => entry.value)).toEqual([0, 1, 2, 3, 4, 5])
  expect(gradeEntries().map(entry => entry.key)).toEqual(['unrated', 'safe', 'good', 'marginal', 'risky', 'problem'])
})

test('reports class shares directly from the run-length encoding', () => {
  const shares = maskShares({width: 2, height: 2, rle: [1, 2, 5, 2]})

  expect(shares[1]).toBeCloseTo(.5)
  expect(shares[5]).toBeCloseTo(.5)
})

test('round-trips values through the run-length encoding', () => {
  const values = [0, 0, 4, 4, 5, 1]

  const mask = encodeRleValues(values, 3, 2)

  expect(mask.rle).toEqual([0, 2, 4, 2, 5, 1, 1, 1])
  expect(decodeRleValues(mask)).toEqual(values)
})

test('resolves the class under a normalized point', () => {
  const mask = {width: 2, height: 2, rle: [0, 1, 2, 1, 3, 1, 1, 1]}

  expect(rleValueAt(mask, [.75, .25])).toBe(2)
  expect(rleValueAt(mask, [.25, .75])).toBe(3)
})

test('keeps the binary fallback palette turquoise for path pixels only', () => {
  expect(AI_BINARY_PALETTE[0][3]).toBe(0)
  expect(AI_BINARY_PALETTE[1]).toEqual([64, 220, 235, 255])
})
