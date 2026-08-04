import type {GradeOntology, NormalizedPoint, TerrainMask} from './types'

export type Rgba = [number, number, number, number]
export type MaskPalette = Record<number, Rgba>

/**
 * Fallback, falls eine Antwort noch ohne `grade_ontology` kommt (aeltere
 * gespeicherte Laeufe). Werte und Farben sind mit GRADE_ONTOLOGY in
 * backend/app/path_model.py abgestimmt und muessen dort mitgepflegt werden.
 */
export const GRADE_ONTOLOGY_FALLBACK: GradeOntology = {
  unrated: {value: 0, label: 'Nicht bewertet / Umgebung', color: '#00000000'},
  safe: {value: 1, label: 'Sicher befahrbar', color: '#1e8c46'},
  good: {value: 2, label: 'Gut befahrbar', color: '#55d96f'},
  marginal: {value: 3, label: 'Knapp befahrbar', color: '#a3ecb4'},
  risky: {value: 4, label: 'Potenziell befahrbar, mit Risiko', color: '#f08c3a'},
  problem: {value: 5, label: 'Problemzone / Hindernis', color: '#e05b52'},
}

export const SAFETY_NOTE = 'KI-Einschätzung der Befahrbarkeit, keine sicherheitsrelevante Fahrfreigabe.'

/** Vergleichsmaske gelabelter Frames: korrekt / übersehen / fälschlich erkannt. */
export const COMPARISON_PALETTE: MaskPalette = {
  0: [0, 0, 0, 0],
  1: [58, 214, 92, 255],
  2: [224, 74, 68, 255],
  3: [239, 196, 55, 255],
}

export const COMPARISON_LEGEND: {value: number; label: string; color: string}[] = [
  {value: 1, label: 'Korrekt erkannter Weg', color: '#3ad65c'},
  {value: 2, label: 'Übersehene Wegfläche', color: '#e04a44'},
  {value: 3, label: 'Fälschlich erkannter Weg', color: '#efc437'},
]

/** Einfarbige KI-Wegmaske ohne Abstufung (Rückfallebene). */
export const AI_BINARY_PALETTE: MaskPalette = {0: [0, 0, 0, 0], 1: [64, 220, 235, 255]}

export function hexToRgba(hex: string, alpha?: number): Rgba {
  const value = hex.replace('#', '')
  const channels =
    value.length >= 6 ? [parseInt(value.slice(0, 2), 16), parseInt(value.slice(2, 4), 16), parseInt(value.slice(4, 6), 16)] : [0, 0, 0]
  const encoded = value.length === 8 ? parseInt(value.slice(6, 8), 16) : 255
  return [channels[0], channels[1], channels[2], alpha ?? encoded]
}

export function paletteFromGradeOntology(ontology: GradeOntology = GRADE_ONTOLOGY_FALLBACK): MaskPalette {
  const palette: MaskPalette = {}
  for (const entry of Object.values(ontology)) palette[entry.value] = hexToRgba(entry.color)
  return palette
}

/** Ontologie-Einträge nach Wert sortiert — stabile Reihenfolge für Legenden. */
export function gradeEntries(ontology: GradeOntology = GRADE_ONTOLOGY_FALLBACK) {
  return Object.entries(ontology)
    .map(([key, entry]) => ({key, ...entry}))
    .sort((left, right) => left.value - right.value)
}

/** Flächenanteil je Klassenwert, direkt aus der Lauflängenkodierung. */
export function maskShares(mask: TerrainMask) {
  const total = Math.max(1, mask.width * mask.height)
  const shares: Record<number, number> = {}
  for (let index = 0; index + 1 < mask.rle.length; index += 2) {
    shares[mask.rle[index]] = (shares[mask.rle[index]] ?? 0) + Math.max(0, mask.rle[index + 1])
  }
  for (const key of Object.keys(shares)) shares[+key] = shares[+key] / total
  return shares
}

export function rleValueAt(mask: TerrainMask, point: NormalizedPoint) {
  const x = Math.min(mask.width - 1, Math.max(0, Math.floor(point[0] * mask.width)))
  const y = Math.min(mask.height - 1, Math.max(0, Math.floor(point[1] * mask.height)))
  const target = y * mask.width + x
  let cursor = 0
  for (let index = 0; index + 1 < mask.rle.length; index += 2) {
    cursor += mask.rle[index + 1]
    if (target < cursor) return mask.rle[index]
  }
  return 0
}

export function decodeRleValues(mask: TerrainMask) {
  const values = new Array<number>(mask.width * mask.height).fill(0)
  let pixel = 0
  for (let index = 0; index + 1 < mask.rle.length && pixel < values.length; index += 2) {
    const end = Math.min(values.length, pixel + Math.max(0, Math.floor(mask.rle[index + 1])))
    values.fill(mask.rle[index], pixel, end)
    pixel = end
  }
  return values
}

export function encodeRleValues(values: number[], width: number, height: number): TerrainMask {
  const size = width * height
  const normalized = values.length === size ? values : new Array<number>(size).fill(0)
  const rle: number[] = []
  if (size) {
    let previous = normalized[0] ?? 0
    let count = 1
    for (let index = 1; index < size; index++) {
      const value = normalized[index] ?? 0
      if (value === previous) count++
      else {
        rle.push(previous, count)
        previous = value
        count = 1
      }
    }
    rle.push(previous, count)
  }
  return {width, height, rle}
}

/**
 * Einzige Stelle, die eine RLE-Maske in Pixel uebersetzt. Fuellt ein
 * ImageData in Maskenaufloesung; Werte ohne Paletteneintrag bleiben
 * transparent.
 */
export function maskToImageData(context: CanvasRenderingContext2D, mask: TerrainMask, palette: MaskPalette) {
  const image = context.createImageData(mask.width, mask.height)
  const pixelCount = mask.width * mask.height
  let pixel = 0
  for (let index = 0; index + 1 < mask.rle.length && pixel < pixelCount; index += 2) {
    const colour = palette[mask.rle[index]] ?? [0, 0, 0, 0]
    const end = Math.min(pixelCount, pixel + Math.max(0, Math.floor(mask.rle[index + 1])))
    if (colour[3] > 0) {
      for (let target = pixel; target < end; target++) {
        const offset = target * 4
        image.data[offset] = colour[0]
        image.data[offset + 1] = colour[1]
        image.data[offset + 2] = colour[2]
        image.data[offset + 3] = colour[3]
      }
    }
    pixel = end
  }
  return image
}

/** Malt eine Maske formatfuellend in ein eigenes Canvas (CSS skaliert es). */
export function paintMaskCanvas(canvas: HTMLCanvasElement, mask: TerrainMask | null, palette: MaskPalette) {
  const context = canvas.getContext('2d')
  if (!context) return
  if (!mask || mask.width < 1 || mask.height < 1 || !mask.rle.length) {
    context.clearRect(0, 0, canvas.width, canvas.height)
    return
  }
  canvas.width = mask.width
  canvas.height = mask.height
  context.putImageData(maskToImageData(context, mask, palette), 0, 0)
}

/** Zeichnet eine Maske skaliert in ein bestehendes Overlay-Canvas. */
export function drawMaskScaled(
  context: CanvasRenderingContext2D,
  mask: TerrainMask,
  targetWidth: number,
  targetHeight: number,
  opacity: number,
  palette: MaskPalette,
) {
  if (mask.width < 1 || mask.height < 1 || !mask.rle.length) return
  const source = document.createElement('canvas')
  source.width = mask.width
  source.height = mask.height
  const sourceContext = source.getContext('2d')
  if (!sourceContext) return
  sourceContext.putImageData(maskToImageData(sourceContext, mask, palette), 0, 0)
  context.save()
  context.globalAlpha = opacity
  context.imageSmoothingEnabled = false
  context.drawImage(source, 0, 0, targetWidth, targetHeight)
  context.restore()
}
