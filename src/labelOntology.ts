import {useEffect, useState} from 'react'
import {getLabelOntology} from './api'
import type {LabelOntology, LabelClass, NormalizedPoint} from './types'

/** Eine markierte Fläche im Labeler: Punkte plus die Metadaten, die das Backend
 *  je Polygon erwartet. */
export type LabelShape = {
  id: string
  class_id: string
  points: NormalizedPoint[]
  certainty: 'certain' | 'uncertain' | 'partially_occluded'
  origin: 'manual' | 'model_proposal' | 'manual_corrected' | 'human_confirmed'
  hard_negative: boolean
  note: string
  uncertainty_reason?: string
  /** Zeitliche Verkettung: dieselbe Stelle über mehrere Frames hinweg. */
  tracking_id?: string | null
  carried_from_frame?: number | null
  edit?: 'new' | 'carried_unchanged' | 'carried_adjusted' | 'corrected'
}

/** Notfallliste, falls die Ontologie nicht geladen werden kann.
 *
 *  Bewusst nur die vier Kernklassen: lieber eingeschränkt weiterlabeln als mit
 *  einer veralteten Kopie der vollen Liste arbeiten, die dann von der des
 *  Backends abweicht. */
export const FALLBACK_CORE: LabelClass[] = [
  {class_id: 'traversable', layer: 'core', label: 'Befahrbarer Boden', color: '#55d96f', value: 1, description: ''},
  {class_id: 'restricted', layer: 'core', label: 'Eingeschränkt befahrbar', color: '#e4c264', value: 4, description: ''},
  {class_id: 'not_traversable', layer: 'core', label: 'Nicht befahrbar', color: '#e05b52', value: 2, description: ''},
  {class_id: 'unknown', layer: 'core', label: 'Nicht bewertbar / verdeckt', color: '#737c78', value: 3, description: ''},
]

export function useLabelOntology() {
  const [ontology, setOntology] = useState<LabelOntology | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    void getLabelOntology()
      .then(result => {
        if (!cancelled) setOntology(result)
      })
      .catch(problem => {
        if (!cancelled) setError(problem instanceof Error ? problem.message : 'Labelklassen konnten nicht geladen werden')
      })
    return () => {
      cancelled = true
    }
  }, [])

  const layers = ontology?.layers ?? []
  const classesOf = (layer: string) => layers.find(item => item.layer === layer)?.classes ?? (layer === 'core' ? FALLBACK_CORE : [])
  const all = layers.flatMap(item => item.classes)

  return {
    ontology,
    error,
    layers,
    classesOf,
    /** Farbe und Beschriftung einer Klasse — ohne Ontologie ein sichtbarer Platzhalter. */
    describe: (class_id: string): LabelClass =>
      all.find(item => item.class_id === class_id) ??
      FALLBACK_CORE.find(item => item.class_id === class_id) ?? {
        class_id,
        layer: 'core',
        label: class_id,
        color: '#8c9690',
        value: null,
        description: '',
      },
  }
}

export const shapeId = (class_id: string, index: number) => `${class_id.replace(/_/g, '-')}-${index + 1}`
