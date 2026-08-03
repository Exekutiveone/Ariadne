import {SAFETY_NOTE, gradeEntries, maskShares} from './masks'
import type {GradeOntology, Grading, TerrainMask} from './types'

const percent = (value: number) => `${Math.round(value * 100)} %`

/**
 * Gemeinsame Legende der Abstufungsmaske. Wird im Labeler, im Modellzentrum
 * und im Analyseplayer identisch verwendet, damit Farben und Beschriftungen
 * nirgends auseinanderlaufen.
 */
export default function GradeLegend({ontology, mask, grading, title = 'Abgestufte Befahrbarkeit'}: {
  ontology?: GradeOntology
  mask?: TerrainMask | null
  grading?: Grading | null
  title?: string
}) {
  const entries = gradeEntries(ontology)
  const shares = mask ? maskShares(mask) : null
  return <div className="grade-legend">
    <b>{title}</b>
    <ul>
      {entries.map(entry => <li key={entry.key}>
        <i style={{background: entry.value === 0 ? 'transparent' : entry.color, borderColor: entry.color}} aria-hidden="true"/>
        <span>{entry.label}</span>
        {shares && <small>{percent(shares[entry.value] ?? 0)}</small>}
      </li>)}
    </ul>
    {grading && <small className="grade-bands">
      Grenzen auf dem normierten Schwellenabstand: sicher ab {grading.bands.safe_min_margin},
      gut ab {grading.bands.good_min_margin}, Risikoband ab {grading.bands.risky_min_margin}.
    </small>}
    <small className="grade-disclaimer">{SAFETY_NOTE}</small>
  </div>
}
