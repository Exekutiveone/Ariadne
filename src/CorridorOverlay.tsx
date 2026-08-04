import {useRef} from 'react'
import type {PointerEvent as ReactPointerEvent} from 'react'
import type {CorridorCheck, NormalizedPoint} from './types'

const polyline = (points: number[][]) => points.map(([x, y]) => `${x},${y}`).join(' ')
/** Korridor als geschlossene Flaeche: rechter Rand hinunter, linker Rand hinauf. */
const wedge = (left: number[][], right: number[][]) => polyline([...right, ...[...left].reverse()])

/** Zeichnet die Geometrie der Korridorprüfung über das Videobild: Fluchtpunkt,
 *  nicht ausgewertete Zone, die drei Korridore und die Trajektorie.
 *
 *  Das viewBox ist 0..1, weil alle Punkte auf das Bild normiert sind. Striche
 *  bekommen `vector-effect`, damit sie durch die nicht-uniforme Skalierung
 *  nicht verzerren. */
export default function CorridorOverlay({
  check,
  activeCorridor,
  onSelectCorridor,
  proposal,
  draft,
  planning,
  aspect,
  onMovePoint,
  onAddPoint,
  onRemovePoint,
}: {
  check: CorridorCheck | null
  activeCorridor: string
  onSelectCorridor: (corridor: string) => void
  proposal: number[][]
  draft: NormalizedPoint[] | null
  planning: boolean
  aspect: number
  onMovePoint: (index: number, point: NormalizedPoint) => void
  onAddPoint: (point: NormalizedPoint) => void
  onRemovePoint: (index: number) => void
}) {
  const surfaceRef = useRef<SVGSVGElement>(null)
  const draggingRef = useRef<number | null>(null)
  if (!check) return null

  const {decomposition, corridors} = check
  const [vanishingX, vanishingY] = decomposition.vanishing_point_normalized
  const skyline = decomposition.first_evaluated_row_normalized
  // Griffe sollen rund erscheinen: horizontal in Bildbreiten, vertikal mit dem
  // Seitenverhaeltnis gegengerechnet.
  const handleX = 0.011
  const handleY = handleX * aspect

  const pointAt = (event: ReactPointerEvent): NormalizedPoint | null => {
    const bounds = surfaceRef.current?.getBoundingClientRect()
    if (!bounds || bounds.width <= 0 || bounds.height <= 0) return null
    return [
      Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width)),
      Math.min(1, Math.max(0, (event.clientY - bounds.top) / bounds.height)),
    ]
  }

  return (
    <svg
      ref={surfaceRef}
      className={`corridor-overlay ${planning ? 'planning' : ''}`}
      viewBox="0 0 1 1"
      preserveAspectRatio="none"
      aria-label="Korridore und Trajektorie"
      onPointerMove={event => {
        if (draggingRef.current === null) return
        const point = pointAt(event)
        if (point) onMovePoint(draggingRef.current, point)
      }}
      onPointerUp={() => {
        draggingRef.current = null
      }}
      onPointerLeave={() => {
        draggingRef.current = null
      }}
      onClick={event => {
        if (!planning || draggingRef.current !== null) return
        const point = pointAt(event as unknown as ReactPointerEvent)
        if (point) onAddPoint(point)
      }}
    >
      {/* A.4: alles oberhalb des Fluchtpunkts wird nicht ausgewertet. */}
      <rect x="0" y="0" width="1" height={skyline} className="corridor-irrelevant" />
      <line x1="0" y1={skyline} x2="1" y2={skyline} className="corridor-skyline" vectorEffect="non-scaling-stroke" />

      {/* Die beiden Linien von den unteren Bildecken zum Fluchtpunkt. */}
      <polyline
        points={polyline(decomposition.relevant_triangle_normalized)}
        className="corridor-triangle"
        vectorEffect="non-scaling-stroke"
      />

      {corridors.map(corridor => (
        <g
          key={corridor.corridor}
          className={`corridor-wedge ${corridor.status} ${corridor.corridor === activeCorridor ? 'active' : ''}`}
          onClick={event => {
            event.stopPropagation()
            onSelectCorridor(corridor.corridor)
          }}
        >
          <polygon points={wedge(corridor.geometry.left, corridor.geometry.right)} />
          <polyline points={polyline(corridor.geometry.center)} className="corridor-centerline" vectorEffect="non-scaling-stroke" />
          <title>{`${corridor.label}: ${corridor.status_label} — ${corridor.reason}`}</title>
        </g>
      ))}

      {/* Vorschlag der KI, gestrichelt; der bearbeitete Verlauf liegt darüber. */}
      {proposal.length > 1 && <polyline points={polyline(proposal)} className="corridor-proposal" vectorEffect="non-scaling-stroke" />}

      {draft && draft.length > 1 && <polyline points={polyline(draft)} className="corridor-draft" vectorEffect="non-scaling-stroke" />}
      {draft?.map((point, index) => (
        <ellipse
          key={index}
          cx={point[0]}
          cy={point[1]}
          rx={handleX}
          ry={handleY}
          className="corridor-handle"
          onPointerDown={event => {
            event.stopPropagation()
            draggingRef.current = index
            ;(event.target as Element).setPointerCapture?.(event.pointerId)
          }}
          onContextMenu={event => {
            event.preventDefault()
            event.stopPropagation()
            onRemovePoint(index)
          }}
        />
      ))}

      {/* Fluchtpunkt zuletzt, damit er immer obenauf liegt. */}
      <g className="corridor-vanishing">
        <line x1={vanishingX - 0.035} y1={vanishingY} x2={vanishingX + 0.035} y2={vanishingY} vectorEffect="non-scaling-stroke" />
        <line
          x1={vanishingX}
          y1={vanishingY - 0.035 * aspect}
          x2={vanishingX}
          y2={vanishingY + 0.035 * aspect}
          vectorEffect="non-scaling-stroke"
        />
        <ellipse cx={vanishingX} cy={vanishingY} rx={0.014} ry={0.014 * aspect} vectorEffect="non-scaling-stroke" />
      </g>
    </svg>
  )
}
