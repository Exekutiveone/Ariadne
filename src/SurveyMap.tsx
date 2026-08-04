import {useMapEvents, MapContainer, Marker, Polyline, TileLayer, Tooltip} from 'react-leaflet'
import type {Point} from './types'
function Clicker({onClick}: {onClick: (p: Point) => void}) {
  useMapEvents({click: e => onClick({lat: +e.latlng.lat.toFixed(6), lng: +e.latlng.lng.toFixed(6)})})
  return null
}
export default function SurveyMap({route, onChange}: {route: Point[]; onChange: (p: Point[]) => void}) {
  const add = (p: Point) => onChange([...route, p])
  return (
    <div className="map-wrap">
      <MapContainer center={route[0] || [48.7, 9]} zoom={route.length ? 15 : 8} scrollWheelZoom className="map">
        <TileLayer attribution="&copy; OpenStreetMap" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        <Clicker onClick={add} />
        {route.map((p, i) => (
          <Marker
            key={i}
            position={p}
            draggable
            eventHandlers={{
              dragend: e => {
                const n = [...route],
                  ll = e.target.getLatLng()
                n[i] = {lat: ll.lat, lng: ll.lng}
                onChange(n)
              },
            }}
          >
            <Tooltip permanent>{i === 0 ? 'A' : i === route.length - 1 ? 'B' : i + 1}</Tooltip>
          </Marker>
        ))}
        <Polyline positions={route} pathOptions={{color: '#d7f26b', weight: 5}} />
      </MapContainer>
      <div className="map-tools">
        <span>Klicken: Wegpunkt setzen · Marker ziehen: korrigieren</span>
        <button type="button" onClick={() => onChange(route.slice(0, -1))} disabled={!route.length}>
          Letzten entfernen
        </button>
        <button type="button" onClick={() => onChange([])} disabled={!route.length}>
          Route löschen
        </button>
      </div>
    </div>
  )
}
