import {useEffect, useMemo, useState} from 'react'
import {getTerrainDashboard, predictTerrainVideo, trainTerrainModel} from './api'
import {terrainCategoryLabel} from './terrainCategories'
import type {TerrainDashboardData, TerrainMetrics, TerrainPredictionRun, TerrainSplitPart} from './types'

const percentFormat = new Intl.NumberFormat('de-DE', {minimumFractionDigits: 1, maximumFractionDigits: 1})
const percent = (value: number) => `${percentFormat.format(value * 100)} %`
const timestamp = (value: string) => new Date(value).toLocaleString('de-DE')
const clock = (milliseconds: number) => {
  const total = Math.round(milliseconds / 1000)
  return `${Math.floor(total / 60)}:${`${total % 60}`.padStart(2, '0')}`
}

function MetricTiles({metrics, label}: {metrics: TerrainMetrics; label: string}) {
  return (
    <div className="global-totals model">
      <div>
        <span>{label} · Frames</span>
        <b>{metrics.frames.toLocaleString('de-DE')}</b>
      </div>
      <div>
        <span>Trefferquote</span>
        <b>{percent(metrics.accuracy)}</b>
      </div>
      <div>
        <span>Klassenmittel</span>
        <b>{percent(metrics.balanced_accuracy)}</b>
      </div>
      <div>
        <span>Unsicher</span>
        <b>{percent(metrics.uncertain_fraction)}</b>
      </div>
      <div>
        <span>Ø Konfidenz</span>
        <b>{percent(metrics.mean_confidence)}</b>
      </div>
    </div>
  )
}

function SplitRow({name, part, hint}: {name: string; part: TerrainSplitPart | null; hint: string}) {
  return (
    <article className={part ? 'included' : ''}>
      <i />
      <div>
        <b>{name}</b>
        <span>{part ? part.classes.map(terrainCategoryLabel).join(' · ') : 'nicht gebildet'}</span>
      </div>
      <strong>{part ? `${part.videos} Videos` : '—'}</strong>
      <small>{part ? `${part.frames.toLocaleString('de-DE')} Frames · ${hint}` : hint}</small>
    </article>
  )
}

function ConfusionMatrix({metrics, classes}: {metrics: TerrainMetrics; classes: string[]}) {
  return (
    <div className="terrain-matrix-scroll">
      <table className="terrain-matrix">
        <caption>Zeilen: tatsächliche Videokategorie · Spalten: Vorhersage</caption>
        <thead>
          <tr>
            <th scope="col">Ist \ Vorhersage</th>
            {classes.map(item => (
              <th key={item} scope="col">
                {terrainCategoryLabel(item)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {metrics.confusion_matrix.map((row, rowIndex) => (
            <tr key={classes[rowIndex]}>
              <th scope="row">{terrainCategoryLabel(classes[rowIndex])}</th>
              {row.map((value, columnIndex) => (
                <td key={classes[columnIndex]} className={rowIndex === columnIndex ? 'diagonal' : value ? 'confused' : ''}>
                  {value}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function TerrainModelPanel() {
  const [data, setData] = useState<TerrainDashboardData | null>(null)
  const [message, setMessage] = useState('')
  const [training, setTraining] = useState(false)
  const [frameStride, setFrameStride] = useState(15)
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.6)
  const [selectedVideoKey, setSelectedVideoKey] = useState('')
  const [predicting, setPredicting] = useState(false)
  const [predictionRun, setPredictionRun] = useState<TerrainPredictionRun | null>(null)

  const load = () =>
    getTerrainDashboard()
      .then(setData)
      .catch(error => setMessage(error instanceof Error ? error.message : 'Terrainmodell konnte nicht geladen werden'))
  useEffect(() => {
    void load()
  }, [])

  useEffect(() => {
    const model = data?.model
    if (!model) return
    setFrameStride(model.dataset.frame_stride)
    setConfidenceThreshold(model.model.confidence_threshold)
  }, [data?.model?.run_id])

  const videos = data?.dataset.videos ?? []
  useEffect(() => {
    if (!selectedVideoKey && videos.length) setSelectedVideoKey(`${videos[0].mission_id}/${videos[0].video_id}`)
  }, [videos, selectedVideoKey])

  const selectedVideo = videos.find(video => `${video.mission_id}/${video.video_id}` === selectedVideoKey) ?? null
  const uncertainFrames = useMemo(() => predictionRun?.frames.filter(frame => frame.uncertain) ?? [], [predictionRun])

  const train = async () => {
    if (training) return
    setTraining(true)
    setPredictionRun(null)
    setMessage(`Terrainmodell wird auf allen Frames kategorisierter Videos trainiert (jeder ${frameStride}. Frame) …`)
    try {
      const model = await trainTerrainModel({frame_stride: frameStride, confidence_threshold: confidenceThreshold})
      await load()
      setMessage(
        `Lauf ${model.run_id} gespeichert: ${percent(model.validation_metrics.accuracy)} auf der Validierung, ${model.classes.length} Klassen.`,
      )
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Terrainmodell konnte nicht trainiert werden')
    } finally {
      setTraining(false)
    }
  }

  const classifyVideo = async () => {
    if (!selectedVideo || predicting) return
    setPredicting(true)
    setMessage(`${selectedVideo.original_name} wird klassifiziert …`)
    try {
      const run = await predictTerrainVideo(selectedVideo.mission_id, selectedVideo.video_id, {
        frame_stride: frameStride,
        confidence_threshold: confidenceThreshold,
      })
      setPredictionRun(run)
      await load()
      setMessage(`Vorhersagelauf ${run.run_id} gespeichert: ${run.summary.frames} Frames, davon ${run.summary.uncertain_frames} unsicher.`)
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Video konnte nicht klassifiziert werden')
    } finally {
      setPredicting(false)
    }
  }

  if (!data)
    return (
      <section className="global-dataset-panel">
        <div className="section-head">
          <h2>Terrainklassifizierung</h2>
        </div>
        <div className="empty">{message || 'Terraindaten werden geladen …'}</div>
      </section>
    )

  const {dataset, model, runs} = data
  const trainable = dataset.totals.classes >= 2
  return (
    <>
      <section className="global-dataset-panel terrain-panel">
        <div className="global-dataset-head">
          <div>
            <h2>Terrainklassifizierung</h2>
            <p>
              Bildausschnitt → Terrainklasse. Jeder Frame erbt die aktuell am Video gesetzte Kategorie; ein Umlabeln des Videos ändert damit
              sofort alle seine Frames.
            </p>
          </div>
          <button disabled={training || !trainable} onClick={() => void train()}>
            {training ? 'TERRAINTRAINING LÄUFT …' : 'TERRAINMODELL TRAINIEREN'}
          </button>
        </div>
        <div className="global-totals">
          <div>
            <span>Kategorisierte Videos</span>
            <b>{dataset.totals.categorized_videos}</b>
          </div>
          <div>
            <span>Ohne Kategorie</span>
            <b>{dataset.totals.uncategorized_videos}</b>
          </div>
          <div>
            <span>Terrainklassen</span>
            <b>{dataset.totals.classes}</b>
          </div>
          <div>
            <span>Missionen</span>
            <b>{dataset.totals.missions}</b>
          </div>
          <div>
            <span>Trainingsläufe</span>
            <b>{runs.training_runs.length}</b>
          </div>
        </div>
        <div className="terrain-class-chips">
          {dataset.classes.map(item => (
            <span key={item.terrain_category}>
              <b>{terrainCategoryLabel(item.terrain_category)}</b>
              <small>
                {item.videos} Videos · {item.missions} Missionen
              </small>
            </span>
          ))}
        </div>
        <div className="terrain-training-controls">
          <label>
            Jeden n-ten Frame
            <input
              aria-label="Schrittweite der Frames"
              type="number"
              min="1"
              max="600"
              step="1"
              value={frameStride}
              onChange={event => {
                const value = Number(event.target.value)
                if (Number.isFinite(value) && value >= 1) setFrameStride(Math.min(600, Math.round(value)))
              }}
            />
            <small>
              Benachbarte Frames eines Videos sind nahezu identisch. Die Schrittweite gilt für Training und Vorhersage und wird im Lauf
              mitgespeichert.
            </small>
          </label>
          <label>
            Konfidenzschwelle · {percent(confidenceThreshold)}
            <input
              aria-label="Konfidenzschwelle"
              type="range"
              min="0.05"
              max="0.99"
              step="0.01"
              value={confidenceThreshold}
              onChange={event => setConfidenceThreshold(+event.target.value)}
            />
            <small>Frames unterhalb dieser Konfidenz gelten als unsicher und werden keiner Klasse verbindlich zugeordnet.</small>
          </label>
        </div>
        {!trainable && (
          <div className="empty">
            Für ein Terrainmodell werden Videos aus mindestens zwei verschiedenen Terrainkategorien benötigt. Kategorien werden im Labeler
            pro Video gesetzt.
          </div>
        )}
        {dataset.totals.uncategorized_videos > 0 && (
          <small className="terrain-hint">
            {dataset.totals.uncategorized_videos} Video(s) ohne Terrainkategorie bleiben außen vor. Sie können trotzdem klassifiziert
            werden.
          </small>
        )}
        {message && (
          <div className="labeling-message" role="status">
            {message}
          </div>
        )}
      </section>

      <section className="global-result-panel terrain-panel">
        <div className="section-head">
          <h2>Aktives Terrainmodell</h2>
          <p>Trainings-, Validierungs- und Testdaten sind strikt nach Video getrennt — kein Video liegt in zwei Teilmengen.</p>
        </div>
        {!model ? (
          <div className="empty">Noch kein Terrainmodell trainiert.</div>
        ) : (
          <>
            <div className="active-model-version">
              <div>
                <span>Modelllauf</span>
                <b>{model.run_id}</b>
                <small>
                  {timestamp(model.created_at)} · jeder {model.dataset.frame_stride}. Frame · Schwelle{' '}
                  {percent(model.model.confidence_threshold)}
                </small>
              </div>
              <strong className="changed">
                {model.classes.length} KLASSEN · {model.dataset.frames.toLocaleString('de-DE')} FRAMES
              </strong>
            </div>
            <MetricTiles metrics={model.validation_metrics} label="Validierung" />
            {model.test_metrics && <MetricTiles metrics={model.test_metrics} label="Test" />}
            <div className="global-mission-list terrain-split">
              <SplitRow name="Training" part={model.split.train} hint="lernt die Zuordnung" />
              <SplitRow name="Validierung" part={model.split.validation} hint="wählt die Konfidenzkalibrierung" />
              <SplitRow name="Test" part={model.split.test} hint="unabhängige Endkontrolle" />
            </div>
            {model.split.notes.map(note => (
              <small key={note} className="terrain-hint">
                {note}
              </small>
            ))}
            <div className="terrain-matrix-scroll">
              <table className="terrain-matrix">
                <caption>Güte je Terrainklasse auf der Validierung</caption>
                <thead>
                  <tr>
                    <th scope="col">Terrainklasse</th>
                    <th scope="col">Frames</th>
                    <th scope="col">Präzision</th>
                    <th scope="col">Trefferquote</th>
                    <th scope="col">F1</th>
                  </tr>
                </thead>
                <tbody>
                  {model.validation_metrics.per_class.map(item => (
                    <tr key={item.terrain_category}>
                      <th scope="row">{terrainCategoryLabel(item.terrain_category)}</th>
                      <td>{item.support}</td>
                      <td>{percent(item.precision)}</td>
                      <td>{percent(item.recall)}</td>
                      <td>{percent(item.f1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <ConfusionMatrix metrics={model.validation_metrics} classes={model.classes} />
            <ul className="terrain-limitations">
              {model.limitations.map(item => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </>
        )}
      </section>

      <section className="global-result-panel terrain-panel">
        <div className="section-head">
          <h2>Video klassifizieren</h2>
          <p>Jeder Vorhersagevorgang wird als eigener Lauf gespeichert und überschreibt keinen früheren.</p>
        </div>
        {!model ? (
          <div className="empty">Trainiere zuerst ein Terrainmodell.</div>
        ) : (
          <>
            <div className="terrain-predict-controls">
              <label>
                Video
                <select
                  value={selectedVideoKey}
                  onChange={event => {
                    setSelectedVideoKey(event.target.value)
                    setPredictionRun(null)
                  }}
                >
                  {videos.map(video => (
                    <option key={`${video.mission_id}/${video.video_id}`} value={`${video.mission_id}/${video.video_id}`}>
                      {video.mission_name} · {video.original_name} · {terrainCategoryLabel(video.terrain_category)}
                    </option>
                  ))}
                </select>
              </label>
              <button disabled={predicting || !selectedVideo} onClick={() => void classifyVideo()}>
                {predicting ? 'VIDEO WIRD KLASSIFIZIERT …' : 'VIDEO KLASSIFIZIEREN'}
              </button>
            </div>
            {predictionRun && (
              <>
                <div className="global-totals model">
                  <div>
                    <span>Frames</span>
                    <b>{predictionRun.summary.frames.toLocaleString('de-DE')}</b>
                  </div>
                  <div>
                    <span>Häufigste Klasse</span>
                    <b>{terrainCategoryLabel(predictionRun.summary.dominant_category)}</b>
                  </div>
                  <div>
                    <span>Unsicher</span>
                    <b>
                      {predictionRun.summary.uncertain_frames} · {percent(predictionRun.summary.uncertain_fraction)}
                    </b>
                  </div>
                  <div>
                    <span>Ø Konfidenz</span>
                    <b>{percent(predictionRun.summary.mean_confidence)}</b>
                  </div>
                  <div>
                    <span>Deckung mit Videolabel</span>
                    <b>
                      {predictionRun.summary.matches_video_category === null
                        ? 'kein Label'
                        : percent(predictionRun.summary.matches_video_category)}
                    </b>
                  </div>
                </div>
                <div className="terrain-class-chips">
                  {Object.entries(predictionRun.summary.counts).map(([category, count]) => (
                    <span key={category}>
                      <b>{terrainCategoryLabel(category)}</b>
                      <small>{count} Frames</small>
                    </span>
                  ))}
                </div>
                {uncertainFrames.length > 0 ? (
                  <div className="terrain-uncertain-list">
                    <b>
                      {uncertainFrames.length} unsichere Frames unterhalb {percent(predictionRun.confidence_threshold)}
                    </b>
                    <div>
                      {uncertainFrames.slice(0, 60).map(frame => (
                        <span key={frame.frame_index}>
                          <i>Frame {frame.frame_index + 1}</i>
                          <small>
                            {clock(frame.timestamp_ms)} · unsicher, am ehesten {terrainCategoryLabel(frame.top_category)} (
                            {percent(frame.confidence)})
                          </small>
                        </span>
                      ))}
                    </div>
                    {uncertainFrames.length > 60 && (
                      <small className="terrain-hint">
                        Es werden die ersten 60 unsicheren Frames angezeigt; der gespeicherte Lauf {predictionRun.run_id} enthält alle.
                      </small>
                    )}
                  </div>
                ) : (
                  <small className="terrain-hint">Kein Frame liegt unter der Konfidenzschwelle.</small>
                )}
              </>
            )}
          </>
        )}
      </section>

      <section className="global-result-panel terrain-panel">
        <div className="section-head">
          <h2>Gespeicherte Terrainläufe</h2>
          <p>Vorhandene Läufe, Ergebnisse und Modelle bleiben unverändert erhalten.</p>
        </div>
        {!runs.training_runs.length ? (
          <div className="empty">Noch keine Läufe gespeichert.</div>
        ) : (
          <div className="global-mission-list">
            {runs.training_runs.map(run => (
              <article key={run.run_id} className={run.active ? 'included' : ''}>
                <i />
                <div>
                  <b>{run.run_id}</b>
                  <span>
                    {timestamp(run.created_at)} · jeder {run.frame_stride}. Frame · {run.classes.length} Klassen
                  </span>
                </div>
                <strong>{percent(run.validation_accuracy)}</strong>
                <small>
                  {run.active ? 'aktiver Lauf · ' : ''}Validierung
                  {run.test_accuracy === null ? ' · kein Test' : ` · Test ${percent(run.test_accuracy)}`}
                </small>
              </article>
            ))}
          </div>
        )}
        {runs.prediction_runs.length > 0 && (
          <div className="global-mission-list terrain-prediction-runs">
            {runs.prediction_runs.map(run => (
              <article key={run.run_id}>
                <i />
                <div>
                  <b>{run.original_name}</b>
                  <span>
                    {run.run_id} · {timestamp(run.created_at)}
                  </span>
                </div>
                <strong>{terrainCategoryLabel(run.dominant_category)}</strong>
                <small>
                  {run.predicted_frames} Frames · {run.uncertain_frames} unsicher · Modell {run.model_run_id}
                </small>
              </article>
            ))}
          </div>
        )}
      </section>
    </>
  )
}
