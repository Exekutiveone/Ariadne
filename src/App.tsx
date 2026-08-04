import {useEffect, useState} from 'react'
import AnalysisView from './AnalysisView'
import GroundTruthLabeler from './GroundTruthLabeler'
import GlobalModelDashboard from './GlobalModelDashboard'
import RunRegistry from './RunRegistry'
import RefinementHistory from './RefinementHistory'
import TrainingProcessPage, {ModelPerformanceDashboard} from './TrainingProcessPage'
import {getAnalysis, getReconstruction, getSegmentation, listMissions, uploadMission} from './api'
import {terrainCategoryLabel, TERRAIN_CATEGORY_OPTIONS} from './terrainCategories'
import type {Analysis, Mission, Reconstruction, RegistryRun, Segmentation, VideoInput} from './types'

export default function App() {
  const [name, setName] = useState('')
  const [videos, setVideos] = useState<VideoInput[]>([])
  const [notes, setNotes] = useState('')
  const [moveStart, setMoveStart] = useState('')
  const [moveEnd, setMoveEnd] = useState('')
  const [pauseStart, setPauseStart] = useState('')
  const [pauseEnd, setPauseEnd] = useState('')
  const [missions, setMissions] = useState<Mission[]>([])
  const [progress, setProgress] = useState<number | null>(null)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState<Mission | null>(null)
  const [analysis, setAnalysis] = useState<{
    mission: Mission
    data: Analysis
    reconstruction: Reconstruction
    segmentation: Segmentation
  } | null>(null)
  const [analysisLoading, setAnalysisLoading] = useState<Mission | null>(null)
  const [labelingMission, setLabelingMission] = useState<Mission | null>(null)
  const [modelCenter, setModelCenter] = useState(false)
  const [registry, setRegistry] = useState(false)
  const [refinementRun, setRefinementRun] = useState<RegistryRun | null>(null)
  const [refinementHistory, setRefinementHistory] = useState(false)
  const [trainingProcess, setTrainingProcess] = useState(false)
  const [performanceDashboard, setPerformanceDashboard] = useState(false)

  const refresh = () =>
    listMissions()
      .then(setMissions)
      .catch(problem => setError(problem.message))

  useEffect(() => {
    void refresh()
  }, [])

  const files = (list: FileList | null) => {
    if (!list) return
    const incoming = [...list].slice(0, 4 - videos.length).map(file => ({
      file,
      direction: 'A_TO_B' as const,
      orientation: (file.name.toLowerCase().includes('portrait') ? 'PORTRAIT' : 'LANDSCAPE') as VideoInput['orientation'],
      terrainCategory: '',
    }))
    setVideos(current => [...current, ...incoming])
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    setSuccess(null)

    const missing: string[] = []
    if (name.trim().length < 3) missing.push('Missionsname (mindestens 3 Zeichen)')
    if (videos.length < 1) missing.push('mindestens ein Video')
    if (videos.length > 4) missing.push('höchstens vier Videos')
    if (videos.some(video => !video.terrainCategory?.trim())) missing.push('für jedes Video eine Terrainkategorie')
    if (pauseStart && pauseEnd && +pauseEnd < +pauseStart) missing.push('Pausenende nach Pausenbeginn')

    if (missing.length) {
      setError(`Noch erforderlich: ${missing.join(' · ')}`)
      return
    }

    try {
      setProgress(0)
      const saved = await uploadMission(
        {
          name: name.trim(),
          movement_start: moveStart || undefined,
          movement_end: moveEnd || undefined,
          pauses: pauseStart && pauseEnd ? [{start_seconds: +pauseStart, end_seconds: +pauseEnd, note: ''}] : [],
          notes,
        },
        videos,
        setProgress,
      )
      setSuccess(saved)
      setName('')
      setVideos([])
      setNotes('')
      refresh()
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : 'Upload fehlgeschlagen')
    } finally {
      setProgress(null)
    }
  }

  const openAnalysis = async (mission: Mission) => {
    setError('')
    setAnalysisLoading(mission)
    try {
      const [data, reconstruction, segmentation] = await Promise.all([
        getAnalysis(mission.id),
        getReconstruction(mission.id),
        getSegmentation(mission.id),
      ])
      setAnalysis({mission, data, reconstruction, segmentation})
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : 'Auswertung nicht verfügbar')
    } finally {
      setAnalysisLoading(null)
    }
  }

  if (labelingMission) {
    return (
      <main>
        <GroundTruthLabeler
          mission={labelingMission}
          onClose={() => setLabelingMission(null)}
          onProcessingComplete={async () => {
            const mission = labelingMission
            setLabelingMission(null)
            await openAnalysis(mission)
          }}
          onMissionUpdated={refresh}
        />
      </main>
    )
  }

  if (analysis) {
    return (
      <main>
        <AnalysisView
          mission={analysis.mission}
          data={analysis.data}
          reconstruction={analysis.reconstruction}
          segmentation={analysis.segmentation}
          onClose={() => setAnalysis(null)}
        />
      </main>
    )
  }

  if (refinementHistory) {
    return <main><RefinementHistory initialRun={refinementRun} onClose={() => { setRefinementHistory(false); setRefinementRun(null) }} /></main>
  }

  if (trainingProcess) return <main><TrainingProcessPage onClose={() => setTrainingProcess(false)} /></main>
  if (performanceDashboard) return <main><ModelPerformanceDashboard onClose={() => setPerformanceDashboard(false)} /></main>

  if (modelCenter) {
    return (
      <main>
        <GlobalModelDashboard onClose={() => setModelCenter(false)} onOpenRefinement={() => { setModelCenter(false); setRefinementRun(null); setRefinementHistory(true) }} />
      </main>
    )
  }

  if (registry) {
    return (
      <main>
        <RunRegistry
          onClose={() => {
            setRegistry(false)
            void refresh()
          }}
          onOpenRefinement={run => {
            setRegistry(false)
            setRefinementRun(run)
            setRefinementHistory(true)
          }}
        />
      </main>
    )
  }

  return (
    <main>
      <header className="app-header">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">A</div>
          <div>
            <span className="eyebrow">ARTEMIS CIVIL SYSTEMS</span>
            <h1>ARIADNE</h1>
            <p>Video intelligence for field operations</p>
          </div>
        </div>
        <div className="header-actions">
          <nav className="app-navigation" aria-label="Arbeitsbereiche">
            <button onClick={() => setRegistry(true)}>RUN-REGISTRY</button>
            <button onClick={() => setTrainingProcess(true)}>KI-TRAININGSPROZESS</button>
            <button onClick={() => setPerformanceDashboard(true)}>MODELL-DASHBOARD</button>
            <button className="nav-primary" onClick={() => setModelCenter(true)}>KI-MODELLZENTRUM</button>
          </nav>
          <div className="status">
            <i />
            SYSTEM BEREIT
          </div>
        </div>
      </header>

      <div className="layout">
        <form onSubmit={submit}>
          <section>
            <div className="step">01</div>
            <div className="section-head">
              <h2>Survey-Mission</h2>
              <p>Bezeichnung der Aufnahme. Eine Georeferenzierung ist für den aktuellen Labeling-Workflow nicht erforderlich.</p>
            </div>
            <label>
              Missionsname
              <input value={name} onChange={e => setName(e.target.value)} placeholder="z. B. Nordhang Vorerkundung" maxLength={120} />
            </label>
          </section>

          <section>
            <div className="step">02</div>
            <div className="section-head">
              <h2>Originalvideos</h2>
              <p>1–4 Dateien, unverändert, mit Aufnahmeparametern und Terrainklasse</p>
            </div>
            <label className="drop">
              <input type="file" accept="video/*" multiple onChange={e => files(e.target.files)} disabled={videos.length >= 4} />
              <b>Videos auswählen</b>
              <span>Große Dateien werden gestreamt; die Oberfläche zeigt den Fortschritt.</span>
            </label>
            <div className="video-list">
              {videos.map((video, index) => (
                <div className="video" key={video.file.name + index}>
                  <div>
                    <b>{video.file.name}</b>
                    <select
                      aria-label={`Terrainkategorie ${index + 1}`}
                      value={video.terrainCategory ?? ''}
                      onChange={e =>
                        setVideos(current =>
                          current.map((item, itemIndex) => (itemIndex === index ? {...item, terrainCategory: e.target.value} : item)),
                        )
                      }
                    >
                      <option value="">Terrainkategorie wählen …</option>
                      {TERRAIN_CATEGORY_OPTIONS.map(option => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                    <small>
                      {(video.file.size / 1024 / 1024).toFixed(1)} MB ·{' '}
                      {video.terrainCategory ? terrainCategoryLabel(video.terrainCategory) : 'Terrainkategorie fehlt'}
                    </small>
                  </div>
                  <select
                    aria-label={`Laufrichtung ${index + 1}`}
                    value={video.direction}
                    onChange={e =>
                      setVideos(current =>
                        current.map((item, itemIndex) =>
                          itemIndex === index ? {...item, direction: e.target.value as VideoInput['direction']} : item,
                        ),
                      )
                    }
                  >
                    <option value="A_TO_B">A → B</option>
                    <option value="B_TO_A">B → A</option>
                  </select>
                  <select
                    aria-label={`Ausrichtung ${index + 1}`}
                    value={video.orientation}
                    onChange={e =>
                      setVideos(current =>
                        current.map((item, itemIndex) =>
                          itemIndex === index ? {...item, orientation: e.target.value as VideoInput['orientation']} : item,
                        ),
                      )
                    }
                  >
                    <option value="LANDSCAPE">Querformat</option>
                    <option value="PORTRAIT">Hochformat</option>
                  </select>
                  <button type="button" onClick={() => setVideos(current => current.filter((_, itemIndex) => itemIndex !== index))}>
                    Entfernen
                  </button>
                </div>
              ))}
            </div>
          </section>

          <section>
            <div className="step">03</div>
            <div className="section-head">
              <h2>Bewegungsdaten</h2>
              <p>Optional: tatsächliche Aufnahmezeit, Pause und Feldnotizen</p>
            </div>
            <div className="coords">
              <label>
                Bewegungsbeginn
                <input type="datetime-local" value={moveStart} onChange={e => setMoveStart(e.target.value)} />
              </label>
              <label>
                Bewegungsende
                <input type="datetime-local" value={moveEnd} onChange={e => setMoveEnd(e.target.value)} />
              </label>
              <label>
                Pause ab Sekunde
                <input type="number" min="0" value={pauseStart} onChange={e => setPauseStart(e.target.value)} />
              </label>
              <label>
                Pause bis Sekunde
                <input type="number" min="0" value={pauseEnd} onChange={e => setPauseEnd(e.target.value)} />
              </label>
            </div>
            <label>
              Notizen
              <textarea
                value={notes}
                onChange={e => setNotes(e.target.value)}
                placeholder="Untergrund, Sicht, Abweichungen, besondere Beobachtungen …"
              />
            </label>
          </section>

          {error && (
            <div role="alert" className="alert error">
              {error}
            </div>
          )}
          {success && (
            <div role="status" className="alert success">
              <b>Mission sicher gespeichert</b>
              <span>Alle benötigten Daten für Goal 2 liegen vor.</span>
              <code>{success.id}</code>
            </div>
          )}
          {progress !== null && (
            <div className="progress">
              <div style={{width: `${progress}%`}} />
              <span>Upload und sichere Ablage · {progress}%</span>
            </div>
          )}
          <div className="requirements" aria-label="Speichervoraussetzungen">
            <span className={name.trim().length >= 3 ? 'done' : ''}>Missionsname</span>
            <span className={videos.length > 0 && videos.length <= 4 ? 'done' : ''}>1–4 Videos</span>
            <span className={videos.length > 0 && videos.length <= 4 && videos.every(video => video.terrainCategory?.trim()) ? 'done' : ''}>
              Terrainkategorien
            </span>
          </div>
          <button className="submit" disabled={progress !== null} type="submit">
            MISSION PERSISTENT SPEICHERN <span>→</span>
          </button>
        </form>

        <aside>
          <h2>Gespeicherte Missionen</h2>
          <p>Bleiben nach einem Neustart verfügbar.</p>
          {missions.length === 0 ? (
            <div className="empty">Noch keine Mission gespeichert.</div>
          ) : (
            missions.map(mission => (
              <article key={mission.id}>
                <span className="ready">GROUND TRUTH BEREIT</span>
                <h3>{mission.name}</h3>
                <p>{new Date(mission.created_at).toLocaleString('de-DE')}</p>
                <div>
                  {mission.route.length ? `${mission.route.length} Wegpunkte · ` : 'ohne Georeferenzierung · '}{mission.videos.length} Videos ·{' '}
                  {mission.videos.map(video => terrainCategoryLabel(video.terrain_category)).join(' · ')}
                </div>
                <button className="analysis-button labeling-entry" type="button" onClick={() => setLabelingMission(mission)}>
                  BEFAHRBAREN WEG LABELN →
                </button>
                <button
                  className="analysis-button secondary"
                  type="button"
                  disabled={analysisLoading !== null || mission.route.length < 2}
                  title={mission.route.length < 2 ? 'Für die Karten-Auswertung muss später eine Georeferenzierung ergänzt werden.' : undefined}
                  onClick={() => void openAnalysis(mission)}
                >
                  {mission.route.length < 2
                    ? 'KARTEN-AUSWERTUNG: GEOREFERENZIERUNG FEHLT'
                    : analysisLoading?.id === mission.id
                      ? 'AUSWERTUNG WIRD GELADEN …'
                      : 'VORHANDENE AUSWERTUNG ÖFFNEN'}
                </button>
                <code>{mission.id.slice(0, 8)}</code>
              </article>
            ))
          )}
        </aside>
      </div>
      {analysisLoading && (
        <div className="analysis-loading" role="status" aria-live="polite">
          <div className="analysis-loading-bar" />
          <b>Auswertung wird geladen</b>
          <span>{analysisLoading.name} — Evidenzdaten, Rekonstruktion und Segmentierung werden vorbereitet.</span>
        </div>
      )}
      <footer>ARIADNE OPERATIONS PLATFORM <span>·</span> EVIDENZBASIERTE AUSWERTUNG <span>·</span> KEINE SICHERHEITSFREIGABE</footer>
    </main>
  )
}
