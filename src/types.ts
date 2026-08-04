export type Point = {lat: number; lng: number}
export type VideoInput = {
  file: File
  direction: 'A_TO_B' | 'B_TO_A'
  orientation: 'PORTRAIT' | 'LANDSCAPE'
  terrainCategory?: string | null
}
export type Mission = {
  id: string
  name: string
  status: 'READY_FOR_GOAL_2'
  created_at: string
  route: Point[]
  videos: {
    id: string
    original_name: string
    size_bytes: number
    terrain_category?: string | null
    fully_not_traversable?: boolean
  }[]
}

/** Zeitspanne, waehrend des Anschauens markiert: in diesem Bereich zeigt KEIN
 *  Frame einen befahrbaren Bereich. Geht als vollstaendig negative Frames
 *  direkt ins Wegtraining. */
export type OffPathInterval = {
  id: string
  schema_version: string
  mission_id: string
  video_id: string
  start_ms: number
  end_ms: number
  note: string
  annotator: string
  created_at: string
}

export type Analysis = {
  mission_id: string
  wrap_up: Record<string, {status: string; basis: string; confidence?: number}>
  route: {
    coordinates: Point[]
    unique_length_m: {value: number; status: string}
    walked_total_m: {value: number; status: string; assumption: string}
  }
  metrics: Record<string, {value: number | null; status: string; reason?: string}>
  keyframes: {
    id: string
    video_id: string
    video_name: string
    timestamp_seconds: number
    route_fraction: number
    position: Point
    image_url: string
    sharpness: number
    features: Record<string, number>
  }[]
  observations: {
    id: string
    frame_id: string
    category: string
    label: string
    confidence: number
    value_status: string
    position: Point
    evidence_url: string
    evidence_urls: string[]
    raw_detection_count: number
    object_status: string
  }[]
  truth_rules: {ground_truth_available: boolean; frame_position_is_object_position: boolean; coverage: string; species_statement: string}
  technical: {
    runtime_seconds: number
    failed_files: number
    unassigned_keyframes: number
    map_objects_without_evidence: number
    mean_sharpness: number
  }
}

export type Reconstruction = {
  source: string
  geojson: {geometry: {coordinates: [number, number][]}}
  reference_route: {coordinates: [number, number][]}
  segments: {
    index: number
    coordinates: [[number, number], [number, number]]
    confidence: number
    status: 'secure' | 'uncertain' | 'not_reconstructed'
    spread_m: number
    evidence: {frame_id: string; image_url: string; video_name: string; timestamp_seconds: number}[]
  }[]
  traversals: {
    video_id: string
    video_name: string
    direction: 'A_TO_B' | 'B_TO_A'
    duration_seconds: number
    tracked_fraction: number
    median_matches: number
    median_inliers: number
    geojson: {type: 'LineString'; coordinates: [number, number][]}
  }[]
  metrics: {
    traversals_aligned: number
    tracked_frame_fraction: number
    median_stable_matches: number
    median_pose_inliers: number
    cross_traversal_rmse_m: number
    max_curve_deviation_m: number
    route_length_m: number
    monotonic_evidence: boolean
  }
  truth_rules: {geometry_source: string; scale: string; accuracy: string}
}

export type SegmentationDetection = {
  detection_id: string
  track_id: string
  instance_id: string
  instance_label: string
  observation_id: string
  class_id: string
  class_label: string
  confidence: number
  bbox: [number, number, number, number]
  polygon: [number, number][]
  area_ratio: number
  tracking_status: 'new' | 'continued' | 'reacquired'
  instance_status: 'tentative' | 'confirmed' | 'cluster' | 'uncertain'
  countable: boolean
  geometry_basis: string
  scores: {objectness: number; classification: number; boundary: number; temporal: number; combined: number}
  quality_flags: string[]
  observed: boolean
  estimated_extent?: [number, number, number, number]
  uncertainty_reason?: string
}

export type TraversabilityClass = 'likely_traversable' | 'limited' | 'not_traversable' | 'unknown'
export type OverlayMode = 'original' | 'vegetation' | 'ground' | 'traversability' | 'annotation' | 'labels' | 'ai_grade'
export type NormalizedPoint = [number, number]
export type TerrainMask = {width: number; height: number; rle: number[]}
export type GroundTruthValue = 0 | 1 | 2 | 3
export type GroundTruthStatus = 'draft' | 'confirmed' | 'skipped'
export type GroundTruthPolygon = {
  id: string
  class_id: string
  points: NormalizedPoint[]
  certainty?: 'certain' | 'uncertain' | 'partially_occluded'
  origin?: 'manual' | 'model_proposal' | 'manual_corrected' | 'human_confirmed'
  hard_negative?: boolean
  note?: string
  /** Zeitliche Verkettung: dieselbe Stelle über mehrere Frames. */
  tracking_id?: string | null
  carried_from_frame?: number | null
  edit?: 'new' | 'carried_unchanged' | 'carried_adjusted' | 'corrected'
}
export type RoiProfile = {
  schema_version: string
  video_id: string
  top_ignore_fraction: number | null
  bottom_ignore_fraction: number | null
  roi: GroundTruthPolygon[]
  note: string
  revision: number
  suggested?: {top_ignore_fraction: number; bottom_ignore_fraction: number}
  applies_as: string
}
export type LabelTrack = {
  tracking_id: string
  class_id: string
  first_frame: number
  last_frame: number
  frame_count: number
  adjusted_count: number
  corrected_count: number
  ended_at_frame: number | null
  class_changes: {frame_index: number; from: string; to: string}[]
  frames: {frame_index: number; edit: string; carried_from_frame: number | null; certainty: string; point_count: number}[]
}
export type LabelTracks = {
  schema_version: string
  video_id: string
  edit_labels: Record<string, string>
  totals: {
    tracks: number
    labelled_frames: number
    untracked_polygons: number
    ended_tracks: number
    adjusted_polygons: number
    corrected_polygons: number
    class_changes: number
  }
  tracks: LabelTrack[]
  note: string
}
export type GroundTruthAnnotation = {
  schema_version: string
  mission_id: string
  video_id: string
  frame_index: number
  timestamp_ms: number
  source_frame_hash: string
  mask?: TerrainMask
  polygons: GroundTruthPolygon[]
  /** Auswertungsbereich — eigene Ebene, schneidet das Bild nicht zu. */
  roi?: GroundTruthPolygon[]
  frame_size?: {width: number; height: number} | null
  status: GroundTruthStatus
  annotator: string
  notes: string
  /** Linear (der Reihe nach) oder Shuffle (gemischte Frames mehrerer Videos)
   *  — welcher Arbeitsmodus aktiv war, als dieser Frame gelabelt wurde. */
  label_mode: 'linear' | 'shuffle'
  revision: number
  updated_at: string
  statistics: {
    polygon_count: number
    point_count: number
    classes: {traversable: number}
    pixels?: {unlabelled: number; traversable: number; not_traversable: number; unknown: number}
    labelled_pixels?: number
    labelled_fraction?: number
  }
}
export type GroundTruthSummary = {
  schema_version: string
  mission_id: string
  ontology: Record<string, {value: GroundTruthValue; label: string; color: string}>
  counts: {total: number; draft: number; confirmed: number; skipped: number}
  items: (Pick<
    GroundTruthAnnotation,
    | 'video_id'
    | 'frame_index'
    | 'timestamp_ms'
    | 'source_frame_hash'
    | 'status'
    | 'annotator'
    | 'revision'
    | 'updated_at'
    | 'statistics'
    | 'label_mode'
  > & {polygons?: GroundTruthPolygon[]; mask?: TerrainMask})[]
}
export type LabelingVideo = {
  video_id: string
  original_name: string
  fps: number
  total_frames: number
  width: number
  height: number
  duration_seconds: number
  terrain_category?: string | null
  fully_not_traversable?: boolean
}
export type LabelingVideoManifest = {
  mission_id: string
  source: 'original_video_metadata_only'
  automatic_processing_started: false
  videos: LabelingVideo[]
}
export type PathModelMetrics = {
  tp: number
  tn: number
  fp: number
  fn: number
  missed_label_fraction: number
  invented_path_fraction: number
  symmetric_penalty_points: number
  symmetric_score: number
  iou: number
  dice: number
  precision: number
  recall: number
}
export type PathModelResult = {
  schema_version: string
  run_id: string
  mission_id: string
  created_at: string
  model: {
    id: string
    type: string
    hardware: 'CPU'
    cloud_used: false
    input_width: number
    feature_count: number
    random_features: number
    threshold: number
    postprocessing: string
  }
  ground_truth: {positive: string; negative: string; confirmed_frames: number; videos: number}
  split: {
    strategy: string
    train_frames: number
    validation_frames: number
    training_pixels_sampled: number
    same_frame_in_train_and_validation: false
  }
  scoring: {rule: string; threshold_selection: PathModelMetrics}
  train_metrics: PathModelMetrics
  validation_metrics: PathModelMetrics
  evidence: {
    kind: string
    video_id: string
    frame_index: number
    timestamp_ms: number
    metrics: PathModelMetrics
    image_url: string
    legend: Record<string, string>
  }[]
  runtime_seconds: number
  limitations: string[]
}
/** Abstufungsklassen der KI-Wegmaske; Werte 0-5 wie GRADE_ONTOLOGY im Backend. */
export type GradeKey = 'unrated' | 'safe' | 'good' | 'marginal' | 'risky' | 'problem'
export type GradeOntology = Record<GradeKey | string, {value: number; label: string; color: string}>
export type Grading = {
  margin: string
  threshold: number
  bands: {safe_min_margin: number; good_min_margin: number; risky_min_margin: number}
  problem_min_area_fraction: number
  problem_neighbourhood_px: number
  problem_clip_px: number
  smoothing: string
  note: string
}
export type PathPrediction = {
  schema_version: string
  model_run_id: string
  video_id: string
  frame_index: number
  timestamp_ms: number
  mask: TerrainMask
  grade_mask?: TerrainMask
  grade_ontology?: GradeOntology
  grading?: Grading
  path_fraction: number
  mean_separation: number
  confidence_note: string
  /** Korridorprüfung desselben Frames — kommt mit, statt eine zweite Inferenz zu kosten. */
  corridors?: CorridorCheck
  source: string
  evaluation?: {
    annotation_status: GroundTruthStatus
    metrics: PathModelMetrics
    comparison_mask: TerrainMask
    legend: Record<string, string>
    refinement_count: number
  }
}
export type PathTrainingJob = {
  job_id: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'interrupted'
  profile: 'quick' | 'overnight'
  duration_hours: number
  pid: number
  started_at: string
  finished_at: string | null
  candidates_completed: number
  maximum_candidates: number
  initial_run_id?: string | null
  best_run_id: string | null
  best_validation_score: number | null
  message: string
  last_candidate_run_id?: string
  last_configuration?: Record<string, number>
}
export type GlobalPathModelResult = {
  schema_version: string
  scope: 'global_cross_mission'
  run_id: string
  created_at: string
  model: {
    id: string
    type: string
    hardware: 'CPU'
    cloud_used: false
    input_width: number
    feature_count: number
    random_features: number
    threshold: number
  }
  dataset: {
    missions: {mission_id: string; name: string; confirmed_frames: number; train_frames: number; validation_frames: number}[]
    confirmed_frames: number
    videos: number
    refinements_included: number
    critical_flags_included: number
  }
  split: {
    strategy: string
    train_frames: number
    validation_frames: number
    training_pixels_sampled: number
    same_frame_in_train_and_validation: false
  }
  train_metrics: PathModelMetrics
  validation_metrics: PathModelMetrics
  evidence: {kind: string; video_id: string; frame_index: number; timestamp_ms: number; metrics: PathModelMetrics; image_url: string}[]
  runtime_seconds: number
  limitations: string[]
}
export type GlobalModelDashboardData = {
  dataset: {
    missions: {mission_id: string; name: string; confirmed_frames: number; videos: number; refinements: number; critical_flags: number}[]
    totals: {missions: number; confirmed_frames: number; videos: number; refinements: number; critical_flags: number}
  }
  model: GlobalPathModelResult | null
}
/* Labelklassen: vom Backend geliefert, damit Liste und Farben nicht doppelt gepflegt werden. */
export type LabelClass = {class_id: string; layer: string; label: string; color: string; value: number | null; description: string}
export type LabelLayer = {layer: string; label: string; exclusive: boolean; classes: LabelClass[]}
export type LabelOntology = {
  schema_version: string
  unlabelled: {key: string; value: number; label: string; color: string}
  layers: LabelLayer[]
  certainty: {value: string; label: string}[]
  origin: {value: string; label: string}[]
  notes: string[]
}

/* Run-Registry: ein Run ist genau ein Originalvideo. */
export type RunStatus = 'unlabeled' | 'queued_for_labeling' | 'labeled' | 'training_ready'
export type RegistryRun = {
  run_id: string
  mission_id: string
  video_id: string
  mission_name: string
  original_name: string
  video_available: boolean
  size_bytes: number
  status: RunStatus
  status_label: string
  terrain_category: string | null
  note: string
  discovered_at: string
  updated_at: string
}
export type RegistryListing = {
  schema_version: string
  database: string
  scan: {added: number; removed: string[]; total: number | null}
  statuses: {value: RunStatus; label: string}[]
  counts: Record<RunStatus, number>
  totals: {runs: number; with_terrain_category: number; missing_video_file: number}
  terrain_categories: {terrain_category: string; runs: number}[]
  runs: RegistryRun[]
  note: string
}

/* Korridor-Bewertung im Bildraum: deterministische Geometrie auf der Maske, kein ML. */
export type CorridorStatus = 'free' | 'blocked' | 'uncertain'
export type CorridorId = 'mitte' | 'rechts' | 'links'
/** geometry/trajectory sind auf das Bild normiert (0..1), nicht in Pixeln des Modellrasters. */
export type CorridorResult = {
  corridor: CorridorId
  label: string
  meaning: string
  status: CorridorStatus
  status_label: string
  reason: string
  rows: {evaluated: number; free: number; uncertain: number; blocked: number}
  bottom_center_x: number
  geometry: {center: number[][]; left: number[][]; right: number[][]}
  trajectory: {points: number[][]; rows: number; source: string}
}
export type ProposedTrajectory = {
  corridor: CorridorId
  label: string
  status: CorridorStatus
  status_label: string
  points: number[][]
  source: string
  note: string
}
export type StoredTrajectory = {
  schema_version: string
  mission_id: string
  video_id: string
  frame_index: number
  timestamp_ms: number
  points: number[][]
  corridor: CorridorId | null
  origin: 'model_proposal' | 'manual_edit' | 'manual'
  note: string
  annotator: string
  coordinate_space: string
  revision: number
  created_at: string
  updated_at: string
}
export type CorridorCheck = {
  schema_version: string
  kind: string
  mask_size: {width: number; height: number}
  decomposition: {
    vanishing_point: {x: number; y: number; source: string; rows_used: number; residual_px: number | null}
    relevant_triangle: number[][]
    irrelevant_zone: {kind: string; first_evaluated_row: number; rows_skipped: number; image_fraction_skipped: number; reason: string}
    evaluated_rows: number
    vanishing_point_normalized: number[]
    relevant_triangle_normalized: number[][]
    first_evaluated_row_normalized: number
  }
  proposed_trajectory: ProposedTrajectory | null
  strip: {
    vehicle_width_m: number
    clearance_m: number
    required_width_m: number
    ground_width_at_bottom_m: number
    required_width_px_at_bottom: number
    scaling: string
    search_band_factor: number
  }
  corridors: CorridorResult[]
  graded_input: boolean
  limitations: string[]
  model_run_id: string
  mission_id: string
  video_id: string
  frame_index: number
  timestamp_ms: number
  path_fraction: number
  source: string
}

/* Videobasierte Terrainklassifizierung. Die Klasse eines Frames ist immer die
   aktuell am Video gesetzte Kategorie; Frames tragen kein eigenes Label. */
export type TerrainClassMetrics = {terrain_category: string; support: number; precision: number; recall: number; f1: number}
export type TerrainMetrics = {
  frames: number
  accuracy: number
  balanced_accuracy: number
  mean_confidence: number
  uncertain_frames: number
  uncertain_fraction: number
  accuracy_on_confident: number
  confusion_matrix: number[][]
  per_class: TerrainClassMetrics[]
}
export type TerrainSplitPart = {videos: number; frames: number; classes: string[]; video_ids: string[]; all_classes: string[]}
export type TerrainDatasetVideo = {
  mission_id: string
  mission_name: string
  video_id: string
  original_name: string
  terrain_category: string | null
  frames?: number
}
export type TerrainModelResult = {
  schema_version: string
  scope: 'video_terrain_classification'
  kind: 'training'
  run_id: string
  created_at: string
  model: {
    id: string
    type: string
    hardware: 'CPU'
    cloud_used: false
    input_width: number
    grid: number
    feature_count: number
    random_features: number
    softmax_scale: number
    confidence_threshold: number
  }
  classes: string[]
  dataset: {
    frame_stride: number
    label_source: string
    categorized_videos: number
    uncategorized_videos: number
    videos: TerrainDatasetVideo[]
    frames: number
  }
  split: {
    strategy: string
    random_frame_split_used: false
    same_video_in_multiple_parts: false
    train: TerrainSplitPart
    validation: TerrainSplitPart
    test: TerrainSplitPart | null
    notes: string[]
  }
  calibration: {softmax_scale: number; selected_on: string; negative_log_likelihood: number}
  train_metrics: TerrainMetrics
  validation_metrics: TerrainMetrics
  test_metrics: TerrainMetrics | null
  runtime_seconds: number
  limitations: string[]
}
/** predicted_category ist null, sobald die Konfidenz unter dem Schwellenwert liegt — dann bleibt nur top_category als unverbindliche Vermutung. */
export type TerrainFramePrediction = {
  frame_index: number
  timestamp_ms: number
  predicted_category: string | null
  top_category: string
  confidence: number
  uncertain: boolean
  scores: Record<string, number>
}
export type TerrainPredictionRun = {
  schema_version: string
  kind: 'prediction'
  run_id: string
  created_at: string
  model_run_id: string
  mission_id: string
  video_id: string
  original_name: string
  video_terrain_category: string | null
  frame_stride: number
  confidence_threshold: number
  classes: string[]
  summary: {
    frames: number
    uncertain_frames: number
    uncertain_fraction: number
    mean_confidence: number
    dominant_category: string
    counts: Record<string, number>
    matches_video_category: number | null
  }
  frames: TerrainFramePrediction[]
  runtime_seconds: number
  limitations: string[]
}
export type TerrainDashboardData = {
  dataset: {
    videos: TerrainDatasetVideo[]
    classes: {terrain_category: string; videos: number; missions: number}[]
    totals: {categorized_videos: number; uncategorized_videos: number; classes: number; missions: number}
    label_source: string
  }
  model: TerrainModelResult | null
  runs: {
    active_run_id: string | null
    training_runs: {
      run_id: string
      kind: string
      created_at: string
      active: boolean
      classes: string[]
      frame_stride: number
      confidence_threshold: number
      validation_accuracy: number
      test_accuracy: number | null
      runtime_seconds: number
    }[]
    prediction_runs: {
      run_id: string
      kind: string
      created_at: string
      model_run_id: string
      mission_id: string
      video_id: string
      original_name: string
      frame_stride: number
      confidence_threshold: number
      predicted_frames: number
      uncertain_frames: number
      dominant_category: string
    }[]
  }
}
export type GlobalVideoAnalysisStatus = {
  job_id: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'interrupted'
  model_run_id: string
  mission_id: string
  video_id: string
  pid: number
  started_at: string
  finished_at: string | null
  processed_frames: number
  total_frames: number
  progress: number
  elapsed_seconds: number
  eta_seconds: number | null
  message: string
}
/** grade_mask ist optional: Analysen aus der Zeit vor Phase 3 enthalten nur die Binärmaske. */
export type GlobalVideoAnalysisFrame = {
  frame_index: number
  timestamp_ms: number
  mask: TerrainMask
  grade_mask?: TerrainMask
  path_fraction: number
  evaluation?: {metrics: PathModelMetrics; comparison_mask: TerrainMask; refinement_count: number}
}
export type GlobalVideoAnalysisResult = {
  schema_version: string
  model_run_id: string
  mission_id: string
  video_id: string
  fps: number
  total_frames: number
  width: number
  height: number
  analyzed_frames: number
  runtime_seconds: number
  frames: GlobalVideoAnalysisFrame[]
}
export type TerrainRegion = {
  region_id: string
  class_id: TraversabilityClass
  polygon: NormalizedPoint[]
  confidence: number
  area_ratio: number
  reasons: string[]
}

export type TerrainFrameEvaluation = {
  source_video_id: string
  source_frame_index: number
  source_frame_timestamp_ms: number
  source_frame_hash: string
  ground: {
    mask: TerrainMask
    regions: {
      region_id: string
      class_id: 'ground'
      polygon: NormalizedPoint[]
      confidence: number
      area_ratio: number
      reasons: string[]
    }[]
    confidence: number
    visible_ratio: number
    source: 'current_video_frame_inference' | string
  }
  traversability: {
    mask: TerrainMask
    regions: TerrainRegion[]
    class_coverage: Record<TraversabilityClass, number>
    overall_class: TraversabilityClass
    overall_confidence: number
  }
  corridor: {
    status: 'available' | 'uncertain' | 'unavailable'
    polygon: NormalizedPoint[]
    centerline: NormalizedPoint[]
    confidence: number
    minimum_width_ratio: number
    minimum_width_m: number | null
    stability_px: number
    stable_frames: number
    green_support_fraction: number
    source_frame_timestamp_ms: number
    temporally_smoothed: boolean
    reasons: string[]
  }
  factors: {
    free_width_score: number
    obstacle_clearance_score: number
    connectivity_score: number
    smoothness_score: number
    bottleneck_clearance_score: number
    visibility_score: number
    calibration_score: number
    temporal_stability_score: number
  }
  quality: {blur_score: number; exposure_score: number; motion_inliers: number; unknown_ratio: number}
  evidence: {representative: boolean; reasons: string[]; image_url?: string; overlay_url?: string}
}

export type SegmentationFrame = {
  video_id: string
  video_name: string
  frame_index: number
  timestamp_ms: number
  quality: {sharpness: number; motion_inliers: number}
  detections: SegmentationDetection[]
  /** Optional so persisted Goal-4 runs from before terrain analysis remain viewable. */
  terrain?: TerrainFrameEvaluation
}

export type SegmentationTrack = {
  track_id: string
  instance_label: string
  class_id: string
  first_timestamp_ms: number
  last_timestamp_ms: number
  observation_count: number
  max_confidence: number
  representative_detection_id: string
  instance_status: 'tentative' | 'confirmed' | 'cluster' | 'uncertain'
  countable: boolean
}

export type Segmentation = {
  schema_version: string
  run_id: string
  mission_id: string
  model: {adapter: string; model_id: string; version: string; hardware: string}
  terrain_model?: {adapter: string; model_id: string; version: string; hardware: string; weights?: string; license?: string}
  configuration: {
    analysis_hz: number
    input_width: number
    min_area_ratio: number
    confirmation_hits: number
    max_track_gap_frames: number
    confidence_meaning: string
  }
  ontology: Record<string, {label: string; color: string; countable: boolean; default_enabled: boolean}>
  terrain_ontology?: Record<TraversabilityClass, {label: string; color: string; value: number}>
  vehicle_configuration?: {
    width_m: number
    safety_margin_per_side_m: number
    required_width_m: number
    source: 'configured_default' | 'configured' | 'default_assumption' | string
    near_field_width_m?: number
    metric_calibration?: string
  }
  terrain_configuration?: {
    near_field_width_m: number
    metric_calibration: string
    source_frames: string
    temporal_motion_minimum_inliers: number
  }
  videos: {
    video_id: string
    video_name: string
    duration_seconds: number
    fps: number
    width: number
    height: number
    analysis_interval_ms: number
    frames: SegmentationFrame[]
    tracks: SegmentationTrack[]
    counts: {visible_individuals_latest_frame: number; confirmed_unique_per_video: {tree: number; shrub: number}}
    metrics: {
      analyzed_frames: number
      raw_detections: number
      tracks: number
      confirmed_tree_instances: number
      confirmed_shrub_instances: number
      average_track_length_frames: number
      short_track_fraction: number
      empty_frame_fraction: number
      median_motion_inliers: number
      inference_seconds: number
      terrain_frames?: number
      terrain_unique_source_hashes?: number
      terrain_source_hash_unique_fraction?: number
      terrain_unique_masks?: number
      terrain_mask_unique_fraction?: number
      terrain_mask_transition_fraction?: number
      terrain_masks_vary?: boolean
      corridor_available_fraction?: number
      corridor_uncertain_fraction?: number
      median_corridor_stability_px?: number
      terrain_overall_class_frames?: Record<TraversabilityClass, number>
      representative_evidence_frames?: number
    }
  }[]
  counts: {confirmed_unique_per_video_sum: {tree: number; shrub: number}; mission_unique: null; mission_unique_reason: string}
  metrics: {
    runtime_seconds: number
    analyzed_frames: number
    raw_detections: number
    tracks: number
    confirmed_tree_instances: number
    confirmed_shrub_instances: number
    empty_frame_fraction: number
    average_track_length_frames: number
    terrain_frames?: number
    terrain_unique_source_hashes?: number
    terrain_source_hash_unique_fraction?: number
    terrain_unique_masks?: number
    terrain_mask_unique_fraction?: number
    terrain_mask_transition_fraction?: number
    terrain_masks_vary?: boolean
    corridor_available_fraction?: number
    corridor_uncertain_fraction?: number
    terrain_overall_class_frames?: Record<TraversabilityClass, number>
    representative_evidence_frames?: number
  }
  truth_rules: {
    ground_truth_available: boolean
    species_inference: boolean
    navigation_grade: boolean
    individual_definition: string
    cross_video_fusion?: boolean
    overlays?: string
    terrain_inference?: string
    terrain_uncertainty?: string
    metric_scale?: string
    safety_disclaimer?: string
  }
}
