export const TERRAIN_CATEGORY_OPTIONS = [
  {
    value: 'asphaltierte_strasse',
    label: 'Asphaltierte Straße',
    description: 'Fester, geschlossener Belag ohne lose Materialien.',
  },
  {
    value: 'schotterweg',
    label: 'Schotterweg',
    description: 'Loser oder halbfester Untergrund mit sichtbarem Schotter.',
  },
  {
    value: 'mischform',
    label: 'Mischform',
    description: 'Konkreter Streckenabschnitt mit gemischten Untergründen, etwa Schotter + Wiese.',
  },
  {
    value: 'feld_landweg_mit_spuren',
    label: 'Feld-/Landweg mit Fahrspuren',
    description: 'Niedergefahrene Spuren mit Mittelstreifen oder wechselnder Fahrspur.',
  },
  {
    value: 'walduntergrund',
    label: 'Walduntergrund',
    description: 'Waldboden, Laub, Nadelstreu, Wurzeln oder ähnliche Waldoberflächen.',
  },
  {
    value: 'wiese_flach',
    label: 'Wiese flach',
    description: 'Flache Wiese mit eher problemloser Befahrbarkeit.',
  },
  {
    value: 'wiese_hoch',
    label: 'Wiese hoch',
    description: 'Hohe Wiese mit deutlich erschwerter Fahrt.',
  },
  {
    value: 'mischkategorie',
    label: 'Mischkategorie',
    description: 'Sammelklasse für mehrere gemeinsam vorkommende Untergründe.',
  },
  {
    value: 'klar_definierter_weg',
    label: 'Klar definierter Weg',
    description: 'Eindeutig erkennbare Trasse oder Spur mit klarer Linienführung.',
  },
] as const

export function terrainCategoryLabel(value?: string | null) {
  const match = TERRAIN_CATEGORY_OPTIONS.find(option => option.value === value)
  return match?.label ?? (value?.trim() || 'Noch keine Kategorie')
}

export function terrainCategoryDescription(value?: string | null) {
  const match = TERRAIN_CATEGORY_OPTIONS.find(option => option.value === value)
  return match?.description ?? 'Die Terrainkategorie kann pro Video frei festgelegt werden.'
}
