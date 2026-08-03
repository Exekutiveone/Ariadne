import '@testing-library/jest-dom/vitest'
import {cleanup} from '@testing-library/react'
import {afterEach} from 'vitest'

// Ohne `globals: true` greift das automatische Cleanup von Testing Library
// nicht; gerenderte Komponenten blieben sonst zwischen Tests derselben Datei
// im DOM stehen und Abfragen fänden mehrere Treffer.
afterEach(cleanup)

Object.defineProperty(HTMLMediaElement.prototype, 'pause', {configurable: true, value: () => undefined})
Object.defineProperty(HTMLMediaElement.prototype, 'play', {configurable: true, value: async () => undefined})
