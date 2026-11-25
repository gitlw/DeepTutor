/**
 * String translations consumed by UI helpers. Keep calls without options
 * separate from calls with an options object, as in i18next's overloads.
 */
export interface Translate {
  (key: string): string;
  (key: string, options: Record<string, unknown>): string;
}
