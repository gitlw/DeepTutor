import { expect, it } from 'vitest'

import { describeReminder } from '@/features/calendar/model/describe'
import { catalogTranslate } from '@/hooks/useTopicSourceLibrary'
import { ensureLanguage, initI18n } from '@/i18n/init'

it('accepts the native i18next translator for calendar labels and interpolation', async () => {
  const i18n = initI18n()
  await ensureLanguage('zh')
  const t: Parameters<typeof describeReminder>[3] = i18n.getFixedT('zh', 'app')

  expect(describeReminder(0, false, 'zh', t)).toBe('开始时')
  expect(describeReminder(5, false, 'zh', t)).toBe('提前 5 分钟')
})

it('forwards catalog translations with and without interpolation options', () => {
  const t = initI18n().getFixedT('en', 'app')
  const translate = catalogTranslate(t, false)

  expect(translate('资源', 'Assigned resources')).toBe('Assigned resources')
  expect(translate('资源', '{{count}} resources selected', { count: 2 })).toBe('2 resources selected')
})
