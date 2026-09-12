import Timeline from './Timeline.jsx'
import { splitDates } from '../data/keyStats.js'

const STEPS = [
  { label: 'Historical Data', detail: 'Full SPY daily history, 2005 onward' },
  {
    label: 'Training',
    range: `${splitDates.train.start} → ${splitDates.train.end}`,
    detail: `${splitDates.train.rows.toLocaleString()} rows`,
  },
  {
    label: 'Validation',
    range: `${splitDates.validation.start} → ${splitDates.validation.end}`,
    detail: `${splitDates.validation.rows.toLocaleString()} rows`,
  },
  {
    label: 'Held-Out Test',
    range: `${splitDates.test.start} → ${splitDates.test.end}`,
    detail: `${splitDates.test.rows.toLocaleString()} rows, never used for fitting`,
  },
]

const RULES = [
  {
    title: 'Future Close is used only to construct the label',
    body: 'Target[t] = 1 if Close[t+1] > Close[t]. Close[t+1] never becomes a feature.',
  },
  {
    title: 'Future observations are never feature inputs',
    body: 'Every raw and engineered feature uses only shift(k) with a positive k — data already observed by time t.',
  },
  {
    title: 'No centered rolling windows',
    body: 'All moving averages and volatility windows look strictly backward, verified directly against the indicator library’s source.',
  },
  {
    title: 'Scaler fitted only on training data',
    body: 'StandardScaler.fit() sees the training partition only; validation and test are transformed, never fit.',
  },
  {
    title: 'No random train/test split',
    body: 'Partitions are sliced by date order alone — never shuffled, never sklearn.train_test_split.',
  },
]

export default function LeakagePrevention() {
  return (
    <section id="leakage" className="mx-auto max-w-6xl px-6 py-20">
      <div className="max-w-2xl">
        <span className="section-eyebrow">Leakage Prevention</span>
        <h2 className="mt-3 text-3xl">Test data stays unseen until evaluation</h2>
        <p className="mt-4 text-slate-400">
          The test window is touched exactly once — for final evaluation. It never influences target
          construction, feature engineering, scaling, or model fitting.
        </p>
      </div>

      <div className="mt-10">
        <Timeline steps={STEPS} />
      </div>

      <div className="mt-10 grid gap-4 sm:grid-cols-2">
        {RULES.map((rule, index) => (
          <div key={rule.title} className="card flex gap-4 p-5">
            <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-accent-500/15 font-mono text-xs font-semibold text-accent-400">
              {index + 1}
            </span>
            <div>
              <h3 className="text-sm font-semibold text-white">{rule.title}</h3>
              <p className="mt-1 text-sm leading-relaxed text-slate-400">{rule.body}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}
