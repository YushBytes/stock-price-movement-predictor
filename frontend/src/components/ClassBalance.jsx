import ProportionBar from './ProportionBar.jsx'
import { testClassBalance } from '../data/classBalance.js'

export default function ClassBalance() {
  const segments = [
    {
      label: 'UP',
      count: testClassBalance.up.count,
      pct: testClassBalance.up.pct,
      className: 'bg-accent-500/70 text-ink-950',
      dotClassName: 'bg-accent-500',
    },
    {
      label: 'DOWN_OR_FLAT',
      count: testClassBalance.downOrFlat.count,
      pct: testClassBalance.downOrFlat.pct,
      className: 'bg-warn-500/60 text-ink-950',
      dotClassName: 'bg-warn-500',
    },
  ]

  return (
    <section className="mx-auto max-w-6xl px-6 py-20">
      <div className="max-w-2xl">
        <span className="section-eyebrow">Class Balance</span>
        <h2 className="mt-3 text-3xl">Test-set class distribution</h2>
        <p className="mt-4 text-slate-400">
          The test set ({testClassBalance.total} observations) is moderately imbalanced toward UP, so
          accuracy is interpreted alongside Precision, Recall, and F1 — not on its own.
        </p>
      </div>

      <div className="card mt-10 p-8">
        <ProportionBar segments={segments} />
      </div>
    </section>
  )
}
