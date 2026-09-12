import MetricBadge from './MetricBadge.jsx'
import { modelResults, bestAccuracyModel, bestF1Model } from '../data/modelResults.js'

const METRICS = [
  { key: 'accuracy', label: 'Accuracy', format: (v) => `${(v * 100).toFixed(2)}%` },
  { key: 'precision', label: 'Precision', format: (v) => `${(v * 100).toFixed(2)}%` },
  { key: 'recall', label: 'Recall', format: (v) => `${(v * 100).toFixed(2)}%` },
  { key: 'f1', label: 'F1', format: (v) => v.toFixed(4) },
]

// Computed from modelResults.js at render time -- never hardcoded.
const bestByMetric = Object.fromEntries(
  METRICS.map((m) => [m.key, modelResults.reduce((best, row) => (row[m.key] > best[m.key] ? row : best), modelResults[0]).id]),
)

export default function ModelComparison() {
  return (
    <section id="results" className="mx-auto max-w-6xl px-6 py-20">
      <div className="max-w-2xl">
        <span className="section-eyebrow">Results</span>
        <h2 className="mt-3 text-3xl">Four-way model comparison</h2>
        <p className="mt-4 text-slate-400">
          Every model below is evaluated on the identical, untouched, chronologically-final 819-row
          test window.
        </p>
      </div>

      <div className="mt-8 flex flex-col gap-3 sm:flex-row">
        <MetricBadge
          label="Best Accuracy"
          modelName={bestAccuracyModel.name}
          value={`${(bestAccuracyModel.accuracy * 100).toFixed(2)}%`}
        />
        <MetricBadge label="Best F1" modelName={bestF1Model.name} value={bestF1Model.f1.toFixed(4)} />
      </div>

      {/* Table layout: sm and up */}
      <div className="card mt-8 hidden overflow-x-auto sm:block">
        <table className="w-full min-w-[560px] border-collapse text-left">
          <thead>
            <tr className="border-b border-white/10 text-xs uppercase tracking-wide text-slate-400">
              <th scope="col" className="px-6 py-4 font-medium">
                Model
              </th>
              {METRICS.map((m) => (
                <th key={m.key} scope="col" className="px-6 py-4 font-medium">
                  {m.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {modelResults.map((row) => (
              <tr key={row.id} className="border-b border-white/5 last:border-0">
                <th scope="row" className="px-6 py-4 text-sm font-medium text-white">
                  {row.name}
                  <span className="ml-2 rounded-full border border-white/10 px-2 py-0.5 text-[10px] font-normal uppercase tracking-wide text-slate-500">
                    {row.kind}
                  </span>
                </th>
                {METRICS.map((m) => (
                  <td
                    key={m.key}
                    className={`px-6 py-4 font-mono text-sm ${
                      bestByMetric[m.key] === row.id ? 'font-semibold text-accent-400' : 'text-slate-300'
                    }`}
                  >
                    {m.format(row[m.key])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Card layout: below sm */}
      <div className="mt-8 flex flex-col gap-4 sm:hidden">
        {modelResults.map((row) => (
          <div key={row.id} className="card p-5">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-white">{row.name}</h3>
              <span className="rounded-full border border-white/10 px-2 py-0.5 text-[10px] uppercase tracking-wide text-slate-500">
                {row.kind}
              </span>
            </div>
            <dl className="mt-4 grid grid-cols-2 gap-3">
              {METRICS.map((m) => (
                <div key={m.key}>
                  <dt className="text-[11px] uppercase tracking-wide text-slate-500">{m.label}</dt>
                  <dd
                    className={`font-mono text-sm ${
                      bestByMetric[m.key] === row.id ? 'font-semibold text-accent-400' : 'text-slate-200'
                    }`}
                  >
                    {m.format(row[m.key])}
                  </dd>
                </div>
              ))}
            </dl>
          </div>
        ))}
      </div>

      <p className="mt-4 text-xs text-slate-500">
        Highlighted values are the best result for that column, derived directly from the data above.
      </p>
    </section>
  )
}
