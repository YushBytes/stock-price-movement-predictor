import { modelResults } from '../data/modelResults.js'

export default function FinalFindings() {
  const engineered = modelResults.find((m) => m.id === 'engineered')
  const maxAccuracy = Math.max(...modelResults.map((m) => m.accuracy))

  return (
    <section id="findings" className="mx-auto max-w-6xl px-6 py-20">
      <div className="max-w-2xl">
        <span className="section-eyebrow text-warn-400">Final Findings</span>
        <h2 className="mt-3 text-3xl">An honest, negative result</h2>
        <p className="mt-4 text-slate-400">
          Technical indicators did not improve this particular model setup. The engineered model did
          not beat either naive baseline.
        </p>
      </div>

      <div className="card mt-10 border-warn-500/20 p-6 sm:p-8">
        <div className="grid gap-4 sm:grid-cols-4">
          {modelResults.map((row) => (
            <div
              key={row.id}
              className={`rounded-xl border p-4 ${
                row.id === 'engineered'
                  ? 'border-warn-500/40 bg-warn-500/[0.06]'
                  : 'border-white/10 bg-white/[0.02]'
              }`}
            >
              <p className="text-xs font-medium text-slate-400">{row.name}</p>
              <p className="mt-2 font-mono text-2xl font-semibold text-white">
                {(row.accuracy * 100).toFixed(2)}%
              </p>
              <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-white/[0.06]">
                <div
                  className={`h-full rounded-full ${row.id === 'engineered' ? 'bg-warn-500' : 'bg-accent-500/70'}`}
                  style={{ width: `${(row.accuracy / maxAccuracy) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>

        <p className="mt-6 text-sm leading-relaxed text-slate-300">
          The engineered model reached <strong className="text-white">{(engineered.accuracy * 100).toFixed(2)}%</strong>{' '}
          accuracy — below Persistence, below Raw Logistic Regression, and below Majority Class,
          which remained the strongest baseline throughout this experiment. Adding technical
          indicators to raw price/volume features did not add usable signal for this model family
          and split. This is reported as-is: a rigorous experiment is allowed to produce a result
          that doesn't favor the more elaborate approach.
        </p>
      </div>
    </section>
  )
}
