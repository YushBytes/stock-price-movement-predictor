const LIMITATIONS = [
  'One instrument — SPY only. Results may not generalize to other symbols or asset classes.',
  'One fixed chronological split — a different date range could produce different numbers.',
  'One primary model family — Logistic Regression, with no hyperparameter search attempted.',
  'No statistical significance testing was performed on any metric difference reported.',
  'No claim of profitability, reliable prediction, or real-world trading performance is made.',
]

export default function Limitations() {
  return (
    <section className="mx-auto max-w-6xl px-6 py-20">
      <div className="max-w-2xl">
        <span className="section-eyebrow">Limitations</span>
        <h2 className="mt-3 text-3xl">What this experiment doesn’t show</h2>
        <p className="mt-4 text-slate-400">
          This result describes one specific experiment. It should not be generalized to all
          markets, instruments, or machine-learning approaches.
        </p>
      </div>

      <ul className="mt-10 grid gap-3 sm:grid-cols-2">
        {LIMITATIONS.map((item) => (
          <li key={item} className="card flex gap-3 p-5 text-sm leading-relaxed text-slate-300">
            <span aria-hidden="true" className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-slate-500" />
            {item}
          </li>
        ))}
      </ul>
    </section>
  )
}
